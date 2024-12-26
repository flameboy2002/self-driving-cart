#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
import cv2
import torch
import depthai as dai
import numpy as np
import serial

# Import necessary utilities for YOLOPv2
from utils.utils import (
    time_synchronized, select_device, increment_path,
    non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, show_seg_result
)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='/home/ele491/YOLOPv2/yolopv2_weights.pt', help='Path to YOLOPv2 weights')
parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels) - should be a multiple of 32')
parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='cpu', help='device to run on, i.e., "cpu" or "cuda"')
parser.add_argument('--nnPath', type=str, default='/home/ele491/depthai/depthai-python/examples/models/mobilenet-ssd_openvino_2021.4_6shave.blob', help="Path to mobilenet detection network blob")
args = parser.parse_args()

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Initialize serial communication with Arduino
try:
    arduino_serial = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
    print("Serial connection to Arduino established!")
except Exception as e:
    print(f"Failed to open serial connection: {e}")
    exit()

# Set up DepthAI pipeline
def setup_pipeline():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nnOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    nnOut.setStreamName("nn")

    # Camera properties
    camRgb.setPreviewSize(300, 300)
    camRgb.setInterleaved(False)
    camRgb.setFps(15)

    # MobileNet properties
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(args.nnPath)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    # Linking
    camRgb.preview.link(xoutRgb.input)
    camRgb.preview.link(nn.input)
    nn.out.link(nnOut.input)

    return pipeline

def detect(device, model, img_size, conf_thres, iou_thres):
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    devicee = select_device(args.device)

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    while True:
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()

            # Lane detection
            img = cv2.resize(frame, (img_size, img_size))
            img = torch.from_numpy(img).to(devicee)
            img = img.float() / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                t1 = time_synchronized()
                [pred, anchor_grid], seg, ll = model(img)
                t2 = time_synchronized()

                pred = split_for_trace_model(pred, anchor_grid)
                pred = non_max_suppression(pred, conf_thres, iou_thres)

                da_seg_mask = driving_area_mask(seg)
                ll_seg_mask = lane_line_mask(ll)

                da_seg_mask_resized = cv2.resize(da_seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                ll_seg_mask_resized = cv2.resize(ll_seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                ll_seg_mask_resized = ll_seg_mask_resized.astype(np.uint8)
                contours, _ = cv2.findContours(ll_seg_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                lane_lines = []
                for contour in contours:
                    if cv2.contourArea(contour) > 50:
                        lane_lines.append(contour)

                for contour in lane_lines:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

                heights, widths = frame.shape[:2]
                lane_centers = []
                for h in range(0, heights, 10):
                    x_coords = []
                    for contour in lane_lines:
                        for point in contour:
                            if abs(point[0][1] - h) < 5:
                                x_coords.append(point[0][0])
                    if len(x_coords) >= 2:
                        center_x = sum(x_coords) // len(x_coords)
                        lane_centers.append((center_x, h))

                if lane_centers:
                    avg_center_x = int(np.mean([x for x, _ in lane_centers]))
                else:
                    avg_center_x = -1

                if avg_center_x != -1:
                    for x, y in lane_centers:
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                    cv2.line(frame, (avg_center_x, 0), (avg_center_x, heights), (255, 255, 0), 2)

                image_center_x = frame.shape[1] // 2
                cv2.line(frame, (image_center_x, 0), (image_center_x, heights), (0, 0, 255), 2)

                if avg_center_x != -1:
                    offset = avg_center_x - image_center_x

                    if offset > 20:
                        command = "CCW\n"
                        direction_text = "CCW"
                    elif offset < -20:
                        command = "CW\n"
                        direction_text = "CW"
                    else:
                        command = "STRAIGHT\n"
                        direction_text = "Straight"
                else:
                    command = "STOP\n"
                    direction_text = "Stop: No Lane Detected"

                try:
                    arduino_serial.write(command.encode())
                    print(f"Sent to Arduino: {command.strip()}")
                except Exception as e:
                    print(f"Failed to send data to Arduino: {e}")

                cv2.putText(frame, direction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'Inference time: {(t2 - t1) * 1000:.2f} ms', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Human detection
            if inDet is not None:
                detections = inDet.detections
                for detection in detections:
                    if labelMap[detection.label] == "person":
                        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        cv2.putText(frame, f"Person {int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 20),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            break

# Main function to start the pipeline and run detection
if __name__ == '__main__':
    # Load model and set device
    device = select_device(args.device)
    model = torch.jit.load(args.weights, map_location=device)
    model.eval()

    # Start DepthAI pipeline and run detection
    with dai.Device(setup_pipeline()) as device:
        detect(device, model, args.img_size, args.conf_thres, args.iou_thres)

    cv2.destroyAllWindows()
    arduino_serial.close()