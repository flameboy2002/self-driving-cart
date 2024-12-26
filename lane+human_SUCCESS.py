#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
import sys
import cv2
import torch
import depthai as dai
import numpy as np

# Import necessary utilities for YOLOPv2
from utils.utils import (
    time_synchronized, select_device, increment_path,
    non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, show_seg_result
)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='/home/ele491/YOLOPv2/yolopv2_weights.pt', help='Path to YOLOPv2 weights')
parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels) - should match model input size')
parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='cpu', help='device to run on, i.e., "cpu" or "cuda"')
args = parser.parse_args()

# Path to YOLOv4 Tiny model for human detection
nnPath = str((Path(__file__).parent / Path('/home/ele491/depthai/depthai-python/examples/models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())

# YOLOv4 Tiny label map
labelMap = [
    "person"#, "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    #"truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    #"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
]

# Set up DepthAI pipeline with YOLOv4 Tiny for human detection and camera preview for YOLOPv2 lane detection
def setup_pipeline():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    nnOut.setStreamName("nn")

    # Camera properties
    camRgb.setPreviewSize(416, 416)  # Set to 416x416 to match model
    camRgb.setInterleaved(False)
    camRgb.setFps(15)

    # YOLOv4 Tiny Network properties
    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(80)
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
    detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
    detectionNetwork.setIouThreshold(0.5)
    detectionNetwork.setBlobPath(nnPath)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Linking camera output to models
    camRgb.preview.link(detectionNetwork.input)
    detectionNetwork.passthrough.link(xoutRgb.input)
    detectionNetwork.out.link(nnOut.input)

    return pipeline

# Normalize detection bounding box coordinates
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

# Function to draw human detections
def draw_human_detections(frame, detections):
    color = (255, 0, 0)
    for detection in detections:
        if labelMap[detection.label] == "person":  # Only show "person" detections
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, "Person", (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)  #here send command to arduino to stop and apply brakes, if not detectesd s
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

# Main detection function
def detect(device, model, img_size, conf_thres, iou_thres):
    # Set up output queue for camera frames and human detections
    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)  # Reduced maxSize to 1
    qDet = device.getOutputQueue(name="nn", maxSize=1, blocking=False)  # Reduced maxSize to 1

    devicee = select_device(args.device)
    frame = None  # Initialize frame to ensure it exists

    while True:
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()

        # YOLOPv2 Lane Detection
        if inRgb is not None:
            frame = inRgb.getCvFrame()

            # Preprocess frame for YOLOPv2
            img = cv2.resize(frame, (img_size, img_size))
            img = torch.from_numpy(img).to(devicee)
            img = img.float() / 255.0  # Normalize to 0-1 range
            img = img.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

            # Run YOLOPv2 lane detection
            with torch.no_grad():
                t1 = time_synchronized()
                [pred, anchor_grid], seg, ll = model(img)
                t2 = time_synchronized()

                # Apply NMS and post-process lane detection
                pred = split_for_trace_model(pred, anchor_grid)
                pred = non_max_suppression(pred, conf_thres, iou_thres)
                da_seg_mask = driving_area_mask(seg)
                ll_seg_mask = lane_line_mask(ll)
                da_seg_mask_resized = cv2.resize(da_seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                ll_seg_mask_resized = cv2.resize(ll_seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                show_seg_result(frame, (da_seg_mask_resized, ll_seg_mask_resized), is_demo=True)

                # Display YOLOPv2 lane detection inference time
                cv2.putText(frame, f'Lane Inference time: {(t2 - t1) * 1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # YOLOv4 Tiny Human Detection
        if inDet is not None and frame is not None:  # Ensure frame is available for drawing
            detections = inDet.detections
            draw_human_detections(frame, detections)

        # Show final frame if available
        if frame is not None:
            cv2.imshow("Lane and Human Detection", frame)

        # Updated delay for stability
        if cv2.waitKey(10) == ord('q'):
            break

# Main function to start pipeline and run detection
if __name__ == '__main__':
    # Suppress the Qt platform warning
    import os
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    device = select_device(args.device)
    model = torch.jit.load(args.weights, map_location=device)
    model.eval()

    # Start DepthAI pipeline and run detection
    with dai.Device(setup_pipeline()) as device:
        detect(device, model, args.img_size, args.conf_thres, args.iou_thres)

    cv2.destroyAllWindows()