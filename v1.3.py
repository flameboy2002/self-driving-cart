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
args = parser.parse_args()

# Initialize serial communication with Arduino
#try:
 #   arduino_serial = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
  #  time.sleep(2)  # Wait for Arduino to initialize
   # print("Serial connection to Arduino established!")
#except Exception as e:
 #   print(f"Failed to open serial connection: {e}")
  #  exit()

# Set up DepthAI pipeline
def setup_pipeline():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")

    # Camera properties
    camRgb.setPreviewSize(args.img_size, args.img_size)
    camRgb.setInterleaved(False)
    camRgb.setFps(15)

    # Linking camera output to stream
    camRgb.preview.link(xoutRgb.input)

    return pipeline

def detect(device, model, img_size, conf_thres, iou_thres):
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    devicee = select_device(args.device)
    while True:
        inRgb = qRgb.tryGet()
        if inRgb is not None:
            frame = inRgb.getCvFrame()

            # Preprocess image for YOLOPv2 model
            img = cv2.resize(frame, (img_size, img_size))
            img = torch.from_numpy(img).to(devicee)
            img = img.float() / 255.0  # Normalize to 0 - 1 range
            img = img.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

            # Run inference
            with torch.no_grad():
                t1 = time_synchronized()
                [pred, anchor_grid], seg, ll = model(img)
                t2 = time_synchronized()

                # Apply NMS
                pred = split_for_trace_model(pred, anchor_grid)
                pred = non_max_suppression(pred, conf_thres, iou_thres)

                # Post-process segmentation masks
                da_seg_mask = driving_area_mask(seg)
                ll_seg_mask = lane_line_mask(ll)

                # Resize masks to match frame size
                da_seg_mask_resized = cv2.resize(da_seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                ll_seg_mask_resized = cv2.resize(ll_seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Extract lane coordinates
                ll_seg_mask_resized = ll_seg_mask_resized.astype(np.uint8)  # Convert to 8-bit unsigned integer
                contours, _ = cv2.findContours(ll_seg_mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                lane_lines = []
                for contour in contours:
                    if cv2.contourArea(contour) > 50:  # Filter small contours
                        lane_lines.append(contour)

                # Draw lane lines
                for contour in lane_lines:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # Green for lane lines

                # Compute Lane Center
                heights, widths = frame.shape[:2]
                lane_centers = []
                for h in range(0, heights, 10):  # Sample vertical positions
                    x_coords = []
                    for contour in lane_lines:
                        for point in contour:
                            if abs(point[0][1] - h) < 5:  # Check vertical alignment
                                x_coords.append(point[0][0])
                    if len(x_coords) >= 2:  # Ensure there are at least two points to find a center
                        center_x = sum(x_coords) // len(x_coords)
                        lane_centers.append((center_x, h))

                if lane_centers:
                    avg_center_x = int(np.mean([x for x, _ in lane_centers]))
                else:
                    avg_center_x = -1  # No center detected

                # Draw Lane Center
                if avg_center_x != -1:
                    for x, y in lane_centers:
                        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blue for lane center points
                    cv2.line(frame, (avg_center_x, 0), (avg_center_x, heights), (255, 255, 0), 2)  # Yellow lane center line

                # Draw Image Center
                image_center_x = frame.shape[1] // 2
                cv2.line(frame, (image_center_x, 0), (image_center_x, heights), (0, 0, 255), 2)  # Red image center line

                # Calculate Steering Direction
                if avg_center_x != -1:
                    offset = avg_center_x - image_center_x  # Positive if right, negative if left

                    if offset > 20:  # Lane center is significantly right of image center
                        command = "CCW\n"  # Clockwise
                        direction_text = "CCW"
                    elif offset < -20:  # Lane center is significantly left of image center
                        command = "CW\n"  # Counter-Clockwise
                        direction_text = "CW"
                    else:  # Lane center is aligned with image center
                        command = "STRAIGHT\n"
                        direction_text = "Straight"
                else:
                    command = "STOP\n"  # No lane detected
                    direction_text = "Stop: No Lane Detected"

                # Send Command to Arduino
                #try:
                 #   arduino_serial.write(command.encode())
                  #  print(f"Sent to Arduino: {command.strip()}")
                #except Exception as e:
                 #   print(f"Failed to send data to Arduino: {e}")

                # Display direction text on the frame
                cv2.putText(frame, direction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Display inference time
                cv2.putText(frame, f'Inference time: {(t2 - t1) * 1000:.2f} ms', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Show frame with annotations
            cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            break

# Main function to start the pipeline and run detection
if __name__ == '__main__':
    # Load model and set device
    device = select_device(args.device)
    model = torch.jit.load(args.weights, map_location=device)
    model.eval()  # Set to evaluation mode

    # Start DepthAI pipeline and run detection
    with dai.Device(setup_pipeline()) as device:
        detect(device, model, args.img_size, args.conf_thres, args.iou_thres)

    cv2.destroyAllWindows()
    arduino_serial.close()  # Close serial connection