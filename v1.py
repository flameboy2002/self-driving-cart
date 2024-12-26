#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
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
parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels) - should be a multiple of 32')
parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='cpu', help='device to run on, i.e., "cpu" or "cuda"')
args = parser.parse_args()

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

# Main detection function
def detect(device, model, img_size, conf_thres, iou_thres):
    # Set up output queue for camera frames
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

                # Show results on frame
                show_seg_result(frame, (da_seg_mask_resized, ll_seg_mask_resized), is_demo=True)

                # Display inference time
                cv2.putText(frame, f'Inference time: {(t2 - t1) * 1000:.2f} ms', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Display frame
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