#!/usr/bin/env python3

import argparse
import cv2
import depthai as dai
import numpy as np
import serial
import time
import torch
from pathlib import Path
from utils.general import non_max_suppression, split_for_trace_model
from utils.torch_utils import select_device, time_synchronized
from utils.lane_detection import driving_area_mask, lane_line_mask, calculate_lane_offset

# Constants and Configuration
nnPathDefault = str((Path(__file__).parent / Path('../models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-s', '--sync', action="store_true", help="Sync RGB output with NN output", default=False)
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
args = parser.parse_args()

if not Path(nnPathDefault).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Serial Setup
try:
    ser1 = serial.Serial("/dev/serial0", 9600, timeout=1)
    print("Serial connection to Arduino established!")
except Exception as e:
    print(f"Failed to open serial connection: {e}")
    exit()

# Pipeline Setup
def create_pipeline():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    nnOut.setStreamName("nn")

    # Properties
    camRgb.setPreviewSize(300, 300)
    camRgb.setInterleaved(False)
    camRgb.setFps(40)
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(args.nnPath)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    # Linking
    if args.sync:
        nn.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)

    camRgb.preview.link(nn.input)
    nn.out.link(nnOut.input)

    return pipeline

# Motor Control
def motorcontrol(command):
    try:
        ser1.write(command.encode())
        print(f"Sent command: {command.strip()}")
        if ser1.in_waiting > 0:
            response = ser1.readline().decode('utf-8').strip()
            print(f"Received from Arduino: {response}")
    except Exception as e:
        print(f"Error in motorcontrol: {e}")

# Braking Process
def process_braking(detections, run_state):
    person_detected = False
    for detection in detections:
        if detection.label == 15:  # '15' corresponds to "person"
            confidence = detection.confidence
            print(f"Person detected with confidence: {confidence:.2f}")
            if confidence >= 0.7:
                person_detected = True
                if not run_state['brake']:
                    motorcontrol("BRAKE\n")
                    run_state['brake'] = True
                break

    if not person_detected and run_state['brake']:
        motorcontrol("RELEASE\n")
        run_state['brake'] = False

# Steering Process
def process_steering(offset, run_state):
    if offset > 20:
        if run_state['steering'] != "CCW":
            motorcontrol("CCW\n")
            run_state['steering'] = "CCW"
    elif offset < -20:
        if run_state['steering'] != "CW":
            motorcontrol("CW\n")
            run_state['steering'] = "CW"
    else:
        if run_state['steering'] != "STALL":
            motorcontrol("STALL\n")
            run_state['steering'] = "STALL"

# Main Detection Loop
def detect(device, model, img_size, conf_thres, iou_thres):
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    devicee = select_device(args.device)
    run_state = {'brake': False, 'steering': None}

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

            # Driving area and lane line masks
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)

            # Resize masks to original frame size
            da_seg_mask_resized = cv2.resize(da_seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            ll_seg_mask_resized = cv2.resize(ll_seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Extract offset for steering
            offset = calculate_lane_offset(ll_seg_mask_resized)
            process_steering(offset, run_state)

        # Person detection and braking
        if inDet is not None:
            detections = inDet.detections
            process_braking(detections, run_state)

        if cv2.waitKey(1) == ord('q'):
            break

# Main Execution
if __name__ == "__main__":
    pipeline = create_pipeline()
    with dai.Device(pipeline) as device:
        detect(device, model, img_size=640, conf_thres=0.25, iou_thres=0.45)

    ser1.close()  # Close the serial connection when done