import sys
import os

# Add the YOLOPv2 directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from lib.models.YOLOPv2 import YOLOPv2  # Import the model directly

# Initialize the model
model = YOLOPv2()

# Load the weights
model.load_state_dict(torch.load('yolopv2_weights.pt', map_location='cpu'))
model.eval()

# Create a sample input
sample_input = torch.rand((1, 3, 640, 640))

# Export to ONNX
torch.onnx.export(
    model,
    sample_input,
    'yolopv2.onnx',
    opset_version=12,
    input_names=['input'],
    output_names=['output']
)

print("Model converted to ONNX successfully.")