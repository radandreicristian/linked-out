import os

import torch
import torchvision.models as models


# Load the pretrained EfficientNet model
model = models.efficientnet_b0(weights="DEFAULT") 
model.eval()  # Set the model to evaluation mode

dummy_input = torch.randn(1, 3, 512, 512)  # Batch size of 1, 3 color channels, 512x512 image size

models_folder_path = "../models"
exported_model_path = os.path.join(models_folder_path, "model.onnx")
if not os.path.exists(models_folder_path):
    os.mkdir(models_folder_path)

# Export the model to an ONNX file
torch.onnx.export(
    model,                      # The model to export
    dummy_input,                # The input to the model
    exported_model_path,        # The output file name
    export_params=True,         # Store the trained parameter weights inside the model file
    opset_version=11,           # ONNX version to export the model to
    do_constant_folding=True,   # Whether to execute constant folding for optimization
    input_names=['input'],      # The model's input names
    output_names=['output'],    # The model's output names
    dynamic_axes={
        'input': {0: 'batch_size'},  # Enable dynamic batching
        'output': {0: 'batch_size'}
    }
)

import onnx
import onnxruntime as ort

# Load the ONNX model
onnx_model = onnx.load(exported_model_path)
onnx.checker.check_model(onnx_model)

# Verify the model with ONNX Runtime
ort_session = ort.InferenceSession(exported_model_path)
outputs = ort_session.run(None, {"input": dummy_input.numpy()})
print(outputs[0].shape)
