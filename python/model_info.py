import os

import torch
import torchvision.models as models


# Load the pretrained EfficientNet model
model = models.efficientnet_b0(weights="DEFAULT") 
model.eval()  # Set the model to evaluation mode

model = torch.nn.Sequential(*list(model.children())[:-1])

for name, layer in model.named_modules():
    print(f"{name}: {layer}")