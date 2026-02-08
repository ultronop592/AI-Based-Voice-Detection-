"""
Image Deepfake Detection Model
EfficientNet-B0 based CNN for detecting deepfake images

Architecture matches train.py:
- EfficientNet-B0 backbone
- Modified classifier[1] to 2 classes (Real/Fake)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def load_image_detector(weights_path: str, device: str = "cpu"):
    """
    Load the trained deepfake image detection model.
    Uses exact architecture from train.py.
    """
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


# For backward compatibility
class DeepfakeCNN(nn.Module):
    """
    Legacy class - use load_image_detector() instead.
    This architecture differs from training, kept for reference only.
    """
    def __init__(self):
        super().__init__()
        self.features = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        ).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)
