import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = efficientnet_b0(
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
