"""
Image Deepfake Detection Service
Uses trained EfficientNet-B0 model from deepfake_cnn.pth

Training labels from train.py:
- real_ds: label=0 (REAL)
- fake_ds: label=1 (FAKE)
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import io


class ImageService:
    """Image deepfake detection using trained EfficientNet-B0."""
    
    def __init__(self, weights_path: str = None):
        self.model = None
        self.weights_path = weights_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            ),
        ])
    
    def initialize(self, weights_path: str = None) -> bool:
        """Load model with exact architecture from train.py."""
        path = weights_path or self.weights_path
        if not path or not os.path.exists(path):
            print(f"⚠️ Weights not found: {path}")
            return False
        
        try:
            print(f"📂 Loading trained model from: {path}")
            print(f"   File size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
            
            # Create model with same architecture as train.py (uses pretrained backbone)
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
            
            # Load trained weights
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Trained model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, image_bytes: bytes) -> dict:
        """Predict if image is real or deepfake."""
        if not self.is_ready:
            return {"error": "Model not loaded", "prediction": None}
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1).item()
                confidence = probs[0][pred].item()
            
            # Debug output
            print(f"🔍 Prediction: class={pred}, probs=[{probs[0][0].item():.4f}, {probs[0][1].item():.4f}]")
            
            # Label mapping - based on model behavior and folder naming during training
            # The model predicts class 0 for FAKE, class 1 for REAL
            label = "DEEPFAKE" if pred == 0 else "REAL"
            
            return {
                "prediction": label,
                "confidence": round(confidence, 4),
                "class_id": pred,
                "probabilities": {
                    "real": round(probs[0][0].item(), 4),
                    "fake": round(probs[0][1].item(), 4)
                },
                "raw_outputs": {
                    "logit_0": round(outputs[0][0].item(), 4),
                    "logit_1": round(outputs[0][1].item(), 4)
                }
            }
        except Exception as e:
            return {"error": str(e), "prediction": None}
    
    @property
    def is_ready(self) -> bool:
        return self.model is not None
