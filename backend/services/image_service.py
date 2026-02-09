"""
Image Deepfake Detection Service
Uses trained EfficientNet-B0 PyTorch model

Label mapping (confirmed):
- Class 0: FAKE/DEEPFAKE
- Class 1: REAL
"""

import os
import numpy as np
from PIL import Image
import io

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DeepfakeCNN(nn.Module):
    """EfficientNet-B0 based deepfake detector."""
    
    def __init__(self, num_classes=2):
        super(DeepfakeCNN, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class ImageService:
    """Image deepfake detection using PyTorch EfficientNet-B0."""
    
    def __init__(self, weights_path: str = None):
        self.model = None
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else None
        self.transform = None
        self._setup_transform()
    
    def _setup_transform(self):
        """Setup image preprocessing transform."""
        if not HAS_TORCH:
            return
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def initialize(self, weights_path: str = None) -> bool:
        """Load PyTorch model."""
        if not HAS_TORCH:
            print("❌ PyTorch not installed")
            return False
            
        path = weights_path or self.weights_path
        if not path or not os.path.exists(path):
            print(f"⚠️ Weights not found: {path}")
            return False
        
        try:
            print(f"📂 Loading PyTorch model from: {path}")
            print(f"   File size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
            print(f"   Device: {self.device}")
            
            # Create model
            self.model = DeepfakeCNN(num_classes=2)
            
            # Load weights
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            
            # Check if state_dict keys match model keys
            if list(state_dict.keys())[0].startswith("model."):
                self.model.load_state_dict(state_dict)
            else:
                # If keys don't start with 'model.', they likely belong to the inner efficientnet
                # But DeepfakeCNN wraps it in self.model
                # So we can try loading into self.model.model
                print("   Loading weights into inner model...")
                try:
                    self.model.model.load_state_dict(state_dict)
                except RuntimeError:
                    # Maybe the keys are for the wrapper but missing prefix?
                    # Let's try adding 'model.' prefix
                    new_state_dict = {"model." + k: v for k, v in state_dict.items()}
                    self.model.load_state_dict(new_state_dict)
            
            # Set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ PyTorch model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        if self.transform is None:
            return None
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image_bytes: bytes) -> dict:
        """Predict if image is real or deepfake."""
        if not self.is_ready:
            return {"error": "Model not loaded", "prediction": None}
        
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_tensor = self.preprocess_image(image)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Class 0 = FAKE, Class 1 = REAL
                fake_prob = float(probabilities[0][0])
                real_prob = float(probabilities[0][1])
                
                # Determine label based on highest probability
                # Determine label based on thresholds
                # Reduce False Positives by requiring higher confidence for DEEPFAKE
                if fake_prob > 0.85:
                    label = "DEEPFAKE"
                    confidence = fake_prob
                elif fake_prob > 0.60:
                     label = "SUSPICIOUS"
                     confidence = fake_prob
                else:
                    label = "REAL"
                    confidence = real_prob
            
            # Debug output
            print(f"🔍 Prediction: {label}, probs=[real:{real_prob:.4f}, fake:{fake_prob:.4f}]")
            
            return {
                "prediction": label,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "real": round(real_prob, 4),
                    "fake": round(fake_prob, 4)
                }
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e), "prediction": None}
    
    @property
    def is_ready(self) -> bool:
        return self.model is not None
