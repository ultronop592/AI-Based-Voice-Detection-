"""
Video Deepfake Detection Service
Uses PyTorch EfficientNet-B0 model for frame-by-frame analysis
"""

import os
import cv2
import numpy as np
import tempfile
import shutil
from PIL import Image

try:
    from moviepy import VideoFileClip
    HAS_MOVIEPY = True
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
        HAS_MOVIEPY = True
    except ImportError:
        HAS_MOVIEPY = False

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


class VideoDeepfakeDetector:
    """
    CNN-based deepfake detector for videos.
    Analyzes frames using PyTorch EfficientNet-B0 model.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else None
        self.transform = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
        
    def load_model(self, model_path: str = None) -> bool:
        """Load the PyTorch model."""
        if not HAS_TORCH:
            print("❌ PyTorch not installed")
            return False
            
        path = model_path or self.model_path
        if not path or not os.path.exists(path):
            print(f"⚠️ Model not found: {path}")
            return False
            
        try:
            print(f"📂 Loading video detection model from: {path}")
            
            # Create model
            self.model = DeepfakeCNN(num_classes=2)
            
            # Load weights
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            
            # Helper to load weights (handling prefix issues)
            if list(state_dict.keys())[0].startswith("model."):
                self.model.load_state_dict(state_dict)
            else:
                try:
                    self.model.model.load_state_dict(state_dict)
                except RuntimeError:
                    new_state_dict = {"model." + k: v for k, v in state_dict.items()}
                    self.model.load_state_dict(new_state_dict)
                
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Video model loaded (PyTorch EfficientNet-B0)")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def crop_face(self, frame: np.ndarray) -> np.ndarray:
        """Detect and crop face from frame. Returns original frame if no face found."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Get largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Add margin
                margin = int(w * 0.2)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(frame.shape[1] - x, w + 2 * margin)
                h = min(frame.shape[0] - y, h + 2 * margin)
                
                return frame[y:y+h, x:x+w]
        except Exception:
            pass
        return frame

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame for model input."""
        if self.transform is None:
            return None
            
        # 1. Crop Face (ROI)
        face_frame = self.crop_face(frame)
        
        # 2. Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        return self.transform(pil_image)

    def extract_frames(self, video_path, num_frames=30):
        """Extract frames from video (Increased to 30 for high density)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()

        return {
            "frames": frames,
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames
        }

    def analyze_video(self, video_path, num_frames=30):
        """Analyze video using CNN model on extracted frames."""
        if self.model is None:
            return {"success": False, "error": "Model not loaded"}
            
        frame_data = self.extract_frames(video_path, num_frames)
        if frame_data is None:
            return {"success": False, "error": "Could not read video"}

        frames = frame_data["frames"]
        if len(frames) == 0:
            return {"success": False, "error": "No frames extracted"}

        # Process frames
        frame_results = []
        
        with torch.no_grad():
            for i, frame in enumerate(frames):
                img_tensor = self.preprocess_frame(frame)
                if img_tensor is None:
                    continue
                    
                # Add batch dimension
                input_batch = img_tensor.unsqueeze(0).to(self.device)
                
                # Predict
                outputs = self.model(input_batch)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Class 0 = FAKE, Class 1 = REAL
                fake_prob = float(probabilities[0][0])
                real_prob = float(probabilities[0][1])
                is_fake = fake_prob > real_prob
                
                frame_results.append({
                    "frame_idx": i,
                    "fake_prob": fake_prob,
                    "real_prob": real_prob,
                    "is_fake": is_fake
                })
        
        if not frame_results:
             return {"success": False, "error": "Frame analysis failed"}

        # Aggregate results
        fake_count = sum(1 for r in frame_results if r["is_fake"])
        real_count = len(frame_results) - fake_count
        
        avg_fake_prob = np.mean([r["fake_prob"] for r in frame_results])
        max_fake_prob = np.max([r["fake_prob"] for r in frame_results])
        avg_real_prob = np.mean([r["real_prob"] for r in frame_results])
        
        # Aggressive Deepfake Detection Heuristic
        # Deepfakes often flicker. A high 'Max' likely indicates a glitch in generation.
        # We combine Peak Risk (Max) with Sustained Risk (Avg).
        
        risk_score = (avg_fake_prob * 0.4) + (max_fake_prob * 0.6)
        
        print(f"🔍 Analysis Metrics: AvgFake={avg_fake_prob:.3f}, MaxFake={max_fake_prob:.3f}, RiskScore={risk_score:.3f}")

        if risk_score > 0.65:
            label = "DEEPFAKE"
            confidence = risk_score
        elif risk_score > 0.35:
            label = "SUSPICIOUS"
            confidence = risk_score
        else:
            label = "REAL"
            confidence = avg_real_prob

        print(f"🎬 Video Final: {label} (Conf={confidence:.3f})")

        return {
            "success": True,
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "real": round(avg_real_prob, 4),
                "fake": round(avg_fake_prob, 4)
            },
            "video_info": {
                "fps": round(frame_data["fps"], 2),
                "duration": round(frame_data["duration"], 2),
                "frames_analyzed": len(frame_results)
            },
            "frame_analysis": {
                "fake_frames": fake_count,
                "real_frames": real_count,
                "total_analyzed": len(frame_results)
            }
        }


class VideoService:
    """Video deepfake detection service."""
    
    def __init__(self, model_path: str = None):
        self.detector = VideoDeepfakeDetector(model_path)
        self.model_path = model_path
        self._initialized = False
        
    def initialize(self, model_path: str = None) -> bool:
        """Initialize the video detector with model."""
        path = model_path or self.model_path
        if path and self.detector.load_model(path):
            self._initialized = True
            print("✅ VideoService initialized with CNN model (PyTorch)")
            return True
        else:
            print("⚠️ VideoService: Model not loaded, service unavailable")
            return False
    
    def predict(self, video_path: str) -> dict:
        """Analyze video for deepfake indicators."""
        if not self._initialized or self.detector.model is None:
            return {"error": "Model not loaded", "prediction": None}
            
        try:
            result = self.detector.analyze_video(video_path)
            if result.get("success"):
                return {
                    "prediction": result["label"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                    "video_info": result["video_info"],
                    "frame_analysis": result.get("frame_analysis", {})
                }
            else:
                return {"error": result.get("error", "Analysis failed"), "prediction": None}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e), "prediction": None}
    
    @property
    def is_ready(self) -> bool:
        return self._initialized and self.detector.model is not None
