"""
Models Package
==============

Contains all ML models for fraud and deepfake detection:
- image_detector: EfficientNet-B0 for deepfake images
- video_detector: Heuristic analyzer for deepfake videos
- text_detector: TF-IDF + LogReg for fraud calls
"""

from models.image_detector import load_image_detector, DeepfakeCNN
from models.video_detector import VideoDeepfakeDetector
from models.text_detector import TextFraudDetector

__all__ = [
    'load_image_detector',
    'DeepfakeCNN',
    'VideoDeepfakeDetector',
    'TextFraudDetector'
]
