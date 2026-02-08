"""
Video Deepfake Detection Service
Uses models/video_detector.py (VideoDeepfakeDetector)
"""

import os
import sys
import cv2
import numpy as np
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

try:
    from models.video_detector import VideoDeepfakeDetector
    HAS_DETECTOR = True
except ImportError:
    HAS_DETECTOR = False


class VideoService:
    """Video deepfake detection using heuristic analysis."""
    
    def __init__(self):
        if HAS_DETECTOR:
            self.detector = VideoDeepfakeDetector()
            print("✅ VideoDeepfakeDetector loaded")
        else:
            self.detector = None
            print("⚠️ Using fallback video analysis")
    
    def predict(self, video_path: str) -> dict:
        """Analyze video for deepfake indicators."""
        
        if self.detector:
            try:
                result = self.detector.analyze_video(video_path)
                if result.get("success"):
                    return {
                        "prediction": result["label"],
                        "deepfake_risk_score": result["deepfake_risk_score"],
                        "confidence": round((100 - abs(result["deepfake_risk_score"] - 50)) / 100, 4),
                        "video_info": result["video_info"],
                        "component_scores": result["component_scores"],
                        "disclaimer": result.get("disclaimer", "Heuristic analysis")
                    }
            except Exception as e:
                print(f"Detector failed: {e}")
        
        return self._fallback_analysis(video_path)
    
    def _fallback_analysis(self, video_path: str) -> dict:
        """Simple fallback analysis."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Could not open video", "prediction": None}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps > 0 else 0
            cap.release()
            
            return {
                "prediction": "UNKNOWN",
                "deepfake_risk_score": 50,
                "confidence": 0.5,
                "video_info": {"fps": round(fps, 2), "duration": round(duration, 2), "frames_analyzed": 0},
                "disclaimer": "Fallback analysis - detector not available"
            }
        except Exception as e:
            return {"error": str(e), "prediction": None}
    
    @property
    def is_ready(self) -> bool:
        return True
