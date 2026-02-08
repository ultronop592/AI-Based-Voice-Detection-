import cv2
import numpy as np
import os
import tempfile
import shutil
from moviepy import VideoFileClip

# ==================================================
# DATASET CONFIGURATION (ONLY CHANGE IF NEEDED)
# ==================================================
DATASET_ROOT = r"D:\deepfake_datasets"

REAL_DIRS = [
    os.path.join(DATASET_ROOT, "real", "Celeb-real"),
    os.path.join(DATASET_ROOT, "real", "YouTube-real"),
    os.path.join(DATASET_ROOT, "DFDC", "train_sample_videos")
]

FAKE_DIRS = [
    os.path.join(DATASET_ROOT, "fake", "Celeb-synthesis"),
    os.path.join(DATASET_ROOT, "fake", "FaceForensics")
]


def resolve_video_path(video_name):
    """
    Searches all real & fake directories for the video.
    Returns full path if found, else None.
    """
    for base_dir in REAL_DIRS + FAKE_DIRS:
        if not os.path.exists(base_dir):
            continue

        for root, _, files in os.walk(base_dir):
            if video_name in files:
                return os.path.join(root, video_name)

    return None


class VideoDeepfakeDetector:
    """
    Heuristic + signal-based deepfake risk analyzer.
    NOT a trained ML model.
    """

    def __init__(self):
        self.analysis_weights = {
            "temporal": 0.30,
            "face": 0.25,
            "frame": 0.35,
            "audio": 0.10
        }

    # --------------------------------------------------
    # Frame Extraction
    # --------------------------------------------------
    def extract_frames(self, video_path, num_frames=10):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                path = os.path.join(temp_dir, f"frame_{i}.jpg")
                cv2.imwrite(path, frame)
                frame_paths.append(path)

        cap.release()

        return {
            "frames": frame_paths,
            "temp_dir": temp_dir,
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames
        }

    # --------------------------------------------------
    # Temporal Analysis
    # --------------------------------------------------
    def analyze_temporal_consistency(self, frames):
        diffs = []

        for i in range(len(frames) - 1):
            img1 = cv2.imread(frames[i])
            img2 = cv2.imread(frames[i + 1])

            if img1 is None or img2 is None:
                continue

            img1 = cv2.resize(img1, (256, 256))
            img2 = cv2.resize(img2, (256, 256))

            diff = cv2.absdiff(img1, img2)
            norm_diff = np.mean(diff) / (np.mean(img1) + 1e-5)
            diffs.append(norm_diff)

        return float(np.mean(diffs) * 100) if diffs else 0.0

    # --------------------------------------------------
    # Face Stability
    # --------------------------------------------------
    def analyze_face_consistency(self, frames):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        positions = []

        for path in frames:
            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                positions.append(faces[0])

        if len(positions) < 2:
            return 0.0

        pos_arr = np.array(positions)
        return float(np.var(pos_arr, axis=0).mean())

    # --------------------------------------------------
    # Frame Artifact Heuristic
    # --------------------------------------------------
    def analyze_frame_artifacts(self, frames):
        scores = []

        for path in frames:
            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(lap_var)

        return float(np.mean(scores)) if scores else 0.0

    # --------------------------------------------------
    # Audio Check
    # --------------------------------------------------
    def analyze_audio(self, video_path):
        try:
            clip = VideoFileClip(video_path)
            return 0.0 if clip.audio is not None else 20.0
        except Exception:
            return 0.0

    # --------------------------------------------------
    # Main Analysis
    # --------------------------------------------------
    def analyze_video(self, video_path):
        frame_data = self.extract_frames(video_path)
        if frame_data is None:
            return {"success": False, "error": "Could not read video"}

        frames = frame_data["frames"]

        temporal = self.analyze_temporal_consistency(frames)
        face = self.analyze_face_consistency(frames)
        frame = self.analyze_frame_artifacts(frames)
        audio = self.analyze_audio(video_path)

        shutil.rmtree(frame_data["temp_dir"], ignore_errors=True)

        temporal = min(temporal, 100)
        face = min(face / 10, 100)
        frame = min(frame / 50, 100)

        final_score = (
            temporal * self.analysis_weights["temporal"]
            + face * self.analysis_weights["face"]
            + frame * self.analysis_weights["frame"]
            + audio * self.analysis_weights["audio"]
        )

        label = (
            "DEEPFAKE" if final_score >= 70 else
            "SUSPICIOUS" if final_score >= 50 else
            "LIKELY_REAL"
        )

        return {
            "success": True,
            "label": label,
            "deepfake_risk_score": round(final_score, 2),
            "video_info": {
                "fps": frame_data["fps"],
                "duration": round(frame_data["duration"], 2),
                "frames_analyzed": len(frames)
            },
            "component_scores": {
                "temporal": round(temporal, 2),
                "face": round(face, 2),
                "frame": round(frame, 2),
                "audio": round(audio, 2)
            },
            "disclaimer": "Heuristic analysis. Not a trained ML classifier."
        }


# ==================================================
# EXAMPLE USAGE
# ==================================================
if __name__ == "__main__":
    detector = VideoDeepfakeDetector()

    found_path = None

    for base_dir in REAL_DIRS + FAKE_DIRS:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    found_path = os.path.join(root, file)
                    break
            if found_path:
                break
        if found_path:
            break

    if not found_path:
        print("No video files found in dataset.")
    else:
        print(f"Testing video: {found_path}")
        result = detector.analyze_video(found_path)

        print("\nRESULT:")
        for k, v in result.items():
            print(f"{k}: {v}")
