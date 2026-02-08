import os
import cv2
import torch
import gc
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------
# CONFIG (STABILITY FIRST)
# -------------------------------------------------
DATASET_ROOT = r"D:\deepfake_datasets"
OUT_ROOT = r"D:\deepfake_datasets\faces"

REAL_DIRS = [
    f"{DATASET_ROOT}/real/Celeb-real",
    f"{DATASET_ROOT}/real/YouTube-real"
]

FAKE_DIRS = [
    f"{DATASET_ROOT}/fake/Celeb-synthesis"
]

MAX_VIDEOS_PER_CLASS = 200   # 👈 LIMIT (changeable)
FRAME_STRIDE = 30            # VERY IMPORTANT
RESIZE_WIDTH = 320           # VERY IMPORTANT
RESIZE_HEIGHT = 240

# 🔴 FORCE CPU (THIS IS THE KEY)
device = "cpu"
print("Face preprocessing device:", device)

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    device=device
)

# -------------------------------------------------
def process_videos(src_dirs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for d in src_dirs:
        video_count = 0

        for root, _, files in os.walk(d):
            for f in files:
                if video_count >= MAX_VIDEOS_PER_CLASS:
                    return

        for root, _, files in os.walk(d):
            for f in files:
                if not f.lower().endswith(".mp4"):
                    continue

                video_path = os.path.join(root, f)
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    continue

                saved = 0
                frame_idx = 0
                max_frames = 150  # HARD LIMIT

                while saved < MAX_FACES_PER_VIDEO and frame_idx < max_frames:
                    frame_idx += 1

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    if frame_idx % FRAME_STRIDE != 0:
                        continue

                    try:
                        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame)

                        face = mtcnn(img)
                    except Exception:
                        continue

                    if face is not None:
                        save_path = os.path.join(
                            out_dir, f"{os.path.splitext(f)[0]}_{saved}.jpg"
                        )

                        face = face.permute(1, 2, 0).numpy()
                        face = (face * 255).astype("uint8")
                        Image.fromarray(face).save(save_path)

                        saved += 1

                cap.release()
                del cap

                gc.collect()

# -------------------------------------------------
print("Extracting REAL faces...")
process_videos(REAL_DIRS, f"{OUT_ROOT}/real")

print("Extracting FAKE faces...")
process_videos(FAKE_DIRS, f"{OUT_ROOT}/fake")

print("\n✅ Face preprocessing finished (STABLE MODE).")
