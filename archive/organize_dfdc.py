import json
import shutil
from pathlib import Path

SRC_DIR = Path("C:/Deepfake_Detection/data/train_sample_videos")
META_PATH = SRC_DIR / "metadata.json"

OUT_REAL = Path("C:/Deepfake_Detection/dataset/raw_videos/real")
OUT_FAKE = Path("C:/Deepfake_Detection/dataset/raw_videos/fake")

OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

for video, info in metadata.items():
    src = SRC_DIR / video
    if not src.exists():
        continue

    if info["label"] == "REAL":
        shutil.copy(src, OUT_REAL / video)
    else:
        shutil.copy(src, OUT_FAKE / video)

print("✔ DFDC sample videos separated into REAL and FAKE")
