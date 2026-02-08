import json
import random
from pathlib import Path

META_PATH = Path("C:/Deepfake_Detection/dataset/metadata/metadata.json")
REAL_DIR = Path("C:/Deepfake_Detection/dataset/raw_videos/real")
FAKE_DIR = Path("C:/Deepfake_Detection/dataset/raw_videos/fake")
SPLIT_DIR = Path("C:/Deepfake_Detection/dataset/splits")

SPLIT_DIR.mkdir(exist_ok=True)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

# Group videos by original
groups = {}

for video, info in metadata.items():
    if info["label"] == "REAL":
        original = video
    else:
        original = info["original"]

    groups.setdefault(original, []).append(video)

originals = list(groups.keys())
random.shuffle(originals)

split_idx = int(0.7 * len(originals))
train_orig = set(originals[:split_idx])
val_orig = set(originals[split_idx:])

train_files, val_files = [], []

for orig, vids in groups.items():
    target = train_files if orig in train_orig else val_files
    for v in vids:
        if (REAL_DIR / v).exists() or (FAKE_DIR / v).exists():
            target.append(v)

# Write splits
(Path(SPLIT_DIR / "train.txt")).write_text("\n".join(train_files))
(Path(SPLIT_DIR / "val.txt")).write_text("\n".join(val_files))

print(f"✔ Train videos: {len(train_files)}")
print(f"✔ Val videos: {len(val_files)}")
