import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image

VIDEO_ROOT = r"D:\deepfake_datasets\unseen_videos"
OUT_ROOT = r"D:\deepfake_datasets\faces_unseen"

os.makedirs(f"{OUT_ROOT}/real", exist_ok=True)
os.makedirs(f"{OUT_ROOT}/fake", exist_ok=True)

mtcnn = MTCNN(image_size=224, device="cpu")

def extract(src, dst):
    for f in os.listdir(src):
        if not f.endswith(".mp4"):
            continue

        cap = cv2.VideoCapture(os.path.join(src, f))
        saved = 0

        while saved < 3:   # only 3 faces per video
            ret, frame = cap.read()
            if not ret:
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face = mtcnn(img)

            if face is not None:
                face = face.permute(1,2,0).numpy()
                face = (face * 255).astype("uint8")
                Image.fromarray(face).save(
                    f"{dst}/{f}_{saved}.jpg"
                )
                saved += 1

        cap.release()

print("Extracting UNSEEN REAL faces...")
extract(f"{VIDEO_ROOT}/real", f"{OUT_ROOT}/real")

print("Extracting UNSEEN FAKE faces...")
extract(f"{VIDEO_ROOT}/fake", f"{OUT_ROOT}/fake")

print("✅ Unseen face extraction complete")
