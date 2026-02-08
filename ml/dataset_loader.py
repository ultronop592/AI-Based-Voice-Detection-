import os
import cv2
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
from PIL import Image
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(image_size=224, margin=20, device=DEVICE)


class DeepfakeDataset(Dataset):
    def __init__(self, root_dirs, label, max_frames=10):
        self.samples = []
        self.label = label
        self.max_frames = max_frames

        for root in root_dirs:
            for r, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(".mp4"):
                        self.samples.append(os.path.join(r, f))

    def __len__(self):
        return len(self.samples)

    def extract_faces(self, video_path):
        cap = cv2.VideoCapture(video_path)
        faces = []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return faces

        indices = random.sample(
            range(total),
            min(self.max_frames, total)
        )

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # OpenCV BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image (REQUIRED)
            img = Image.fromarray(frame)

            # MTCNN forward
            face = mtcnn(img)

            if face is not None:
                faces.append(face)

        cap.release()
        return faces


    def __getitem__(self, idx):
        video = self.samples[idx]
        faces = self.extract_faces(video)

        if len(faces) == 0:
            # fallback tensor if no face detected
            face = torch.zeros(3, 224, 224)
        else:
            face = random.choice(faces)

        return face, torch.tensor(self.label, dtype=torch.long)

