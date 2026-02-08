import os
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATASET_ROOT = r"D:\deepfake_datasets\faces"
MODEL_PATH = "deepfake_cnn.pth"
BATCH_SIZE = 32   # 🔥 SAFE for 4GB GPU

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------
# DATASET
# -------------------------------------------------
class FaceEvalDataset(Dataset):
    def __init__(self, folder, label, transform):
        self.images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".jpg")
        ]
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.label

# -------------------------------------------------
# MODEL (MATCH TRAINING)
# -------------------------------------------------
model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features, 2
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# -------------------------------------------------
# TRANSFORMS
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
real_ds = FaceEvalDataset(
    os.path.join(DATASET_ROOT, "real"), label=0, transform=transform
)
fake_ds = FaceEvalDataset(
    os.path.join(DATASET_ROOT, "fake"), label=1, transform=transform
)

dataset = torch.utils.data.ConcatDataset([real_ds, fake_ds])

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -------------------------------------------------
# INFERENCE (BATCHED)
# -------------------------------------------------
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y.numpy())

# -------------------------------------------------
# METRICS
# -------------------------------------------------
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n===== METRICS =====")
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nFormat:")
print("[[TN  FP]")
print(" [FN  TP]]")
