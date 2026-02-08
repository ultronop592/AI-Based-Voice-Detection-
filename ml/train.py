import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from ml.model import DeepfakeCNN

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATASET_ROOT = r"D:\deepfake_datasets\faces"
BATCH_SIZE = 96         # 🔥 GPU friendly
EPOCHS = 10
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------
# DATASET
# -------------------------------------------------
class FaceDataset(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.images = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(".jpg")
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.label, dtype=torch.long)

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
real_ds = FaceDataset(
    os.path.join(DATASET_ROOT, "real"),
    label=0,
    transform=transform
)

fake_ds = FaceDataset(
    os.path.join(DATASET_ROOT, "fake"),
    label=1,
    transform=transform
)

dataset = torch.utils.data.ConcatDataset([real_ds, fake_ds])

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,     # 🔥 FIX
    pin_memory=True
)

print("Real images:", len(real_ds))
print("Fake images:", len(fake_ds))
print("Total images:", len(dataset))

# -------------------------------------------------
# MODEL (EfficientNet-B0)
# -------------------------------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

# Class weights: [REAL, FAKE]
weights = torch.tensor([11278 / 2194, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
torch.save(model.state_dict(), "deepfake_cnn.pth")
print("\n✅ Training complete. Model saved as deepfake_cnn.pth")
