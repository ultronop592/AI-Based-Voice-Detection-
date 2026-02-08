import torch
from torchvision import transforms
from PIL import Image
from ml.model import DeepfakeCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DeepfakeCNN().to(device)
model.load_state_dict(torch.load("deepfake_cnn.pth", map_location=device))
model.eval()


mtcnn = MTCNN(image_size=224, device=device)

def predict_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    preds = []

    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)

        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(face)
                pred = torch.argmax(logits, dim=1).item()
                preds.append(pred)

        count += 1

    cap.release()

    if not preds:
        return "UNKNOWN"

    return "DEEPFAKE" if np.mean(preds) > 0.5 else "REAL"
