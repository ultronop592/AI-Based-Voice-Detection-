
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Define model structure
class DeepfakeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeCNN, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def check_model():
    model_path = r"c:\AI Based Model\AI-Based-Voice-Detection-\backend\weights\deepfake_cnn.pth"
    device = torch.device("cpu")
    
    print(f"Loading model from {model_path}")
    model = DeepfakeCNN(num_classes=2)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
         # Helper to load weights (handling prefix issues)
        if list(state_dict.keys())[0].startswith("model."):
            model.load_state_dict(state_dict)
        else:
            try:
                model.model.load_state_dict(state_dict)
            except RuntimeError:
                new_state_dict = {"model." + k: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy images
    # 1. Pure black image (artificial)
    img_black = Image.new('RGB', (224, 224), color='black')
    # 2. Pure white image (artificial)
    img_white = Image.new('RGB', (224, 224), color='white')
    # 3. Random noise image (artificial/fake-like)
    img_noise = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for name, img in [("Black", img_black), ("White", img_white), ("Noise", img_noise)]:
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            fake_prob = probs[0][0].item()
            real_prob = probs[0][1].item()
            
            print(f"Image: {name}")
            print(f"  Raw Output: {output.numpy()}")
            print(f"  Probs: Fake={fake_prob:.4f}, Real={real_prob:.4f}")
            print(f"  Current Logic (Real > Fake): {'REAL' if real_prob > fake_prob else 'DEEPFAKE'}")
            print("-" * 30)

if __name__ == "__main__":
    check_model()
