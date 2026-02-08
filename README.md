# AI Fraud & Deepfake Detection System

A comprehensive AI-powered system for detecting:
- **Fraud Calls** - Text-based fraud detection using TF-IDF + Logistic Regression
- **Deepfake Images** - EfficientNet-B0 CNN classifier
- **Deepfake Videos** - Heuristic analysis with temporal/face/frame/audio checks
- **Audio Fraud** - Speech-to-text conversion + fraud text detection

## 📁 Folder Structure

```
AI-Based-Voice-Detection-/
├── backend/               # FastAPI REST API
│   ├── main.py
│   ├── services/
│   └── requirements.txt
├── frontend/              # Next.js Web UI
│   ├── src/app/
│   └── package.json
├── models/                # ML Models
│   ├── image_detector.py
│   ├── video_detector.py
│   ├── text_detector.py
│   └── weights/
│       └── deepfake_cnn.pth
├── scripts/               # Utility Scripts
│   ├── train_image_model.py
│   └── generate_audio.py
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Backend
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Run Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Open App
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/text` | POST | Analyze text for fraud |
| `/predict/audio` | POST | Upload audio → transcribe → detect fraud |
| `/predict/image` | POST | Detect deepfake in image |
| `/predict/video` | POST | Analyze video for deepfake |
| `/health` | GET | Check service status |

## 🔧 Models Used

| Detection | Model | Accuracy |
|-----------|-------|----------|
| Image | EfficientNet-B0 | ~95% |
| Text | TF-IDF + LogReg | ~92% |
| Video | Heuristic Analysis | Risk Score |

## 📊 Dataset

Place your dataset in `../Dataset/`:
- `fraud_calls_multilingual.csv` - Text fraud training data
- `audio/fraud/` - Fraud audio samples
- `audio/normal/` - Normal audio samples
