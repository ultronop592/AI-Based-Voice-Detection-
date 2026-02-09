# AI Security Hub (Pro)

An enterprise-grade **AI Security Platform** for detecting digital fraud, deepfakes, and social engineering attacks across multiple media formats.

## 🛡️ Core Capabilities

- **Unified Intelligence Narrative**: Centralized dashboard for all security operations.
- **Multi-Modal Detection**:
  - **📝 Text Intelligence**: Detects financial fraud, urgency, and coercion patterns in transcripts.
  - **🖼️ Image Authenticity**: Pinpoints Deepfake manipulation in uploaded images.
  - **🎥 Video Forensics**: Analyzes temporal inconsistencies and facial landmarks (Heuristic & AI).
  - **🎙️ Audio Analysis**: Identifies synthetic voice artifacts and stress patterns.
- **Risk Scoring & Explainability**: 
  - 0-100% granular risk scores (Safety vs Threat).
  - Detailed "Why AI flagged this" explanations.
  - Actionable recommendations for users.
- **Enterprise UI**: Dark-mode, high-contrast "Security Blue" (Safe) and "Alert Red" (Risk) theme.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** (Backend)
- **Node.js 18+** (Frontend)

### 📦 Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/ultronop592/AI-Based-Voice-Detection-.git
cd AI-Based-Voice-Detection-
```

#### 2. Backend Setup
```bash
cd backend
python -m venv venv
# Activate: venv\Scripts\activate (Win) or source venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

#### 3. Frontend Setup
```bash
cd frontend
npm install
```

---

## ▶️ Running the Hub

### Start Backend (Terminal 1)
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Start Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```
Access the Hub at: **http://localhost:3000**

---

## 📁 Project Structure

```
AI-Based-Voice-Detection-/
├── backend/
│   ├── main.py              # FastAPI Entrypoint
│   ├── services/            # Intelligence Services (Text, Image, Video, Audio)
│   ├── debug_model.py       # Model Debugging Tool
│   ├── data/                # Datasets & Logs
│   └── requirements.txt     # Python Dependencies
│
├── frontend/
│   ├── src/app/             # Next.js App Router
│   │   ├── page.tsx         # Dashboard / Landing Page
│   │   ├── image/           # Image Analysis UI
│   │   ├── video/           # Video Analysis UI
│   │   ├── text/            # Text Analysis UI
│   │   └── audio/           # Audio Analysis UI
│   └── components/          # Reusable UI Components
│
└── models/
    └── weights/             # Trained AI Models (CNN, EfficientNet, etc.)
```

---

## 🔍 API Services

| Service | Endpoint | Type | Description |
|---------|----------|------|-------------|
| **Health** | `/health` | GET | System status check |
| **Text** | `/predict/text` | POST | NLP-based fraud pattern detection |
| **Audio** | `/predict/audio` | POST | Synthetic voice & stress analysis |
| **Image** | `/predict/image` | POST | Deepfake artifact detection |
| **Video** | `/predict/video` | POST | Temporal forensic analysis |

---

## 📄 License
MIT License
