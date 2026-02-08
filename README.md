# AI Fraud & Deepfake Detection System

A full-stack application for detecting fraudulent calls and deepfake media using advanced AI/ML models.

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** (for backend)
- **Node.js 18+** (for frontend)
- **Git** (for version control)

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ultronop592/AI-Based-Voice-Detection-.git
cd AI-Based-Voice-Detection-
```

### 2. Backend Setup
```bash
# Navigate to backend
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('stopwords')"
```

### 3. Frontend Setup
```bash
# Navigate to frontend (from project root)
cd frontend

# Install dependencies
npm install
```

---

## ▶️ Running the Application

### Start Backend (Terminal 1)
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
Backend will be available at: **http://localhost:8000**

### Start Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```
Frontend will be available at: **http://localhost:3000**

---

## 🔗 URLs Reference

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | Main UI |
| Backend API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Health Check | http://localhost:8000/health | Service status |

---

## 📁 Project Structure

```
AI-Based-Voice-Detection-/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── render.yaml          # Render deployment config
│   ├── data/                # Dataset files
│   └── services/            # API services
│       ├── text_service.py
│       ├── image_service.py
│       ├── video_service.py
│       └── audio_service.py
│
├── frontend/
│   ├── src/
│   │   ├── app/             # Next.js pages
│   │   │   ├── page.tsx     # Home page
│   │   │   ├── text/        # Text detection
│   │   │   ├── audio/       # Audio detection
│   │   │   ├── image/       # Image detection
│   │   │   └── video/       # Video detection
│   │   └── components/
│   │       └── ui/          # UI components
│   ├── package.json
│   └── tailwind.config.ts
│
├── models/
│   └── weights/             # ML model weights
│       └── deepfake_cnn.pth
│
└── README.md
```

---

## 🔍 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check all services status |
| POST | `/predict/text` | Analyze text for fraud |
| POST | `/predict/audio` | Upload audio for analysis |
| POST | `/predict/image` | Detect deepfake in image |
| POST | `/predict/video` | Analyze video for deepfakes |

### Example API Request
```bash
# Text Detection
curl -X POST http://localhost:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your bank account is blocked. Share OTP now!"}'
```

---

## 🛠️ Troubleshooting

### Backend Issues
```bash
# If "Module not found" errors:
pip install -r requirements.txt

# If port already in use:
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :8000
kill -9 <PID>
```

### Frontend Issues
```bash
# If "npm not found":
# Install Node.js from https://nodejs.org

# If dependency errors:
rm -rf node_modules package-lock.json
npm install

# If port 3000 in use:
npm run dev -- -p 3001
```

---

## 🚀 Deployment (Render)

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Create New → Web Service
4. Connect GitHub repo
5. Set **Root Directory**: `backend`
6. Render auto-detects `render.yaml`

---

## 📄 License

MIT License - Feel free to use and modify!
