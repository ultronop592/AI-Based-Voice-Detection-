"""
AI Fraud & Deepfake Detection - FastAPI Backend
===============================================
"""

import os
import sys
import tempfile
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.text_service import TextService
from services.image_service import ImageService
from services.video_service import VideoService
from services.audio_service import AudioService

# -------------------------------------------------
# Configuration - Environment-aware paths
# -------------------------------------------------
# For Render: dataset and weights should be in backend folder
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(BACKEND_DIR, "data", "fraud_calls_multilingual.csv")
)
WEIGHTS_PATH = os.environ.get(
    "WEIGHTS_PATH", 
    os.path.join(BACKEND_DIR, "weights", "deepfake_cnn.pth")
)

# Fallback to parent directory Dataset if backend/data doesn't exist
if not os.path.exists(DATASET_PATH):
    alt_path = os.path.join(PROJECT_ROOT, "..", "Dataset", "fraud_calls_multilingual.csv")
    if os.path.exists(alt_path):
        DATASET_PATH = os.path.normpath(alt_path)

# -------------------------------------------------
# Services
# -------------------------------------------------
text_service = TextService(DATASET_PATH)
image_service = ImageService(WEIGHTS_PATH)
video_service = VideoService()
audio_service = AudioService()

# -------------------------------------------------
# Lifespan
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting AI Fraud & Deepfake Detection API...")
    print(f"📁 Backend dir: {BACKEND_DIR}")
    print(f"📊 Dataset: {DATASET_PATH}")
    print(f"🧠 Weights: {WEIGHTS_PATH}")
    
    # Initialize services
    if os.path.exists(DATASET_PATH):
        text_service.initialize()
        audio_service.set_text_service(text_service)
    else:
        print(f"⚠️ Dataset not found: {DATASET_PATH}")
    
    if os.path.exists(WEIGHTS_PATH):
        image_service.initialize()
    else:
        print(f"⚠️ Weights not found: {WEIGHTS_PATH}")
    
    print("✅ API ready!")
    yield
    print("👋 Shutting down...")

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="AI Fraud & Deepfake Detection API",
    description="4 detection endpoints: text, audio, image, video",
    version="2.0.0",
    lifespan=lifespan
)

# CORS - Allow frontend origins
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Models
# -------------------------------------------------
class TextRequest(BaseModel):
    text: str

# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/")
async def root():
    return {"name": "AI Fraud & Deepfake Detection API", "version": "2.0.0"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "services": {
            "text": text_service.is_ready,
            "audio": audio_service.is_ready,
            "image": image_service.is_ready,
            "video": video_service.is_ready
        }
    }

@app.post("/predict/text")
async def predict_text(request: TextRequest):
    """Detect fraud from text."""
    if not text_service.is_ready:
        raise HTTPException(503, "Text service not ready - dataset not loaded")
    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")
    return text_service.predict(request.text)

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """Upload audio → transcribe → detect fraud."""
    if not audio_service.is_ready:
        raise HTTPException(503, "Audio service not ready")
    contents = await file.read()
    result = audio_service.predict(contents, file.filename or "audio.wav")
    if result.get("error") and not result.get("prediction"):
        raise HTTPException(400, result["error"])
    return result

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Detect deepfake in image."""
    if not image_service.is_ready:
        raise HTTPException(503, "Image service not ready - model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    contents = await file.read()
    return image_service.predict(contents)

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """Analyze video for deepfake."""
    if not video_service.is_ready:
        raise HTTPException(503, "Video service not ready")
    
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename or "video.mp4")
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        return video_service.predict(temp_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
