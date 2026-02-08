"""
Audio Analysis Service - Speech-to-Text + Fraud Detection
"""

import os
import tempfile
import speech_recognition as sr
from pydub import AudioSegment

from services.text_service import TextService


class AudioService:
    """Service for audio-based fraud detection using speech-to-text."""
    
    def __init__(self, text_service: TextService = None):
        self.text_service = text_service
        self.recognizer = sr.Recognizer()
    
    def set_text_service(self, text_service: TextService):
        """Set the text service for fraud detection after transcription."""
        self.text_service = text_service
    
    def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> dict:
        """Convert audio to text using speech recognition."""
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        wav_path = os.path.join(temp_dir, "converted.wav")
        
        try:
            # Save uploaded file
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            
            # Convert to WAV if needed
            try:
                audio = AudioSegment.from_file(temp_path)
                audio.export(wav_path, format="wav")
            except Exception:
                # Already WAV or can be read directly
                wav_path = temp_path
            
            # Transcribe
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
            
            # Try multiple recognition engines
            transcript = None
            
            # Try Google (free, no API key needed for short audio)
            try:
                transcript = self.recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                pass
            
            if transcript is None:
                return {
                    "error": "Could not transcribe audio. Ensure audio is clear and contains speech.",
                    "transcript": None
                }
            
            return {
                "transcript": transcript,
                "error": None
            }
            
        except Exception as e:
            return {"error": str(e), "transcript": None}
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def predict(self, audio_bytes: bytes, filename: str = "audio.wav") -> dict:
        """Transcribe audio and detect fraud."""
        if self.text_service is None or not self.text_service.is_ready:
            return {"error": "Text service not available", "prediction": None}
        
        # Step 1: Transcribe
        transcription = self.transcribe_audio(audio_bytes, filename)
        
        if transcription.get("error") or not transcription.get("transcript"):
            return {
                "error": transcription.get("error", "Transcription failed"),
                "prediction": None,
                "transcript": None
            }
        
        transcript = transcription["transcript"]
        
        # Step 2: Analyze text for fraud
        result = self.text_service.predict(transcript)
        
        return {
            "transcript": transcript,
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "text_cleaned": result.get("text_cleaned"),
            "error": None
        }
    
    @property
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self.text_service is not None and self.text_service.is_ready
