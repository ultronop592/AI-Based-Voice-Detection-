"""
Text Fraud Detection Service
Uses the same TF-IDF + Logistic Regression approach from DeepFake Audio/call_fraud_detection.py
"""

import os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TextService:
    """Service for text-based fraud call detection using TF-IDF + Logistic Regression."""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path
        self.model = None
        self.vectorizer = None
        self.stop_words = None
    
    def initialize(self) -> bool:
        """
        Initialize the service by training the model on the dataset.
        Uses the exact same approach as call_fraud_detection.py.
        """
        try:
            # Download NLTK data
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            
            # Load dataset
            if not self.dataset_path or not os.path.exists(self.dataset_path):
                print(f"⚠️ Dataset not found: {self.dataset_path}")
                return False
            
            print(f"📂 Loading dataset from: {self.dataset_path}")
            data = pd.read_csv(self.dataset_path)
            
            # Map labels (same as original)
            data['label'] = data['label'].map({'fraud': 1, 'normal': 0})
            data.dropna(inplace=True)
            
            # Clean text
            print("🔄 Cleaning text data...")
            data['text'] = data['text'].apply(self._clean_text)
            
            # TF-IDF with bigrams (same as original)
            print("🔄 Training TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=2
            )
            X = self.vectorizer.fit_transform(data['text'])
            y = data['label']
            
            # Train model
            print("🔄 Training Logistic Regression model...")
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X, y)
            
            print("✅ Model trained successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean text using the same method as call_fraud_detection.py."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = ''.join(c for c in text if c not in string.punctuation)
        words = text.split()
        if self.stop_words:
            words = [w for w in words if w not in self.stop_words]
        return ' '.join(words)
    
    def predict(self, text: str) -> dict:
        """Predict if text is fraud or genuine."""
        if not self.is_ready:
            return {"error": "Model not initialized", "prediction": None}
        
        try:
            cleaned = self._clean_text(text)
            vector = self.vectorizer.transform([cleaned])
            prediction = self.model.predict(vector)[0]
            proba = self.model.predict_proba(vector)[0]
            
            label = "Fraud Call" if prediction == 1 else "Genuine Call"
            confidence = float(proba[prediction])
            
            return {
                "prediction": label,
                "confidence": round(confidence, 4),
                "text_cleaned": cleaned[:100] + "..." if len(cleaned) > 100 else cleaned
            }
        except Exception as e:
            return {"error": str(e), "prediction": None}
    
    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None
