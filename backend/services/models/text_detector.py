"""
Text Fraud Detection Model
TF-IDF + Logistic Regression for detecting fraudulent call transcripts

Based on DeepFake Audio/call_fraud_detection.py
"""

import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TextFraudDetector:
    """
    Fraud call detector using TF-IDF + Logistic Regression.
    Trains on fraud_calls_multilingual.csv dataset.
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.stop_words = None
    
    def train(self, csv_path: str) -> bool:
        """Train the model from CSV dataset."""
        try:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            
            # Load data
            data = pd.read_csv(csv_path)
            data['label'] = data['label'].map({'fraud': 1, 'normal': 0})
            data.dropna(inplace=True)
            
            # Clean text
            data['text'] = data['text'].apply(self._clean_text)
            
            # TF-IDF with bigrams
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=2
            )
            X = self.vectorizer.fit_transform(data['text'])
            y = data['label']
            
            # Train
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(X, y)
            
            return True
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean text for prediction."""
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
            return {"error": "Model not trained", "prediction": None}
        
        cleaned = self._clean_text(text)
        vector = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vector)[0]
        proba = self.model.predict_proba(vector)[0]
        
        return {
            "prediction": "Fraud Call" if pred == 1 else "Genuine Call",
            "confidence": float(proba[pred]),
            "text_cleaned": cleaned[:100] + "..." if len(cleaned) > 100 else cleaned
        }
    
    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None
