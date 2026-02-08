import pandas as pd
import nltk
import string
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

# 1️⃣ Load dataset
data = pd.read_csv("fraud_calls_multilingual.csv")

# Labels already exist, just map them
data['label'] = data['label'].map({'fraud': 1, 'normal': 0})



# Remove empty rows
data.dropna(inplace=True)

# 2️⃣ Text cleaning
def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

data['text'] = data['text'].apply(clean_text)

# 3️⃣ TF-IDF with bigrams
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2
)

X = vectorizer.fit_transform(data['text'])
y = data['label']

# 4️⃣ Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5️⃣ Train model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6️⃣ Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Test with your own call text
def predict_call(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)
    return "Fraud Call" if result[0] == 1 else "Genuine Call"

print("\nTest Prediction:")
print(predict_call("Your bank account is blocked share OTP immediately"))
