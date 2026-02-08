import pandas as pd
from googletrans import Translator
import time

translator = Translator()

data = pd.read_csv("fraud_calls.csv")

translated_texts = []

print("⏳ Translating to Hindi...")

for i, text in enumerate(data["text"]):
    try:
        translated = translator.translate(text, src="en", dest="hi").text
    except:
        translated = text

    translated_texts.append(translated)

    # progress update
    if i % 10 == 0:
        print(f"Translated {i} lines")

    time.sleep(0.3)  # VERY IMPORTANT (prevents blocking)

data["text"] = translated_texts

data.to_csv("fraud_calls_hindi.csv", index=False, encoding="utf-8")

print("✅ Hindi CSV created successfully")
