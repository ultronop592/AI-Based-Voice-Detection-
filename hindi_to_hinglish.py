import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Load Hindi dataset
data = pd.read_csv("fraud_calls_hindi.csv")

def to_hinglish(text):
    try:
        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except:
        return text

print("⏳ Converting Hindi to Hinglish...")

data["text"] = data["text"].apply(to_hinglish)

data.to_csv("fraud_calls_hinglish.csv", index=False, encoding="utf-8")

print("✅ Hinglish CSV created successfully")
