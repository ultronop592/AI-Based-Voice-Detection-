import pyttsx3
import pandas as pd
import os

DATASET_PATH = "C:/AI Based Model/Dataset"

# Limit for faster generation
MAX_FRAUD = 250
MAX_NORMAL = 250

engine = pyttsx3.init()
engine.setProperty("rate", 155)
engine.setProperty("volume", 1.0)

os.makedirs("C:/AI Based Model/Dataset/audio/fraud", exist_ok=True)
os.makedirs("C:/AI Based Model/Dataset/audio/normal", exist_ok=True)

fraud_count = 0
normal_count = 0

print(f"Generating limited sample: {MAX_FRAUD} fraud + {MAX_NORMAL} normal audio files")
print("This should take about 5-10 minutes...")

for csv_file in os.listdir(DATASET_PATH):
    if csv_file.endswith(".csv"):
        print(f"\nProcessing: {csv_file}")
        df = pd.read_csv(os.path.join(DATASET_PATH, csv_file))

        for _, row in df.iterrows():
            # Stop if we have enough samples
            if fraud_count >= MAX_FRAUD and normal_count >= MAX_NORMAL:
                break
                
            text = str(row["text"]).strip()
            label = str(row["label"]).strip().lower()

            if len(text) < 5:
                continue

            if label == "fraud" and fraud_count < MAX_FRAUD:
                filename = f"C:/AI Based Model/Dataset/audio/fraud/fraud_{fraud_count}.wav"
                engine.save_to_file(text, filename)
                fraud_count += 1
                if fraud_count % 50 == 0:
                    print(f"  Fraud: {fraud_count}/{MAX_FRAUD}")
                    
            elif label == "normal" and normal_count < MAX_NORMAL:
                filename = f"C:/AI Based Model/Dataset/audio/normal/normal_{normal_count}.wav"
                engine.save_to_file(text, filename)
                normal_count += 1
                if normal_count % 50 == 0:
                    print(f"  Normal: {normal_count}/{MAX_NORMAL}")
        
        # Exit loop if done
        if fraud_count >= MAX_FRAUD and normal_count >= MAX_NORMAL:
            break

print("\nSynthesizing audio files... Please wait...")
engine.runAndWait()

print(f"\nDone! Generated {fraud_count} fraud + {normal_count} normal = {fraud_count + normal_count} audio files")
