import pandas as pd

# load datasets
eng = pd.read_csv("fraud_calls.csv")
hin = pd.read_csv("fraud_calls_hindi.csv")
hing = pd.read_csv("fraud_calls_hinglish.csv")

# merge
final_data = pd.concat([eng, hin, hing], ignore_index=True)

# shuffle
final_data = final_data.sample(frac=1).reset_index(drop=True)

# save final dataset
final_data.to_csv("fraud_calls_multilingual.csv", index=False)

print("✅ Merged dataset created:", final_data.shape)
