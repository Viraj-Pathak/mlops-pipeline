"""
Convert raw UCI Cleveland data → heart.csv with proper headers.
Also creates train/current_batch splits for drift detection.
"""
import pandas as pd
import numpy as np

COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target",
]

df = pd.read_csv("data/raw/heart_raw.csv", header=None, names=COLS, na_values="?")
print(f"Raw rows: {len(df)}, missing: {df.isna().sum().sum()}")

df = df.dropna().reset_index(drop=True)

# Binarise target (0 = no disease, 1 = disease)
df["target"] = (df["target"] > 0).astype(int)

# Cast to int where appropriate
for col in ["sex","cp","fbs","restecg","exang","slope","ca","thal","target"]:
    df[col] = df[col].astype(int)

df.to_csv("data/raw/heart.csv", index=False)
print(f"Saved data/raw/heart.csv  shape={df.shape}  positive_rate={df['target'].mean():.2%}")

# ── Train / current-batch split for drift detection ──────────────────────────
import os
os.makedirs("data/processed", exist_ok=True)

train = df.sample(frac=0.7, random_state=42)
current = df.drop(train.index)

train.to_csv("data/processed/train.csv", index=False)
current.to_csv("data/processed/current_batch.csv", index=False)
print(f"Train={len(train)}  current_batch={len(current)}")
