"""
Data loading and preprocessing utilities.
Uses the UCI Heart Disease dataset (heart.csv).
Download: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
"""

import pandas as pd

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]
TARGET_COL = "target"


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = df.dropna()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    """Lightweight feature engineering — can be extended."""
    X = X.copy()
    # Bin age into decades for an extra ordinal feature
    X["age_bin"] = (X["age"] // 10).astype(int)
    return X
