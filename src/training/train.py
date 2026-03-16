"""
Train an XGBoost model on the UCI Heart Disease dataset.
Tracks experiment with MLflow: params, metrics, and model artifact.
"""
import argparse

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.training.data_utils import load_data


def train(data_path: str, n_estimators: int = 100, max_depth: int = 4, lr: float = 0.1):
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("heart-disease-prediction")

    with mlflow.start_run():
        # ── Data ─────────────────────────────────────────────────────────────
        X, y = load_data(data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ── Model ─────────────────────────────────────────────────────────────
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": lr,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # ── Metrics ───────────────────────────────────────────────────────────
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        mlflow.log_metrics(metrics)
        print(metrics)

        # ── Artifacts ─────────────────────────────────────────────────────────
        mlflow.xgboost.log_model(model, name="model")

        # Save scaler for the serving layer
        import os

        import joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.joblib")
        mlflow.log_artifact("models/scaler.joblib")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw/heart.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    train(args.data_path, args.n_estimators, args.max_depth, args.lr)
