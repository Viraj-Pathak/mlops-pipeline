"""
FastAPI inference endpoint.
Loads the latest MLflow model + scaler on first request (lazy load).
"""
import os
import joblib
import mlflow.xgboost
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ── Config ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/heart-disease-prediction/Production")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")

_model = None
_scaler = None


def get_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.xgboost.load_model(MODEL_URI)
    return _model


def get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _scaler


app = FastAPI(title="Heart Disease Predictor", version="1.0")


class PatientFeatures(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
    }}}

    age: float
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float
    chol: float
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)


class Prediction(BaseModel):
    prediction: int
    probability: float
    risk_label: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientFeatures):
    features = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps,
        patient.chol, patient.fbs, patient.restecg, patient.thalach,
        patient.exang, patient.oldpeak, patient.slope, patient.ca, patient.thal,
    ]])
    features_scaled = get_scaler().transform(features)
    pred = int(get_model().predict(features_scaled)[0])
    prob = float(get_model().predict_proba(features_scaled)[0][1])

    return Prediction(
        prediction=pred,
        probability=round(prob, 4),
        risk_label="HIGH" if prob >= 0.5 else "LOW",
    )
