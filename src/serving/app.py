"""
FastAPI inference endpoint.
Loads the latest MLflow model + scaler and serves predictions.
"""
import os
import joblib
import mlflow.xgboost
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Heart Disease Predictor", version="1.0")

# ── Load artifacts at startup ────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = os.getenv("MODEL_URI", "models:/heart-disease-prediction/Production")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.xgboost.load_model(MODEL_URI)
scaler = joblib.load(SCALER_PATH)


class PatientFeatures(BaseModel):
    age: float = Field(..., example=52)
    sex: int = Field(..., ge=0, le=1, example=1)
    cp: int = Field(..., ge=0, le=3, example=0)
    trestbps: float = Field(..., example=125)
    chol: float = Field(..., example=212)
    fbs: int = Field(..., ge=0, le=1, example=0)
    restecg: int = Field(..., ge=0, le=2, example=1)
    thalach: float = Field(..., example=168)
    exang: int = Field(..., ge=0, le=1, example=0)
    oldpeak: float = Field(..., example=1.0)
    slope: int = Field(..., ge=0, le=2, example=2)
    ca: int = Field(..., ge=0, le=4, example=2)
    thal: int = Field(..., ge=0, le=3, example=3)


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
    features_scaled = scaler.transform(features)
    pred = int(model.predict(features_scaled)[0])
    prob = float(model.predict_proba(features_scaled)[0][1])

    return Prediction(
        prediction=pred,
        probability=round(prob, 4),
        risk_label="HIGH" if prob >= 0.5 else "LOW",
    )
