"""Unit tests for the serving layer (no model loaded — uses mocks)."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ── Fixture: mock heavy dependencies before importing app ────────────────────
@pytest.fixture(autouse=True)
def mock_mlflow_and_joblib(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([1])
    fake_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    fake_scaler = MagicMock()
    fake_scaler.transform.side_effect = lambda x: x

    monkeypatch.setattr("mlflow.xgboost.load_model", lambda *a, **kw: fake_model)
    monkeypatch.setattr("joblib.load", lambda *a, **kw: fake_scaler)
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda *a, **kw: None)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.serving.app import app
    return TestClient(app)


SAMPLE_PAYLOAD = {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
    "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_schema(client):
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "probability" in body
    assert body["risk_label"] in ("HIGH", "LOW")


def test_predict_high_probability_gives_high_label(client):
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    body = r.json()
    # mock returns prob=0.8 → HIGH
    assert body["risk_label"] == "HIGH"
    assert body["prediction"] == 1
