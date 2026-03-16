# End-to-End MLOps Pipeline — Heart Disease Prediction

A production-grade MLOps pipeline demonstrating the full model lifecycle:
data versioning → experiment tracking → REST serving → drift monitoring → CI/CD.

**Use case:** Binary classification of heart disease risk using the
[UCI Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
(303 patients, 13 clinical features).

---

## Architecture

```
┌─────────────┐    DVC     ┌──────────────┐   MLflow   ┌──────────────────┐
│  Raw Data   │──────────▶ │  Processed   │──────────▶ │  XGBoost Model   │
│  (heart.csv)│            │  Train/Test  │            │  + Scaler        │
└─────────────┘            └──────────────┘            └────────┬─────────┘
                                                                 │
                                              ┌──────────────────▼──────────────┐
                                              │  FastAPI  /predict endpoint     │
                                              │  Docker container  :8000        │
                                              └──────────────────┬──────────────┘
                                                                 │
                                              ┌──────────────────▼──────────────┐
                                              │  Evidently AI drift reports     │
                                              │  Scheduled via GitHub Actions   │
                                              └─────────────────────────────────┘
```

## Stack

| Layer | Tool |
|---|---|
| Training | XGBoost, scikit-learn |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Serving | FastAPI + Uvicorn |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Drift monitoring | Evidently AI |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset → data/raw/heart.csv
#    https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

# 3. Train (tracks to local MLflow)
mlflow server --host 0.0.0.0 --port 5000 &
python -m src.training.train --data-path data/raw/heart.csv

# 4. View experiment
open http://localhost:5000

# 5. Serve (after promoting model in MLflow UI to "Production")
uvicorn src.serving.app:app --reload

# 6. Test the endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,
       "restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,
       "slope":2,"ca":2,"thal":3}'
```

## Running with Docker

```bash
docker build -t mlops-pipeline .
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  mlops-pipeline
```

## Running Tests

```bash
pytest tests/ -v
```

## Drift Detection

```bash
python -m src.monitoring.drift_report \
  --reference data/processed/train.csv \
  --current data/processed/current_batch.csv
# HTML report → reports/drift_report.html
```

## CI/CD

GitHub Actions runs on every push to `main`:
1. **Lint** with Ruff
2. **Unit tests** with pytest
3. **Docker build + push** to Docker Hub (main branch only)
4. **Drift check** (manual trigger / scheduled)

See [.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## Results

| Metric | Value |
|---|---|
| Accuracy | ~0.87 |
| F1 Score | ~0.88 |
| ROC-AUC | ~0.93 |

*(Results on 20% held-out test set; exact values logged in MLflow)*
