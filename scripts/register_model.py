"""
Registers the latest MLflow run's model to the Model Registry
and promotes it to 'Production'. Run once after training.
"""
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlruns/mlflow.db"
EXPERIMENT_NAME = "heart-disease-prediction"
MODEL_NAME = "heart-disease-prediction"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# Get the best run by ROC-AUC
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.roc_auc DESC"],
    max_results=1,
)
if not runs:
    raise RuntimeError("No runs found. Run train.py first.")

best_run = runs[0]
run_id = best_run.info.run_id
roc_auc = best_run.data.metrics.get("roc_auc", 0)
print(f"Best run: {run_id}  ROC-AUC={roc_auc:.4f}")

# Register model
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registered version {mv.version}")

# Promote to Production (new MLflow uses aliases instead of stages)
try:
    client.set_registered_model_alias(MODEL_NAME, "Production", mv.version)
    print(f"Set alias 'Production' -> version {mv.version}")
except Exception:
    # Older MLflow — use transition_model_version_stage
    client.transition_model_version_stage(
        name=MODEL_NAME, version=mv.version, stage="Production"
    )
    print(f"Promoted version {mv.version} to Production stage")
