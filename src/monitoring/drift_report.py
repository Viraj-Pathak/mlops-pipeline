"""
Data drift detection using Evidently AI.
Compares a reference dataset (training data) against a current batch.
Outputs an HTML report to reports/ and prints a drift summary.
"""
import argparse
import json
import os
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import DatasetDriftMetric

from src.training.data_utils import FEATURE_COLS


def run_drift_report(reference_path: str, current_path: str, output_dir: str = "reports"):
    reference = pd.read_csv(reference_path)[FEATURE_COLS]
    current = pd.read_csv(current_path)[FEATURE_COLS]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "drift_report.html")
    report.save_html(html_path)
    print(f"Drift report saved → {html_path}")

    # Extract summary for CI gate
    result = report.as_dict()
    drift_metric = result["metrics"][0]["result"]
    n_drifted = drift_metric["number_of_drifted_columns"]
    share_drifted = drift_metric["share_of_drifted_columns"]

    summary = {
        "n_drifted_features": n_drifted,
        "share_drifted": round(share_drifted, 3),
        "drift_detected": drift_metric["dataset_drift"],
    }
    print(json.dumps(summary, indent=2))

    json_path = os.path.join(output_dir, "drift_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Fail fast if drift is detected (useful for CI)
    if summary["drift_detected"]:
        raise SystemExit(
            f"DRIFT DETECTED: {n_drifted} features drifted "
            f"({share_drifted:.0%}). Trigger retraining."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/train.csv")
    parser.add_argument("--current", default="data/processed/current_batch.csv")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    run_drift_report(args.reference, args.current, args.output_dir)
