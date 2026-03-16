"""
Data drift detection using Evidently AI 0.7+.
Compares a reference dataset (training data) against a current batch.
Outputs an HTML report to reports/ and prints a drift summary.
"""
import argparse
import json
import os
import pandas as pd

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently.metrics import DriftedColumnsCount

from src.training.data_utils import FEATURE_COLS


def run_drift_report(reference_path: str, current_path: str, output_dir: str = "reports"):
    reference_df = pd.read_csv(reference_path)[FEATURE_COLS]
    current_df = pd.read_csv(current_path)[FEATURE_COLS]

    reference = Dataset.from_pandas(reference_df, data_definition=DataDefinition())
    current = Dataset.from_pandas(current_df, data_definition=DataDefinition())

    report = Report(metrics=[DataDriftPreset(), DriftedColumnsCount()])
    my_run = report.run(reference_data=reference, current_data=current)

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "drift_report.html")
    my_run.save_html(html_path)
    print(f"Drift report saved -> {html_path}")

    # Extract summary from Evidently 0.7+ dict structure
    result_dict = my_run.dict()
    metrics_list = result_dict.get("metrics", [])

    drifted_cols = None
    share_drifted = None
    for m in metrics_list:
        if "DriftedColumnsCount" in str(m.get("metric_name", "")):
            val = m.get("value", {})
            drifted_cols = int(val.get("count", 0))
            share_drifted = float(val.get("share", 0.0))
            break

    # Default drift threshold: flag if >30% of features drift
    drift_threshold = 0.3
    summary = {
        "n_drifted_features": drifted_cols,
        "share_drifted": round(share_drifted, 3) if share_drifted is not None else None,
        "drift_detected": (share_drifted or 0.0) > drift_threshold,
    }
    print(json.dumps(summary, indent=2))

    json_path = os.path.join(output_dir, "drift_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    if summary["drift_detected"]:
        raise SystemExit(
            f"DRIFT DETECTED: {drifted_cols} features drifted. Trigger retraining."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/train.csv")
    parser.add_argument("--current", default="data/processed/current_batch.csv")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    run_drift_report(args.reference, args.current, args.output_dir)
