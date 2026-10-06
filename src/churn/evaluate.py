"""DVC stage 3 -- evaluate: score the model on the validation split.

Writes metrics/metrics.json, which is declared as `metrics` in dvc.yaml with
`cache: false` -- so it goes into Git and `dvc metrics diff` can compare commits.

Run:  python -m src.churn.evaluate      (or: dvc repro evaluate)
"""

from __future__ import annotations
from typing import Any

import joblib
import mlflow
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.churn import config

log = config.get_logger(__name__)


def main() -> None:
    params: dict[str, Any] = config.load_params()
    target = params["prepare"]["target"]
    evaluate_params = params["evaluate"]

    pipe = joblib.load(config.MODEL_PATH)
    valid_df: DataFrame = pd.read_csv(config.VALID_CSV)
    X = valid_df.drop(columns=[target])
    y = valid_df[target]

    proba = pipe.predict_proba(X)[:, 1]
    y_pred = (proba >= evaluate_params["threshold"]).astype(int)

    metrics: dict[str, float | int] = {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "n_valid": int(len(valid_df)),
        "positive_rate": round(float(y.mean()), 4),
    }

    config.write_json(config.METRICS_PATH, metrics)
    log.info("Metrics: %s", metrics)

    run_id = config.MLFLOW_RUN_ID_PATH.read_text().strip()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({
            "threshold": evaluate_params["threshold"],
            "min_f1": evaluate_params["min_f1"],
        })
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_artifact(str(config.METRICS_PATH))

    if metrics["f1"] < evaluate_params["min_f1"]:
        log.warning(
            "F1 %.4f is BELOW the gate %.4f -- in week 6 this will fail the pipeline",
            metrics["f1"],
            evaluate_params["min_f1"],
        )


if __name__ == "__main__":
    main()
