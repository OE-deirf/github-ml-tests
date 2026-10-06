"""Metric computation -- ONE implementation, used by both train.py and evaluate.py.

If the training script and the evaluation script compute metrics differently,
you will spend a day chasing a difference that does not exist. One function.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def compute_metrics(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
) -> dict[str, float]:
    proba = pipe.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }


def slice_metrics(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    column: str,
    threshold: float = 0.5,
    min_rows: int = 30,
) -> dict[str, float]:
    """Per-segment F1. The model card and the promotion gate both need this:
    a good average can hide one badly broken segment."""
    out: dict[str, float] = {}
    if column not in X.columns:
        return out
    for value, idx in X.groupby(column).groups.items():
        if len(idx) < min_rows:
            continue
        sub_metrics = compute_metrics(pipe, X.loc[idx], y.loc[idx], threshold)
        key = f"f1_slice_{column}_{value}".replace(" ", "_")
        out[key] = sub_metrics["f1"]
    return out
