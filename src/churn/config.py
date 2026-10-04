"""Shared configuration and helpers for the churn pipeline.

Single source of truth for paths and for reading ``params.yaml``.
No magic constants scattered across the stage modules.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

# --- Paths -----------------------------------------------------------------
# config.py -> churn -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
print("Project root:", PROJECT_ROOT)
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
METRICS_DIR: Path = PROJECT_ROOT / "metrics"

RAW_CSV: Path = RAW_DIR / "churn.csv"
TRAIN_CSV: Path = PROCESSED_DIR / "train.csv"
VALID_CSV: Path = PROCESSED_DIR / "valid.csv"
MODEL_PATH: Path = MODELS_DIR / "model.joblib"
METRICS_PATH: Path = METRICS_DIR / "metrics.json"

PARAMS_PATH: Path = PROJECT_ROOT / "params.yaml"


# --- Params ----------------------------------------------------------------
def load_params(path: Path | None = None) -> dict[str, Any]:
    """Read ``params.yaml``. Every hyperparameter lives there, never in code."""
    path = path or PARAMS_PATH
    with path.open("r", encoding="utf-8") as fh:
        params: dict[str, Any] = yaml.safe_load(fh)
    return params


# --- Logging ---------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Structured-ish stdlib logging; never ``print()`` in pipeline code."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# --- Small IO helpers ------------------------------------------------------
def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as file_handler:
        json.dump(payload, file_handler, indent=2, sort_keys=True)
        file_handler.write("\n")
