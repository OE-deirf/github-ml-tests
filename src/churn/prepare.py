"""DVC stage 1 -- prepare: load raw CSV, clean, split into train/valid.

Run:  python -m src.churn.prepare      (or: dvc repro prepare)
"""

from __future__ import annotations
from logging import Logger
from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.churn import config

log: Logger = config.get_logger(__name__)


def clean(df: pd.DataFrame, target: str, drop_columns: list[str]) -> pd.DataFrame:
    """Deterministic cleaning: drop admin columns, normalise the target to 0/1."""
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])

    # 'Churn' arrives as True/False (bool) or as the strings "True"/"False".
    if df[target].dtype == bool:
        df[target] = df[target].astype(int)
    else:
        df[target] = (
            df[target].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
        )
    if df[target].isna().any():
        raise ValueError(f"Unparseable values in target column '{target}'")

    before: int = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) != before:
        log.warning("Dropped %d duplicate rows", before - len(df))
    return df


def main() -> None:
    params: dict[str, Any] = config.load_params()
    seed: int = params["seed"]
    prepare_params = params["prepare"]
    target = prepare_params["target"]

    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"{config.RAW_CSV} not found. Run: python scripts/fetch_data.py"
        )

    log.info("Reading %s", config.RAW_CSV)
    df = pd.read_csv(config.RAW_CSV)
    log.info("Raw shape: %s", df.shape)

    df: DataFrame = clean(df, target=target, drop_columns=prepare_params["drop_columns"])

    stratify = df[target] if prepare_params["stratify"] else None
    train_df, test_df = train_test_split(
        df,
        test_size=prepare_params["test_size"],
        random_state=seed,
        stratify=stratify,
    )

    config.ensure_dirs(config.PROCESSED_DIR)
    train_df.to_csv(config.TRAIN_CSV, index=False)
    test_df.to_csv(config.VALID_CSV, index=False)

    log.info(
        "train=%d valid=%d churn_rate_train=%.4f churn_rate_valid=%.4f",
        len(train_df),
        len(test_df),
        train_df[target].mean(),
        test_df[target].mean(),
    )


if __name__ == "__main__":
    main()
