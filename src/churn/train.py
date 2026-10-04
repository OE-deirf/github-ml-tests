"""DVC stage 2 -- train: fit a scikit-learn Pipeline and persist it.

The preprocessing lives INSIDE the Pipeline on purpose: that is what removes the
training/serving skew class of bugs (week 5-6). Never ship separate scaler.pkl /
encoder.pkl files.

Run:  python -m src.churn.train      (or: dvc repro train)
"""

from __future__ import annotations
from logging import Logger
from typing import Any

import joblib
import pandas as pd
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn import config

log: Logger = config.get_logger(__name__)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot for object/category columns, scaling for numeric ones."""
    categorical: Any = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric: list[str] = [c for c in X.columns if c not in categorical]
    log.info("categorical=%d numeric=%d", len(categorical), len(numeric))

    transformer: ColumnTransformer = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # handle_unknown='ignore': unseen categories at serving time
                        # must not crash the API (week 7).
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )

    return transformer


def build_estimator(
        model_name: str, seed: int, train_params: dict[str, Any]
) -> RandomForestClassifier | LogisticRegression:
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=train_params["n_estimators"],
            max_depth=train_params["max_depth"],
            min_samples_leaf=train_params["min_samples_leaf"],
            class_weight=train_params["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight=train_params["class_weight"],
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name!r}")


def build_pipeline(
        X: pd.DataFrame, model_name: str, seed: int, train_params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(X)),
            ("model", build_estimator(model_name, seed, train_params)),
        ]
    )


def main() -> None:
    params: dict[str, Any] = config.load_params()
    seed: int = params["seed"]
    target: str = params["prepare"]["target"]
    train_params: dict[str, Any] = params["train"]

    train_df: DataFrame = pd.read_csv(config.TRAIN_CSV)
    X: DataFrame = train_df.drop(columns=[target])
    y: Series = train_df[target]

    pipe: Pipeline = build_pipeline(X, train_params["model"], seed, train_params)
    log.info("Fitting %s on %d rows", train_params["model"], len(X))
    pipe.fit(X, y)

    config.ensure_dirs(config.MODELS_DIR)
    joblib.dump(pipe, config.MODEL_PATH)
    log.info("Model written to %s", config.MODEL_PATH)


if __name__ == "__main__":
    main()
