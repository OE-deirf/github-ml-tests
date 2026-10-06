"""DVC stage 2 -- train: fit a scikit-learn Pipeline and persist it.

The preprocessing lives INSIDE the Pipeline on purpose: that is what removes the
training/serving skew class of bugs (week 5-6). Never ship separate scaler.pkl /
encoder.pkl files.

Run:  python -m src.churn.train      (or: dvc repro train)
"""

from __future__ import annotations

from typing import Any

import joblib
import mlflow
import mlflow.sklearn as sklearn
import mlflow.data.pandas_dataset as mlflow_pandas

from datetime import datetime
from logging import Logger

import pandas as pd
from pandas import DataFrame

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn import config, scoring

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


def convert_int_to_float(df: DataFrame) -> DataFrame:
    int_cols = df.select_dtypes(include="int").columns
    df[int_cols] = df[int_cols].astype("float64")
    return df


def main() -> None:
    params: dict[str, Any] = config.load_params()
    seed: int = params["seed"]
    target: str = params["prepare"]["target"]
    train_params: dict[str, Any] = params["train"]
    prepare_params: dict[str, Any] = params["prepare"]

    train_df: DataFrame = convert_int_to_float(pd.read_csv(config.TRAIN_CSV))
    valid_df: DataFrame = convert_int_to_float(pd.read_csv(config.VALID_CSV))

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_valid, y_valid = valid_df.drop(columns=[target]), valid_df[target]

    pipeline: Pipeline = build_pipeline(X_train, train_params["model"], seed, train_params)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        dataset: mlflow_pandas.PandasDataset = mlflow_pandas.from_pandas(
            train_df,
            name=config.TRAIN_CSV.name,
            targets=target
        )

        mlflow_params: dict[str, int | Any] = {
            "seed": seed,
            "model": train_params["model"],
            "class_weight": train_params["class_weight"],
            "test_size": prepare_params["test_size"],
            "stratify": prepare_params["stratify"],
        }

        mlflow.log_input(dataset, context="training")
        # model-specific hyperparams (absent for logistic_regression)
        for key in ("n_estimators", "max_depth", "min_samples_leaf"):
            if train_params.get(key) is not None:
                mlflow_params[key] = train_params[key]
        mlflow.log_params({"mlflow_params": mlflow_params})
        mlflow.log_params({"params": config.flatten(params)})

        pipeline.fit(X_train, y_train)

        threshold = params["evaluate"]["threshold"]
        metrics: dict[str, float] = scoring.compute_metrics(pipeline, X_valid, y_valid, threshold)
        metrics.update(
            scoring.slice_metrics(pipeline, X_valid, y_valid, "International plan", threshold)
        )
        metrics["n_valid"] = float(len(valid_df))
        mlflow.log_metrics(metrics)

        lineage_tags: dict[str, Any] = config.lineage_tags()
        mlflow.set_tags({"git_sha": lineage_tags["git_sha"], "dvc_data_hash": lineage_tags["dvc_data_hash"]})
        dirty: str = lineage_tags["git_sha"][1:6] if not lineage_tags['dirty'] else "dirty"
        run_name: str = f"{dirty} {datetime.now():%m%d_%H%M}"
        mlflow.set_tag("mlflow.runName", run_name)

        config.ensure_dirs(config.MODELS_DIR)
        joblib.dump(pipeline, config.MODEL_PATH)
        log.info("Model written to %s", config.MODEL_PATH)

        registered_model: str | None = config.MLFLOW_REGISTERED_MODEL or None
        sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=registered_model,
            input_example=X_train,
        )

        config.MLFLOW_RUN_ID_PATH.write_text(run.info.run_id)
        log.info("MLflow run_id: %s", run.info.run_id)


if __name__ == "__main__":
    main()
