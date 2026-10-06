"""Security-focused tests for src/churn/train.py

Test groups by security category:

  [SEC-INJECT]    joblib.dump() serializes the Pipeline as a pickle binary.
                  No HMAC or digital-signature file is written alongside the model,
                  so the evaluate stage cannot verify the artifact has not been
                  tampered with between DVC stages.  A crafted replacement
                  model.joblib executes arbitrary code when evaluate.py calls
                  joblib.load().
  [SEC-TYPE]      Missing type guards in build_estimator() and main(): wrong-typed
                  inputs (None model_name, string seed, non-dict params) produce
                  internal Python / sklearn errors instead of controlled messages.
  [SEC-INFOLEAKW] KeyError messages expose the params.yaml schema (expected keys
                  such as 'n_estimators', 'seed', 'train', 'prepare').
                  ValueError for an unknown model name echoes the raw input value.
  [SEC-SILENT]    Silent failures: (1) single-class target trains a degenerate
                  model that will fail at evaluate time without any warning;
                  (2) an existing model file is silently overwritten on each run.
  [SEC-DOS]       No upper bound on n_estimators; a crafted params.yaml with
                  n_estimators=1_000_000 is accepted at construction time and
                  would exhaust memory at fit time.  No row limit on the training
                  CSV.
  [SEC-PATH]      config.MODEL_PATH is used with no project-boundary check; a
                  path outside the project root is written without error.

Run:
    pytest tests/churn/test_train_security.py -v

CURRENT BEHAVIOUR / EXPECTED BEHAVIOUR annotations mark confirmed security gaps.
Fixing them requires modifying the source, which is out of scope here.

Security hardening suggestions
-------------------------------
joblib.dump():
  - Write an HMAC / SHA-256 hash file alongside the model artifact:
      model_bytes = config.MODEL_PATH.read_bytes()
      config.MODEL_PATH.with_suffix(".sha256").write_text(
          hashlib.sha256(model_bytes).hexdigest() + "\\n"
      )
  - Track the hash file in DVC so it travels with the model.
  - In evaluate.py, verify the hash before calling joblib.load().

build_estimator():
  - Enforce a strict allowlist instead of the open-ended if/elif chain:
      ALLOWED_MODELS = frozenset({"random_forest", "logistic_regression"})
      if not isinstance(model_name, str) or model_name not in ALLOWED_MODELS:
          raise ValueError(
              f"model must be one of {sorted(ALLOWED_MODELS)}, got {model_name!r}"
          )
  - Add explicit isinstance guards for seed and hyperparameter values before
    passing them to sklearn.
  - Cap n_estimators to a safe maximum:
      if n_estimators > MAX_N_ESTIMATORS:
          raise ValueError(f"n_estimators exceeds maximum: {MAX_N_ESTIMATORS}")

main():
  - Validate params before key access:
      if not isinstance(params, dict):
          raise TypeError("load_params() must return a dict")
      for key in ("seed", "prepare", "train"):
          if key not in params:
              raise ValueError(f"params.yaml is missing required key: '{key}'")
  - Validate data immediately after loading:
      if train_df.empty:
          raise ValueError("Training CSV is empty")
      if len(y.unique()) < 2:
          raise ValueError("Training target must have at least 2 classes")
  - Warn before overwriting an existing model:
      if config.MODEL_PATH.exists():
          log.warning("Overwriting existing model at %s", config.MODEL_PATH)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import numpy as np
from numpy.random import Generator
import pandas as pd
import pytest
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.pipeline import Pipeline

from src.churn import config
from src.churn.train import build_estimator, build_preprocessor, main


# ---------------------------------------------------------------------------
# Pickle code-execution probe (module-level so it is importable by pickle)
# ---------------------------------------------------------------------------

_PROBE_REGISTRY: list[str] = []


def _probe_callback() -> None:
    """Appended to _PROBE_REGISTRY when the pickled reduction is called at load time."""
    _PROBE_REGISTRY.append("joblib_load_executed_code")


class _PickleCodeProbe:
    """Benign stand-in for a malicious pickle payload.

    __reduce__ stores (_probe_callback, ()) in the serialised bytes.
    When joblib.load() deserialises the file it calls _probe_callback(),
    demonstrating the arbitrary-code-execution vector.
    """

    def __reduce__(self) -> tuple[Any, Any]:
        return (_probe_callback, ())


# ---------------------------------------------------------------------------
# Helper returned by the train_env fixture
# ---------------------------------------------------------------------------


class _TrainEnv:
    def __init__(
        self,
        csv_path: Path,
        valid_csv_path: Path,
        model_path: Path,
        base_params: dict[str, Any],
    ) -> None:
        self.csv_path: Path = csv_path
        self.valid_csv_path: Path = valid_csv_path
        self.model_path: Path = model_path
        self.base_params: dict[str, Any] = base_params

    def run(self, params: dict[str, Any] | None = None) -> None:
        _params: dict[str, Any] = params if params is not None else self.base_params
        with patch("src.churn.train.config.load_params", return_value=_params):
            main()

    def load_model(self) -> Any:
        return joblib.load(self.model_path)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_params() -> dict[str, Any]:
    """Minimal valid params dict mirroring the prepare/train/evaluate sections of params.yaml."""
    return {
        "seed": 42,
        "prepare": {
            "target": "Churn",
            "test_size": 0.2,
            "stratify": True,
        },
        "evaluate": {
            "threshold": 0.5,
        },
        "train": {
            "model": "logistic_regression",
            "class_weight": "balanced",
        },
    }


@pytest.fixture()
def train_df() -> pd.DataFrame:
    """Eight-row training DataFrame with binary Churn target (both classes present)."""
    return pd.DataFrame({
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 65.0, 30.0, 77.0, 12.5],
        "Tenure": [1, 12, 24, 6, 36, 48, 60, 3],
        "Churn": [1, 0, 1, 0, 1, 0, 0, 1],
    })


@pytest.fixture()
def valid_df() -> pd.DataFrame:
    """Four-row validation DataFrame with binary Churn target (both classes present)."""
    return pd.DataFrame({
        "MonthlyCharges": [30.0, 55.0, 50.0, 40.0],
        "Tenure": [2, 10, 20, 5],
        "Churn": [1, 0, 1, 0],
    })


@pytest.fixture()
def train_env(
    tmp_path: Path,
    base_params: dict[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> _TrainEnv:
    """Wire filesystem paths to tmp_path and return a _TrainEnv helper."""
    csv_path: Path = tmp_path / "train.csv"
    train_df.to_csv(csv_path, index=False)
    valid_csv_path: Path = tmp_path / "valid.csv"
    valid_df.to_csv(valid_csv_path, index=False)
    model_path: Path = tmp_path / "model.joblib"

    monkeypatch.setattr(config, "TRAIN_CSV", csv_path)
    monkeypatch.setattr(config, "VALID_CSV", valid_csv_path)
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "MLFLOW_RUN_ID_PATH", tmp_path / "mlflow_run_id")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    monkeypatch.setattr(config, "MLFLOW_REGISTERED_MODEL", "")

    return _TrainEnv(
        csv_path=csv_path,
        valid_csv_path=valid_csv_path,
        model_path=model_path,
        base_params=base_params,
    )


# ===========================================================================
# TestBuildPreprocessorNormalOperation
# ===========================================================================


class TestBuildPreprocessorNormalOperation:

    def test_returns_column_transformer_for_mixed_dataframe(self) -> None:
        """Positive test: build_preprocessor() returns a ColumnTransformer for
        a DataFrame containing both numeric and categorical columns."""
        from sklearn.compose import ColumnTransformer

        df = pd.DataFrame({
            "MonthlyCharges": [29.85, 56.95],
            "Contract": ["Month-to-month", "One year"],
            "Churn": [1, 0],
        })
        result: ColumnTransformer = build_preprocessor(df)
        assert isinstance(result, ColumnTransformer)

    def test_all_numeric_columns_produces_valid_transformer(self) -> None:
        """Positive test: a DataFrame with only numeric columns is handled without error."""
        from sklearn.compose import ColumnTransformer

        df = pd.DataFrame({"FeatureA": [1.0, 2.0, 3.0], "FeatureB": [4.0, 5.0, 6.0]})
        result: ColumnTransformer = build_preprocessor(df)
        assert isinstance(result, ColumnTransformer)


# ===========================================================================
# TestBuildPreprocessorInputValidation  [SEC-TYPE, SEC-SILENT]
# ===========================================================================


class TestBuildPreprocessorInputValidation:

    def test_empty_dataframe_accepted_without_error(self) -> None:
        """[SEC-SILENT] build_preprocessor() accepts an empty DataFrame silently.

        CURRENT BEHAVIOUR: ColumnTransformer is constructed with empty column lists.
        EXPECTED BEHAVIOUR: raise ValueError("DataFrame must not be empty")

        SECURITY GAP CONFIRMED: a corrupted data pipeline that passes an empty
        DataFrame through is not detected at preprocessing time.
        """
        empty_df = pd.DataFrame()
        result: ColumnTransformer = build_preprocessor(empty_df)  # must NOT raise
        assert result is not None, (
            "SECURITY GAP CONFIRMED: empty DataFrame accepted without validation"
        )

    def test_none_dataframe_causes_uncontrolled_attribute_error(self) -> None:
        """[SEC-TYPE] Passing None as df raises AttributeError from pandas internals,
        not a controlled TypeError from build_preprocessor().

        CURRENT BEHAVIOUR: AttributeError: 'NoneType' object has no attribute
          'select_dtypes'
        EXPECTED BEHAVIOUR: raise TypeError("df must be a DataFrame, got NoneType")
        """
        with pytest.raises((AttributeError, TypeError)):
            build_preprocessor(None)  # type: ignore[arg-type]

    def test_dict_instead_of_dataframe_causes_uncontrolled_error(self) -> None:
        """[SEC-TYPE] Passing a plain dict raises an uncontrolled AttributeError
        from pandas, not a controlled TypeError from build_preprocessor().

        CURRENT BEHAVIOUR: AttributeError: 'dict' object has no attribute 'select_dtypes'
        EXPECTED BEHAVIOUR: raise TypeError("df must be a DataFrame, got dict")
        """
        with pytest.raises((AttributeError, TypeError)):
            build_preprocessor({"col": [1, 2, 3]})  # type: ignore[arg-type]


# ===========================================================================
# TestBuildEstimatorNormalOperation
# ===========================================================================


class TestBuildEstimatorNormalOperation:

    def test_random_forest_returns_random_forest_classifier(self) -> None:
        """Positive test: 'random_forest' builds a RandomForestClassifier."""
        from sklearn.ensemble import RandomForestClassifier

        estimator: RandomForestClassifier | LogisticRegression = build_estimator(
            "random_forest",
            42,
            {
                "n_estimators": 10,
                "max_depth": 3,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
            },
        )
        assert isinstance(estimator, RandomForestClassifier)

    def test_logistic_regression_returns_logistic_regression(self) -> None:
        """Positive test: 'logistic_regression' builds a LogisticRegression."""
        from sklearn.linear_model import LogisticRegression

        estimator: RandomForestClassifier | LogisticRegression = build_estimator(
            "logistic_regression",
            42,
            {"class_weight": "balanced"},
        )
        assert isinstance(estimator, LogisticRegression)

    def test_unknown_model_name_raises_value_error(self) -> None:
        """Positive negative test: an unrecognised model name raises ValueError.

        This is a correctly handled case -- the gap is in the error message content
        (see TestBuildEstimatorInputValidation).
        """
        with pytest.raises(ValueError):
            build_estimator("gradient_boosting", 42, {})


# ===========================================================================
# TestBuildEstimatorInputValidation  [SEC-TYPE, SEC-INFOLEAKW]
# ===========================================================================


class TestBuildEstimatorInputValidation:

    def test_unknown_model_name_error_message_echoes_raw_input(self) -> None:
        """[SEC-INFOLEAKW] The ValueError for an unknown model name echoes the raw
        input value back in the error message.

        CURRENT BEHAVIOUR: ValueError("Unknown model: '../../../../etc/passwd'")
        EXPECTED BEHAVIOUR: a fixed message that does not reflect untrusted input:
          ValueError("model must be one of: random_forest, logistic_regression")

        SECURITY GAP CONFIRMED: if model_name originates from user-supplied data
        (e.g. an edited params.yaml), the unvalidated value is reflected in the
        error response.
        """
        arbitrary_input = "../../../../etc/passwd"
        with pytest.raises(ValueError) as exc_info:
            build_estimator(arbitrary_input, 42, {})
        assert arbitrary_input in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: input value echoed in ValueError message"
        )

    def test_none_model_name_falls_through_to_value_error_with_none_in_message(
        self,
    ) -> None:
        """[SEC-TYPE] None as model_name has no type guard -- it falls through all
        if/elif branches and raises ValueError with None in the message.

        CURRENT BEHAVIOUR: ValueError("Unknown model: None")
        EXPECTED BEHAVIOUR: raise TypeError("model_name must be a str, got NoneType")

        SECURITY GAP CONFIRMED: no isinstance(model_name, str) check before
        the equality comparisons.
        """
        with pytest.raises(ValueError) as exc_info:
            build_estimator(None, 42, {})  # type: ignore[arg-type]
        assert "None" in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: None accepted without type guard"
        )

    def test_missing_hyperparameter_raises_key_error_exposing_schema(self) -> None:
        """[SEC-INFOLEAKW] A missing hyperparameter key raises KeyError whose
        message exposes the params.yaml schema expected by build_estimator().

        CURRENT BEHAVIOUR: KeyError('n_estimators') -- expected key name leaked.
        EXPECTED BEHAVIOUR:
          raise ValueError("Missing required hyperparameter: 'n_estimators'")

        SECURITY GAP CONFIRMED: KeyError message reveals the expected structure
        of params.yaml to any caller who can observe the exception.
        """
        with pytest.raises(KeyError) as exc_info:
            build_estimator("random_forest", 42, {})  # no n_estimators
        assert "n_estimators" in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: KeyError exposes params.yaml schema"
        )

    def test_string_seed_accepted_silently_at_construction_time(self) -> None:
        """[SEC-TYPE] A string value for seed is accepted by LogisticRegression
        at construction time without raising any error from our code or sklearn.

        CURRENT BEHAVIOUR: LogisticRegression(random_state="not_a_seed") is
          constructed silently -- no error at construction, and lbfgs solver
          may not exercise random_state at fit time either.
        EXPECTED BEHAVIOUR: raise TypeError("seed must be an int, got str")
          from build_estimator() itself, before sklearn is invoked.

        SECURITY GAP CONFIRMED: no isinstance check on seed before forwarding
        to sklearn -- a string seed propagates silently through construction.
        """
        estimator: RandomForestClassifier | LogisticRegression = build_estimator(
            "logistic_regression",
            "not_a_seed",  # pyright: ignore[reportArgumentType]
            {"class_weight": "balanced"},
        )
        assert estimator is not None, (
            "SECURITY GAP CONFIRMED: string seed accepted without type guard"
        )


# ===========================================================================
# TestBuildEstimatorDOSVectors  [SEC-DOS]
# ===========================================================================


class TestBuildEstimatorDOSVectors:

    def test_huge_n_estimators_accepted_at_construction_time(self) -> None:
        """[SEC-DOS] A very large n_estimators value is accepted by build_estimator()
        without any validation at construction time.

        CURRENT BEHAVIOUR: RandomForestClassifier(n_estimators=1_000_000) is
          returned without error.  Calling .fit() on this estimator would exhaust
          process memory and block the DVC pipeline.
        EXPECTED BEHAVIOUR:
          raise ValueError("n_estimators exceeds maximum allowed value")

        SECURITY GAP CONFIRMED: a crafted params.yaml can trigger an OOM
        denial-of-service at fit time with no early detection.
        """
        from sklearn.ensemble import RandomForestClassifier

        estimator = build_estimator(
            "random_forest",
            42,
            {
                "n_estimators": 1_000_000,
                "max_depth": 3,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
            },
        )
        assert isinstance(estimator, RandomForestClassifier), (
            "SECURITY GAP CONFIRMED: n_estimators=1_000_000 accepted without guard"
        )

    def test_string_n_estimators_accepted_silently_at_construction_time(
        self,
    ) -> None:
        """[SEC-TYPE] A string value for n_estimators is accepted by
        RandomForestClassifier at construction time without any error.

        CURRENT BEHAVIOUR: RandomForestClassifier(n_estimators="100") is
          constructed silently -- the type error is deferred to fit() time.
        EXPECTED BEHAVIOUR: raise ValueError("n_estimators must be an int, got str")
          from build_estimator() itself, before sklearn is invoked.

        SECURITY GAP CONFIRMED: hyperparameter type validation is absent from
        build_estimator(), allowing invalid values to propagate undetected
        until fit time (or silently corrupt training results).
        """
        from sklearn.ensemble import RandomForestClassifier

        estimator: RandomForestClassifier | LogisticRegression = build_estimator(
            "random_forest",
            42,
            {
                "n_estimators": "100",
                "max_depth": 3,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
            },
        )
        assert isinstance(estimator, RandomForestClassifier), (
            "SECURITY GAP CONFIRMED: string n_estimators accepted without type guard"
        )


# ===========================================================================
# TestMainNormalOperation
# ===========================================================================


class TestMainNormalOperation:

    def test_model_file_created_at_model_path(self, train_env: _TrainEnv) -> None:
        """Positive test: main() writes a model artifact to config.MODEL_PATH."""
        train_env.run()
        assert train_env.model_path.exists()

    def test_written_model_is_sklearn_pipeline(self, train_env: _TrainEnv) -> None:
        """Positive test: the artifact loaded from MODEL_PATH is a sklearn Pipeline."""
        train_env.run()
        pipe = train_env.load_model()
        assert isinstance(pipe, Pipeline)

    def test_model_pipeline_contains_preprocess_and_model_steps(
        self, train_env: _TrainEnv
    ) -> None:
        """Positive test: the Pipeline has exactly the two expected named steps."""
        train_env.run()
        pipe = train_env.load_model()
        assert list(pipe.named_steps.keys()) == ["preprocess", "model"]


# ===========================================================================
# TestMainParamsSecurity  [SEC-TYPE, SEC-INFOLEAKW]
# ===========================================================================


class TestMainParamsSecurity:

    def test_none_params_causes_uncontrolled_type_error(self) -> None:
        """[SEC-TYPE] When load_params() returns None, main() raises an uncontrolled
        TypeError from Python's subscript on NoneType, not a controlled error.

        CURRENT BEHAVIOUR: TypeError: 'NoneType' object is not subscriptable
        EXPECTED BEHAVIOUR:
          raise TypeError("load_params() must return a dict, got NoneType")
        """
        with pytest.raises(TypeError):
            with patch("src.churn.train.config.load_params", return_value=None):
                main()

    def test_missing_seed_key_raises_key_error_exposing_schema(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-INFOLEAKW] A params dict without 'seed' raises KeyError('seed'),
        exposing the expected params.yaml schema to the caller.

        CURRENT BEHAVIOUR: KeyError('seed')
        EXPECTED BEHAVIOUR:
          raise ValueError("params.yaml is missing required key: 'seed'")

        SECURITY GAP CONFIRMED: KeyError message exposes the internal params
        structure expected by main().
        """
        params_no_seed: dict[str, dict[str, str]] = {
            "prepare": {"target": "Churn"},
            "train": {"model": "logistic_regression", "class_weight": "balanced"},
        }
        with pytest.raises(KeyError) as exc_info:
            train_env.run(params=params_no_seed)
        assert "seed" in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: KeyError exposes params.yaml schema"
        )

    def test_missing_train_section_raises_key_error_exposing_schema(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-INFOLEAKW] A params dict without 'train' raises KeyError('train'),
        exposing the expected params.yaml schema.

        CURRENT BEHAVIOUR: KeyError('train')
        EXPECTED BEHAVIOUR:
          raise ValueError("params.yaml is missing required section: 'train'")

        SECURITY GAP CONFIRMED: KeyError message exposes params.yaml schema.
        """
        params_no_train: dict[str, int | dict[str, str]] = {"seed": 42, "prepare": {"target": "Churn"}}
        with pytest.raises(KeyError) as exc_info:
            train_env.run(params=params_no_train)
        assert "train" in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: KeyError exposes params.yaml schema"
        )

    def test_missing_prepare_section_raises_key_error_exposing_schema(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-INFOLEAKW] A params dict without 'prepare' raises KeyError('prepare'),
        exposing the expected params.yaml schema.

        CURRENT BEHAVIOUR: KeyError('prepare')
        EXPECTED BEHAVIOUR:
          raise ValueError("params.yaml is missing required section: 'prepare'")

        SECURITY GAP CONFIRMED: KeyError message exposes params.yaml schema.
        """
        params_no_prepare: dict[str, int | dict[str, str]] = {
            "seed": 42,
            "train": {"model": "logistic_regression", "class_weight": "balanced"},
        }
        with pytest.raises(KeyError) as exc_info:
            train_env.run(params=params_no_prepare)
        assert "prepare" in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: KeyError exposes params.yaml schema"
        )

    def test_string_seed_raises_uncontrolled_error_at_estimator_construction(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-TYPE] A string value for 'seed' passes through main()'s key access
        silently and raises an internal sklearn error at estimator construction.

        CURRENT BEHAVIOUR: sklearn raises InvalidParameterError (ValueError subclass)
          with an internal message about random_state.
        EXPECTED BEHAVIOUR:
          raise ValueError("params.yaml: seed must be an int, got str")
          from main() itself, before any sklearn call.

        SECURITY GAP CONFIRMED: seed is not validated before being forwarded
        to build_estimator() as the random_state argument.
        """
        params_str_seed: dict[str, Any | str] = {**train_env.base_params, "seed": "forty-two"}
        with pytest.raises((TypeError, ValueError)):
            train_env.run(params=params_str_seed)

    def test_none_target_raises_uncontrolled_error_from_pandas(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-TYPE] When params['prepare']['target'] is None, pandas raises an
        uncontrolled error instead of a controlled ValueError from main().

        CURRENT BEHAVIOUR: KeyError or ValueError from pandas when None is passed
          to DataFrame.drop(columns=[None]).
        EXPECTED BEHAVIOUR:
          raise ValueError("params.yaml: prepare.target must be a non-empty string")

        SECURITY GAP CONFIRMED: target is not type-validated before being used
        as a DataFrame column selector.
        """
        params_none_target: dict[str, Any | dict[str, None]] = {**train_env.base_params, "prepare": {"target": None}}
        with pytest.raises((KeyError, ValueError, TypeError)):
            train_env.run(params=params_none_target)


# ===========================================================================
# TestMainDataSecurity  [SEC-SILENT, SEC-DOS, SEC-INFOLEAKW]
# ===========================================================================


class TestMainDataSecurity:

    def test_empty_training_csv_raises_uncontrolled_sklearn_error(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-SILENT] An empty training CSV does not raise a controlled error
        from main() -- the failure is delayed until sklearn's fit() call.

        CURRENT BEHAVIOUR: ValueError from sklearn ("Found array with 0 sample(s)")
          raised LATE at fit() time.
        EXPECTED BEHAVIOUR: raise ValueError("Training CSV is empty")
          checked EARLY after pd.read_csv(), with a controlled message.

        SECURITY GAP CONFIRMED: (a) the check is delayed; (b) the error message
        is an uncontrolled sklearn internal string.
        """
        pd.DataFrame(columns=["MonthlyCharges", "Tenure", "Churn"]).to_csv(
            train_env.csv_path, index=False
        )
        with pytest.raises(ValueError):
            train_env.run()

    def test_single_class_target_raises_at_metric_computation(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-SILENT] Training on a single-class target now raises IndexError during
        scoring.compute_metrics() rather than silently writing a degenerate model.

        CURRENT BEHAVIOUR: RandomForestClassifier is fitted on y=[0,0,0,0].
          scoring.compute_metrics() calls predict_proba(X_valid)[:, 1] on the
          degenerate model, which returns a 1-column array → IndexError.
          joblib.dump() is never reached, so no model artifact is written.
        EXPECTED BEHAVIOUR:
          raise ValueError("Training target must have at least 2 classes")
          checked BEFORE fit(), with a controlled message.

        SECURITY GAP PARTIALLY FIXED: the pipeline now fails before the artifact
        is written.  The error is still uncontrolled (IndexError) instead of a
        targeted ValueError raised early.
        """
        single_class_df = pd.DataFrame({
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "Tenure": [1, 12, 24, 6],
            "Churn": [0, 0, 0, 0],
        })
        single_class_df.to_csv(train_env.csv_path, index=False)
        params_rf: dict[str, Any | dict[str, str | int]] = {
            **train_env.base_params,
            "train": {
                "model": "random_forest",
                "n_estimators": 5,
                "max_depth": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
            },
        }
        with pytest.raises((IndexError, ValueError)):
            train_env.run(params=params_rf)
        assert not train_env.model_path.exists(), (
            "Model must not be written when metric computation fails on single-class data"
        )

    def test_missing_target_column_in_csv_raises_key_error_exposing_column_name(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-INFOLEAKW] When the target column is absent from the training CSV,
        pandas raises KeyError whose message contains the column name.

        CURRENT BEHAVIOUR: KeyError('[\"Churn\"] not found in axis')
          -- the target column name is leaked in the error message.
        EXPECTED BEHAVIOUR:
          raise ValueError("Target column not found in training CSV")

        SECURITY GAP CONFIRMED: the target column name is reflected in the
        uncontrolled error response.
        """
        no_target_df = pd.DataFrame({
            "MonthlyCharges": [29.85, 56.95],
            "Tenure": [1, 12],
        })
        no_target_df.to_csv(train_env.csv_path, index=False)
        with pytest.raises(KeyError) as exc_info:
            train_env.run()
        assert "Churn" in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: KeyError exposes target column name"
        )

    def test_no_row_limit_on_training_csv(self, train_env: _TrainEnv) -> None:
        """[SEC-DOS] main() imposes no upper bound on the number of rows in the
        training CSV.

        CURRENT BEHAVIOUR: a 20 000-row DataFrame is accepted and fit without error.
        EXPECTED BEHAVIOUR:
          raise ValueError("Training set too large: 20 000 rows > MAX_TRAIN_ROWS")

        SECURITY GAP CONFIRMED: a crafted training CSV with millions of rows can
        exhaust process memory and block the DVC pipeline.
        """
        rng: Generator = np.random.default_rng(42)
        large_df = pd.DataFrame({
            "MonthlyCharges": rng.uniform(10, 100, 20_000),
            "Tenure": rng.integers(1, 72, 20_000),
            "Churn": rng.integers(0, 2, 20_000),
        })
        large_df.to_csv(train_env.csv_path, index=False)
        train_env.run()  # must NOT raise
        assert train_env.model_path.exists(), (
            "SECURITY GAP CONFIRMED: 20 000-row training CSV accepted without size limit"
        )


# ===========================================================================
# TestMainModelPersistence  [SEC-INJECT, SEC-SILENT, SEC-PATH]
# ===========================================================================


class TestMainModelPersistence:

    def test_no_hash_file_written_alongside_model(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-INJECT] main() does not write an HMAC / SHA-256 hash file alongside
        the model artifact.

        CURRENT BEHAVIOUR: only models/model.joblib is created; no model.sha256 file.
        EXPECTED BEHAVIOUR: models/model.sha256 is also written so that evaluate.py
          can verify the artifact has not been tampered with between DVC stages.

        SECURITY GAP CONFIRMED: the evaluate stage has no cryptographic anchor to
        detect model substitution between the train and evaluate stages.
        """
        train_env.run()
        hash_path: Path = train_env.model_path.with_suffix(".sha256")
        assert not hash_path.exists(), (
            "SECURITY GAP CONFIRMED: no .sha256 integrity file written alongside model"
        )

    def test_existing_model_silently_overwritten_on_second_run(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-SILENT] A second invocation of main() silently overwrites the
        existing model file with no warning and no backup.

        CURRENT BEHAVIOUR: joblib.dump() replaces the file without any log entry
          at WARNING level or above.
        EXPECTED BEHAVIOUR: log.warning("Overwriting existing model at %s", MODEL_PATH)
          or a hard error requiring an explicit --force flag.

        SECURITY GAP CONFIRMED: an attacker who can trigger re-training can
        silently replace a production model artifact.
        """
        train_env.run()
        train_env.model_path.write_bytes(b"SENTINEL_ORIGINAL_MODEL")
        train_env.run()  # second run
        content: bytes = train_env.model_path.read_bytes()
        assert content != b"SENTINEL_ORIGINAL_MODEL", (
            "SECURITY GAP CONFIRMED: existing model silently overwritten with no warning"
        )

    def test_substituted_model_executes_code_at_joblib_load_time(
        self, train_env: _TrainEnv
    ) -> None:
        """[SEC-INJECT] A pickle payload substituted for model.joblib after training
        executes arbitrary code when joblib.load() is called (as in evaluate.py).

        Full attack chain:
          1. train.py writes models/model.joblib -- no hash file written.
          2. Attacker replaces model.joblib with a crafted pickle payload.
          3. evaluate.py calls joblib.load() -- payload executes.

        Without the hash file from step 1, step 3 is indistinguishable from
        loading a legitimate model.

        SECURITY GAP CONFIRMED: the train stage provides no integrity anchor
        to detect model substitution.
        """
        train_env.run()
        assert train_env.model_path.exists()

        # Step 2: attacker substitutes the model
        _PROBE_REGISTRY.clear()
        joblib.dump(_PickleCodeProbe(), train_env.model_path)
        # __reduce__ fires at dump time; clear the registry to prove the
        # callback fires at LOAD time, not dump time.
        _PROBE_REGISTRY.clear()

        # Step 3: evaluate.py loads the (substituted) model
        joblib.load(train_env.model_path)

        assert "joblib_load_executed_code" in _PROBE_REGISTRY, (
            "SECURITY GAP CONFIRMED: substituted pickle payload executed code at load time"
        )

    def test_model_path_outside_project_root_not_blocked(
        self,
        train_env: _TrainEnv,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """[SEC-PATH] main() accepts a MODEL_PATH that points outside the project
        root without raising a boundary-check error.

        CURRENT BEHAVIOUR: the model file is written to any writable path,
          including paths outside the project tree.
        EXPECTED BEHAVIOUR:
          raise ValueError("MODEL_PATH must be inside the project root")

        SECURITY GAP CONFIRMED: a misconfigured or maliciously overridden MODEL_PATH
        can write the model artifact to an unintended filesystem location.
        """
        outside_path: Path = tmp_path / "outside_project" / "model.joblib"
        outside_path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, "MODEL_PATH", outside_path)
        monkeypatch.setattr(config, "MODELS_DIR", outside_path.parent)

        train_env.run()

        assert outside_path.exists(), (
            "SECURITY GAP CONFIRMED: model written to path outside project root"
        )
