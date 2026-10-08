"""Security-focused tests for src/churn/scoring.py

Test groups by security category:

  [SEC-TYPE]       Missing type guards on all parameters: pipe, X, y, and threshold
                   in compute_metrics(); column and min_rows in slice_metrics().
                   Wrong-typed inputs produce uncontrolled AttributeError / TypeError
                   from numpy or sklearn internals rather than a controlled message
                   from scoring.py itself.
  [SEC-THRESHOLD]  No range validation on threshold in compute_metrics() or
                   slice_metrics(): degenerate values (0.0, 1.0, negative, > 1.0)
                   produce pathological all-positive or all-negative predictions
                   without any validation error or warning.
  [SEC-SILENT]     Silent failures: (1) single-class y causes roc_auc_score to return
                   NaN (sklearn >= 1.4) with an UndefinedMetricWarning, not a
                   controlled early error from compute_metrics(); (2) empty X / y raises
                   late from sklearn rather than from an early guard; (3) NaN in y or
                   in predict_proba() output propagates undetected.
  [SEC-SHAPE]      No guard on predict_proba() output shape: a 1-column (degenerate
                   single-class model) array causes IndexError: index 1 is out of
                   bounds with no controlled message from compute_metrics().
  [SEC-INJECT]     slice_metrics() builds metric keys by f-string interpolation of the
                   caller-supplied column name and the column's group values, applying
                   only a space-to-underscore substitution. Arbitrary characters
                   including path separators, JSON special characters, and newlines are
                   embedded in the key unchanged, potentially corrupting metrics.json
                   or downstream log records.
  [SEC-DOS]        No upper bound on the number of rows: arbitrarily large DataFrames
                   are accepted by compute_metrics() and slice_metrics() without any
                   size guard, allowing process memory exhaustion.

Run:
    pytest tests/churn/test_scoring_security.py -v

CURRENT BEHAVIOUR / EXPECTED BEHAVIOUR annotations mark confirmed security gaps.
Fixing them requires modifying the source, which is out of scope here.

Security hardening suggestions
-------------------------------
compute_metrics():
  - Add explicit type guards at the top of the function:
      if not isinstance(X, pd.DataFrame):
          raise TypeError(f"X must be a DataFrame, got {type(X).__name__}")
      if not isinstance(y, pd.Series):
          raise TypeError(f"y must be a Series, got {type(y).__name__}")
      if not isinstance(threshold, (int, float)):
          raise TypeError(f"threshold must be numeric, got {type(threshold).__name__}")
      if not (0.0 <= threshold <= 1.0):
          raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
  - Guard the predict_proba output shape before the column index:
      proba_2d = pipe.predict_proba(X)
      if proba_2d.ndim < 2 or proba_2d.shape[1] < 2:
          raise ValueError(
              "predict_proba() must return a 2-column array; "
              f"got shape {proba_2d.shape} -- model may be degenerate"
          )
      proba = proba_2d[:, 1]
  - Guard against degenerate input before calling sklearn metrics:
      if len(X) == 0:
          raise ValueError("X must not be empty")
      if pd.isna(y).any():
          raise ValueError("y must not contain NaN")
      if np.isnan(proba).any():
          raise ValueError("predict_proba() returned NaN values")
      if len(np.unique(y)) < 2:
          raise ValueError("y must contain at least 2 distinct classes for roc_auc")
  - Enforce a row-count cap:
      MAX_ROWS: int = 500_000
      if len(X) > MAX_ROWS:
          raise ValueError(f"X has {len(X)} rows; maximum allowed is {MAX_ROWS}")

slice_metrics():
  - Sanitise the metric key using a strict character allowlist:
      import re
      safe_col = re.sub(r"[^A-Za-z0-9_]", "_", str(column))
      safe_val = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
      key = f"f1_slice_{safe_col}_{safe_val}"
  - Add type guards:
      if not isinstance(column, str):
          raise TypeError(f"column must be a str, got {type(column).__name__}")
      if not isinstance(min_rows, int) or min_rows < 1:
          raise ValueError(f"min_rows must be a positive int, got {min_rows!r}")
"""
from __future__ import annotations

import math
import warnings
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from numpy._typing._array_like import NDArray
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.churn.scoring import compute_metrics, slice_metrics


# ---------------------------------------------------------------------------
# Shared fixtures for compute_metrics tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def X() -> pd.DataFrame:
    """Six-row feature DataFrame (no target column)."""
    return pd.DataFrame({
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 65.0, 30.0],
        "Tenure": [1, 12, 24, 6, 36, 48],
    })


@pytest.fixture()
def y() -> pd.Series:
    """Binary target Series aligned with X (3 positives, 3 negatives)."""
    return pd.Series([1, 0, 1, 0, 1, 0], name="Churn")


@pytest.fixture()
def mock_pipe() -> MagicMock:
    """Mock Pipeline: predict_proba returns well-separated 2-column scores for 6 rows."""
    mock = MagicMock(spec=Pipeline)
    _proba: NDArray[Any] = np.array([
        [0.20, 0.80],  # row 0 → high positive-class probability (true label 1)
        [0.80, 0.20],  # row 1 → low positive-class probability  (true label 0)
        [0.30, 0.70],  # row 2 → high (true label 1)
        [0.75, 0.25],  # row 3 → low  (true label 0)
        [0.25, 0.75],  # row 4 → high (true label 1)
        [0.70, 0.30],  # row 5 → low  (true label 0)
    ])
    mock.predict_proba.return_value = _proba
    return mock


# ---------------------------------------------------------------------------
# Shared fixture for slice_metrics tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def slice_data() -> tuple[pd.DataFrame, pd.Series, MagicMock]:
    """Ten-row dataset with a binary 'Group' column (5 rows each) for slice_metrics."""
    n_per_group = 5
    X_slice = pd.DataFrame({
        "Group": ["A"] * n_per_group + ["B"] * n_per_group,
        "Feature": list(range(2 * n_per_group)),
    })
    y_slice = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], name="Churn")
    pipe = MagicMock(spec=Pipeline)
    pipe.predict_proba.side_effect = lambda df: np.column_stack(  # pyright: ignore[reportUnknownLambdaType]
        [np.full(len(df), 0.4), np.full(len(df), 0.6)]
    )
    return X_slice, y_slice, pipe


# ===========================================================================
# TestComputeMetricsNormalOperation
# ===========================================================================


class TestComputeMetricsNormalOperation:

    def test_returns_all_five_expected_metric_keys(
        self, mock_pipe: MagicMock, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """Positive test: compute_metrics() returns a dict with all 5 expected keys."""
        result = compute_metrics(mock_pipe, X, y)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in result, f"Expected metric key '{key}' is missing from the result."

    def test_all_values_are_floats_in_zero_to_one_range(
        self, mock_pipe: MagicMock, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """Positive test: all returned metric values must be floats in [0.0, 1.0]."""
        result = compute_metrics(mock_pipe, X, y)
        for key, val in result.items():
            assert isinstance(val, float), f"Metric '{key}' is not a float: {val!r}"
            assert 0.0 <= val <= 1.0, f"Metric '{key}' = {val} is outside [0, 1]."

    def test_perfect_classifier_gives_f1_of_one(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """Positive test: a pipe that perfectly separates all classes produces f1=1.0."""
        perfect_pipe = MagicMock()
        perfect_pipe.predict_proba.return_value = np.array([
            [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],
            [1.0, 0.0], [0.0, 1.0], [1.0, 0.0],
        ])
        result = compute_metrics(perfect_pipe, X, y)
        assert result["f1"] == 1.0

    def test_high_threshold_forces_all_predictions_to_negative(
        self, mock_pipe: MagicMock, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """Positive test: threshold=0.9 classifies all rows as negative (max proba=0.80)."""
        result = compute_metrics(mock_pipe, X, y, threshold=0.9)
        assert result["recall"] == 0.0


# ===========================================================================
# TestComputeMetricsTypeValidation  [SEC-TYPE]
# ===========================================================================


class TestComputeMetricsTypeValidation:

    def test_none_pipe_raises_uncontrolled_attribute_error(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """[SEC-TYPE] Passing None as pipe raises AttributeError from Python, not a
        controlled TypeError from compute_metrics().

        CURRENT BEHAVIOUR: AttributeError: 'NoneType' object has no attribute 'predict_proba'
        EXPECTED BEHAVIOUR: raise TypeError("pipe must be a Pipeline, got NoneType")

        SECURITY GAP CONFIRMED: no isinstance(pipe, Pipeline) guard before the call.
        """
        with pytest.raises((AttributeError, TypeError)):
            compute_metrics(None, X, y)  # type: ignore[arg-type]

    def test_none_X_raises_uncontrolled_error_from_sklearn(
        self, y: pd.Series
    ) -> None:
        """[SEC-TYPE] Passing None as X is forwarded to predict_proba() without a type
        check. A real sklearn pipeline (or any pipe that accesses len(X)) raises an
        uncontrolled error from Python / numpy, not a controlled TypeError from
        compute_metrics().

        CURRENT BEHAVIOUR: TypeError: object of type 'NoneType' has no len()
          raised inside the pipe, not in compute_metrics().
        EXPECTED BEHAVIOUR: raise TypeError("X must be a DataFrame, got NoneType")

        SECURITY GAP CONFIRMED: no isinstance(X, pd.DataFrame) guard before the
        predict_proba() call.
        """
        # Use a pipe whose predict_proba accesses len(X) -- equivalent to sklearn
        # validation -- so the TypeError originates inside predict_proba, not in
        # compute_metrics (which has no guard of its own).
        realistic_pipe = MagicMock()
        realistic_pipe.predict_proba.side_effect = \
            lambda X: np.full((len(X), 2), 0.5)  # pyright: ignore[reportUnknownLambdaType]

        with pytest.raises((TypeError, ValueError)):
            compute_metrics(realistic_pipe, None, y)  # type: ignore[arg-type]

    def test_none_y_raises_uncontrolled_error_from_sklearn(
        self, mock_pipe: MagicMock, X: pd.DataFrame
    ) -> None:
        """[SEC-TYPE] Passing None as y is forwarded to sklearn metric functions,
        raising an uncontrolled error rather than a controlled message from compute_metrics().

        CURRENT BEHAVIOUR: TypeError or ValueError from sklearn metrics internals.
        EXPECTED BEHAVIOUR: raise TypeError("y must be a Series, got NoneType")

        SECURITY GAP CONFIRMED: no isinstance(y, pd.Series) guard.
        """
        with pytest.raises((AttributeError, TypeError, ValueError)):
            compute_metrics(mock_pipe, X, None)  # type: ignore[arg-type]

    def test_string_threshold_raises_uncontrolled_type_error(
        self, mock_pipe: MagicMock, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """[SEC-TYPE] A string threshold causes TypeError from numpy's >= operator,
        not a controlled ValueError from compute_metrics().

        CURRENT BEHAVIOUR: TypeError from numpy comparison ('>=' unsupported for str).
        EXPECTED BEHAVIOUR: raise ValueError("threshold must be a float in [0, 1], got str")

        SECURITY GAP CONFIRMED: no isinstance(threshold, (int, float)) guard.
        """
        with pytest.raises((TypeError, ValueError)):
            compute_metrics(mock_pipe, X, y, threshold="0.5")  # type: ignore[arg-type]

    def test_none_threshold_raises_uncontrolled_type_error(
        self, mock_pipe: MagicMock, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """[SEC-TYPE] None threshold causes TypeError from numpy's >= operator,
        not a controlled ValueError from compute_metrics().

        CURRENT BEHAVIOUR: TypeError: '>=' not supported between 'ndarray' and 'NoneType'
        EXPECTED BEHAVIOUR: raise ValueError("threshold must be a float in [0, 1], got None")

        SECURITY GAP CONFIRMED: no None check before the threshold comparison.
        """
        with pytest.raises((TypeError, ValueError)):
            compute_metrics(mock_pipe, X, y, threshold=None)  # type: ignore[arg-type]

    def test_dict_instead_of_dataframe_raises_uncontrolled_error(
        self, y: pd.Series
    ) -> None:
        """[SEC-TYPE] A plain dict passed as X is forwarded to predict_proba() without
        a type check. A pipe that uses len(X) to size its output returns a 1-row array
        for a 1-key dict, and sklearn raises an uncontrolled ValueError about
        inconsistent sample counts rather than a controlled TypeError from compute_metrics().

        CURRENT BEHAVIOUR: ValueError from sklearn ("inconsistent number of samples")
          raised inside the metric computation, not in compute_metrics().
        EXPECTED BEHAVIOUR: raise TypeError("X must be a DataFrame, got dict")

        SECURITY GAP CONFIRMED: no isinstance(X, pd.DataFrame) guard at the entry point.
        """
        realistic_pipe = MagicMock()
        # len({"feature": [1.0, 2.0]}) == 1 → produces a 1-row proba array, but y has
        # 6 elements → accuracy_score(y_6, y_pred_1) raises ValueError from sklearn.
        realistic_pipe.predict_proba.side_effect = \
            lambda X: np.full((len(X), 2), 0.5)  # pyright: ignore[reportUnknownLambdaType]

        with pytest.raises((TypeError, ValueError)):
            compute_metrics(realistic_pipe, {"feature": [1.0, 2.0]}, y)  # type: ignore[arg-type]


# ===========================================================================
# TestComputeMetricsThresholdBoundaries  [SEC-THRESHOLD]
# ===========================================================================


class TestComputeMetricsThresholdBoundaries:

    def test_threshold_zero_produces_all_positive_predictions_silently(
        self, mock_pipe: MagicMock, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """[SEC-THRESHOLD] threshold=0.0 makes (proba >= 0.0) True for every sample,
        producing 100% positive predictions.  No boundary check prevents this.

        CURRENT BEHAVIOUR: recall=1.0 returned silently (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError("threshold must be in (0, 1), got 0.0")

        SECURITY GAP CONFIRMED: a crafted threshold inflates recall to 1.0, masking
        real model performance while potentially passing downstream quality gates.
        """
        result = compute_metrics(mock_pipe, X, y, threshold=0.0)
        assert result["recall"] == 1.0, (
            "SECURITY GAP CONFIRMED: threshold=0.0 accepted without validation; "
            "recall is inflated to 1.0."
        )

    def test_threshold_one_produces_all_negative_predictions_silently(
        self, mock_pipe: MagicMock, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """[SEC-THRESHOLD] threshold=1.0 makes (proba >= 1.0) False for every sample
        (no probability reaches exactly 1.0), producing all-zero predictions.

        CURRENT BEHAVIOUR: recall=0.0 returned silently (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError("threshold must be < 1.0, got 1.0")

        SECURITY GAP CONFIRMED: threshold=1.0 drives recall to 0.0 and f1 to 0.0,
        which should block model promotion but only triggers an advisory warning in
        evaluate.py's quality gate.
        """
        result = compute_metrics(mock_pipe, X, y, threshold=1.0)
        assert result["recall"] == 0.0, (
            "SECURITY GAP CONFIRMED: threshold=1.0 accepted without validation."
        )

    def test_negative_threshold_accepted_silently(
        self, mock_pipe: MagicMock, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """[SEC-THRESHOLD] A negative threshold (e.g. -1.0) behaves identically to
        threshold=0.0: every prediction is positive. No lower-bound check is present.

        CURRENT BEHAVIOUR: recall=1.0 returned silently (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError("threshold must be >= 0.0, got -1.0")

        SECURITY GAP CONFIRMED: no lower-bound validation on the threshold parameter.
        """
        result = compute_metrics(mock_pipe, X, y, threshold=-1.0)
        assert result["recall"] == 1.0, (
            "SECURITY GAP CONFIRMED: negative threshold accepted without validation."
        )

    def test_threshold_above_one_produces_all_negative_predictions_silently(
        self, mock_pipe: MagicMock, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """[SEC-THRESHOLD] A threshold above 1.0 (e.g. 2.0) makes every prediction
        negative because no probability can reach 2.0. No upper-bound check is present.

        CURRENT BEHAVIOUR: recall=0.0 returned silently (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError("threshold must be <= 1.0, got 2.0")

        SECURITY GAP CONFIRMED: no upper-bound validation beyond 1.0.
        """
        result = compute_metrics(mock_pipe, X, y, threshold=2.0)
        assert result["recall"] == 0.0, (
            "SECURITY GAP CONFIRMED: threshold=2.0 accepted without validation."
        )


# ===========================================================================
# TestComputeMetricsProbShape  [SEC-SHAPE]
# ===========================================================================


class TestComputeMetricsProbShape:

    def test_single_column_proba_causes_index_error_no_shape_guard(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """[SEC-SHAPE] When predict_proba() returns a 1-column array (degenerate
        single-class model), the [:, 1] index slice raises an uncontrolled IndexError.
        There is no shape guard before the column access in compute_metrics().

        CURRENT BEHAVIOUR: IndexError: index 1 is out of bounds for axis 1 with size 1
        EXPECTED BEHAVIOUR: raise ValueError(
            "predict_proba() must return a 2-column array; got 1 column"
        )

        SECURITY GAP CONFIRMED: no proba.shape validation in compute_metrics().
        """
        single_col_pipe = MagicMock()
        single_col_pipe.predict_proba.return_value = np.array(
            [[0.8], [0.2], [0.7], [0.3], [0.75], [0.4]]
        )
        with pytest.raises(IndexError):
            compute_metrics(single_col_pipe, X, y)

    def test_zero_column_proba_causes_index_error_no_shape_guard(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """[SEC-SHAPE] A 0-column proba array (entirely malformed output) causes the
        same uncontrolled IndexError, with no shape guard in compute_metrics().

        CURRENT BEHAVIOUR: IndexError raised from numpy (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError("predict_proba() returned an empty array")
        """
        zero_col_pipe = MagicMock()
        zero_col_pipe.predict_proba.return_value = np.empty((6, 0))
        with pytest.raises(IndexError):
            compute_metrics(zero_col_pipe, X, y)


# ===========================================================================
# TestComputeMetricsSilentFailures  [SEC-SILENT]
# ===========================================================================


class TestComputeMetricsSilentFailures:

    def test_single_class_y_silently_produces_nan_roc_auc(
        self, mock_pipe: MagicMock, X: pd.DataFrame
    ) -> None:
        """[SEC-SILENT] When y contains only one class, compute_metrics() has no early
        guard: roc_auc_score returns NaN (sklearn >= 1.4) after emitting
        UndefinedMetricWarning, without a controlled error from compute_metrics().

        CURRENT BEHAVIOUR: roc_auc=NaN returned silently (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError(
            "y must contain at least 2 distinct classes for roc_auc computation"
        ) from compute_metrics() before calling roc_auc_score.

        SECURITY GAP CONFIRMED: single-class y reaches roc_auc_score without an early
        guard; the resulting NaN propagates to metrics.json as an invalid JSON value.
        """
        y_single_class = pd.Series([0, 0, 0, 0, 0, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_metrics(mock_pipe, X, y_single_class)
        assert math.isnan(result["roc_auc"]), (
            "SECURITY GAP CONFIRMED: single-class y produced roc_auc=NaN silently "
            "with no early guard in compute_metrics()."
        )

    def test_empty_X_and_y_raises_uncontrolled_sklearn_error(self) -> None:
        """[SEC-SILENT] Empty X and y raise a late, uncontrolled ValueError from
        sklearn metric functions rather than an early guard in compute_metrics().

        CURRENT BEHAVIOUR: ValueError from sklearn ("Found array with 0 sample(s)")
          raised inside metric computation, not at the top of compute_metrics().
        EXPECTED BEHAVIOUR: raise ValueError("X must not be empty") early, before
          any call to predict_proba() or sklearn metrics.

        SECURITY GAP CONFIRMED: empty input is not validated at the entry point.
        """
        empty_pipe = MagicMock()
        empty_pipe.predict_proba.return_value = np.empty((0, 2))
        empty_X = pd.DataFrame({"MonthlyCharges": pd.Series(dtype=float)})
        empty_y = pd.Series(dtype=int)

        with pytest.raises((ValueError, ZeroDivisionError)):
            compute_metrics(empty_pipe, empty_X, empty_y)

    def test_nan_in_y_causes_uncontrolled_sklearn_error(
        self, mock_pipe: MagicMock, X: pd.DataFrame
    ) -> None:
        """[SEC-SILENT] NaN values in y are passed directly to sklearn metric functions
        with no guard in compute_metrics(), causing an uncontrolled error.

        CURRENT BEHAVIOUR: ValueError or TypeError from sklearn (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError("y must not contain NaN") before any
          metric computation.

        SECURITY GAP CONFIRMED: no pd.isna(y).any() check before metric computation.
        """
        y_with_nan = pd.Series([1.0, float("nan"), 1.0, 0.0, 1.0, 0.0])
        with pytest.raises((ValueError, TypeError)):
            compute_metrics(mock_pipe, X, y_with_nan)

    def test_nan_in_proba_raises_uncontrolled_sklearn_error(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """[SEC-SILENT] NaN values returned by predict_proba() are not validated in
        compute_metrics() before being passed to sklearn's roc_auc_score. sklearn's own
        check_array() detects the NaN and raises an uncontrolled ValueError with no
        controlled message from compute_metrics() itself.

        CURRENT BEHAVIOUR: ValueError("Input contains NaN.") raised inside roc_auc_score,
          not a controlled message from compute_metrics() (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError("predict_proba() returned NaN values")
          from compute_metrics() after the predict_proba() call, before any sklearn metric.

        SECURITY GAP CONFIRMED: no np.isnan(proba).any() guard in compute_metrics();
        the caller receives an opaque sklearn internal error instead of a clear message.
        """
        nan_pipe = MagicMock()
        nan_pipe.predict_proba.return_value = np.array([
            [float("nan"), float("nan")],  # row 0: NaN proba (true label 1)
            [0.80, 0.20],
            [0.30, 0.70],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.70, 0.30],
        ])
        with pytest.raises(ValueError) as exc_info:
            compute_metrics(nan_pipe, X, y)
        # The error originates from sklearn's check_array, not from compute_metrics.
        # A controlled implementation would raise "predict_proba() returned NaN values"
        # before reaching sklearn.
        assert "NaN" in str(exc_info.value) or "nan" in str(exc_info.value).lower(), (
            "SECURITY GAP CONFIRMED: ValueError originates from sklearn internals "
            "('Input contains NaN.'), not from a guard in compute_metrics()."
        )


# ===========================================================================
# TestComputeMetricsDOS  [SEC-DOS]
# ===========================================================================


class TestComputeMetricsDOS:

    def test_large_input_accepted_without_row_limit(self) -> None:
        """[SEC-DOS] compute_metrics() imposes no upper bound on the number of rows.
        A 100 000-row dataset is accepted without any size check.

        CURRENT BEHAVIOUR: large input is processed without error (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError(
            "X has N rows; maximum allowed is MAX_ROWS"
        )

        SECURITY GAP CONFIRMED: no row-count guard in compute_metrics(); a crafted
        validation set with millions of rows can exhaust process memory.
        """
        n = 100_000
        rng = np.random.default_rng(0)
        large_X = pd.DataFrame({"Feature": rng.uniform(0, 1, n)})
        large_y = pd.Series(np.tile([1, 0], n // 2))

        large_pipe = MagicMock()
        large_pipe.predict_proba.return_value = np.column_stack(
            [rng.uniform(0, 1, n), rng.uniform(0, 1, n)]
        )

        result = compute_metrics(large_pipe, large_X, large_y)
        assert result is not None, (
            f"SECURITY GAP CONFIRMED: {n:,}-row input accepted by compute_metrics() "
            "with no row-count guard."
        )


# ===========================================================================
# TestSliceMetricsNormalOperation
# ===========================================================================


class TestSliceMetricsNormalOperation:

    def test_missing_column_returns_empty_dict(
        self, slice_data: tuple[pd.DataFrame, pd.Series, MagicMock]
    ) -> None:
        """Positive test: slice_metrics() returns {} when column is not in X."""
        X_slice, y_slice, pipe = slice_data
        result = slice_metrics(pipe, X_slice, y_slice, "NonExistent", min_rows=1)
        assert result == {}

    def test_groups_above_min_rows_produce_f1_slice_keys(
        self, slice_data: tuple[pd.DataFrame, pd.Series, MagicMock]
    ) -> None:
        """Positive test: groups with at least min_rows rows produce f1_slice_* keys."""
        X_slice, y_slice, pipe = slice_data
        result = slice_metrics(pipe, X_slice, y_slice, "Group", min_rows=1)
        assert len(result) == 2
        assert all(k.startswith("f1_slice_Group_") for k in result)

    def test_groups_below_min_rows_are_excluded(
        self, slice_data: tuple[pd.DataFrame, pd.Series, MagicMock]
    ) -> None:
        """Positive test: groups with fewer than min_rows rows are silently skipped."""
        X_slice, y_slice, pipe = slice_data
        # Both groups have 5 rows; min_rows=6 means both are skipped.
        result = slice_metrics(pipe, X_slice, y_slice, "Group", min_rows=6)
        assert result == {}


# ===========================================================================
# TestSliceMetricsTypeValidation  [SEC-TYPE]
# ===========================================================================


class TestSliceMetricsTypeValidation:

    def test_none_column_silently_returns_empty_dict(
        self, slice_data: tuple[pd.DataFrame, pd.Series, MagicMock]
    ) -> None:
        """[SEC-TYPE] Passing None as column passes the `column not in X.columns`
        membership test (None is not a string column name) and silently returns {},
        giving the caller no indication that the column argument was wrong.

        CURRENT BEHAVIOUR: {} returned silently -- None treated as absent column.
        EXPECTED BEHAVIOUR: raise TypeError("column must be a str, got NoneType")

        SECURITY GAP CONFIRMED: None is accepted silently without a type guard.
        """
        X_slice, y_slice, pipe = slice_data
        result = slice_metrics(pipe, X_slice, y_slice, None, min_rows=1)  # type: ignore[arg-type]
        assert result == {}, (
            "SECURITY GAP CONFIRMED: None column silently returns {} with no TypeError."
        )

    def test_integer_column_silently_returns_empty_dict(
        self, slice_data: tuple[pd.DataFrame, pd.Series, MagicMock]
    ) -> None:
        """[SEC-TYPE] An integer value for column (e.g. 0) is not found among string
        column names, so slice_metrics() silently returns {} with no type error.

        CURRENT BEHAVIOUR: {} returned silently (assertion PASSES).
        EXPECTED BEHAVIOUR: raise TypeError("column must be a str, got int")

        SECURITY GAP CONFIRMED: wrong column type is accepted silently without a guard.
        """
        X_slice, y_slice, pipe = slice_data
        result = slice_metrics(pipe, X_slice, y_slice, 0, min_rows=1)  # type: ignore[arg-type]
        assert result == {}, (
            "SECURITY GAP CONFIRMED: integer column silently returns {}."
        )

    def test_string_min_rows_causes_uncontrolled_type_error(
        self, slice_data: tuple[pd.DataFrame, pd.Series, MagicMock]
    ) -> None:
        """[SEC-TYPE] A string value for min_rows causes TypeError from Python's
        `len(idx) < min_rows` comparison, not a controlled error from slice_metrics().

        CURRENT BEHAVIOUR: TypeError: '<' not supported between 'int' and 'str'.
        EXPECTED BEHAVIOUR: raise ValueError(
            "min_rows must be a positive int, got str"
        ) from slice_metrics() before iterating over groups.

        SECURITY GAP CONFIRMED: no isinstance(min_rows, int) guard.
        """
        X_slice, y_slice, pipe = slice_data
        with pytest.raises(TypeError):
            slice_metrics(pipe, X_slice, y_slice, "Group", min_rows="5")  # type: ignore[arg-type]

    def test_none_min_rows_causes_uncontrolled_type_error(
        self, slice_data: tuple[pd.DataFrame, pd.Series, MagicMock]
    ) -> None:
        """[SEC-TYPE] None for min_rows causes TypeError from the < comparison,
        not a controlled ValueError from slice_metrics().

        CURRENT BEHAVIOUR: TypeError: '<' not supported between 'int' and 'NoneType'.
        EXPECTED BEHAVIOUR: raise ValueError("min_rows must be a positive int, got None")

        SECURITY GAP CONFIRMED: no None guard on min_rows.
        """
        X_slice, y_slice, pipe = slice_data
        with pytest.raises(TypeError):
            slice_metrics(pipe, X_slice, y_slice, "Group", min_rows=None)  # type: ignore[arg-type]


# ===========================================================================
# TestSliceMetricsKeyInjection  [SEC-INJECT]
# ===========================================================================


class TestSliceMetricsKeyInjection:
    """
    slice_metrics() builds the metric key as:
        key = f"f1_slice_{column}_{value}".replace(" ", "_")

    Only spaces are substituted; all other characters from the caller-controlled
    column name and the column's group values are embedded verbatim in the key.
    A downstream consumer writing the dict to metrics.json, a log, or a filename
    inherits these arbitrary characters unchanged.
    """

    def test_path_traversal_in_column_name_embedded_in_key(self) -> None:
        """[SEC-INJECT] A column name containing path-traversal sequences
        (e.g. '../../conf') is embedded verbatim in the metric key after the
        single space-to-underscore substitution.

        CURRENT BEHAVIOUR: key contains '../../' (assertion PASSES).
        EXPECTED BEHAVIOUR: key characters are restricted to [A-Za-z0-9_] via
          re.sub(r'[^A-Za-z0-9_]', '_', column) before interpolation.

        SECURITY GAP CONFIRMED: unsanitised column names embed arbitrary characters
        in metric keys, which may corrupt metrics.json or downstream log records.
        """
        X_inject = pd.DataFrame({
            "../../conf": ["value_a"] * 5 + ["value_b"] * 5,
            "Feature": range(10),
        })
        y_inject = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        pipe = MagicMock()
        pipe.predict_proba.side_effect = lambda df: np.column_stack(  # pyright: ignore[reportUnknownLambdaType]
            [np.full(len(df), 0.4), np.full(len(df), 0.6)]
        )

        result = slice_metrics(pipe, X_inject, y_inject, "../../conf", min_rows=1)

        assert len(result) > 0, "Expected at least one slice key to be produced."
        for key in result:
            assert "../" in key, (
                "SECURITY GAP CONFIRMED: path traversal characters in column name "
                "are embedded verbatim in the metric key."
            )

    def test_json_special_chars_in_group_value_embedded_in_key(self) -> None:
        """[SEC-INJECT] Group values that contain JSON special characters (e.g. a
        double-quote) are embedded verbatim in the metric key, potentially corrupting
        JSON serialisation of the metrics dict downstream.

        CURRENT BEHAVIOUR: key contains the double-quote character (assertion PASSES).
        EXPECTED BEHAVIOUR: group values are sanitised before key interpolation.

        SECURITY GAP CONFIRMED: unsanitised group values can produce keys that produce
        malformed JSON or are misinterpreted by downstream consumers.
        """
        X_inject = pd.DataFrame({
            "Category": ['"injected"'] * 5 + ["normal"] * 5,
            "Feature": range(10),
        })
        y_inject = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        pipe = MagicMock()
        pipe.predict_proba.side_effect = lambda df: np.column_stack(  # pyright: ignore[reportUnknownLambdaType]
            [np.full(len(df), 0.4), np.full(len(df), 0.6)]
        )

        result = slice_metrics(pipe, X_inject, y_inject, "Category", min_rows=1)

        assert len(result) > 0, "Expected at least one slice key to be produced."
        combined_keys = " ".join(result.keys())
        assert '"' in combined_keys, (
            "SECURITY GAP CONFIRMED: double-quote in group value embedded verbatim "
            "in metric key, potentially corrupting JSON serialisation."
        )

    def test_newline_in_column_name_embedded_in_key(self) -> None:
        """[SEC-INJECT] A column name containing a newline character is embedded
        verbatim in the metric key; replace() only substitutes spaces, not newlines.

        CURRENT BEHAVIOUR: key contains a newline character (assertion PASSES).
        EXPECTED BEHAVIOUR: re.sub sanitisation strips all non-alphanumeric characters.

        SECURITY GAP CONFIRMED: newlines in metric keys corrupt plain-text log records
        and may be exploited for log injection attacks.
        """
        col_with_newline = "Group\nInjection"
        X_inject = pd.DataFrame({
            col_with_newline: ["A"] * 5 + ["B"] * 5,
            "Feature": range(10),
        })
        y_inject = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        pipe = MagicMock()
        pipe.predict_proba.side_effect = lambda df: np.column_stack(  # pyright: ignore[reportUnknownLambdaType]
            [np.full(len(df), 0.4), np.full(len(df), 0.6)]
        )

        result = slice_metrics(pipe, X_inject, y_inject, col_with_newline, min_rows=1)

        assert len(result) > 0, "Expected at least one slice key to be produced."
        combined_keys = "".join(result.keys())
        assert "\n" in combined_keys, (
            "SECURITY GAP CONFIRMED: newline in column name embedded verbatim in "
            "the metric key (replace() only removes spaces, not newlines)."
        )


# ===========================================================================
# TestSliceMetricsDOS  [SEC-DOS]
# ===========================================================================


class TestSliceMetricsDOS:

    def test_large_dataframe_accepted_without_row_limit(self) -> None:
        """[SEC-DOS] slice_metrics() imposes no upper bound on the number of rows.
        A 50 000-row DataFrame with a groupable column is accepted without any size check.

        CURRENT BEHAVIOUR: large DataFrame processed without error (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError(
            "X has N rows; maximum allowed is MAX_ROWS"
        )

        SECURITY GAP CONFIRMED: no row-count guard in slice_metrics(); a crafted
        dataset with millions of rows can exhaust process memory.
        """
        n = 50_000
        rng = np.random.default_rng(1)
        large_X = pd.DataFrame({
            "Group": np.tile(["A", "B"], n // 2),
            "Feature": rng.uniform(0, 1, n),
        })
        large_y = pd.Series(np.tile([1, 0], n // 2))

        pipe = MagicMock()
        pipe.predict_proba.side_effect = lambda df: np.column_stack(  # pyright: ignore[reportUnknownLambdaType]
            [np.full(len(df), 0.4), np.full(len(df), 0.6)]
        )

        result = slice_metrics(pipe, large_X, large_y, "Group", min_rows=1)
        assert result is not None, (
            f"SECURITY GAP CONFIRMED: {n:,}-row DataFrame accepted by slice_metrics() "
            "with no row-count guard."
        )
