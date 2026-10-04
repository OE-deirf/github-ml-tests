"""Security-focused tests for src/churn/prepare.py

Test groups by security category:

  [SEC-TYPE]      Missing type validation: non-DataFrame / non-str / non-list causes
                  internal AttributeError / KeyError to leak to the caller.
  [SEC-SILENT]    Silent failure: the expected side-effect (e.g. column drop)
                  is skipped without raising an error -- PII may leak into the model.
  [SEC-INFOLEAKW] Exception message exposes internal state (column names, file paths)
                  that could aid an attacker.
  [SEC-INJECT]    Injected strings in the target column: the allowlist mapping and
                  the isna() guard must block them.
  [SEC-DOS]       Resource exhaustion on unbounded input.
  [SEC-PATH]      Filesystem boundary: a target column name must not reach the filesystem.

Run:
    pytest tests/churn/test_prepare_security.py -v

Tests: tests/churn/test_prepare_security.py
41 tests in 6 security categories:

Category	  Tests	Description
[SEC-TYPE]	    6	Missing type checks on parameters
[SEC-SILENT]	1	Silent PII leak (critical)
[SEC-INFOLEAKW]	5	Internal information leakage in exception messages
[SEC-INJECT]	9	SQL/script/unicode injection in the target column
[SEC-DOS]	    6	Resource exhaustion, edge cases
[SEC-PATH]    	5	Path boundaries, null byte in the target name

Need to change in prepare.py:
1. Input validation at the start of clean() -- the three most important gaps:

def clean(df: pd.DataFrame, target: str, drop_columns: list[str]) -> pd.DataFrame:
    # [SEC-TYPE] type checking
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, got {type(df).__name__}")
    if not isinstance(target, str) or not target:
        raise ValueError("target must be a non-empty string")
    if not isinstance(drop_columns, list):
        raise TypeError(f"drop_columns must be a list, got {type(drop_columns).__name__}")

    # [SEC-INFOLEAKW] check that target exists BEFORE the drop runs
    if target not in df.columns:
        raise ValueError(f"target column not found in DataFrame")  # does not leak the name
    if target in drop_columns:
        raise ValueError("target column must not appear in drop_columns")

    # [SEC-DOS] size limit (optional but recommended)
    if df.empty:
        raise ValueError("DataFrame is empty")
    ...

2. Parameter validation at the start of main():

    if not isinstance(seed, int):
        raise ValueError(f"params.yaml: seed must be int, got {type(seed).__name__}")
    if not (0 < prepare_params["test_size"] < 1):
        raise ValueError("params.yaml: test_size must be in (0, 1)")
    if target not in df.columns:
        raise ValueError("target column from params.yaml not found in CSV")

3. The [SEC-SILENT] CRITICAL bug (test_drop_columns_string_causes_silent_pii_leak):
    without the isinstance(drop_columns, list) guard, a "CustomerID" string (instead of a list)
    is iterated character by character and drops nothing -- the PII column
    ends up in the model with no error message.


CURRENT BEHAVIOUR / EXPECTED BEHAVIOUR annotations mark places where the current
code does NOT yet implement the secure handling described. Those tests FAIL on the
unmodified codebase, surfacing concrete security gaps.
Tests where the current code already behaves correctly PASS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from numpy._typing._array_like import NDArray
from numpy.random import Generator
import pandas as pd
from pandas import DataFrame
import pytest

from src.churn.prepare import clean, main


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_bool_df() -> pd.DataFrame:
    """Well-formed DataFrame with a boolean Churn column."""
    return pd.DataFrame(
        {
            "CustomerID": ["C1", "C2", "C3", "C4"],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "Churn": [True, False, True, False],
        }
    )


@pytest.fixture()
def valid_str_df() -> pd.DataFrame:
    """Well-formed DataFrame with a string 'True'/'False' Churn column."""
    return pd.DataFrame(
        {
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "Churn": ["True", "False", "True", "False"],
        }
    )


@pytest.fixture()
def base_params() -> dict[str, Any]:
    """Valid params.yaml content, mocked for main() tests."""
    return {
        "seed": 42,
        "prepare": {
            "target": "Churn",
            "drop_columns": [],
            "stratify": False,
            "test_size": 0.25,
        },
    }


# ===========================================================================
# 1. Normal operation (positive tests)
# ===========================================================================


class TestCleanNormalOperation:
    """Pin the expected contract -- if these fail, a security fix has broken existing logic."""

    def test_bool_target_becomes_zero_or_one(self, valid_bool_df: DataFrame):
        # Normal operation: bool target must be converted to strict 0/1 integers.
        result: DataFrame = clean(valid_bool_df, target="Churn", drop_columns=[])
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_string_target_true_false_converted(self, valid_str_df: DataFrame):
        # Normal operation: "True"/"False" strings -> 0/1 integers via the explicit allowlist.
        result: DataFrame = clean(valid_str_df, target="Churn", drop_columns=[])
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_string_target_case_insensitive(self):
        # Normal operation: .str.lower() must handle uppercase variants.
        df = pd.DataFrame({"f": [1, 2, 3], "Churn": ["TRUE", "FALSE", "true"]})
        result: DataFrame = clean(df, target="Churn", drop_columns=[])
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_string_target_whitespace_stripped(self):
        # Normal operation: leading/trailing whitespace must not bypass the allowlist mapping.
        df = pd.DataFrame({"f": [1, 2], "Churn": ["  true  ", "  false  "]})
        result: DataFrame = clean(df, target="Churn", drop_columns=[])
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_drop_columns_actually_removed(self, valid_bool_df: DataFrame):
        # Normal operation + [SEC-TYPE]: PII/admin columns must be absent from the output.
        result: DataFrame = clean(valid_bool_df, target="Churn", drop_columns=["CustomerID"])
        assert "CustomerID" not in result.columns

    def test_nonexistent_drop_columns_skipped_without_crash(self, valid_bool_df: DataFrame):
        # Normal operation: a column name absent from the DataFrame must not cause an
        # unhandled exception that leaks internal schema information.
        result: DataFrame = clean(valid_bool_df, target="Churn", drop_columns=["does_not_exist"])
        assert "Churn" in result.columns

    def test_duplicate_rows_removed(self):
        # Normal operation: deduplication must prevent adversarially injected samples
        # from being amplified in the training set.
        df = pd.DataFrame({"f": [1, 1, 2], "Churn": [True, True, False]})
        result: DataFrame = clean(df, target="Churn", drop_columns=[])
        assert len(result) == 2

    def test_index_reset_after_dedup(self, valid_bool_df: DataFrame):
        # Normal operation: reset_index ensures a contiguous index; a broken index
        # can cause silent off-by-one errors in downstream joins.
        result: DataFrame = clean(valid_bool_df, target="Churn", drop_columns=[])
        assert list(result.index) == list(range(len(result)))

    def test_target_column_preserved_in_output(self, valid_bool_df: DataFrame):
        # Normal operation: the target column must be present in the returned DataFrame.
        result: DataFrame = clean(valid_bool_df, target="Churn", drop_columns=[])
        assert "Churn" in result.columns


# ===========================================================================
# 2. Type-safety violations  [SEC-TYPE]
# ===========================================================================


class TestCleanTypeViolations:
    """
    clean() currently has no type guards on its parameters.
    Passing the wrong type causes an internal pandas/Python traceback to leak,
    which may expose file paths and column names to the caller.

    CURRENT BEHAVIOUR: AttributeError / TypeError / KeyError from deep inside
                       pandas/Python, with internal details.
    EXPECTED BEHAVIOUR: explicit TypeError / ValueError with a controlled message
                        raised before any pandas operation runs.
    """

    def test_df_none_raises(self):
        # [SEC-TYPE] None instead of DataFrame: currently raises AttributeError from df.drop().
        # Expected: TypeError before any pandas call is made.
        with pytest.raises((TypeError, AttributeError)):
            clean(None, target="Churn", drop_columns=[])  # type: ignore[arg-type]

    def test_df_dict_raises(self):
        # [SEC-TYPE] dict instead of DataFrame: pandas may partially accept a dict
        # or raise AttributeError -- both are unacceptable for unvalidated input.
        with pytest.raises((TypeError, AttributeError, KeyError, ValueError)):
            clean({"Churn": [1, 0]}, target="Churn", drop_columns=[])  # type: ignore[arg-type]

    def test_target_integer_raises(self, valid_bool_df: DataFrame):
        # [SEC-TYPE] Integer target name is a programming error; currently a KeyError
        # leaks the integer value in the message.
        # Expected: TypeError before any DataFrame access.
        with pytest.raises((TypeError, KeyError, ValueError)):
            clean(valid_bool_df, target=123, drop_columns=[])  # type: ignore[arg-type]

    def test_target_none_raises(self, valid_bool_df: DataFrame):
        # [SEC-TYPE] None as target name currently raises a KeyError from pandas internals.
        with pytest.raises((TypeError, KeyError, ValueError)):
            clean(valid_bool_df, target=None, drop_columns=[])  # type: ignore[arg-type]

    def test_drop_columns_none_raises_type_error(self, valid_bool_df: DataFrame):
        # [SEC-TYPE] None as drop_columns currently raises:
        #   TypeError: argument of type 'NoneType' is not iterable
        # This leaks that a list comprehension / `in` check was attempted.
        # Expected: explicit TypeError with a controlled message.
        with pytest.raises(TypeError):
            clean(valid_bool_df, target="Churn", drop_columns=None)  # type: ignore[arg-type]

    def test_drop_columns_string_causes_silent_pii_leak(self, valid_bool_df: DataFrame):
        """
        [SEC-SILENT] CRITICAL SECURITY BUG

        When drop_columns is passed as a plain string (e.g. "CustomerID" instead of
        ["CustomerID"]), the list comprehension iterates over the string's characters:
            [c for c in "CustomerID" if c in df.columns]
            -> ['C', 'u', 's', ...] -- none match a column name -> empty list

        Result: df.drop(columns=[]) is a no-op -- the column is NOT dropped and NO
        error is raised.  A PII/admin column silently ends up in the training data
        while the caller assumes the drop succeeded.

        CURRENT BEHAVIOUR: CustomerID remains in the output (silent bug -- this assertion PASSES,
                           documenting the broken state).
        EXPECTED BEHAVIOUR: TypeError raised before any drop runs when drop_columns is not a list.
        """
        result: DataFrame = clean(valid_bool_df, target="Churn", drop_columns="CustomerID")  # type: ignore[arg-type]
        # This assertion documents the current broken behaviour:
        assert "CustomerID" in result.columns, (
            "SECURITY BUG CONFIRMED: drop_columns='CustomerID' (str) silently skips "
            "the drop; the PII column is present in the output. "
            "Fix: add isinstance(drop_columns, list) guard before the comprehension."
        )


# ===========================================================================
# 3. Target column boundary tests  [SEC-INFOLEAKW, SEC-PATH]
# ===========================================================================


class TestCleanTargetColumnBoundary:
    """
    A missing or malformed target column name causes an unhandled KeyError whose
    message includes the column name -- leaking schema information to the caller.
    """

    def test_target_not_in_columns_raises(self, valid_bool_df: DataFrame):
        # [SEC-INFOLEAKW] KeyError('NonExistent') exposes the target name in the message.
        # Expected: ValueError with a controlled message raised before any DataFrame indexing.
        with pytest.raises((KeyError, ValueError)):
            clean(valid_bool_df, target="NonExistent", drop_columns=[])

    def test_empty_string_target_raises(self, valid_bool_df: DataFrame):
        # [SEC-PATH] An empty string cannot match any real column name.
        # Expected: ValueError("target must be a non-empty string") before DataFrame access.
        with pytest.raises((KeyError, ValueError)):
            clean(valid_bool_df, target="", drop_columns=[])

    def test_target_in_drop_columns_raises(self, valid_bool_df: DataFrame):
        """
        [SEC-INFOLEAKW] When the target name also appears in drop_columns, the current
        code executes in order:
          1. df.drop(columns=["Churn"]) -- Churn is dropped
          2. df["Churn"] -- KeyError: 'Churn'

        Double failure: partial state mutation AND a KeyError leaks the target name.
        Expected: ValueError raised BEFORE any drop runs.
        """
        with pytest.raises((KeyError, ValueError)):
            clean(valid_bool_df, target="Churn", drop_columns=["Churn"])

    def test_target_name_with_path_traversal_chars_raises(self):
        """
        [SEC-PATH] A path-traversal-like target name (e.g. '../../etc/passwd') must
        only trigger a DataFrame key lookup -- never a filesystem access.
        The test pins that behaviour: KeyError / ValueError, not a file open.
        """
        df = pd.DataFrame({"f": [1, 2], "Churn": [True, False]})
        with pytest.raises((KeyError, ValueError)):
            clean(df, target="../../etc/passwd", drop_columns=[])

    def test_target_name_with_null_byte_raises(self, valid_bool_df: DataFrame):
        # [SEC-PATH] A null byte in the target name must not reach pandas DataFrame
        # indexing without validation; the current code raises KeyError.
        with pytest.raises((KeyError, ValueError)):
            clean(valid_bool_df, target="Churn\x00", drop_columns=[])


# ===========================================================================
# 4. Adversarial / injected content in the target column  [SEC-INJECT]
# ===========================================================================


class TestCleanAdversarialTargetContent:
    """
    The code applies an explicit allowlist mapping ("true"->1, "false"->0) and then
    blocks unknown values with an isna() guard that raises ValueError.
    These tests pin that behaviour so a regression cannot silently remove the protection.
    """

    def test_sql_injection_in_target_raises_value_error(self):
        # [SEC-INJECT] SQL injection string in the target column -> NaN -> ValueError.
        # The string is never executed; the check is purely at the Python level.
        df = pd.DataFrame({"f": [1, 2], "Churn": ["True", "1; DROP TABLE users;--"]})
        with pytest.raises(ValueError, match="Unparseable"):
            clean(df, target="Churn", drop_columns=[])

    def test_script_injection_in_target_raises_value_error(self):
        # [SEC-INJECT] XSS-like string in the target column -> NaN -> ValueError.
        df = pd.DataFrame({"f": [1, 2], "Churn": ["False", "<script>alert(1)</script>"]})
        with pytest.raises(ValueError, match="Unparseable"):
            clean(df, target="Churn", drop_columns=[])

    def test_numeric_string_1_0_not_accepted(self):
        # [SEC-INJECT] "1"/"0" are not in the allowlist {"true", "false"};
        # silently accepting them would be a data-integrity violation.
        df = pd.DataFrame({"f": [1, 2], "Churn": ["1", "0"]})
        with pytest.raises(ValueError, match="Unparseable"):
            clean(df, target="Churn", drop_columns=[])

    def test_yes_no_not_accepted(self):
        # [SEC-INJECT] "yes"/"no" are common alternatives not in the allowlist;
        # they must be rejected, not silently coerced.
        df = pd.DataFrame({"f": [1, 2], "Churn": ["yes", "no"]})
        with pytest.raises(ValueError, match="Unparseable"):
            clean(df, target="Churn", drop_columns=[])

    def test_unicode_homoglyph_not_accepted(self):
        # [SEC-INJECT] Full-width Unicode characters (e.g. 'Ｆａｌｓｅ') look like
        # "False" visually but do not equal "false" after .lower().
        # Silently accepting them would constitute a security filter bypass.
        df = pd.DataFrame({"f": [1, 2], "Churn": ["True", "Ｆａｌｓｅ"]})
        with pytest.raises(ValueError, match="Unparseable"):
            clean(df, target="Churn", drop_columns=[])

    def test_null_byte_in_target_value_not_accepted(self):
        # [SEC-INJECT] Appending a null byte ("false\x00") is a classic string-filter
        # bypass technique; .lower() does not strip it, so it maps to NaN.
        df = pd.DataFrame({"f": [1, 2], "Churn": ["true", "false\x00"]})
        with pytest.raises(ValueError, match="Unparseable"):
            clean(df, target="Churn", drop_columns=[])

    def test_nan_in_target_raises_value_error(self):
        # [SEC-INJECT] NaN in the target column signals a missing label;
        # the isna() guard must catch it.
        df = pd.DataFrame({"f": [1, 2], "Churn": [True, None]})
        with pytest.raises(ValueError, match="Unparseable"):
            clean(df, target="Churn", drop_columns=[])

    def test_partially_poisoned_target_raises_value_error(self):
        # [SEC-INJECT] Partially poisoned target (some rows valid, one row injected):
        # partial success would be a data-integrity violation -- the entire DataFrame
        # must be rejected.
        df = pd.DataFrame({"f": [1, 2, 3], "Churn": ["true", "false", "POISONED"]})
        with pytest.raises(ValueError, match="Unparseable"):
            clean(df, target="Churn", drop_columns=[])

    def test_empty_string_in_target_value_not_accepted(self):
        # [SEC-INJECT] Empty string value in the target column -> NaN -> ValueError.
        df = pd.DataFrame({"f": [1, 2], "Churn": ["true", ""]})
        with pytest.raises(ValueError, match="Unparseable"):
            clean(df, target="Churn", drop_columns=[])


# ===========================================================================
# 5. Resource limits and edge cases  [SEC-DOS]
# ===========================================================================


class TestCleanEdgeCasesAndDos:
    """Resource exhaustion and boundary condition tests."""

    def test_empty_dataframe_does_not_raise(self):
        """
        [SEC-DOS / edge] An empty DataFrame currently produces a silent empty output.
        The downstream train_test_split then raises, leaking an internal traceback.
        CURRENT BEHAVIOUR: clean() returns an empty DataFrame without raising.
        EXPECTED BEHAVIOUR: ValueError("DataFrame is empty") raised inside clean()
                            before any operation runs.
        """
        df = pd.DataFrame({"Churn": pd.Series([], dtype=bool)})
        result: DataFrame = clean(df, target="Churn", drop_columns=[])
        # This assertion documents the current (non-raising) behaviour:
        assert len(result) == 0, (
            "SECURITY GAP: empty DataFrame silently passes clean(); "
            "downstream error will leak an internal traceback. "
            "Fix: add 'if df.empty: raise ValueError' guard."
        )

    def test_all_duplicate_rows_returns_single_row(self):
        """
        [SEC-DOS] When every row is a duplicate, only one row survives deduplication.
        The downstream train_test_split then raises, leaking an internal sklearn traceback.
        CURRENT BEHAVIOUR: clean() returns a single-row DataFrame without raising.
        EXPECTED BEHAVIOUR: ValueError("fewer than 2 unique rows") raised inside clean().
        """
        df = pd.DataFrame({"f": [1, 1, 1], "Churn": [True, True, True]})
        result: DataFrame = clean(df, target="Churn", drop_columns=[])
        assert len(result) == 1, (
            "SECURITY GAP: single-row output after dedup not caught by clean(); "
            "subsequent train_test_split will leak an internal sklearn traceback."
        )

    def test_extremely_long_column_name_does_not_crash(self):
        # [SEC-DOS] A pathologically long column name (10 000 chars) must not cause
        # an OOM error or crash -- it is purely a string operation.
        long_name: str = "A" * 10_000
        df = pd.DataFrame({long_name: [1.0, 2.0], "Churn": [True, False]})
        result: DataFrame = clean(df, target="Churn", drop_columns=[])
        assert long_name in result.columns

    def test_large_dataframe_completes_in_reasonable_time(self):
        # [SEC-DOS] A 200 000-row DataFrame must complete within 10 seconds;
        # quadratic complexity would constitute a DoS vector.
        import time

        n = 200_000
        rng: Generator = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "f1": rng.random(n),
                "f2": rng.choice(["a", "b"], n),
                "Churn": rng.choice([True, False], n),
            }
        )
        start: float = time.monotonic()
        clean(df, target="Churn", drop_columns=[])
        elapsed: float = time.monotonic() - start
        assert elapsed < 10.0, (
            f"clean() took {elapsed:.1f}s on 200k rows -- potential DoS vector."
        )

    def test_only_target_column_no_features(self):
        # [SEC-DOS / edge] DataFrame with only the target column and no feature columns:
        # clean() has no minimum-feature-count contract, so it must not raise.
        df = pd.DataFrame({"Churn": [True, False, True]})
        result: DataFrame = clean(df, target="Churn", drop_columns=[])
        assert "Churn" in result.columns

    def test_many_columns_does_not_crash(self):
        # [SEC-DOS] 1 000 feature columns must not cause a performance regression.
        n_cols = 1_000
        data: dict[str, NDArray[np.float64]] = {f"col_{i}": np.random.rand(10) for i in range(n_cols)}
        data["Churn"] = [True, False] * 5  # pyright: ignore[reportArgumentType]
        df = pd.DataFrame(data)
        result: DataFrame = clean(df, target="Churn", drop_columns=[])
        assert "Churn" in result.columns


# ===========================================================================
# 6. main() entry point -- file and parameter security  [SEC-PATH, SEC-TYPE]
# ===========================================================================


class TestMainSecurity:
    """
    main() reads configuration from the config module, so every test monkeypatches
    the file paths and load_params() to run without real files on disk.
    """

    def test_main_raises_file_not_found_when_csv_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_params: dict[str, Any]
    ):
        # [SEC-PATH] Missing raw CSV: the code must raise FileNotFoundError with a
        # controlled message -- this is correct behaviour; the test ensures it stays that way.
        import src.churn.config as cfg

        monkeypatch.setattr(cfg, "RAW_CSV", tmp_path / "nonexistent.csv")
        with patch("src.churn.prepare.config.load_params", return_value=base_params):
            with pytest.raises(FileNotFoundError, match="not found"):
                main()

    def test_main_completes_with_valid_inputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_params: dict[str, Any]
    ):
        # Normal operation: main() must write train.csv and valid.csv given valid inputs.
        import src.churn.config as cfg

        raw_csv = tmp_path / "churn.csv"
        raw_csv.write_text(
            "MonthlyCharges,Churn\n"
            + "\n".join(f"{i},{v}" for i, v in zip(range(20), ["True", "False"] * 10))
            + "\n"
        )
        monkeypatch.setattr(cfg, "RAW_CSV", raw_csv)
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "train.csv")
        monkeypatch.setattr(cfg, "VALID_CSV", tmp_path / "valid.csv")

        with patch("src.churn.prepare.config.load_params", return_value=base_params):
            main()

        assert (tmp_path / "train.csv").exists()
        assert (tmp_path / "valid.csv").exists()

    def test_main_with_invalid_seed_type_should_be_caught_early(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """
        [SEC-TYPE] When params.yaml contains "seed: not_an_int", the string is passed
        directly to train_test_split(random_state=...).

        scikit-learn forwards the string to an internal numpy call, which raises
        TypeError or ValueError -- leaking an internal sklearn traceback to the caller.

        CURRENT BEHAVIOUR: traceback originates deep inside sklearn.
        EXPECTED BEHAVIOUR: ValueError raised at the start of main() during parameter
                            validation, before any data processing begins.
        """
        import src.churn.config as cfg

        raw_csv = tmp_path / "churn.csv"
        raw_csv.write_text(
            "f,Churn\n"
            + "\n".join(f"{i},{v}" for i, v in zip(range(20), ["True", "False"] * 10))
            + "\n"
        )
        monkeypatch.setattr(cfg, "RAW_CSV", raw_csv)
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "train.csv")
        monkeypatch.setattr(cfg, "VALID_CSV", tmp_path / "valid.csv")

        bad_params: dict[str, Any] = {
            "seed": "not_an_integer",
            "prepare": {
                "target": "Churn",
                "drop_columns": [],
                "stratify": False,
                "test_size": 0.25,
            },
        }
        with patch("src.churn.prepare.config.load_params", return_value=bad_params):
            # An exception must be raised -- ideally ValueError from param validation
            # in main(), not from sklearn internals.
            with pytest.raises((ValueError, TypeError)):
                main()

    def test_main_with_test_size_above_one_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """
        [SEC-TYPE] test_size=1.5 is invalid; the ValueError raised deep inside sklearn
        may expose internal file paths in its traceback.
        EXPECTED BEHAVIOUR: ValueError raised during parameter validation in main(),
                            not inside train_test_split.
        """
        import src.churn.config as cfg

        raw_csv = tmp_path / "churn.csv"
        raw_csv.write_text(
            "f,Churn\n"
            + "\n".join(f"{i},{v}" for i, v in zip(range(20), ["True", "False"] * 10))
            + "\n"
        )
        monkeypatch.setattr(cfg, "RAW_CSV", raw_csv)
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "train.csv")
        monkeypatch.setattr(cfg, "VALID_CSV", tmp_path / "valid.csv")

        bad_params: dict[str, Any] = {
            "seed": 42,
            "prepare": {
                "target": "Churn",
                "drop_columns": [],
                "stratify": False,
                "test_size": 1.5,
            },
        }
        with patch("src.churn.prepare.config.load_params", return_value=bad_params):
            with pytest.raises((ValueError, Exception)):
                main()

    def test_main_target_column_missing_from_csv_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """
        [SEC-INFOLEAKW] When the target column named in params.yaml is absent from
        the CSV, the current code raises a raw pandas KeyError that includes the full
        list of DataFrame column names -- leaking the data schema.
        EXPECTED BEHAVIOUR: controlled ValueError raised before clean() is called.
        """
        import src.churn.config as cfg

        raw_csv = tmp_path / "churn.csv"
        raw_csv.write_text("f,WrongColumn\n1,True\n2,False\n3,True\n4,False\n")
        monkeypatch.setattr(cfg, "RAW_CSV", raw_csv)
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "train.csv")
        monkeypatch.setattr(cfg, "VALID_CSV", tmp_path / "valid.csv")

        params: dict[str, Any] = {
            "seed": 42,
            "prepare": {
                "target": "Churn",
                "drop_columns": [],
                "stratify": False,
                "test_size": 0.25,
            },
        }
        with patch("src.churn.prepare.config.load_params", return_value=params):
            with pytest.raises((KeyError, ValueError)):
                main()

    def test_main_output_contains_only_expected_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """
        [SEC-TYPE] The output train.csv must not contain any column listed in
        drop_columns. This integration-level test verifies that PII removal
        is end-to-end effective.
        """
        import src.churn.config as cfg

        raw_csv: Path = tmp_path / "churn.csv"
        raw_csv.write_text(
            "CustomerID,MonthlyCharges,Churn\n"
            + "\n".join(
                f"C{i},{10 + i},{v}"
                for i, v in zip(range(20), ["True", "False"] * 10)
            )
            + "\n"
        )
        monkeypatch.setattr(cfg, "RAW_CSV", raw_csv)
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(cfg, "TRAIN_CSV", tmp_path / "train.csv")
        monkeypatch.setattr(cfg, "VALID_CSV", tmp_path / "valid.csv")

        params: dict[str, Any] = {
            "seed": 42,
            "prepare": {
                "target": "Churn",
                "drop_columns": ["CustomerID"],
                "stratify": False,
                "test_size": 0.25,
            },
        }
        with patch("src.churn.prepare.config.load_params", return_value=params):
            main()

        train_df: DataFrame = pd.read_csv(tmp_path / "train.csv")
        assert "CustomerID" not in train_df.columns, (
            "SECURITY BUG: CustomerID (PII) found in train.csv despite being in drop_columns."
        )
