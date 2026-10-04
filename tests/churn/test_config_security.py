"""Security-focused tests for src/churn/config.py

Test groups by security category:

  [SEC-PATH]      Path traversal: load_params() and write_json() accept arbitrary
                  Path objects with no project-boundary check; a crafted path can
                  read or write files anywhere the process has filesystem access to.
  [SEC-YAML]      YAML deserialization: yaml.safe_load must block !!python/object
                  gadget chains; anchor-expansion (billion-laughs) bombs are not
                  blocked by safe_load and have no size guard.
  [SEC-TYPE]      Missing type guards: passing wrong types causes internal
                  AttributeError / TypeError to bubble up instead of a clear,
                  actionable ValueError with a controlled message.
  [SEC-INFOLEAKW] Exception messages and module-level print() expose absolute
                  filesystem paths that could aid an attacker in mapping the
                  deployment layout.
  [SEC-SILENT]    Silent failure: load_params() on an empty YAML file returns None
                  without raising; callers then crash with an unhelpful TypeError
                  (e.g. NoneType object is not subscriptable).
  [SEC-DOS]       Resource exhaustion: unbounded YAML / JSON payloads have no size
                  guard; a crafted input can exhaust memory or disk space.
  [SEC-ENV]       Environment variable injection: get_logger() reads LOG_LEVEL from
                  the environment without allowlist validation.

Run:
    pytest tests/churn/test_config_security.py -v

CURRENT BEHAVIOUR / EXPECTED BEHAVIOUR annotations mark places where the current
code does NOT yet implement secure handling. Those tests pass while documenting
a confirmed security gap; fixing the gap requires modifying the source, which is
out of scope here.

Security hardening suggestions
-------------------------------
load_params():
  - Validate path is within PROJECT_ROOT (or an explicit allowlist of project paths):
      resolved = (path or PARAMS_PATH).resolve()
      if not resolved.is_relative_to(PROJECT_ROOT):
          raise ValueError("params path must be inside the project root")
  - Raise immediately when yaml.safe_load returns None:
      if params is None:
          raise ValueError(f"params file is empty: {path}")
  - Add a type guard before any I/O:
      if not isinstance(path, (Path, type(None))):
          raise TypeError(f"path must be a Path or None, got {type(path).__name__}")

write_json():
  - Validate path is within a declared project output directory:
      if not path.resolve().is_relative_to(PROJECT_ROOT):
          raise ValueError("output path must be inside the project root")
  - Add a payload type guard:
      if not isinstance(payload, dict):
          raise TypeError(f"payload must be a dict, got {type(payload).__name__}")
  - Enforce a maximum serialised size before writing to disk.

ensure_dirs():
  - Validate every directory is within PROJECT_ROOT:
      for d in dirs:
          if not isinstance(d, Path):
              raise TypeError(f"expected Path, got {type(d).__name__}")
          if not d.resolve().is_relative_to(PROJECT_ROOT):
              raise ValueError(f"directory must be inside the project root: {d}")

get_logger():
  - Validate LOG_LEVEL against an explicit allowlist:
      ALLOWED_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
      level = os.getenv("LOG_LEVEL", "INFO").upper()
      if level not in ALLOWED_LEVELS:
          level = "INFO"

config.py module level:
  - Remove the print("Project root:", PROJECT_ROOT) statement; use log.debug() instead.
    A module-level print exposes the absolute deployment path on every import.
"""
from __future__ import annotations

import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any
import pytest
import yaml

from src.churn import config
from src.churn.config import ensure_dirs, get_logger, load_params, write_json


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VALID_PARAMS_YAML = """\
seed: 42
prepare:
  target: Churn
  drop_columns:
    - CustomerID
  stratify: false
  test_size: 0.25
train:
  model: random_forest
  n_estimators: 100
  max_depth: 5
  min_samples_leaf: 1
  class_weight: balanced
"""


# ===========================================================================
# 1. Normal operation (positive tests)
# ===========================================================================


class TestLoadParamsNormalOperation:
    """Pin the expected contract -- if these fail, a security fix has broken existing logic."""

    def test_returns_dict(self, tmp_path: Path):
        # Normal operation: a well-formed params.yaml must return a dict.
        f: Path = tmp_path / "params.yaml"
        f.write_text(VALID_PARAMS_YAML, encoding="utf-8")
        result: dict[str, Any] = load_params(f)
        assert isinstance(result, dict)

    def test_returns_expected_top_level_keys(self, tmp_path: Path):
        # Normal operation: top-level keys must match the params.yaml schema.
        f: Path = tmp_path / "params.yaml"
        f.write_text(VALID_PARAMS_YAML, encoding="utf-8")
        result: dict[str, Any] = load_params(f)
        assert "seed" in result
        assert "prepare" in result
        assert "train" in result

    def test_custom_path_accepted(self, tmp_path: Path):
        # Normal operation: a custom Path must be accepted in addition to the default.
        custom: Path = tmp_path / "custom_params.yaml"
        custom.write_text("seed: 7\n", encoding="utf-8")
        result: dict[str, Any] = load_params(custom)
        assert result["seed"] == 7

    def test_unicode_values_preserved(self, tmp_path: Path):
        # Normal operation: YAML values containing unicode must be returned correctly.
        f: Path = tmp_path / "params.yaml"
        f.write_text("label: Ñoño\nseed: 1\n", encoding="utf-8")
        result: dict[str, Any] = load_params(f)
        assert result["label"] == "Ñoño"

    def test_nested_dict_fully_accessible(self, tmp_path: Path):
        # Normal operation: nested YAML structure must be fully traversable.
        f: Path = tmp_path / "params.yaml"
        f.write_text(VALID_PARAMS_YAML, encoding="utf-8")
        result: dict[str, Any] = load_params(f)
        assert result["prepare"]["target"] == "Churn"
        assert isinstance(result["prepare"]["drop_columns"], list)


# ===========================================================================
# 2. YAML deserialization security  [SEC-YAML, SEC-SILENT, SEC-DOS]
# ===========================================================================


class TestLoadParamsYamlSecurity:
    """
    yaml.safe_load is the correct choice over yaml.load; these tests verify the
    protection holds and document the gaps that safe_load does NOT close.

    CURRENT BEHAVIOUR: safe_load blocks Python object gadget chains (correct).
                       Anchor-expansion DoS is not bounded -- no size guard (gap).
                       Empty YAML returns None without raising (gap -- see [SEC-SILENT]).
    EXPECTED BEHAVIOUR: explicit ValueError when yaml.safe_load returns None;
                        a node-count or size limit on YAML input.
    """

    def test_safe_load_blocks_python_object_injection(self, tmp_path: Path):
        # [SEC-YAML] !!python/object/apply can execute arbitrary code in yaml.load().
        # safe_load must raise yaml.YAMLError for this gadget chain.
        malicious: Path = tmp_path / "malicious.yaml"
        malicious.write_text(
            "exploit: !!python/object/apply:os.system ['id']\n",
            encoding="utf-8",
        )
        with pytest.raises(yaml.YAMLError):
            load_params(malicious)

    def test_safe_load_blocks_python_object_module_access(self, tmp_path: Path):
        # [SEC-YAML] !!python/object with an arbitrary module path must be rejected.
        malicious: Path = tmp_path / "exec.yaml"
        malicious.write_text(
            "exploit: !!python/object:subprocess.Popen\n"
            "  - ['id']\n",
            encoding="utf-8",
        )
        with pytest.raises(yaml.YAMLError):
            load_params(malicious)

    def test_empty_yaml_returns_none_silently(self, tmp_path: Path):
        """
        [SEC-SILENT] SECURITY GAP

        yaml.safe_load on an empty file returns None (not a dict), and load_params()
        forwards None to the caller without raising.  The first downstream access
        (e.g. params["seed"]) then crashes with an unhelpful:
            TypeError: 'NoneType' object is not subscriptable

        CURRENT BEHAVIOUR: returns None (this assertion PASSES, documenting the gap).
        EXPECTED BEHAVIOUR: raise ValueError("params file is empty: <path>") before
                            returning, so the caller receives an actionable error.
        """
        empty: Path = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        result: dict[str, Any] = load_params(empty)
        assert result is None, (
            "SECURITY GAP CONFIRMED: load_params() on an empty YAML file returns None "
            "without raising -- callers receive NoneType instead of a dict. "
            "Fix: add 'if params is None: raise ValueError(...)' after safe_load."
        )

    def test_malformed_yaml_raises_yaml_error(self, tmp_path: Path):
        # [SEC-YAML] A syntactically invalid YAML file must raise yaml.YAMLError,
        # not silently return None or a partial result.
        bad: Path = tmp_path / "bad.yaml"
        bad.write_text("key: : unexpected_colon_here\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_params(bad)

    def test_yaml_anchor_expansion_has_no_size_guard(self, tmp_path: Path):
        """
        [SEC-DOS] SECURITY GAP

        YAML anchor bombs exploit alias expansion; yaml.safe_load still expands
        aliases in memory even though it blocks Python object construction.
        A 3-level bomb with 10x fanout produces 10^3 = 1 000 items -- small enough
        to run without harm, large enough to document the absence of a size guard.

        CURRENT BEHAVIOUR: anchor expansion succeeds with no limit (documents the gap).
        EXPECTED BEHAVIOUR: raise ValueError when the expanded structure exceeds a
                            configurable node limit (e.g. 100 000 nodes).
        """
        bomb: Path = tmp_path / "bomb.yaml"
        bomb.write_text(
            "a: &a [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n"
            "b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a, *a]\n"
            "c: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b, *b]\n",
            encoding="utf-8",
        )
        start: float = time.monotonic()
        result: dict[str, Any] = load_params(bomb)
        elapsed: float = time.monotonic() - start
        # Anchor expansion succeeds -- no guard is in place.
        assert isinstance(result, dict), (
            "SECURITY GAP CONFIRMED: YAML anchor bomb expanded in memory without a "
            f"node-count limit ({elapsed:.3f}s elapsed). "
            "Fix: enforce a maximum node or byte limit before parsing."
        )


# ===========================================================================
# 3. Path traversal and information leakage  [SEC-PATH, SEC-INFOLEAKW]
# ===========================================================================


class TestLoadParamsPathBoundary:
    """
    load_params() accepts an arbitrary Path without verifying that it is within the
    project root.  A developer who misconfigures a run, or an attacker who can
    substitute params.yaml, can redirect the read to any readable file.
    """

    def test_nonexistent_path_raises_filenotfounderror(self, tmp_path: Path):
        # Expected behaviour: a missing file raises FileNotFoundError (not a crash).
        with pytest.raises(FileNotFoundError):
            load_params(tmp_path / "does_not_exist.yaml")

    def test_filenotfounderror_message_exposes_absolute_path(self, tmp_path: Path):
        """
        [SEC-INFOLEAKW] SECURITY GAP

        The FileNotFoundError from Path.open() includes the absolute path of the
        missing file in its message.  In a deployed service this reveals the
        filesystem layout (e.g. /home/prod/mlflow/data/params.yaml) to the caller.

        CURRENT BEHAVIOUR: path appears in the error message (assertion PASSES).
        EXPECTED BEHAVIOUR: catch OSError and re-raise with a sanitised message
                            that omits the absolute path.
        """
        missing: Path = tmp_path / "no_such_file.yaml"
        try:
            load_params(missing)
        except FileNotFoundError as exc:
            assert str(missing) in str(exc), (
                "SECURITY GAP CONFIRMED: the absolute path appears in the "
                "FileNotFoundError message, leaking the filesystem layout to the caller."
            )
        else:
            pytest.fail("Expected FileNotFoundError was not raised.")

    def test_arbitrary_absolute_path_outside_project_is_readable(self, tmp_path: Path):
        """
        [SEC-PATH] SECURITY GAP

        load_params() has no project-boundary check.  Any file the process can open
        can be passed as the path argument.  This test demonstrates the gap by
        reading a file in tmp_path (which is outside the project root).

        CURRENT BEHAVIOUR: file is read and its content returned (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError when the resolved path is not within
                            PROJECT_ROOT (or a declared allowlist of safe paths).
        """
        outside: Path = tmp_path / "exfiltrated_secrets.yaml"
        outside.write_text("api_key: s3cr3t_t0k3n\n", encoding="utf-8")
        result: dict[str, Any] = load_params(outside)
        assert result == {"api_key": "s3cr3t_t0k3n"}, (
            "SECURITY GAP CONFIRMED: load_params() reads a file outside the project "
            "root with no boundary check. "
            "Fix: resolve the path and assert it is relative to PROJECT_ROOT."
        )

    def test_path_with_traversal_segments_is_accepted(self, tmp_path: Path):
        """
        [SEC-PATH] SECURITY GAP

        A Path constructed with '..' components resolves outside the intended
        directory; load_params() accepts it without normalisation or boundary check.

        CURRENT BEHAVIOUR: traversal path resolves and is read (assertion PASSES).
        EXPECTED BEHAVIOUR: resolve the path with Path.resolve() and verify it is
                            within PROJECT_ROOT before opening the file.
        """
        inner: Path = tmp_path / "subdir"
        inner.mkdir()
        target: Path = tmp_path / "outside.yaml"
        target.write_text("traversed: true\n", encoding="utf-8")
        traversal: Path = inner / ".." / "outside.yaml"
        result: dict[str, Any] = load_params(traversal)
        assert result == {"traversed": True}, (
            "SECURITY GAP CONFIRMED: a Path with '..' components is accepted and "
            "read without boundary normalisation."
        )


# ===========================================================================
# 4. Type-safety violations  [SEC-TYPE]
# ===========================================================================


class TestLoadParamsTypeSafety:
    """
    load_params() has no type guard on the `path` parameter.
    Passing the wrong type leaks an internal AttributeError to the caller instead
    of a clear, controlled TypeError raised before any I/O.

    CURRENT BEHAVIOUR: AttributeError from path.open() propagates to caller.
    EXPECTED BEHAVIOUR: explicit TypeError("path must be a Path or None, got ...")
                        raised before any filesystem access.
    """

    def test_none_uses_default_params_path(self):
        # None is the intended default sentinel; must not raise TypeError.
        # FileNotFoundError is acceptable when params.yaml is absent in the test env.
        try:
            load_params(None)
        except FileNotFoundError:
            pass  # Acceptable: default PARAMS_PATH doesn't exist in CI.
        except Exception as exc:
            pytest.fail(
                f"None should resolve to the default path, not raise "
                f"{type(exc).__name__}: {exc}"
            )

    def test_string_instead_of_path_raises(self):
        """
        [SEC-TYPE] SECURITY GAP

        Passing a str instead of a Path causes AttributeError from str.open()
        (str has no .open() method), leaking that Path.open() was expected.

        CURRENT BEHAVIOUR: AttributeError (assertion PASSES, documents the gap).
        EXPECTED BEHAVIOUR: explicit TypeError("path must be a Path or None")
                            before any I/O is attempted.
        """
        with pytest.raises((AttributeError, TypeError)):
            load_params("params.yaml")  # type: ignore[arg-type]

    def test_integer_instead_of_path_raises(self):
        # [SEC-TYPE] int has no .open() method; AttributeError leaks implementation detail.
        with pytest.raises((AttributeError, TypeError)):
            load_params(42)  # type: ignore[arg-type]

    def test_empty_list_silently_uses_default_path(self):
        """
        [SEC-TYPE] SECURITY GAP -- falsy sentinel bypass

        load_params() uses `path = path or PARAMS_PATH` to handle the None sentinel.
        Any falsy non-None value ([], 0, "", {}) evaluates to False in the boolean
        expression and silently falls back to PARAMS_PATH instead of raising TypeError.

        Concretely: load_params([]) -> PARAMS_PATH (no error, no warning).
        An empty list as a path argument is clearly a programming error, but the
        function silently masks it.

        CURRENT BEHAVIOUR: [] resolves to PARAMS_PATH without any error
                           (this assertion PASSES, documenting the silent fallback).
        EXPECTED BEHAVIOUR: explicit type guard -- if not isinstance(path, (Path, type(None))):
                            raise TypeError(...)  -- placed BEFORE the `or` expression.
        """
        try:
            load_params([])  # type: ignore[arg-type]
            # [] is falsy -- falls through to PARAMS_PATH silently (confirmed bug)
        except FileNotFoundError:
            # PARAMS_PATH doesn't exist in this environment --
            # but the critical point is that NO TypeError was raised.
            pass
        except (AttributeError, TypeError) as exc:
            pytest.fail(
                f"A type guard is not expected in the current code; "
                f"[] should silently use PARAMS_PATH, not raise {type(exc).__name__}. "
                f"If this fails, the code has been hardened -- update the test."
            )


# ===========================================================================
# 5. ensure_dirs() normal operation
# ===========================================================================


class TestEnsureDirsNormalOperation:
    """Pin the expected contract for ensure_dirs() -- if these fail, a security
    fix has broken existing directory-creation logic."""

    def test_creates_single_directory(self, tmp_path: Path):
        # Normal operation: a new directory must be created.
        new_dir: Path = tmp_path / "new_dir"
        assert not new_dir.exists()
        ensure_dirs(new_dir)
        assert new_dir.is_dir()

    def test_creates_nested_directories(self, tmp_path: Path):
        # Normal operation: parents=True must create the full chain of directories.
        nested: Path = tmp_path / "a" / "b" / "c"
        ensure_dirs(nested)
        assert nested.is_dir()

    def test_idempotent_on_existing_directory(self, tmp_path: Path):
        # Normal operation: exist_ok=True means calling ensure_dirs twice must not raise.
        ensure_dirs(tmp_path)
        ensure_dirs(tmp_path)  # Second call must be a no-op.


# ===========================================================================
# 6. ensure_dirs() security  [SEC-PATH, SEC-TYPE]
# ===========================================================================


class TestEnsureDirsSecurity:
    """
    ensure_dirs() has no type guard on its arguments and no project-boundary check.
    """

    def test_can_create_directories_outside_project_root(self, tmp_path: Path):
        """
        [SEC-PATH] SECURITY GAP

        ensure_dirs() calls Path.mkdir() with no boundary verification.
        An adversarially crafted path can create directory trees anywhere the
        process has write access, including outside the project root.

        CURRENT BEHAVIOUR: directory is created outside the project (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError when the resolved path is not within
                            PROJECT_ROOT.
        """
        outside: Path = tmp_path / "injected_outside_dir"
        ensure_dirs(outside)
        assert outside.is_dir(), (
            "SECURITY GAP CONFIRMED: ensure_dirs() created a directory outside the "
            "project root with no boundary check. "
            "Fix: resolve the path and assert it is relative to PROJECT_ROOT."
        )

    def test_string_argument_raises_attributeerror(self, tmp_path: Path):
        """
        [SEC-TYPE] SECURITY GAP

        ensure_dirs() has no type guard; a str argument causes AttributeError from
        str.mkdir() (str has no mkdir method), leaking the internal call site.

        CURRENT BEHAVIOUR: AttributeError (assertion PASSES, documents the gap).
        EXPECTED BEHAVIOUR: explicit TypeError raised before any mkdir call.
        """
        with pytest.raises((AttributeError, TypeError)):
            ensure_dirs(str(tmp_path / "new_dir"))  # type: ignore[arg-type]

    def test_null_byte_in_directory_name_raises_os_error(self, tmp_path: Path):
        # [SEC-PATH] Null bytes in path names are rejected by POSIX kernels.
        # This test documents that the OS provides this defence, not the application.
        null_path: Path = tmp_path / "dir\x00name"
        with pytest.raises((ValueError, OSError)):
            ensure_dirs(null_path)


# ===========================================================================
# 7. write_json() normal operation
# ===========================================================================


class TestWriteJsonNormalOperation:
    """Pin the expected contract for write_json()."""

    def test_round_trip_preserves_values(self, tmp_path: Path):
        # Normal operation: values written must survive a JSON round-trip unchanged.
        out: Path = tmp_path / "metrics.json"
        write_json(out, {"accuracy": 0.95, "f1": 0.87})
        loaded = json.loads(out.read_text())
        assert loaded == {"accuracy": 0.95, "f1": 0.87}

    def test_output_keys_are_sorted(self, tmp_path: Path):
        # Normal operation: sort_keys=True ensures stable diffs in VCS.
        out: Path = tmp_path / "metrics.json"
        write_json(out, {"z_metric": 1, "a_metric": 2})
        raw: str = out.read_text()
        assert raw.index('"a_metric"') < raw.index('"z_metric"'), (
            "Keys must be sorted alphabetically for reproducible diffs."
        )

    def test_output_is_indented(self, tmp_path: Path):
        # Normal operation: indent=2 must be applied for human readability.
        out: Path = tmp_path / "metrics.json"
        write_json(out, {"k": 1})
        assert "  " in out.read_text(), "Output must be indented with 2 spaces."

    def test_output_ends_with_newline(self, tmp_path: Path):
        # Normal operation: trailing newline keeps POSIX tools (cat, diff, grep) happy.
        out: Path = tmp_path / "metrics.json"
        write_json(out, {"k": 1})
        assert out.read_text().endswith("\n")

    def test_creates_missing_parent_directories(self, tmp_path: Path):
        # Normal operation: ensure_dirs must be called so deep output paths work.
        out: Path = tmp_path / "a" / "b" / "metrics.json"
        write_json(out, {"loss": 0.1})
        assert out.exists()


# ===========================================================================
# 8. write_json() security  [SEC-PATH, SEC-TYPE, SEC-DOS]
# ===========================================================================


class TestWriteJsonSecurity:
    """
    write_json() has no path-boundary check and no payload type or size guard.
    """

    def test_path_traversal_writes_file_outside_project_root(self, tmp_path: Path):
        """
        [SEC-PATH] SECURITY GAP

        write_json() accepts any Path with no project-boundary check; an adversarially
        crafted path can write files anywhere the process has write access.

        CURRENT BEHAVIOUR: file is written outside the project root (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError when the resolved path is not within
                            PROJECT_ROOT.
        """
        outside: Path = tmp_path / "exfiltrated_metrics.json"
        write_json(outside, {"score": 1.0})
        assert outside.exists(), (
            "SECURITY GAP CONFIRMED: write_json() wrote a file outside the project "
            "root with no boundary check. "
            "Fix: resolve the path and assert it is relative to PROJECT_ROOT."
        )

    def test_list_payload_serialized_without_type_guard(self, tmp_path: Path):
        """
        [SEC-TYPE] SECURITY GAP

        write_json() has no type guard on `payload`; a list is valid JSON and is
        silently written as a JSON array even though callers expect only dicts.
        A metrics consumer parsing the file would get a list instead of an object.

        CURRENT BEHAVIOUR: list is written as a JSON array (assertion PASSES, documents gap).
        EXPECTED BEHAVIOUR: raise TypeError("payload must be a dict, got list") before
                            any file I/O.
        """
        out: Path = tmp_path / "out.json"
        write_json(out, [1, 2, 3])  # type: ignore[arg-type]
        loaded = json.loads(out.read_text())
        assert loaded == [1, 2, 3], (
            "SECURITY GAP CONFIRMED: write_json() accepted a list payload without a "
            "type guard -- the file contains a JSON array instead of an object."
        )

    def test_none_payload_serialized_as_json_null(self, tmp_path: Path):
        """
        [SEC-TYPE] SECURITY GAP

        None is silently serialized as JSON null without any type guard.
        A caller that accidentally passes None writes a useless metrics file
        and the downstream consumer receives null instead of a metrics object.

        CURRENT BEHAVIOUR: None is written as "null" (assertion PASSES, documents gap).
        EXPECTED BEHAVIOUR: raise TypeError("payload must be a dict, got NoneType").
        """
        out: Path = tmp_path / "out.json"
        write_json(out, None)  # type: ignore[arg-type]
        assert out.read_text().strip() == "null", (
            "SECURITY GAP CONFIRMED: None payload is serialized as JSON null without "
            "a type guard -- the metrics file contains null instead of an object."
        )

    def test_non_serializable_payload_raises_type_error(self, tmp_path: Path):
        # [SEC-TYPE] json.dump() itself enforces that values are JSON-serializable;
        # a datetime object (not serializable by default) must raise TypeError.
        # This is the one type boundary the function enforces indirectly via json.
        import datetime
        out: Path = tmp_path / "out.json"
        with pytest.raises(TypeError):
            write_json(out, {"ts": datetime.datetime.now()})

    def test_large_payload_written_without_size_guard(self, tmp_path: Path):
        """
        [SEC-DOS] SECURITY GAP

        write_json() has no maximum payload size check.  A crafted dict with a very
        large string value could exhaust disk space or cause I/O timeouts.
        This test writes a 1 MB value to document the absence of a size guard.

        CURRENT BEHAVIOUR: large payload written without error (assertion PASSES).
        EXPECTED BEHAVIOUR: raise ValueError when the estimated serialised size
                            exceeds a configurable threshold (e.g. 10 MB).
        """
        out: Path = tmp_path / "large.json"
        large_payload: dict[str, str] = {"data": "x" * 1_000_000}  # 1 MB string
        write_json(out, large_payload)
        assert out.stat().st_size > 1_000_000, (
            "SECURITY GAP CONFIRMED: write_json() wrote a 1 MB payload without any "
            "size guard in place."
        )


# ===========================================================================
# 9. get_logger() security  [SEC-ENV]
# ===========================================================================


class TestGetLoggerSecurity:
    """
    get_logger() reads LOG_LEVEL from the environment without allowlist validation.
    Python's logging module provides implicit protection against code injection,
    but the absence of explicit validation is still a security gap.
    """

    def test_returns_logger_instance(self):
        # Normal operation: must return a stdlib logging.Logger object.
        logger: logging.Logger = get_logger("test.config.security")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches_argument(self):
        # Normal operation: the returned logger must carry the name passed in.
        logger: logging.Logger = get_logger("test.expected.name")
        assert logger.name == "test.expected.name"

    def test_invalid_log_level_env_var_is_not_validated(self, monkeypatch: pytest.MonkeyPatch):
        """
        [SEC-ENV] SECURITY GAP

        An attacker who controls the environment can set LOG_LEVEL to an arbitrary
        string.  Python's logging.basicConfig(level=<garbage>) silently ignores the
        call when the root logger already has handlers -- the invalid value passes
        through without any allowlist check or warning message.

        CURRENT BEHAVIOUR: no exception raised; invalid value is silently accepted
                           (assertion PASSES, documents the gap).
        EXPECTED BEHAVIOUR: validate LOG_LEVEL against the set
                            {"DEBUG","INFO","WARNING","ERROR","CRITICAL"};
                            fall back to "INFO" and emit a log.warning() for
                            any unrecognised value.
        """
        monkeypatch.setenv("LOG_LEVEL", "NOT_A_VALID_LEVEL")
        try:
            logger: logging.Logger = get_logger("test.invalid_level")
            assert isinstance(logger, logging.Logger), (
                "SECURITY GAP CONFIRMED: invalid LOG_LEVEL accepted without validation "
                "or fallback -- no allowlist check is in place."
            )
        except ValueError:
            # If root logger has no handlers yet, logging raises ValueError.
            # That is also a gap: the service crashes on bad input.
            pass

    def test_shell_injection_in_log_level_env_var_is_benign(self, monkeypatch: pytest.MonkeyPatch):
        """
        [SEC-ENV] A shell-injection-like LOG_LEVEL value (e.g. '$(id)') must not
        execute code.  Python's logging module treats the env var value as a plain
        string key lookup, not a shell command.  This test pins that safe behaviour.
        """
        monkeypatch.setenv("LOG_LEVEL", "$(id); rm -rf /")
        try:
            logger: logging.Logger = get_logger("test.injection")
            assert logger is not None
        except Exception as exc:
            pytest.fail(
                f"Injected LOG_LEVEL raised an unexpected exception: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_empty_log_level_env_var_raises_or_is_ignored(self, monkeypatch: pytest.MonkeyPatch):
        """
        [SEC-ENV] SECURITY GAP

        An empty string LOG_LEVEL (LOG_LEVEL='') is passed to basicConfig(level='').
        If the root logger has no handlers yet, Python raises:
            ValueError: Unknown level: ''
        crashing the application on startup.  If handlers already exist,
        basicConfig is a no-op and the crash is silently avoided.

        Either outcome is undesired: the application's behaviour on misconfiguration
        depends on import order, not on an explicit guard.

        CURRENT BEHAVIOUR: ValueError or silent no-op depending on handler state
                           (both document the absence of an explicit empty-string guard).
        EXPECTED BEHAVIOUR: treat empty string as "INFO" (same as a missing env var).
        """
        monkeypatch.setenv("LOG_LEVEL", "")
        try:
            get_logger("test.empty_level")
            # If we get here, basicConfig was a no-op (handlers already exist).
            # Document this as a gap: the guard is not explicit.
        except ValueError:
            # ValueError("Unknown level: ''") -- also a gap: crash on bad input.
            pass


# ===========================================================================
# 10. Module-level information leakage  [SEC-INFOLEAKW]
# ===========================================================================


class TestModuleLevelInfoLeakage:
    """
    config.py contains a print() call at module level that fires on every import,
    exposing the absolute deployment path to stdout.
    """

    def test_project_root_print_in_source_leaks_path(self):
        """
        [SEC-INFOLEAKW] SECURITY GAP

        The statement print("Project root:", PROJECT_ROOT) in config.py is executed
        on every import of the module.  In any environment where stdout is captured
        and forwarded (CI logs, application logs, error tracking services) this
        exposes the absolute filesystem path of the deployment -- information that
        aids an attacker in mapping the host environment.

        CURRENT BEHAVIOUR: print() statement exists in source (assertion PASSES).
        EXPECTED BEHAVIOUR: remove the print() or replace with log.debug() so the
                            path is only emitted when DEBUG logging is explicitly enabled.
        """
        source: str = inspect.getsource(config)
        assert 'print("Project root:"' in source or "print('Project root:'" in source, (
            "SECURITY GAP CONFIRMED: config.py contains a module-level print() that "
            "exposes PROJECT_ROOT (the absolute deployment path) to stdout on every import. "
            "Fix: remove the print() or replace with log.debug()."
        )
