"""Tests for Phase 7.4 trial-file Python syntax checks."""

from __future__ import annotations

from pathlib import Path

from gpuboost.schemas.trial import TrialWorkspace
from gpuboost.trial.syntax_check import is_python_file, run_python_syntax_check
from gpuboost.trial.workspace import create_trial_workspace


def test_py_file_with_valid_syntax_returns_passed(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path, "train.py", "x = 1\n")

    status, step = run_python_syntax_check(workspace)

    assert status == "passed"
    assert step.name == "syntax_check"
    assert step.status == "passed"
    assert step.error is None
    assert step.duration_sec is not None


def test_py_file_with_invalid_syntax_returns_failed(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path, "train.py", "if True\n")

    status, step = run_python_syntax_check(workspace)

    assert status == "failed"
    assert step.name == "syntax_check"
    assert step.status == "failed"
    assert step.error
    assert step.duration_sec is not None


def test_txt_file_returns_skipped_by_default(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path, "notes.txt", "x = 1\n")

    status, step = run_python_syntax_check(workspace)

    assert status == "skipped"
    assert step.name == "syntax_check"
    assert step.status == "skipped"
    assert step.error is None


def test_txt_file_with_force_python_can_pass(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path, "notes.txt", "x = 1\n")

    status, step = run_python_syntax_check(workspace, force_python=True)

    assert status == "passed"
    assert step.status == "passed"


def test_missing_trial_file_returns_failed(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path, "train.py", "x = 1\n")
    Path(workspace.trial_file).unlink()

    status, step = run_python_syntax_check(workspace)

    assert status == "failed"
    assert step.status == "failed"
    assert "does not exist" in str(step.error)


def test_unsafe_workspace_where_trial_file_is_original_returns_failed(
    tmp_path: Path,
) -> None:
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")
    workspace = TrialWorkspace(
        original_file=str(original.resolve()),
        workspace_path=str(tmp_path.resolve()),
        trial_file=str(original.resolve()),
        cleanup_enabled=True,
        created_at="2026-01-01T00:00:00+00:00",
    )

    status, step = run_python_syntax_check(workspace)

    assert status == "failed"
    assert step.status == "failed"


def test_trial_file_outside_workspace_returns_failed(tmp_path: Path) -> None:
    original, workspace = _make_workspace(tmp_path, "train.py", "x = 1\n")
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("x = 1\n", encoding="utf-8")
    unsafe_workspace = TrialWorkspace(
        original_file=str(original.resolve()),
        workspace_path=workspace.workspace_path,
        trial_file=str(outside_file.resolve()),
        cleanup_enabled=True,
        created_at=workspace.created_at,
    )

    status, step = run_python_syntax_check(unsafe_workspace)

    assert status == "failed"
    assert step.status == "failed"


def test_syntax_check_does_not_execute_code(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(
        tmp_path,
        "train.py",
        'raise RuntimeError("should not execute")\n',
    )

    status, step = run_python_syntax_check(workspace)

    assert status == "passed"
    assert step.status == "passed"


def test_is_python_file_true_false_case_insensitive() -> None:
    assert is_python_file("train.py") is True
    assert is_python_file("TRAIN.PY") is True
    assert is_python_file("notes.txt") is False
    assert is_python_file("script.py.txt") is False


def test_trial_step_has_syntax_check_name_and_duration(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path, "train.py", "x = 1\n")

    _status, step = run_python_syntax_check(workspace)

    assert step.name == "syntax_check"
    assert step.duration_sec is not None


def _make_workspace(
    tmp_path: Path,
    filename: str,
    source_text: str,
) -> tuple[Path, TrialWorkspace]:
    original = tmp_path / filename
    original.write_text(source_text, encoding="utf-8")
    workspace, _steps = create_trial_workspace(str(original), base_dir=str(tmp_path))
    return original, workspace
