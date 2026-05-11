"""Tests for Phase 7.5 trial test command runner."""

from __future__ import annotations

import sys
from pathlib import Path

from gpuboost.schemas.trial import TrialWorkspace
from gpuboost.trial.test_command import (
    TEST_COMMAND_WARNING,
    run_trial_test_command,
    should_run_test_command,
)
from gpuboost.trial.workspace import create_trial_workspace


def test_none_command_returns_skipped_and_does_not_run(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    _original, workspace = _make_workspace(tmp_path)

    status, step = run_trial_test_command(workspace, None)

    assert status == "skipped"
    assert step.name == "run_test_command"
    assert step.status == "skipped"
    assert step.stdout is None
    assert step.stderr is None
    assert step.exit_code is None
    assert not marker.exists()


def test_blank_command_returns_skipped(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path)

    status, step = run_trial_test_command(workspace, "   ")

    assert status == "skipped"
    assert step.status == "skipped"


def test_successful_command_returns_passed(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path)

    status, step = run_trial_test_command(workspace, _python_command("print('ok')"))

    assert status == "passed"
    assert step.status == "passed"
    assert step.exit_code == 0


def test_failing_command_returns_failed_and_captures_exit_code(
    tmp_path: Path,
) -> None:
    _original, workspace = _make_workspace(tmp_path)
    command = _python_command("import sys; sys.exit(2)")

    status, step = run_trial_test_command(workspace, command)

    assert status == "failed"
    assert step.status == "failed"
    assert step.exit_code == 2


def test_stdout_is_captured(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path)

    _status, step = run_trial_test_command(workspace, _python_command("print('ok')"))

    assert step.stdout == "ok\n"


def test_stderr_is_captured(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path)
    command = _python_command("import sys; print('err', file=sys.stderr)")

    _status, step = run_trial_test_command(workspace, command)

    assert step.stderr == "err\n"


def test_command_runs_in_workspace_not_original_directory(tmp_path: Path) -> None:
    original_dir = tmp_path / "original"
    workspace_root = tmp_path / "trial-root"
    original_dir.mkdir()
    original = original_dir / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")
    workspace, _steps = create_trial_workspace(
        str(original),
        base_dir=str(workspace_root),
    )
    command = _python_command(
        "from pathlib import Path; "
        "Path('workspace-marker.txt').write_text('ok', encoding='utf-8')"
    )

    status, step = run_trial_test_command(workspace, command)

    assert status == "passed"
    assert step.status == "passed"
    assert (Path(workspace.workspace_path) / "workspace-marker.txt").exists()
    assert not (original_dir / "workspace-marker.txt").exists()


def test_timeout_returns_failed(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path)
    command = _python_command("import time; time.sleep(2)")

    status, step = run_trial_test_command(workspace, command, timeout_sec=1)

    assert status == "failed"
    assert step.status == "failed"
    assert "timed out" in str(step.error)


def test_missing_trial_file_returns_failed(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path)
    Path(workspace.trial_file).unlink()

    status, step = run_trial_test_command(workspace, _python_command("print('ok')"))

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

    status, step = run_trial_test_command(workspace, _python_command("print('ok')"))

    assert status == "failed"
    assert step.status == "failed"


def test_trial_file_outside_workspace_returns_failed(tmp_path: Path) -> None:
    original, workspace = _make_workspace(tmp_path)
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("x = 1\n", encoding="utf-8")
    unsafe_workspace = TrialWorkspace(
        original_file=str(original.resolve()),
        workspace_path=workspace.workspace_path,
        trial_file=str(outside_file.resolve()),
        cleanup_enabled=True,
        created_at=workspace.created_at,
    )

    status, step = run_trial_test_command(
        unsafe_workspace,
        _python_command("print('ok')"),
    )

    assert status == "failed"
    assert step.status == "failed"


def test_warning_is_included_when_command_runs(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path)

    _status, step = run_trial_test_command(workspace, _python_command("print('ok')"))

    assert TEST_COMMAND_WARNING in step.warnings


def test_should_run_test_command_behavior() -> None:
    assert should_run_test_command(None) is False
    assert should_run_test_command("") is False
    assert should_run_test_command("   ") is False
    assert should_run_test_command("python -c \"print('ok')\"") is True


def _make_workspace(tmp_path: Path) -> tuple[Path, TrialWorkspace]:
    original = tmp_path / "train.py"
    original.write_text("x = 1\n", encoding="utf-8")
    workspace, _steps = create_trial_workspace(str(original), base_dir=str(tmp_path))
    return original, workspace


def _python_command(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'
