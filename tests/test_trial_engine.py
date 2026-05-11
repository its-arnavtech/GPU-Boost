"""Tests for Phase 7.6 safe trial engine."""

from __future__ import annotations

import sys
from pathlib import Path

from gpuboost.schemas.patch_plan import PatchEdit, PatchPlan, PatchSuggestion
from gpuboost.trial import engine
from gpuboost.trial.engine import run_patch_trial


def test_successful_trial_applies_patch_to_copy_only(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(
        str(original),
        _plan([_edit(1, 1, "value = 1\n", "value = 2\n")]),
        base_dir=str(tmp_path),
    )

    assert result.status == "passed"
    assert result.patch_applied is True
    assert original.read_text(encoding="utf-8") == "value = 1\n"
    assert result.workspace is not None
    assert not Path(result.workspace.workspace_path).exists()


def test_original_file_unchanged_after_successful_trial(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(
        str(original),
        _plan([_edit(1, 1, "value = 1\n", "value = 2\n")]),
        base_dir=str(tmp_path),
    )

    assert result.original_file_unchanged is True
    assert original.read_text(encoding="utf-8") == "value = 1\n"


def test_cleanup_enabled_true_removes_workspace(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(str(original), _plan([]), base_dir=str(tmp_path))

    assert result.workspace is not None
    assert not Path(result.workspace.workspace_path).exists()


def test_cleanup_enabled_false_keeps_workspace(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(
        str(original),
        _plan([]),
        cleanup_enabled=False,
        base_dir=str(tmp_path),
    )

    assert result.workspace is not None
    assert Path(result.workspace.workspace_path).exists()
    assert result.step_by_name("cleanup_workspace").status == "skipped"


def test_no_edits_patch_plan_returns_skipped_and_original_unchanged(
    tmp_path: Path,
) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(str(original), _plan([]), base_dir=str(tmp_path))

    assert result.status == "skipped"
    assert result.patch_applied is False
    assert result.original_file_unchanged is True


def test_invalid_patch_plan_returns_failed(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")
    plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="error",
        error="bad plan",
    )

    result = run_patch_trial(str(original), plan, base_dir=str(tmp_path))

    assert result.status == "failed"
    assert result.step_by_name("apply_patch").status == "failed"


def test_syntax_failure_returns_failed(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(
        str(original),
        _plan([_edit(1, 1, "value = 1\n", "if True\n")]),
        base_dir=str(tmp_path),
    )

    assert result.status == "failed"
    assert result.syntax_check_status == "failed"


def test_test_command_success_returns_passed(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(
        str(original),
        _plan([_edit(1, 1, "value = 1\n", "value = 2\n")]),
        test_command=_python_command("print('ok')"),
        base_dir=str(tmp_path),
    )

    assert result.status == "passed"
    assert result.test_status == "passed"


def test_test_command_failure_returns_failed(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(
        str(original),
        _plan([_edit(1, 1, "value = 1\n", "value = 2\n")]),
        test_command=_python_command("import sys; sys.exit(3)"),
        base_dir=str(tmp_path),
    )

    assert result.status == "failed"
    assert result.test_status == "failed"


def test_test_command_skipped_when_none(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(
        str(original),
        _plan([_edit(1, 1, "value = 1\n", "value = 2\n")]),
        test_command=None,
        base_dir=str(tmp_path),
    )

    assert result.test_status == "skipped"
    assert result.step_by_name("run_test_command").status == "skipped"


def test_original_file_mutation_detection_forces_error(monkeypatch, tmp_path) -> None:
    original = _write_source(tmp_path, "value = 1\n")
    monkeypatch.setattr(engine, "verify_original_unchanged", lambda *_args: False)

    result = run_patch_trial(str(original), _plan([]), base_dir=str(tmp_path))

    assert result.status == "error"
    assert result.original_file_unchanged is False


def test_steps_include_expected_workflow_steps(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(str(original), _plan([]), base_dir=str(tmp_path))

    assert [step.name for step in result.steps] == [
        "create_workspace",
        "copy_source",
        "apply_patch",
        "syntax_check",
        "run_test_command",
        "cleanup_workspace",
    ]


def test_warnings_aggregate(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    result = run_patch_trial(
        str(original),
        _plan(
            [
                _edit(1, 1, "value = 1\n", "value = 2\n"),
                _edit(99, 99, "", "missing\n"),
            ]
        ),
        base_dir=str(tmp_path),
    )

    assert "Skipped edit update: invalid line range." in result.warnings


def test_no_source_file_editing(tmp_path: Path) -> None:
    original = _write_source(tmp_path, "value = 1\n")

    run_patch_trial(
        str(original),
        _plan([_edit(1, 1, "value = 1\n", "value = 2\n")]),
        base_dir=str(tmp_path),
    )

    assert original.read_text(encoding="utf-8") == "value = 1\n"


def _write_source(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "train.py"
    path.write_text(source, encoding="utf-8")
    return path


def _edit(
    start_line: int,
    end_line: int,
    original_text: str,
    replacement_text: str,
) -> PatchEdit:
    return PatchEdit(
        filepath="train.py",
        start_line=start_line,
        end_line=end_line,
        original_text=original_text,
        replacement_text=replacement_text,
        description="update",
    )


def _plan(edits: list[PatchEdit]) -> PatchPlan:
    return PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        suggestions=[
            PatchSuggestion(
                id="suggestion",
                title="Suggestion",
                category="general",
                severity="info",
                confidence="medium",
                filepath="train.py",
                edits=edits,
            )
        ],
    )


def _python_command(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'
