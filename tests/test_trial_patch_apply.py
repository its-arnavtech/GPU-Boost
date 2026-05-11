"""Tests for Phase 7.3 patch application inside trial workspaces."""

from __future__ import annotations

from pathlib import Path

from gpuboost.schemas.patch_plan import PatchEdit, PatchPlan, PatchSuggestion
from gpuboost.schemas.trial import TrialWorkspace
from gpuboost.trial.patch_apply import (
    apply_patch_plan_to_trial_file,
    collect_patch_edits,
    trial_file_is_inside_workspace,
)
from gpuboost.trial.workspace import create_trial_workspace, hash_file


def test_applies_patch_to_trial_file(tmp_path: Path) -> None:
    original, workspace = _make_workspace(tmp_path, "a = 1\n")
    before_hash = hash_file(str(original))
    plan = _plan([_edit(1, 1, "a = 1\n", "a = 2\n", "update a")])

    applied, step = apply_patch_plan_to_trial_file(workspace, plan)

    assert applied is True
    assert step.name == "apply_patch"
    assert step.status == "passed"
    assert step.warnings == []
    assert original.read_text(encoding="utf-8") == "a = 1\n"
    assert hash_file(str(original)) == before_hash
    assert Path(workspace.trial_file).read_text(encoding="utf-8") == "a = 2\n"


def test_no_edits_returns_skipped_and_file_unchanged(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path, "a = 1\n")
    before_text = Path(workspace.trial_file).read_text(encoding="utf-8")
    plan = _plan([])

    applied, step = apply_patch_plan_to_trial_file(workspace, plan)

    assert applied is False
    assert step.status == "skipped"
    assert Path(workspace.trial_file).read_text(encoding="utf-8") == before_text


def test_patch_plan_status_error_returns_failed_and_file_unchanged(
    tmp_path: Path,
) -> None:
    _original, workspace = _make_workspace(tmp_path, "a = 1\n")
    before_text = Path(workspace.trial_file).read_text(encoding="utf-8")
    plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="error",
        error="Cannot build patch plan.",
    )

    applied, step = apply_patch_plan_to_trial_file(workspace, plan)

    assert applied is False
    assert step.status == "failed"
    assert step.error == "Cannot build patch plan."
    assert Path(workspace.trial_file).read_text(encoding="utf-8") == before_text


def test_original_text_mismatch_returns_failed_and_original_unchanged(
    tmp_path: Path,
) -> None:
    original, workspace = _make_workspace(tmp_path, "a = 1\n")
    before_hash = hash_file(str(original))
    before_trial_text = Path(workspace.trial_file).read_text(encoding="utf-8")
    plan = _plan([_edit(1, 1, "a = 100\n", "a = 2\n", "update a")])

    applied, step = apply_patch_plan_to_trial_file(workspace, plan)

    assert applied is False
    assert step.status == "failed"
    assert step.warnings == [
        "Skipped edit update a: original text did not match source."
    ]
    assert hash_file(str(original)) == before_hash
    assert Path(workspace.trial_file).read_text(encoding="utf-8") == before_trial_text


def test_warnings_from_patch_apply_appear_on_trial_step(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path, "a = 1\nb = 2\n")
    plan = _plan(
        [
            _edit(1, 1, "a = 1\n", "a = 10\n", "update a"),
            _edit(99, 99, "", "z = 0\n", "invalid edit"),
        ]
    )

    applied, step = apply_patch_plan_to_trial_file(workspace, plan)

    assert applied is True
    assert step.status == "passed"
    assert step.warnings == ["Skipped edit invalid edit: invalid line range."]
    assert Path(workspace.trial_file).read_text(encoding="utf-8") == (
        "a = 10\nb = 2\n"
    )


def test_unsafe_workspace_where_trial_file_is_original_refuses_to_write(
    tmp_path: Path,
) -> None:
    original = tmp_path / "train.py"
    original.write_text("a = 1\n", encoding="utf-8")
    workspace = TrialWorkspace(
        original_file=str(original.resolve()),
        workspace_path=str(tmp_path.resolve()),
        trial_file=str(original.resolve()),
        cleanup_enabled=True,
        created_at="2026-01-01T00:00:00+00:00",
    )
    plan = _plan([_edit(1, 1, "a = 1\n", "a = 2\n", "update a")])

    applied, step = apply_patch_plan_to_trial_file(workspace, plan)

    assert applied is False
    assert step.status == "failed"
    assert original.read_text(encoding="utf-8") == "a = 1\n"


def test_trial_file_outside_workspace_refuses_to_write(tmp_path: Path) -> None:
    original, workspace = _make_workspace(tmp_path, "a = 1\n")
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("a = 1\n", encoding="utf-8")
    unsafe_workspace = TrialWorkspace(
        original_file=workspace.original_file,
        workspace_path=workspace.workspace_path,
        trial_file=str(outside_file.resolve()),
        cleanup_enabled=True,
        created_at=workspace.created_at,
    )
    plan = _plan([_edit(1, 1, "a = 1\n", "a = 2\n", "update a")])

    applied, step = apply_patch_plan_to_trial_file(unsafe_workspace, plan)

    assert applied is False
    assert step.status == "failed"
    assert outside_file.read_text(encoding="utf-8") == "a = 1\n"
    assert original.read_text(encoding="utf-8") == "a = 1\n"


def test_missing_trial_file_returns_failed_step(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path, "a = 1\n")
    Path(workspace.trial_file).unlink()
    plan = _plan([_edit(1, 1, "a = 1\n", "a = 2\n", "update a")])

    applied, step = apply_patch_plan_to_trial_file(workspace, plan)

    assert applied is False
    assert step.status == "failed"
    assert "does not exist" in str(step.error)


def test_collect_patch_edits_preserves_order() -> None:
    first = _edit(1, 1, "a\n", "A\n", "first")
    second = _edit(2, 2, "b\n", "B\n", "second")
    third = _edit(3, 3, "c\n", "C\n", "third")
    plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        suggestions=[
            _suggestion("first", [first, second]),
            _suggestion("second", [third]),
        ],
    )

    assert collect_patch_edits(plan) == [first, second, third]


def test_trial_file_is_inside_workspace_true_and_false(tmp_path: Path) -> None:
    _original, workspace = _make_workspace(tmp_path, "a = 1\n")
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("a = 1\n", encoding="utf-8")
    outside_workspace = TrialWorkspace(
        original_file=workspace.original_file,
        workspace_path=workspace.workspace_path,
        trial_file=str(outside_file.resolve()),
        cleanup_enabled=True,
        created_at=workspace.created_at,
    )

    assert trial_file_is_inside_workspace(workspace) is True
    assert trial_file_is_inside_workspace(outside_workspace) is False


def _make_workspace(
    tmp_path: Path,
    source_text: str,
) -> tuple[Path, TrialWorkspace]:
    original = tmp_path / "train.py"
    original.write_text(source_text, encoding="utf-8")
    workspace, _steps = create_trial_workspace(str(original), base_dir=str(tmp_path))
    return original, workspace


def _edit(
    start_line: int,
    end_line: int,
    original_text: str,
    replacement_text: str,
    description: str,
) -> PatchEdit:
    return PatchEdit(
        filepath="train.py",
        start_line=start_line,
        end_line=end_line,
        original_text=original_text,
        replacement_text=replacement_text,
        description=description,
    )


def _suggestion(suggestion_id: str, edits: list[PatchEdit]) -> PatchSuggestion:
    return PatchSuggestion(
        id=suggestion_id,
        title="Suggestion",
        category="general",
        severity="info",
        confidence="medium",
        filepath="train.py",
        edits=edits,
    )


def _plan(edits: list[PatchEdit]) -> PatchPlan:
    return PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        suggestions=[_suggestion("suggestion", edits)],
    )
