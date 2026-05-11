"""Tests for Phase 7.1 trial workspace schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from gpuboost.schemas.trial import (
    TrialResult,
    TrialStep,
    TrialWorkspace,
    create_timestamp,
)


def test_trial_workspace_creation() -> None:
    workspace = _make_workspace()

    assert workspace.original_file == "train.py"
    assert workspace.workspace_path == ".gpuboost/trials/trial_001"
    assert workspace.trial_file == ".gpuboost/trials/trial_001/train.py"
    assert workspace.cleanup_enabled is True
    assert workspace.created_at == "2026-01-01T00:00:00+00:00"
    assert workspace.metadata == {"phase": "7.1", "safe": True}


def test_trial_step_creation_with_defaults() -> None:
    step = TrialStep(name="syntax_check", status="pending")

    assert step.name == "syntax_check"
    assert step.status == "pending"
    assert step.started_at is None
    assert step.ended_at is None
    assert step.duration_sec is None
    assert step.message == ""
    assert step.stdout is None
    assert step.stderr is None
    assert step.exit_code is None
    assert step.warnings == []
    assert step.error is None


def test_trial_result_creation() -> None:
    workspace = _make_workspace()
    step = _make_step()
    result = TrialResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="passed",
        workspace=workspace,
        steps=[step],
        patch_applied=True,
        syntax_check_status="passed",
        test_command="python -m pytest tests/test_train.py",
        test_status="passed",
        original_file_unchanged=True,
        warnings=["Trial workspace is schema-only."],
        error=None,
    )

    assert result.generated_at == "2026-01-01T00:00:02+00:00"
    assert result.status == "passed"
    assert result.workspace == workspace
    assert result.steps == [step]
    assert result.patch_applied is True
    assert result.syntax_check_status == "passed"
    assert result.test_command == "python -m pytest tests/test_train.py"
    assert result.test_status == "passed"
    assert result.original_file_unchanged is True
    assert result.warnings == ["Trial workspace is schema-only."]
    assert result.error is None


def test_to_dict_nesting_works() -> None:
    result = TrialResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="passed",
        workspace=_make_workspace(),
        steps=[_make_step()],
        patch_applied=True,
        syntax_check_status="passed",
        test_status="skipped",
    )

    data = result.to_dict()

    assert data["workspace"]["original_file"] == "train.py"
    assert data["workspace"]["metadata"]["phase"] == "7.1"
    assert data["steps"][0]["name"] == "create_workspace"
    assert data["steps"][0]["warnings"] == []
    assert data["patch_applied"] is True
    assert data["test_status"] == "skipped"
    assert data["warnings"] == []
    assert data["error"] is None


def test_json_serialization_works() -> None:
    result = TrialResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="passed",
        workspace=_make_workspace(),
        steps=[_make_step()],
    )

    serialized = json.dumps(result.to_dict())
    deserialized = json.loads(serialized)

    assert deserialized["workspace"]["trial_file"] == (
        ".gpuboost/trials/trial_001/train.py"
    )
    assert deserialized["steps"][0]["message"] == "Workspace created."
    assert deserialized["original_file_unchanged"] is True


def test_default_list_and_dict_fields_are_isolated_between_instances() -> None:
    first_workspace = TrialWorkspace(
        original_file="first.py",
        workspace_path=".gpuboost/trials/first",
        trial_file=".gpuboost/trials/first/first.py",
        cleanup_enabled=True,
        created_at="2026-01-01T00:00:00+00:00",
    )
    second_workspace = TrialWorkspace(
        original_file="second.py",
        workspace_path=".gpuboost/trials/second",
        trial_file=".gpuboost/trials/second/second.py",
        cleanup_enabled=True,
        created_at="2026-01-01T00:00:01+00:00",
    )
    first_step = TrialStep(name="syntax_check", status="pending")
    second_step = TrialStep(name="run_test_command", status="pending")
    first_result = TrialResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="partial",
    )
    second_result = TrialResult(
        generated_at="2026-01-01T00:00:03+00:00",
        status="skipped",
    )

    first_workspace.metadata["attempt"] = 1
    first_step.warnings.append("Syntax check was skipped.")
    first_result.steps.append(first_step)
    first_result.warnings.append("Trial was partial.")

    assert first_workspace.metadata == {"attempt": 1}
    assert second_workspace.metadata == {}
    assert first_step.warnings == ["Syntax check was skipped."]
    assert second_step.warnings == []
    assert first_result.steps == [first_step]
    assert second_result.steps == []
    assert first_result.warnings == ["Trial was partial."]
    assert second_result.warnings == []


def test_create_timestamp_returns_non_empty_utc_iso_string() -> None:
    timestamp = create_timestamp()
    parsed = datetime.fromisoformat(timestamp)

    assert timestamp
    assert parsed.tzinfo == timezone.utc


def test_has_failures_returns_false_for_passed() -> None:
    result = TrialResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="passed",
        steps=[TrialStep(name="syntax_check", status="passed")],
    )

    assert result.has_failures() is False


def test_has_failures_returns_true_for_failed_status() -> None:
    result = TrialResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="failed",
    )

    assert result.has_failures() is True


def test_has_failures_returns_true_when_a_step_failed() -> None:
    result = TrialResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="partial",
        steps=[
            TrialStep(name="create_workspace", status="passed"),
            TrialStep(name="syntax_check", status="failed"),
        ],
    )

    assert result.has_failures() is True


def test_step_by_name_finds_a_step() -> None:
    syntax_step = TrialStep(name="syntax_check", status="passed")
    result = TrialResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="passed",
        steps=[TrialStep(name="create_workspace", status="passed"), syntax_step],
    )

    assert result.step_by_name("syntax_check") == syntax_step


def test_step_by_name_returns_none_when_missing() -> None:
    result = TrialResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="passed",
        steps=[TrialStep(name="create_workspace", status="passed")],
    )

    assert result.step_by_name("run_test_command") is None


def test_trial_result_can_represent_skipped_syntax_and_test_command() -> None:
    result = TrialResult(
        generated_at="2026-01-01T00:00:02+00:00",
        status="skipped",
        workspace=None,
        steps=[
            TrialStep(
                name="syntax_check",
                status="skipped",
                message="Syntax check was not requested.",
            ),
            TrialStep(
                name="run_test_command",
                status="skipped",
                message="No test command was provided.",
            ),
        ],
        patch_applied=False,
        syntax_check_status="skipped",
        test_command=None,
        test_status="skipped",
        original_file_unchanged=True,
    )

    assert result.syntax_check_status == "skipped"
    assert result.test_command is None
    assert result.test_status == "skipped"
    assert result.step_by_name("syntax_check").status == "skipped"
    assert result.step_by_name("run_test_command").status == "skipped"


def _make_workspace() -> TrialWorkspace:
    return TrialWorkspace(
        original_file="train.py",
        workspace_path=".gpuboost/trials/trial_001",
        trial_file=".gpuboost/trials/trial_001/train.py",
        cleanup_enabled=True,
        created_at="2026-01-01T00:00:00+00:00",
        metadata={"phase": "7.1", "safe": True},
    )


def _make_step() -> TrialStep:
    return TrialStep(
        name="create_workspace",
        status="passed",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        duration_sec=1.0,
        message="Workspace created.",
        exit_code=0,
    )
