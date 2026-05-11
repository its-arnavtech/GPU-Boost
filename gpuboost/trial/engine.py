"""End-to-end safe trial workspace engine."""

from __future__ import annotations

from gpuboost.schemas.patch_plan import PatchPlan
from gpuboost.schemas.trial import TrialResult, TrialStep, TrialWorkspace
from gpuboost.schemas.trial import create_timestamp
from gpuboost.trial.patch_apply import apply_patch_plan_to_trial_file
from gpuboost.trial.syntax_check import run_python_syntax_check
from gpuboost.trial.test_command import run_trial_test_command
from gpuboost.trial.workspace import (
    cleanup_trial_workspace,
    create_trial_workspace,
    hash_file,
    verify_original_unchanged,
)


def run_patch_trial(
    original_file: str,
    patch_plan: PatchPlan,
    test_command: str | None = None,
    cleanup_enabled: bool = True,
    force_python_syntax: bool = False,
    base_dir: str | None = None,
    test_timeout_sec: int = 60,
) -> TrialResult:
    """Run a safe patch trial against a copied file."""

    steps: list[TrialStep] = []
    workspace: TrialWorkspace | None = None
    patch_applied = False
    syntax_check_status: str | None = None
    test_status: str | None = None
    original_file_unchanged = True
    error: str | None = None

    try:
        before_hash = hash_file(original_file)
        workspace, workspace_steps = create_trial_workspace(
            original_file,
            cleanup_enabled=cleanup_enabled,
            base_dir=base_dir,
        )
        steps.extend(workspace_steps)

        patch_applied, patch_step = apply_patch_plan_to_trial_file(
            workspace,
            patch_plan,
        )
        steps.append(patch_step)

        syntax_check_status, syntax_step = run_python_syntax_check(
            workspace,
            force_python=force_python_syntax,
        )
        steps.append(syntax_step)

        test_status, test_step = run_trial_test_command(
            workspace,
            test_command,
            timeout_sec=test_timeout_sec,
        )
        steps.append(test_step)

        original_file_unchanged = verify_original_unchanged(original_file, before_hash)
        status = _compute_trial_status(
            patch_applied=patch_applied,
            patch_step=patch_step,
            syntax_check_status=syntax_check_status,
            test_status=test_status,
            original_file_unchanged=original_file_unchanged,
        )
        error = _first_error(steps)
    except Exception as exc:  # noqa: BLE001 - trial engine converts failures
        status = "error"
        error = str(exc)
    finally:
        if workspace is not None:
            cleanup_step = cleanup_trial_workspace(workspace)
            if cleanup_step.status == "failed":
                cleanup_message = cleanup_step.error or cleanup_step.message
                cleanup_step.warnings.append(cleanup_message)
            steps.append(cleanup_step)

    warnings = _aggregate_warnings(steps)
    if not original_file_unchanged:
        status = "error"
        error = "Original file changed during trial."

    return TrialResult(
        generated_at=create_timestamp(),
        status=status,
        workspace=workspace,
        steps=steps,
        patch_applied=patch_applied,
        syntax_check_status=syntax_check_status,
        test_command=test_command if test_command and test_command.strip() else None,
        test_status=test_status,
        original_file_unchanged=original_file_unchanged,
        warnings=warnings,
        error=error if status in {"failed", "error"} else None,
    )


def _compute_trial_status(
    *,
    patch_applied: bool,
    patch_step: TrialStep,
    syntax_check_status: str,
    test_status: str,
    original_file_unchanged: bool,
) -> str:
    if not original_file_unchanged:
        return "error"
    if patch_step.status == "failed":
        return "failed"
    if syntax_check_status == "failed":
        return "failed"
    if test_status == "failed":
        return "failed"
    if patch_applied:
        return "passed"
    return "skipped"


def _aggregate_warnings(steps: list[TrialStep]) -> list[str]:
    warnings: list[str] = []
    for step in steps:
        warnings.extend(step.warnings)
    return warnings


def _first_error(steps: list[TrialStep]) -> str | None:
    for step in steps:
        if step.status == "failed" and step.error:
            return step.error
    return None
