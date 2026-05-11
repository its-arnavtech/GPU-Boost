"""Apply reviewable patch plans inside safe trial workspaces."""

from __future__ import annotations

import time
from pathlib import Path

from gpuboost.patching.diff import apply_patch_edits_to_text
from gpuboost.schemas.patch_plan import PatchEdit, PatchPlan
from gpuboost.schemas.trial import TrialStep, TrialWorkspace, create_timestamp
from gpuboost.trial.workspace import workspace_is_separate_from_original


def apply_patch_plan_to_trial_file(
    workspace: TrialWorkspace,
    patch_plan: PatchPlan,
) -> tuple[bool, TrialStep]:
    """Apply a patch plan to the trial file only."""

    started_at = create_timestamp()
    start_time = time.perf_counter()

    if patch_plan.status != "ok":
        return False, _finish_step(
            status="failed",
            message="Patch plan status is not ok; no trial patch was applied.",
            started_at=started_at,
            start_time=start_time,
            error=patch_plan.error or f"Patch plan status: {patch_plan.status}",
        )

    edits = collect_patch_edits(patch_plan)
    if not edits:
        return False, _finish_step(
            status="skipped",
            message="Patch plan contains no edits; trial file was unchanged.",
            started_at=started_at,
            start_time=start_time,
        )

    safety_error = _trial_write_safety_error(workspace)
    if safety_error is not None:
        return False, _finish_step(
            status="failed",
            message="Refused to apply patch plan to unsafe trial workspace.",
            started_at=started_at,
            start_time=start_time,
            error=safety_error,
        )

    try:
        trial_file = Path(workspace.trial_file).expanduser().resolve()
        source_text = trial_file.read_text(encoding="utf-8")
        modified_text, warnings = apply_patch_edits_to_text(source_text, edits)
        if modified_text == source_text:
            return False, _finish_step(
                status="failed",
                message="No patch edits were applied; trial file was unchanged.",
                started_at=started_at,
                start_time=start_time,
                warnings=warnings,
                error="Patch edits did not modify the trial file.",
            )

        trial_file.write_text(modified_text, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return False, _finish_step(
            status="failed",
            message="Failed to apply patch plan to trial file.",
            started_at=started_at,
            start_time=start_time,
            error=str(exc),
        )

    return True, _finish_step(
        status="passed",
        message=f"Applied patch plan to trial file: {trial_file}",
        started_at=started_at,
        start_time=start_time,
        warnings=warnings,
    )


def trial_file_is_inside_workspace(workspace: TrialWorkspace) -> bool:
    """Return whether the trial file resolves inside the workspace directory."""

    try:
        workspace_path = Path(workspace.workspace_path).expanduser().resolve()
        trial_file = Path(workspace.trial_file).expanduser().resolve()
        trial_file.relative_to(workspace_path)
    except (OSError, ValueError):
        return False

    return trial_file != workspace_path


def collect_patch_edits(patch_plan: PatchPlan) -> list[PatchEdit]:
    """Collect patch edits in suggestion order, preserving edit order."""

    return [
        edit
        for suggestion in patch_plan.suggestions
        for edit in suggestion.edits
    ]


def _trial_write_safety_error(workspace: TrialWorkspace) -> str | None:
    if not workspace_is_separate_from_original(workspace):
        return "Trial workspace paths are not separate from the original file."
    if not trial_file_is_inside_workspace(workspace):
        return "Trial file is not inside the trial workspace directory."

    try:
        original_file = Path(workspace.original_file).expanduser().resolve()
        trial_file = Path(workspace.trial_file).expanduser().resolve()
    except OSError as exc:
        return f"Unable to resolve trial paths safely: {exc}"

    if trial_file == original_file:
        return "Trial file matches the original file path."
    if not trial_file.exists():
        return f"Trial file does not exist: {trial_file}"
    if not trial_file.is_file():
        return f"Trial path is not a file: {trial_file}"

    return None


def _finish_step(
    *,
    status: str,
    message: str,
    started_at: str,
    start_time: float,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> TrialStep:
    return TrialStep(
        name="apply_patch",
        status=status,
        started_at=started_at,
        ended_at=create_timestamp(),
        duration_sec=time.perf_counter() - start_time,
        message=message,
        warnings=list(warnings or []),
        error=error,
    )
