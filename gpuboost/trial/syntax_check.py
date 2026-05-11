"""Python syntax checks for safe trial workspace files."""

from __future__ import annotations

import py_compile
import tempfile
import time
from pathlib import Path

from gpuboost.schemas.trial import TrialStep, TrialWorkspace, create_timestamp
from gpuboost.trial.patch_apply import trial_file_is_inside_workspace
from gpuboost.trial.workspace import workspace_is_separate_from_original


def run_python_syntax_check(
    workspace: TrialWorkspace,
    force_python: bool = False,
) -> tuple[str, TrialStep]:
    """Validate Python syntax for a trial file without executing it."""

    started_at = create_timestamp()
    start_time = time.perf_counter()

    safety_error = _syntax_check_safety_error(workspace)
    if safety_error is not None:
        return "failed", _finish_step(
            status="failed",
            message="Refused to syntax check unsafe trial workspace.",
            started_at=started_at,
            start_time=start_time,
            error=safety_error,
        )

    trial_file = Path(workspace.trial_file).expanduser().resolve()
    if not is_python_file(str(trial_file)) and not force_python:
        return "skipped", _finish_step(
            status="skipped",
            message=f"Skipped syntax check for non-Python file: {trial_file}",
            started_at=started_at,
            start_time=start_time,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="gpuboost-syntax-") as temp_dir:
            pyc_path = Path(temp_dir) / "trial_syntax.pyc"
            py_compile.compile(
                str(trial_file),
                cfile=str(pyc_path),
                doraise=True,
            )
    except py_compile.PyCompileError as exc:
        return "failed", _finish_step(
            status="failed",
            message=f"Python syntax check failed for trial file: {trial_file}",
            started_at=started_at,
            start_time=start_time,
            error=str(exc),
        )
    except OSError as exc:
        return "failed", _finish_step(
            status="failed",
            message="Unable to read trial file for syntax check.",
            started_at=started_at,
            start_time=start_time,
            error=str(exc),
        )

    return "passed", _finish_step(
        status="passed",
        message=f"Python syntax check passed for trial file: {trial_file}",
        started_at=started_at,
        start_time=start_time,
    )


def is_python_file(filepath: str) -> bool:
    """Return whether a filepath has a Python file suffix."""

    return Path(filepath).suffix.lower() == ".py"


def _syntax_check_safety_error(workspace: TrialWorkspace) -> str | None:
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
    error: str | None = None,
    warnings: list[str] | None = None,
) -> TrialStep:
    return TrialStep(
        name="syntax_check",
        status=status,
        started_at=started_at,
        ended_at=create_timestamp(),
        duration_sec=time.perf_counter() - start_time,
        message=message,
        warnings=list(warnings or []),
        error=error,
    )
