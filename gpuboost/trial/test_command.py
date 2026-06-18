"""Run explicit user-provided test commands inside trial workspaces."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path

from gpuboost.schemas.trial import TrialStep, TrialWorkspace, create_timestamp
from gpuboost.trial.patch_apply import trial_file_is_inside_workspace
from gpuboost.trial.workspace import workspace_is_separate_from_original


TEST_COMMAND_WARNING = "User-provided test commands may execute arbitrary code."


def run_trial_test_command(
    workspace: TrialWorkspace,
    test_command: str | None,
    timeout_sec: int = 60,
) -> tuple[str, TrialStep]:
    """Run an explicit test command inside the trial workspace."""

    started_at = create_timestamp()
    start_time = time.perf_counter()

    if not should_run_test_command(test_command):
        return "skipped", _finish_step(
            status="skipped",
            message="No test command was provided; skipped trial test command.",
            started_at=started_at,
            start_time=start_time,
        )

    safety_error = _test_command_safety_error(workspace)
    if safety_error is not None:
        return "failed", _finish_step(
            status="failed",
            message="Refused to run test command in unsafe trial workspace.",
            started_at=started_at,
            start_time=start_time,
            error=safety_error,
        )

    workspace_path = Path(workspace.workspace_path).expanduser().resolve()
    command = test_command.strip() if test_command is not None else ""

    try:
        argv = _parse_test_command(command)
    except ValueError as exc:
        return "failed", _finish_step(
            status="failed",
            message="Trial test command could not be parsed.",
            started_at=started_at,
            start_time=start_time,
            error=str(exc),
            warnings=[TEST_COMMAND_WARNING],
        )

    try:
        # The command is tokenized with shlex and executed WITHOUT a shell
        # (shell=False). This eliminates shell injection: metacharacters such as
        # ``&&``, ``;``, ``|`` or ``$(...)`` are passed through as literal
        # arguments instead of being interpreted by a shell, so a single test
        # command can never chain or substitute additional commands.
        completed = subprocess.run(
            argv,
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return "failed", _finish_step(
            status="failed",
            message=f"Trial test command timed out after {timeout_sec} seconds.",
            started_at=started_at,
            start_time=start_time,
            stdout=exc.stdout if isinstance(exc.stdout, str) else None,
            stderr=exc.stderr if isinstance(exc.stderr, str) else None,
            error=f"Command timed out after {timeout_sec} seconds.",
            warnings=[TEST_COMMAND_WARNING],
        )
    except OSError as exc:
        return "failed", _finish_step(
            status="failed",
            message="Trial test command could not start.",
            started_at=started_at,
            start_time=start_time,
            error=str(exc),
            warnings=[TEST_COMMAND_WARNING],
        )

    if completed.returncode == 0:
        return "passed", _finish_step(
            status="passed",
            message="Trial test command passed.",
            started_at=started_at,
            start_time=start_time,
            stdout=completed.stdout or None,
            stderr=completed.stderr or None,
            exit_code=completed.returncode,
            warnings=[TEST_COMMAND_WARNING],
        )

    return "failed", _finish_step(
        status="failed",
        message=f"Trial test command failed with exit code {completed.returncode}.",
        started_at=started_at,
        start_time=start_time,
        stdout=completed.stdout or None,
        stderr=completed.stderr or None,
        exit_code=completed.returncode,
        error=f"Command exited with code {completed.returncode}.",
        warnings=[TEST_COMMAND_WARNING],
    )


def should_run_test_command(test_command: str | None) -> bool:
    """Return whether a test command was explicitly provided."""

    return bool(test_command and test_command.strip())


def _parse_test_command(command: str) -> list[str]:
    """Tokenize a user test command into argv for shell-free execution.

    The command is split with :func:`shlex.split` so it can be run with
    ``shell=False``. This is the security boundary that prevents shell
    injection: any shell metacharacters become ordinary literal arguments.
    """

    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"Could not parse test command: {exc}") from exc

    if not argv:
        raise ValueError("Test command did not contain an executable to run.")

    # Resolve the executable against PATH/PATHEXT so plain names like "pytest"
    # work cross-platform without relying on a shell. If it cannot be resolved
    # the original token is kept and subprocess will raise a clean OSError.
    resolved = shutil.which(argv[0])
    if resolved is not None:
        argv[0] = resolved

    return argv


def _test_command_safety_error(workspace: TrialWorkspace) -> str | None:
    if not workspace_is_separate_from_original(workspace):
        return "Trial workspace paths are not separate from the original file."
    if not trial_file_is_inside_workspace(workspace):
        return "Trial file is not inside the trial workspace directory."

    try:
        original_file = Path(workspace.original_file).expanduser().resolve()
        workspace_path = Path(workspace.workspace_path).expanduser().resolve()
        trial_file = Path(workspace.trial_file).expanduser().resolve()
    except OSError as exc:
        return f"Unable to resolve trial paths safely: {exc}"

    if trial_file == original_file:
        return "Trial file matches the original file path."
    if workspace_path == original_file.parent:
        return "Workspace path matches the original file parent directory."
    if not workspace_path.exists():
        return f"Trial workspace does not exist: {workspace_path}"
    if not workspace_path.is_dir():
        return f"Trial workspace path is not a directory: {workspace_path}"
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
    stdout: str | None = None,
    stderr: str | None = None,
    exit_code: int | None = None,
    error: str | None = None,
    warnings: list[str] | None = None,
) -> TrialStep:
    return TrialStep(
        name="run_test_command",
        status=status,
        started_at=started_at,
        ended_at=create_timestamp(),
        duration_sec=time.perf_counter() - start_time,
        message=message,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        warnings=list(warnings or []),
        error=error,
    )
