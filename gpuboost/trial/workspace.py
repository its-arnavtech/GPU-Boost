"""Temporary workspace helpers for safe patch trials."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from pathlib import Path

from gpuboost.schemas.trial import TrialStep, TrialWorkspace, create_timestamp


_HASH_CHUNK_SIZE = 1024 * 1024


def create_trial_workspace(
    original_file: str,
    cleanup_enabled: bool = True,
    base_dir: str | None = None,
) -> tuple[TrialWorkspace, list[TrialStep]]:
    """Create a temporary workspace and copy the original file into it."""

    source_path = Path(original_file).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Original file does not exist: {original_file}")
    if source_path.is_dir():
        raise ValueError(
            f"Original path must be a file, not a directory: {original_file}"
        )
    if not source_path.is_file():
        raise ValueError(f"Original path must be a regular file: {original_file}")

    original_path = source_path.resolve()
    steps: list[TrialStep] = []

    started_at = create_timestamp()
    start_time = time.perf_counter()
    workspace_dir = _make_workspace_dir(base_dir)
    ended_at = create_timestamp()
    steps.append(
        TrialStep(
            name="create_workspace",
            status="passed",
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=time.perf_counter() - start_time,
            message=f"Created trial workspace: {workspace_dir}",
        )
    )

    trial_file = workspace_dir / original_path.name
    started_at = create_timestamp()
    start_time = time.perf_counter()
    shutil.copy2(original_path, trial_file)
    ended_at = create_timestamp()
    steps.append(
        TrialStep(
            name="copy_source",
            status="passed",
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=time.perf_counter() - start_time,
            message=f"Copied source file to trial workspace: {trial_file}",
        )
    )

    workspace = TrialWorkspace(
        original_file=str(original_path),
        workspace_path=str(workspace_dir),
        trial_file=str(trial_file.resolve()),
        cleanup_enabled=cleanup_enabled,
        created_at=create_timestamp(),
        metadata={
            "original_filename": original_path.name,
            "original_size_bytes": original_path.stat().st_size,
            "original_sha256": hash_file(str(original_path)),
        },
    )

    return workspace, steps


def cleanup_trial_workspace(workspace: TrialWorkspace) -> TrialStep:
    """Clean up a trial workspace when cleanup is enabled."""

    started_at = create_timestamp()
    start_time = time.perf_counter()

    if not workspace.cleanup_enabled:
        return TrialStep(
            name="cleanup_workspace",
            status="skipped",
            started_at=started_at,
            ended_at=create_timestamp(),
            duration_sec=time.perf_counter() - start_time,
            message="Cleanup is disabled for this trial workspace.",
        )

    safety_error = _workspace_delete_safety_error(workspace)
    if safety_error is not None:
        return TrialStep(
            name="cleanup_workspace",
            status="failed",
            started_at=started_at,
            ended_at=create_timestamp(),
            duration_sec=time.perf_counter() - start_time,
            message="Refused to delete unsafe trial workspace path.",
            error=safety_error,
        )

    workspace_path = Path(workspace.workspace_path).expanduser().resolve(strict=False)
    if not workspace_path.exists():
        return TrialStep(
            name="cleanup_workspace",
            status="skipped",
            started_at=started_at,
            ended_at=create_timestamp(),
            duration_sec=time.perf_counter() - start_time,
            message=f"Trial workspace does not exist: {workspace_path}",
        )

    if not workspace_path.is_dir():
        return TrialStep(
            name="cleanup_workspace",
            status="failed",
            started_at=started_at,
            ended_at=create_timestamp(),
            duration_sec=time.perf_counter() - start_time,
            message=(
                "Refused to delete trial workspace path because it is not a "
                "directory."
            ),
            error=f"Workspace path is not a directory: {workspace_path}",
        )

    shutil.rmtree(workspace_path)
    return TrialStep(
        name="cleanup_workspace",
        status="passed",
        started_at=started_at,
        ended_at=create_timestamp(),
        duration_sec=time.perf_counter() - start_time,
        message=f"Deleted trial workspace: {workspace_path}",
    )


def verify_original_unchanged(original_file: str, before_hash: str) -> bool:
    """Return whether the original file still matches a previous SHA256 hash."""

    return hash_file(original_file) == before_hash


def hash_file(filepath: str) -> str:
    """Return the SHA256 hex digest for a file."""

    path = Path(filepath).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {filepath}")
    if not path.is_file():
        raise ValueError(f"Path must be a file: {filepath}")

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_is_separate_from_original(workspace: TrialWorkspace) -> bool:
    """Return whether trial paths are absolute and distinct from the original."""

    try:
        original_raw = Path(workspace.original_file).expanduser()
        workspace_raw = Path(workspace.workspace_path).expanduser()
        trial_raw = Path(workspace.trial_file).expanduser()
        original_path = original_raw.resolve(strict=False)
        workspace_path = workspace_raw.resolve(strict=False)
        trial_path = trial_raw.resolve(strict=False)
    except OSError:
        return False

    return (
        original_raw.is_absolute()
        and workspace_raw.is_absolute()
        and trial_raw.is_absolute()
        and workspace_path != original_path.parent
        and workspace_path != original_path
        and workspace_path != trial_path
        and trial_path != original_path
    )


def build_step(
    name: str,
    status: str,
    message: str,
    started_at: str | None = None,
    ended_at: str | None = None,
    error: str | None = None,
    warnings: list[str] | None = None,
) -> TrialStep:
    """Build a trial step with isolated warning lists."""

    return TrialStep(
        name=name,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        message=message,
        warnings=list(warnings or []),
        error=error,
    )


def _make_workspace_dir(base_dir: str | None) -> Path:
    if base_dir is None:
        return Path(tempfile.mkdtemp(prefix="gpuboost-trial-")).resolve()

    parent_dir = Path(base_dir).expanduser().resolve()
    parent_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix="gpuboost-trial-", dir=str(parent_dir))
    ).resolve()


def _workspace_delete_safety_error(workspace: TrialWorkspace) -> str | None:
    raw_workspace_path = workspace.workspace_path.strip()
    if not raw_workspace_path:
        return "Workspace path is empty."

    try:
        workspace_path = Path(raw_workspace_path).expanduser().resolve(strict=False)
        original_path = Path(workspace.original_file).expanduser().resolve(strict=False)
    except OSError as exc:
        return f"Unable to resolve workspace path safely: {exc}"

    if workspace_path == Path(workspace_path.anchor):
        return f"Workspace path is root-like: {workspace_path}"
    if workspace_path == original_path.parent:
        return "Workspace path matches the original file parent directory."
    if workspace_path == original_path:
        return "Workspace path matches the original file path."

    return None
