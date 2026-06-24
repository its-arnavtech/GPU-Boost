"""Helpers for locating an optional GPUBoost source repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_REPOSITORY_MARKERS = (
    Path("pyproject.toml"),
    Path("gpuboost/__init__.py"),
)


@dataclass(frozen=True)
class RepositoryContext:
    """Describes whether a GPUBoost source repository is available."""

    root: Path | None
    status: str
    message: str

    @property
    def applicable(self) -> bool:
        """Return whether repository-only checks should run."""

        return self.root is not None


def resolve_repository_context(
    repo_root: str | Path | None = None,
    *,
    start: Path | None = None,
) -> RepositoryContext:
    """Return the source-repository context for repo-root-aware checks."""

    if repo_root is not None:
        requested = Path(repo_root).expanduser()
        if not requested.exists():
            return RepositoryContext(
                root=None,
                status="invalid",
                message=f"Requested repo root does not exist: {requested}",
            )
        resolved = requested.resolve()
        if not resolved.is_dir():
            return RepositoryContext(
                root=None,
                status="invalid",
                message=f"Requested repo root is not a directory: {resolved}",
            )
        if _is_repository_root(resolved):
            return RepositoryContext(
                root=resolved,
                status="explicit",
                message=f"Using explicit GPUBoost source repository: {resolved}",
            )
        return RepositoryContext(
            root=None,
            status="invalid",
            message=(
                "Requested repo root is not a GPUBoost source repository: "
                f"{resolved}"
            ),
        )

    detected = find_repository_root(start=start)
    if detected is not None:
        return RepositoryContext(
            root=detected,
            status="detected",
            message=f"Detected GPUBoost source repository: {detected}",
        )
    return RepositoryContext(
        root=None,
        status="not_found",
        message=(
            "No GPUBoost source repository detected from the current "
            "working directory."
        ),
    )


def find_repository_root(start: Path | None = None) -> Path | None:
    """Search start and its parents for the GPUBoost source repository root."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if _is_repository_root(candidate):
            return candidate
    return None


def _is_repository_root(path: Path) -> bool:
    return all((path / marker).exists() for marker in _REPOSITORY_MARKERS)
