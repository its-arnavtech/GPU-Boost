"""Dataclass schemas for GPUBoost safe trial workspaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


TrialValue = str | int | float | bool | None


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TrialWorkspace:
    """A safe temporary workspace for testing copied source files."""

    original_file: str
    workspace_path: str
    trial_file: str
    cleanup_enabled: bool
    created_at: str
    metadata: dict[str, TrialValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the workspace as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class TrialStep:
    """One step in a safe patch-trial workflow."""

    name: str
    status: str
    started_at: str | None = None
    ended_at: str | None = None
    duration_sec: float | None = None
    message: str = ""
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the step as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class TrialResult:
    """Result envelope for a safe patch-trial workflow."""

    generated_at: str
    status: str
    workspace: TrialWorkspace | None = None
    steps: list[TrialStep] = field(default_factory=list)
    patch_applied: bool = False
    syntax_check_status: str | None = None
    test_command: str | None = None
    test_status: str | None = None
    original_file_unchanged: bool = True
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the result as JSON-serializable data."""

        return asdict(self)

    def has_failures(self) -> bool:
        """Return whether the trial result or any step failed."""

        return self.status in {"failed", "error"} or any(
            step.status == "failed" for step in self.steps
        )

    def step_by_name(self, name: str) -> TrialStep | None:
        """Return the first step with a matching name."""

        return next((step for step in self.steps if step.name == name), None)
