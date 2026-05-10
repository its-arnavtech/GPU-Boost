"""Dataclass schemas for GPUBoost patch planning output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class PatchEdit:
    """One proposed source-code replacement."""

    filepath: str
    start_line: int
    end_line: int
    original_text: str
    replacement_text: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Return the edit as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class PatchSuggestion:
    """A reviewable suggested code change for one or more findings."""

    id: str
    title: str
    category: str
    severity: str
    confidence: str
    filepath: str
    finding_ids: list[str] = field(default_factory=list)
    summary: str = ""
    rationale: str = ""
    edits: list[PatchEdit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the suggestion as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class PatchPlan:
    """All patch suggestions for one file."""

    generated_at: str
    filepath: str
    status: str
    suggestions: list[PatchSuggestion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the patch plan as JSON-serializable data."""

        return asdict(self)
