"""Dataclass schemas for GPUBoost Phase 4 code analysis output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class CodeFinding:
    """A single static code analysis finding."""

    id: str
    title: str
    category: str
    severity: str
    confidence: str
    filepath: str
    line: int | None
    column: int | None
    end_line: int | None
    end_column: int | None
    summary: str
    rationale: str
    suggested_action: str
    code_snippet: str | None
    related_recommendation_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class CodeAnalysisResult:
    """Complete Phase 4 code analysis result for one file."""

    generated_at: str
    filepath: str
    status: str
    findings: list[CodeFinding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the code analysis result as JSON-serializable data."""

        return asdict(self)
