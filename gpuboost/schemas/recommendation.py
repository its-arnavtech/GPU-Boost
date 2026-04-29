"""Dataclass schemas for GPUBoost Phase 3 advisor output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Recommendation:
    """A single advisor recommendation."""

    id: str
    title: str
    category: str
    priority: int
    impact: str
    confidence: str
    effort: str
    estimated_speedup: float | None
    summary: str
    rationale: str
    suggested_action: str
    code_snippet: str | None
    related_metrics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the recommendation as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class AdvisorResult:
    """Complete Phase 3 advisor result."""

    generated_at: str
    recommendations: list[Recommendation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the advisor result as JSON-serializable data."""

        return asdict(self)
