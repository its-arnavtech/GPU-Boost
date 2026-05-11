"""Dataclass schemas for GPUBoost Phase 8 benchmark comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


ComparisonMetricValue = float | int | bool | str | None


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class BenchmarkMetricDelta:
    """A before/after delta for one benchmark metric."""

    name: str
    unit: str | None
    before: ComparisonMetricValue
    after: ComparisonMetricValue
    absolute_delta: float | None
    percent_delta: float | None
    direction: str
    higher_is_better: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Return the metric delta as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class ComparisonSection:
    """A group of related benchmark metric deltas."""

    title: str
    metrics: list[BenchmarkMetricDelta] = field(default_factory=list)
    verdict: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the section as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class ComparisonResult:
    """Result envelope for a before/after benchmark comparison."""

    generated_at: str
    status: str
    baseline_label: str
    optimized_label: str
    sections: list[ComparisonSection] = field(default_factory=list)
    overall_verdict: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison result as JSON-serializable data."""

        return asdict(self)

    def has_regressions(self) -> bool:
        """Return whether any section or metric reports a regression."""

        return any(
            section.verdict == "regressed"
            or any(metric.direction == "regressed" for metric in section.metrics)
            for section in self.sections
        )

    def has_improvements(self) -> bool:
        """Return whether any section or metric reports an improvement."""

        return any(
            section.verdict == "improved"
            or any(metric.direction == "improved" for metric in section.metrics)
            for section in self.sections
        )
