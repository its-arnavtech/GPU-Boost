"""Dataclass schemas for GPUBoost Phase 9 local run history."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


HistoryValue = str | int | float | bool | None


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class HistoryRunRecord:
    """Schema-only summary of one local GPUBoost agent run."""

    run_id: str
    created_at: str
    status: str
    command: str
    schema_version: str
    goal_kind: str
    goal_description: str
    script_path: str | None = None
    script_sha256: str | None = None
    gpu_name: str | None = None
    cuda_available: bool | None = None
    benchmark_summary: dict[str, HistoryValue] = field(default_factory=dict)
    advisor_summary: dict[str, HistoryValue] = field(default_factory=dict)
    code_summary: dict[str, HistoryValue] = field(default_factory=dict)
    patch_summary: dict[str, HistoryValue] = field(default_factory=dict)
    trial_summary: dict[str, HistoryValue] = field(default_factory=dict)
    comparison_summary: dict[str, HistoryValue] = field(default_factory=dict)
    action_statuses: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, HistoryValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the run record as JSON-serializable data."""

        return asdict(self)

    def has_error(self) -> bool:
        """Return whether this record represents a failed run."""

        return self.status == "error" or self.error is not None

    def has_trial(self) -> bool:
        """Return whether this record contains trial summary data."""

        return bool(self.trial_summary)

    def has_comparison(self) -> bool:
        """Return whether this record contains comparison summary data."""

        return bool(self.comparison_summary)


@dataclass(slots=True)
class HistorySummary:
    """Schema-only summary of local GPUBoost run history."""

    generated_at: str
    total_runs: int
    runs: list[HistoryRunRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the history summary as JSON-serializable data."""

        return asdict(self)

    def latest(self) -> HistoryRunRecord | None:
        """Return the newest run if this summary contains any runs."""

        if not self.runs:
            return None
        return self.runs[0]


@dataclass(slots=True)
class HistoryCompareResult:
    """Schema-only result for comparing two local history records."""

    generated_at: str
    status: str
    left_run_id: str
    right_run_id: str
    summary: str
    changed_fields: dict[str, HistoryValue] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the history comparison result as JSON-serializable data."""

        return asdict(self)
