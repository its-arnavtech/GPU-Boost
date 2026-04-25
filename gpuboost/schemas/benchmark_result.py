"""Dataclass schemas for GPUBoost Phase 2 benchmark output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


MetricValue = float | int | str | bool | None


@dataclass(slots=True)
class BenchmarkMetric:
    """A single named benchmark measurement."""

    name: str
    value: MetricValue
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the metric as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class BenchmarkResult:
    """Result for one benchmark."""

    name: str
    status: str
    started_at: str
    ended_at: str
    duration_sec: float
    metrics: list[BenchmarkMetric] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the benchmark result as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class BenchmarkSuiteResult:
    """Result for a Phase 2 benchmark suite run."""

    generated_at: str
    gpu_name: str | None
    cuda_available: bool
    device_index: int | None
    results: list[BenchmarkResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the suite result as JSON-serializable data."""

        return asdict(self)

