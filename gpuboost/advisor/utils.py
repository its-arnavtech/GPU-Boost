"""Utility helpers for GPUBoost advisor inputs and display values."""

from __future__ import annotations

from typing import Any

from gpuboost.schemas.benchmark_result import BenchmarkResult, BenchmarkSuiteResult


_MISSING = object()


def get_result_by_name(
    suite: BenchmarkSuiteResult,
    name: str,
) -> BenchmarkResult | None:
    """Return a benchmark result by exact name, if present."""

    for result in suite.results:
        if result.name == name:
            return result
    return None


def get_metric(
    result: BenchmarkResult | None,
    metric_name: str,
    default: Any = None,
) -> Any:
    """Return a metric value by name, or a default when unavailable."""

    if result is None:
        return default

    for metric in result.metrics:
        if metric.name == metric_name:
            return metric.value

    return default


def metric_exists(result: BenchmarkResult | None, metric_name: str) -> bool:
    """Return whether a metric name is present, regardless of its value."""

    return get_metric(result, metric_name, default=_MISSING) is not _MISSING


def speedup_to_percent(speedup: float) -> float:
    """Convert a speedup ratio into percent improvement."""

    return (speedup - 1.0) * 100


def format_speedup(speedup: float | None) -> str:
    """Format a speedup ratio for display."""

    if speedup is None:
        return "unknown"
    return f"{speedup:.2f}x"
