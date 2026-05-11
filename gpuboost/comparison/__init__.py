"""Helpers for comparing GPUBoost benchmark outputs."""

from gpuboost.comparison.engine import (
    calculate_metric_delta,
    compare_benchmarks,
    group_comparable_metrics,
    overall_verdict,
    section_verdict,
)
from gpuboost.comparison.metrics import (
    DEFAULT_COMPARISON_METRICS,
    HIGHER_IS_BETTER_METRICS,
    extract_named_metrics,
    get_metric_unit,
    get_metric_value,
    has_metric,
    iter_benchmark_metrics,
    metric_higher_is_better,
)

__all__ = [
    "DEFAULT_COMPARISON_METRICS",
    "HIGHER_IS_BETTER_METRICS",
    "calculate_metric_delta",
    "compare_benchmarks",
    "extract_named_metrics",
    "get_metric_unit",
    "get_metric_value",
    "group_comparable_metrics",
    "has_metric",
    "iter_benchmark_metrics",
    "metric_higher_is_better",
    "overall_verdict",
    "section_verdict",
]
