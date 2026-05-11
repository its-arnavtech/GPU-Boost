"""Metric extraction helpers for GPUBoost benchmark comparisons."""

from __future__ import annotations


MetricValue = float | int | bool | str | None

DEFAULT_COMPARISON_METRICS = [
    "best_fp32_tflops",
    "best_fp16_tflops",
    "fp16_speedup_ratio",
    "fp32_samples_per_sec",
    "amp_samples_per_sec",
    "amp_speedup_ratio",
    "best_images_per_sec",
    "speedup_vs_batch_1",
    "best_batch_size",
    "max_successful_batch_size",
]

HIGHER_IS_BETTER_METRICS = {
    "best_fp32_tflops": True,
    "best_fp16_tflops": True,
    "fp16_speedup_ratio": True,
    "fp32_samples_per_sec": True,
    "amp_samples_per_sec": True,
    "amp_speedup_ratio": True,
    "best_images_per_sec": True,
    "speedup_vs_batch_1": True,
    "best_batch_size": True,
    "max_successful_batch_size": True,
    "median_fp32_step_ms": False,
    "median_amp_step_ms": False,
    "batch_1_median_ms": False,
    "batch_8_median_ms": False,
    "batch_16_median_ms": False,
    "batch_32_median_ms": False,
    "batch_64_median_ms": False,
    "batch_128_median_ms": False,
}


def iter_benchmark_metrics(benchmark: dict) -> list[dict]:
    """Return flattened metric records from a benchmark suite dictionary."""

    if not isinstance(benchmark, dict):
        return []

    results = benchmark.get("results")
    if not isinstance(results, list):
        return []

    records = []
    for result in results:
        if not isinstance(result, dict):
            continue

        benchmark_name = result.get("name")
        metrics = result.get("metrics")
        if not isinstance(metrics, list):
            continue

        for metric in metrics:
            if not isinstance(metric, dict):
                continue

            records.append(
                {
                    "benchmark_name": benchmark_name,
                    "metric_name": metric.get("name"),
                    "value": metric.get("value"),
                    "unit": metric.get("unit"),
                }
            )

    return records


def get_metric_value(
    benchmark: dict,
    metric_name: str,
    benchmark_name: str | None = None,
) -> MetricValue:
    """Return the first matching metric value, or None if it is missing."""

    record = _find_metric_record(benchmark, metric_name, benchmark_name)
    if record is None:
        return None
    return record["value"]


def get_metric_unit(
    benchmark: dict,
    metric_name: str,
    benchmark_name: str | None = None,
) -> str | None:
    """Return the first matching metric unit, or None if it is missing."""

    record = _find_metric_record(benchmark, metric_name, benchmark_name)
    if record is None:
        return None

    unit = record["unit"]
    if isinstance(unit, str) or unit is None:
        return unit
    return None


def has_metric(
    benchmark: dict,
    metric_name: str,
    benchmark_name: str | None = None,
) -> bool:
    """Return whether a metric exists, regardless of its stored value."""

    return _find_metric_record(benchmark, metric_name, benchmark_name) is not None


def extract_named_metrics(
    benchmark: dict,
    metric_names: list[str],
) -> dict[str, MetricValue]:
    """Return a mapping of existing metric names to values."""

    extracted = {}
    for metric_name in metric_names:
        if has_metric(benchmark, metric_name):
            extracted[metric_name] = get_metric_value(benchmark, metric_name)
    return extracted


def metric_higher_is_better(metric_name: str) -> bool:
    """Return whether larger values should be considered better."""

    return HIGHER_IS_BETTER_METRICS.get(metric_name, True)


def _find_metric_record(
    benchmark: dict,
    metric_name: str,
    benchmark_name: str | None,
) -> dict | None:
    for record in iter_benchmark_metrics(benchmark):
        if record["metric_name"] != metric_name:
            continue
        if benchmark_name is not None and record["benchmark_name"] != benchmark_name:
            continue
        return record
    return None
