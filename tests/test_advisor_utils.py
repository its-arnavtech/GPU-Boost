"""Tests for Phase 3 advisor utility helpers."""

import pytest

from gpuboost.advisor.utils import (
    format_speedup,
    get_metric,
    get_result_by_name,
    metric_exists,
    speedup_to_percent,
)
from gpuboost.schemas.benchmark_result import (
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkSuiteResult,
)


def test_get_result_by_name_finds_existing_result() -> None:
    suite = _make_suite(
        [
            _make_result("matmul"),
            _make_result("mixed_precision"),
        ],
    )

    result = get_result_by_name(suite, "mixed_precision")

    assert result is not None
    assert result.name == "mixed_precision"


def test_get_result_by_name_returns_none_for_missing_result() -> None:
    suite = _make_suite([_make_result("matmul")])

    assert get_result_by_name(suite, "dataloader") is None


def test_get_result_by_name_handles_empty_results() -> None:
    suite = _make_suite([])

    assert get_result_by_name(suite, "matmul") is None


def test_get_metric_returns_metric_value() -> None:
    result = _make_result(
        "matmul",
        metrics=[BenchmarkMetric(name="speedup", value=1.26, unit="x")],
    )

    assert get_metric(result, "speedup") == 1.26


def test_get_metric_returns_default_when_missing() -> None:
    result = _make_result("matmul")

    assert get_metric(result, "missing", default="fallback") == "fallback"


def test_get_metric_handles_result_none() -> None:
    assert get_metric(None, "speedup", default=0) == 0


def test_metric_exists_returns_true_when_metric_value_is_zero() -> None:
    result = _make_result(
        "matmul",
        metrics=[BenchmarkMetric(name="batch_size", value=0)],
    )

    assert metric_exists(result, "batch_size") is True


def test_metric_exists_returns_true_when_metric_value_is_false() -> None:
    result = _make_result(
        "matmul",
        metrics=[BenchmarkMetric(name="cuda_graphs_enabled", value=False)],
    )

    assert metric_exists(result, "cuda_graphs_enabled") is True


def test_metric_exists_returns_true_when_metric_value_is_none() -> None:
    result = _make_result(
        "matmul",
        metrics=[BenchmarkMetric(name="optional_value", value=None)],
    )

    assert metric_exists(result, "optional_value") is True


def test_metric_exists_returns_false_when_metric_name_missing() -> None:
    result = _make_result(
        "matmul",
        metrics=[BenchmarkMetric(name="speedup", value=1.26)],
    )

    assert metric_exists(result, "missing") is False


def test_speedup_to_percent_math() -> None:
    assert speedup_to_percent(1.26) == 26.0
    assert speedup_to_percent(2.0) == 100.0
    assert speedup_to_percent(0.8) == pytest.approx(-20.0)


def test_format_speedup_behavior() -> None:
    assert format_speedup(None) == "unknown"
    assert format_speedup(1.26) == "1.26x"
    assert format_speedup(2) == "2.00x"
    assert format_speedup(0.8) == "0.80x"


def _make_suite(results: list[BenchmarkResult]) -> BenchmarkSuiteResult:
    return BenchmarkSuiteResult(
        generated_at="2026-01-01T00:00:00+00:00",
        gpu_name="NVIDIA Test GPU",
        cuda_available=True,
        device_index=0,
        results=results,
    )


def _make_result(
    name: str,
    metrics: list[BenchmarkMetric] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        status="ok",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        duration_sec=1.0,
        metrics=metrics or [],
    )
