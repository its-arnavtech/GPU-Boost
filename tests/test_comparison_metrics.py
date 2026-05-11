"""Tests for Phase 8.2 benchmark metric extraction helpers."""

from __future__ import annotations

from gpuboost.comparison.metrics import (
    DEFAULT_COMPARISON_METRICS,
    extract_named_metrics,
    get_metric_unit,
    get_metric_value,
    has_metric,
    iter_benchmark_metrics,
    metric_higher_is_better,
)


def test_iter_benchmark_metrics_flattens_metrics() -> None:
    records = iter_benchmark_metrics(_make_benchmark())

    assert records == [
        {
            "benchmark_name": "Matrix Multiply",
            "metric_name": "best_fp32_tflops",
            "value": 12.5,
            "unit": "TFLOPS",
        },
        {
            "benchmark_name": "Mixed Precision",
            "metric_name": "amp_speedup_ratio",
            "value": 1.2,
            "unit": "x",
        },
        {
            "benchmark_name": "Mixed Precision",
            "metric_name": "zero_metric",
            "value": 0,
            "unit": "items/sec",
        },
        {
            "benchmark_name": "Mixed Precision",
            "metric_name": "false_metric",
            "value": False,
            "unit": None,
        },
        {
            "benchmark_name": "Mixed Precision",
            "metric_name": "none_metric",
            "value": None,
            "unit": None,
        },
    ]


def test_get_metric_value_finds_metric() -> None:
    assert get_metric_value(_make_benchmark(), "amp_speedup_ratio") == 1.2


def test_get_metric_value_respects_benchmark_name() -> None:
    benchmark = {
        "results": [
            {
                "name": "First",
                "metrics": [{"name": "shared_metric", "value": 1, "unit": "x"}],
            },
            {
                "name": "Second",
                "metrics": [{"name": "shared_metric", "value": 2, "unit": "x"}],
            },
        ]
    }

    assert get_metric_value(benchmark, "shared_metric", "Second") == 2


def test_get_metric_value_returns_none_when_missing() -> None:
    assert get_metric_value(_make_benchmark(), "missing_metric") is None


def test_get_metric_unit_works() -> None:
    assert get_metric_unit(_make_benchmark(), "best_fp32_tflops") == "TFLOPS"


def test_has_metric_true_for_value_zero() -> None:
    assert has_metric(_make_benchmark(), "zero_metric") is True


def test_has_metric_true_for_value_false() -> None:
    assert has_metric(_make_benchmark(), "false_metric") is True


def test_has_metric_true_for_value_none() -> None:
    assert has_metric(_make_benchmark(), "none_metric") is True


def test_extract_named_metrics_returns_existing_only() -> None:
    values = extract_named_metrics(
        _make_benchmark(),
        ["best_fp32_tflops", "amp_speedup_ratio", "missing_metric"],
    )

    assert values == {
        "best_fp32_tflops": 12.5,
        "amp_speedup_ratio": 1.2,
    }


def test_invalid_benchmark_structures_return_safe_results() -> None:
    invalid_cases = [
        None,
        {},
        {"results": None},
        {"results": "not a list"},
        {"results": [None, {"name": "No metrics"}, {"metrics": "bad"}]},
    ]

    for benchmark in invalid_cases:
        assert iter_benchmark_metrics(benchmark) == []
        assert get_metric_value(benchmark, "anything") is None
        assert get_metric_unit(benchmark, "anything") is None
        assert has_metric(benchmark, "anything") is False
        assert extract_named_metrics(benchmark, ["anything"]) == {}


def test_metric_higher_is_better_known_throughput_true() -> None:
    assert metric_higher_is_better("best_fp32_tflops") is True


def test_metric_higher_is_better_known_timing_false() -> None:
    assert metric_higher_is_better("batch_32_median_ms") is False


def test_metric_higher_is_better_unknown_defaults_true() -> None:
    assert metric_higher_is_better("custom_metric") is True


def test_default_comparison_metrics_contains_expected_names() -> None:
    assert "best_fp32_tflops" in DEFAULT_COMPARISON_METRICS
    assert "amp_speedup_ratio" in DEFAULT_COMPARISON_METRICS
    assert "max_successful_batch_size" in DEFAULT_COMPARISON_METRICS


def _make_benchmark() -> dict:
    return {
        "results": [
            {
                "name": "Matrix Multiply",
                "metrics": [
                    {
                        "name": "best_fp32_tflops",
                        "value": 12.5,
                        "unit": "TFLOPS",
                    }
                ],
            },
            {
                "name": "Mixed Precision",
                "metrics": [
                    {"name": "amp_speedup_ratio", "value": 1.2, "unit": "x"},
                    {"name": "zero_metric", "value": 0, "unit": "items/sec"},
                    {"name": "false_metric", "value": False, "unit": None},
                    {"name": "none_metric", "value": None, "unit": None},
                ],
            },
        ]
    }
