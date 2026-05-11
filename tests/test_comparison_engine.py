"""Tests for Phase 8.3 before/after comparison engine."""

from __future__ import annotations

from gpuboost.comparison.engine import (
    calculate_metric_delta,
    compare_benchmarks,
    group_comparable_metrics,
    overall_verdict,
    section_verdict,
)
from gpuboost.schemas.comparison import BenchmarkMetricDelta, ComparisonSection


def test_calculate_metric_delta_numeric_improvement_higher_is_better() -> None:
    delta = calculate_metric_delta("amp_speedup_ratio", 1.1, 1.25, "x")

    assert delta.absolute_delta == 0.1499999999999999
    assert round(delta.percent_delta, 2) == 13.64
    assert delta.direction == "improved"
    assert delta.higher_is_better is True
    assert delta.summary == "amp_speedup_ratio improved from 1.10x to 1.25x (+13.64%)."


def test_calculate_metric_delta_numeric_regression_higher_is_better() -> None:
    delta = calculate_metric_delta("best_fp32_tflops", 10.0, 9.0, "TFLOPS")

    assert delta.absolute_delta == -1.0
    assert delta.percent_delta == -10.0
    assert delta.direction == "regressed"


def test_calculate_metric_delta_timing_improvement_lower_is_better() -> None:
    delta = calculate_metric_delta("batch_32_median_ms", 20.0, 18.0, "ms")

    assert delta.absolute_delta == -2.0
    assert delta.percent_delta == -10.0
    assert delta.direction == "improved"
    assert delta.higher_is_better is False


def test_calculate_metric_delta_unchanged_within_tolerance() -> None:
    delta = calculate_metric_delta(
        "best_images_per_sec",
        100.0,
        100.5,
        "images/sec",
        tolerance_pct=1.0,
    )

    assert delta.percent_delta == 0.5
    assert delta.direction == "unchanged"


def test_calculate_metric_delta_before_zero_uses_absolute_delta() -> None:
    delta = calculate_metric_delta("best_batch_size", 0, 16, None)

    assert delta.absolute_delta == 16.0
    assert delta.percent_delta is None
    assert delta.direction == "improved"


def test_calculate_metric_delta_bool_treated_categorical() -> None:
    delta = calculate_metric_delta("custom_flag", False, True)

    assert delta.absolute_delta is None
    assert delta.percent_delta is None
    assert delta.direction == "unknown"
    assert delta.summary == "custom_flag changed from False to True."


def test_calculate_metric_delta_string_changed_gives_unknown() -> None:
    delta = calculate_metric_delta("status", "ok", "failed")

    assert delta.absolute_delta is None
    assert delta.percent_delta is None
    assert delta.direction == "unknown"


def test_calculate_metric_delta_none_values_unchanged_when_both_none() -> None:
    delta = calculate_metric_delta("nullable_metric", None, None)

    assert delta.direction == "unchanged"
    assert delta.absolute_delta is None
    assert delta.percent_delta is None


def test_percent_delta_calculation_is_correct() -> None:
    delta = calculate_metric_delta("amp_speedup_ratio", 2.0, 3.0, "x")

    assert delta.percent_delta == 50.0


def test_lower_is_better_metric_regression_works() -> None:
    delta = calculate_metric_delta("median_amp_step_ms", 10.0, 12.0, "ms")

    assert delta.absolute_delta == 2.0
    assert delta.percent_delta == 20.0
    assert delta.direction == "regressed"


def test_section_verdict_improved() -> None:
    assert section_verdict([_delta("improved")]) == "improved"


def test_section_verdict_regressed() -> None:
    assert section_verdict([_delta("regressed")]) == "regressed"


def test_section_verdict_mixed() -> None:
    assert section_verdict([_delta("improved"), _delta("regressed")]) == "mixed"


def test_section_verdict_unchanged() -> None:
    assert section_verdict([_delta("unchanged"), _delta("unchanged")]) == "unchanged"


def test_section_verdict_unknown() -> None:
    assert section_verdict([]) == "unknown"
    assert section_verdict([_delta("unknown")]) == "unknown"


def test_overall_verdict_improved() -> None:
    assert overall_verdict([ComparisonSection("A", verdict="improved")]) == "improved"


def test_overall_verdict_regressed() -> None:
    assert overall_verdict([ComparisonSection("A", verdict="regressed")]) == "regressed"


def test_overall_verdict_mixed() -> None:
    sections = [
        ComparisonSection("A", verdict="improved"),
        ComparisonSection("B", verdict="regressed"),
    ]

    assert overall_verdict(sections) == "mixed"


def test_overall_verdict_unchanged() -> None:
    sections = [
        ComparisonSection("A", verdict="unchanged"),
        ComparisonSection("B", verdict="unchanged"),
    ]

    assert overall_verdict(sections) == "unchanged"


def test_overall_verdict_unknown() -> None:
    assert overall_verdict([]) == "unknown"
    assert overall_verdict([ComparisonSection("A", verdict="unknown")]) == "unknown"


def test_group_comparable_metrics_groups_by_baseline_benchmark_name() -> None:
    grouped = group_comparable_metrics(
        _baseline_benchmark(),
        _optimized_benchmark(),
        ["best_fp32_tflops", "amp_speedup_ratio"],
    )

    assert list(grouped) == ["Matrix Multiply", "Mixed Precision"]
    assert grouped["Matrix Multiply"][0] == (
        "best_fp32_tflops",
        10.0,
        12.0,
        "TFLOPS",
    )
    assert grouped["Mixed Precision"][0] == ("amp_speedup_ratio", 1.1, 1.25, "x")


def test_compare_benchmarks_returns_ok_for_comparable_metrics() -> None:
    result = compare_benchmarks(
        _baseline_benchmark(),
        _optimized_benchmark(),
        metric_names=["best_fp32_tflops", "amp_speedup_ratio"],
    )

    assert result.status == "ok"
    assert result.error is None
    assert result.overall_verdict == "improved"
    assert result.warnings == []
    assert [section.title for section in result.sections] == [
        "Matrix Multiply",
        "Mixed Precision",
    ]


def test_compare_benchmarks_returns_partial_when_some_metrics_missing() -> None:
    result = compare_benchmarks(
        _baseline_benchmark(),
        _optimized_benchmark(),
        metric_names=["best_fp32_tflops", "best_images_per_sec"],
    )

    assert result.status == "partial"
    assert result.error is None
    assert len(result.sections) == 1
    assert result.warnings == [
        "Metric missing from baseline result: best_images_per_sec",
        "Metric missing from optimized result: best_images_per_sec",
    ]


def test_compare_benchmarks_returns_error_when_no_metrics_comparable() -> None:
    result = compare_benchmarks(
        _baseline_benchmark(),
        _optimized_benchmark(),
        metric_names=["best_images_per_sec"],
    )

    assert result.status == "error"
    assert result.sections == []
    assert result.overall_verdict == "unknown"
    assert result.error == "No comparable metrics were found."
    assert "No comparable metrics were found." in result.warnings


def test_compare_benchmarks_uses_labels() -> None:
    result = compare_benchmarks(
        _baseline_benchmark(),
        _optimized_benchmark(),
        metric_names=["best_fp32_tflops"],
        baseline_label="before",
        optimized_label="after",
    )

    assert result.baseline_label == "before"
    assert result.optimized_label == "after"


def test_compare_benchmarks_warns_for_metric_missing_from_optimized() -> None:
    optimized = {
        "results": [
            {
                "name": "Matrix Multiply",
                "metrics": [{"name": "best_fp32_tflops", "value": 12.0}],
            }
        ]
    }

    result = compare_benchmarks(
        _baseline_benchmark(),
        optimized,
        metric_names=["amp_speedup_ratio"],
    )

    assert result.status == "error"
    assert result.warnings == [
        "Metric missing from optimized result: amp_speedup_ratio",
        "No comparable metrics were found.",
    ]


def test_compare_benchmarks_warns_for_metric_missing_from_baseline() -> None:
    baseline = {
        "results": [
            {
                "name": "Matrix Multiply",
                "metrics": [{"name": "best_fp32_tflops", "value": 10.0}],
            }
        ]
    }

    result = compare_benchmarks(
        baseline,
        _optimized_benchmark(),
        metric_names=["amp_speedup_ratio"],
    )

    assert result.status == "error"
    assert result.warnings == [
        "Metric missing from baseline result: amp_speedup_ratio",
        "No comparable metrics were found.",
    ]


def test_compare_benchmarks_creates_deterministic_sections() -> None:
    result = compare_benchmarks(
        _baseline_benchmark(),
        _optimized_benchmark(),
        metric_names=["amp_speedup_ratio", "best_fp32_tflops"],
    )

    assert [(section.title, section.metrics[0].name) for section in result.sections] == [
        ("Mixed Precision", "amp_speedup_ratio"),
        ("Matrix Multiply", "best_fp32_tflops"),
    ]


def test_compare_benchmarks_uses_default_metrics_when_none() -> None:
    result = compare_benchmarks(_baseline_benchmark(), _optimized_benchmark())

    assert result.status == "partial"
    assert result.sections[0].metrics[0].name == "best_fp32_tflops"


def _delta(direction: str) -> BenchmarkMetricDelta:
    return BenchmarkMetricDelta(
        name=f"{direction}_metric",
        unit=None,
        before=1.0,
        after=2.0,
        absolute_delta=1.0,
        percent_delta=100.0,
        direction=direction,
        higher_is_better=True,
        summary=direction,
    )


def _baseline_benchmark() -> dict:
    return {
        "results": [
            {
                "name": "Matrix Multiply",
                "metrics": [
                    {"name": "best_fp32_tflops", "value": 10.0, "unit": "TFLOPS"},
                    {"name": "median_amp_step_ms", "value": 10.0, "unit": "ms"},
                ],
            },
            {
                "name": "Mixed Precision",
                "metrics": [
                    {"name": "amp_speedup_ratio", "value": 1.1, "unit": "x"},
                ],
            },
        ]
    }


def _optimized_benchmark() -> dict:
    return {
        "results": [
            {
                "name": "Matrix Multiply",
                "metrics": [
                    {"name": "best_fp32_tflops", "value": 12.0, "unit": "TFLOPS"},
                    {"name": "median_amp_step_ms", "value": 9.0, "unit": "ms"},
                ],
            },
            {
                "name": "Mixed Precision",
                "metrics": [
                    {"name": "amp_speedup_ratio", "value": 1.25, "unit": "x"},
                ],
            },
        ]
    }
