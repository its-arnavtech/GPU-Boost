"""Tests for Phase 3 advisor rules."""

from gpuboost.advisor.rules import mixed_precision_rule, tensor_core_rule
from gpuboost.schemas.benchmark_result import (
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkSuiteResult,
)


def test_mixed_precision_rule_returns_enable_recommendation() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Mixed Precision",
                metrics=[
                    BenchmarkMetric(name="amp_speedup_ratio", value=1.26),
                    BenchmarkMetric(name="fp32_samples_per_sec", value=100.0),
                    BenchmarkMetric(name="amp_samples_per_sec", value=126.0),
                    BenchmarkMetric(name="median_fp32_step_ms", value=10.0),
                    BenchmarkMetric(name="median_amp_step_ms", value=7.9),
                ],
            ),
        ],
    )

    recommendations = mixed_precision_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "mixed_precision_enable"
    assert recommendation.title == "Enable mixed precision"
    assert recommendation.impact == "high"
    assert recommendation.confidence == "high"
    assert recommendation.effort == "low"
    assert recommendation.estimated_speedup == 1.26
    assert "1.26x" in recommendation.summary
    assert "torch.amp.autocast" in (recommendation.code_snippet or "")
    assert recommendation.related_metrics == [
        "amp_speedup_ratio",
        "fp32_samples_per_sec",
        "amp_samples_per_sec",
        "median_fp32_step_ms",
        "median_amp_step_ms",
    ]


def test_mixed_precision_rule_returns_warning_when_amp_slower() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Mixed Precision",
                metrics=[BenchmarkMetric(name="amp_speedup_ratio", value=0.92)],
            ),
        ],
    )

    recommendations = mixed_precision_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "mixed_precision_do_not_enable_blindly"
    assert recommendation.impact == "low"
    assert recommendation.confidence == "medium"
    assert recommendation.estimated_speedup == 0.92
    assert recommendation.code_snippet is None
    assert recommendation.related_metrics == ["amp_speedup_ratio"]


def test_mixed_precision_rule_returns_limited_benefit_recommendation() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Mixed Precision",
                metrics=[BenchmarkMetric(name="amp_speedup_ratio", value=1.05)],
            ),
        ],
    )

    recommendations = mixed_precision_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "mixed_precision_limited_benefit"
    assert recommendation.title == "Mixed precision benefit is limited"
    assert recommendation.impact == "low"
    assert recommendation.confidence == "medium"
    assert recommendation.estimated_speedup == 1.05
    assert recommendation.code_snippet is None


def test_mixed_precision_rule_returns_empty_when_result_missing() -> None:
    assert mixed_precision_rule(_make_suite([])) == []


def test_tensor_core_rule_returns_recommendation_for_strong_fp16_speedup() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Matrix Multiplication",
                metrics=[
                    BenchmarkMetric(name="fp16_speedup_ratio", value=2.25),
                    BenchmarkMetric(name="tensor_cores_likely_active", value=True),
                    BenchmarkMetric(name="best_fp16_tflops", value=45.0),
                    BenchmarkMetric(name="best_fp32_tflops", value=20.0),
                ],
            ),
        ],
    )

    recommendations = tensor_core_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "tensor_core_friendly_workloads"
    assert recommendation.title == "Prioritize Tensor Core-friendly workloads"
    assert recommendation.impact == "high"
    assert recommendation.confidence == "high"
    assert recommendation.effort == "medium"
    assert recommendation.estimated_speedup == 2.25
    assert "2.25x" in recommendation.summary
    assert recommendation.related_metrics == [
        "fp16_speedup_ratio",
        "tensor_cores_likely_active",
        "best_fp16_tflops",
        "best_fp32_tflops",
    ]


def test_tensor_core_rule_returns_weak_acceleration_warning() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Matrix Multiplication",
                metrics=[
                    BenchmarkMetric(name="fp16_speedup_ratio", value=1.25),
                    BenchmarkMetric(name="tensor_cores_likely_active", value=False),
                ],
            ),
        ],
    )

    recommendations = tensor_core_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "tensor_core_acceleration_weak"
    assert recommendation.impact == "medium"
    assert recommendation.confidence == "medium"
    assert recommendation.effort == "medium"
    assert recommendation.estimated_speedup == 1.25
    assert recommendation.related_metrics == ["fp16_speedup_ratio"]


def test_tensor_core_rule_returns_empty_when_result_missing() -> None:
    assert tensor_core_rule(_make_suite([])) == []


def test_warnings_from_benchmark_result_are_copied_into_recommendation() -> None:
    warnings = ["synthetic benchmark only"]
    result = _make_result(
        "Mixed Precision",
        metrics=[BenchmarkMetric(name="amp_speedup_ratio", value=1.3)],
        warnings=warnings,
    )
    suite = _make_suite([result])

    recommendation = mixed_precision_rule(suite)[0]
    result.warnings.append("mutated after recommendation")

    assert recommendation.warnings == ["synthetic benchmark only"]
    assert recommendation.confidence == "medium"


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
    *,
    metrics: list[BenchmarkMetric],
    status: str = "ok",
    warnings: list[str] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        status=status,
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        duration_sec=1.0,
        metrics=metrics,
        warnings=warnings or [],
    )
