"""Tests for Phase 3 advisor rules."""

from gpuboost.advisor.rules import (
    batch_size_rule,
    dataloader_rule,
    mixed_precision_rule,
    tensor_core_rule,
    warning_propagation_rule,
)
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


def test_batch_size_rule_recommends_larger_batch_when_speedup_high() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Batch Size Sweep",
                metrics=[
                    BenchmarkMetric(name="best_batch_size", value=16),
                    BenchmarkMetric(name="best_images_per_sec", value=800.0),
                    BenchmarkMetric(name="batch_1_images_per_sec", value=100.0),
                    BenchmarkMetric(name="speedup_vs_batch_1", value=1.5),
                    BenchmarkMetric(name="max_successful_batch_size", value=32),
                ],
            ),
        ],
    )

    recommendations = batch_size_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "batch_size_increase"
    assert recommendation.title == "Use a larger batch size"
    assert recommendation.impact == "high"
    assert recommendation.confidence == "high"
    assert recommendation.estimated_speedup == 1.5
    assert "batch_size=16" in recommendation.suggested_action
    assert recommendation.code_snippet == "DataLoader(dataset, batch_size=16, ...)"
    assert recommendation.related_metrics == [
        "best_batch_size",
        "best_images_per_sec",
        "batch_1_images_per_sec",
        "speedup_vs_batch_1",
        "max_successful_batch_size",
    ]


def test_batch_size_rule_adds_limited_scaling_warning() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Batch Size Sweep",
                metrics=[
                    BenchmarkMetric(name="best_batch_size", value=4),
                    BenchmarkMetric(name="speedup_vs_batch_1", value=1.05),
                ],
            ),
        ],
    )

    recommendations = batch_size_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "batch_size_scaling_limited"
    assert recommendation.impact == "medium"
    assert recommendation.confidence == "medium"
    assert recommendation.estimated_speedup == 1.05
    assert recommendation.related_metrics == ["best_batch_size", "speedup_vs_batch_1"]


def test_batch_size_rule_prefers_increase_over_limited_when_scaling_is_good() -> None:
    # best_batch_size <= 4 but scaling is strong: recommend increasing the batch
    # size and suppress the contradictory "scaling limited" warning.
    suite = _make_suite(
        [
            _make_result(
                "Batch Size Sweep",
                metrics=[
                    BenchmarkMetric(name="best_batch_size", value=4),
                    BenchmarkMetric(name="speedup_vs_batch_1", value=1.3),
                ],
            ),
        ],
    )

    recommendations = batch_size_rule(suite)

    assert [recommendation.id for recommendation in recommendations] == [
        "batch_size_increase",
    ]


def test_batch_size_rule_returns_empty_when_result_missing() -> None:
    assert batch_size_rule(_make_suite([])) == []


def test_dataloader_rule_recommends_workers_when_best_num_workers_positive() -> None:
    suite = _make_suite(
        [
            _make_result(
                "DataLoader",
                metrics=[
                    BenchmarkMetric(name="best_num_workers", value=4),
                    BenchmarkMetric(name="best_pin_memory", value=False),
                    BenchmarkMetric(name="best_samples_per_sec", value=1200.0),
                ],
            ),
        ],
    )

    recommendations = dataloader_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "dataloader_tune_workers"
    assert recommendation.title == "Tune DataLoader workers"
    assert recommendation.impact == "medium"
    assert recommendation.confidence == "high"
    assert recommendation.estimated_speedup is None
    assert "num_workers=4" in recommendation.summary
    assert recommendation.code_snippet == (
        "DataLoader(dataset, num_workers=4, pin_memory=False)"
    )


def test_dataloader_rule_recommends_pinned_memory_when_speedup_high() -> None:
    suite = _make_suite(
        [
            _make_result(
                "DataLoader",
                metrics=[
                    BenchmarkMetric(name="best_num_workers", value=None),
                    BenchmarkMetric(name="best_pin_memory", value=True),
                    BenchmarkMetric(name="pin_memory_speedup_ratio", value=1.18),
                ],
            ),
        ],
    )

    recommendations = dataloader_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "dataloader_enable_pinned_memory"
    assert recommendation.impact == "medium"
    assert recommendation.estimated_speedup == 1.18
    assert "1.18x" in recommendation.summary
    assert recommendation.code_snippet == "batch = batch.to('cuda', non_blocking=True)"
    assert recommendation.related_metrics == [
        "best_pin_memory",
        "pin_memory_speedup_ratio",
    ]


def test_dataloader_rule_adds_num_workers_zero_information() -> None:
    suite = _make_suite(
        [
            _make_result(
                "DataLoader",
                metrics=[BenchmarkMetric(name="best_num_workers", value=0)],
            ),
        ],
    )

    recommendations = dataloader_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "dataloader_workers_may_not_help"
    assert recommendation.impact == "low"
    assert recommendation.confidence == "medium"
    assert recommendation.related_metrics == ["best_num_workers"]


def test_dataloader_rule_can_return_multiple_recommendations() -> None:
    suite = _make_suite(
        [
            _make_result(
                "DataLoader",
                metrics=[
                    BenchmarkMetric(name="best_num_workers", value=2),
                    BenchmarkMetric(name="best_pin_memory", value=True),
                    BenchmarkMetric(name="best_samples_per_sec", value=1400.0),
                    BenchmarkMetric(name="pin_memory_speedup_ratio", value=1.2),
                ],
            ),
        ],
    )

    recommendations = dataloader_rule(suite)

    assert [recommendation.id for recommendation in recommendations] == [
        "dataloader_tune_workers",
        "dataloader_enable_pinned_memory",
    ]


def test_warning_propagation_rule_creates_recommendation_for_warnings() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Batch Size Sweep",
                metrics=[],
                warnings=["Batch sweep hit an out-of-memory condition."],
            ),
        ],
    )

    recommendations = warning_propagation_rule(suite)

    assert len(recommendations) == 1
    recommendation = recommendations[0]
    assert recommendation.id == "warning_batch_size_sweep"
    assert recommendation.title == "Review benchmark warning: Batch Size Sweep"
    assert recommendation.category == "warning"
    assert recommendation.impact == "medium"
    assert recommendation.confidence == "high"
    assert recommendation.summary == "Batch sweep hit an out-of-memory condition."
    assert recommendation.warnings == ["Batch sweep hit an out-of-memory condition."]


def test_warning_id_slug_is_stable() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Batch Size Sweep",
                metrics=[],
                warnings=["warning"],
            ),
        ],
    )

    recommendation = warning_propagation_rule(suite)[0]

    assert recommendation.id == "warning_batch_size_sweep"


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
