"""Tests for Phase 3 advisor engine."""

from __future__ import annotations

from gpuboost.advisor import engine
from gpuboost.advisor.engine import generate_advisor_result
from gpuboost.schemas.benchmark_result import (
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkSuiteResult,
)
from gpuboost.schemas.recommendation import AdvisorResult, Recommendation


def test_generate_advisor_result_returns_advisor_result() -> None:
    result = generate_advisor_result(_make_suite([]))

    assert isinstance(result, AdvisorResult)


def test_engine_combines_recommendations_from_multiple_rules() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Mixed Precision",
                metrics=[BenchmarkMetric(name="amp_speedup_ratio", value=1.3)],
            ),
            _make_result(
                "Matrix Multiplication",
                metrics=[
                    BenchmarkMetric(name="fp16_speedup_ratio", value=2.1),
                    BenchmarkMetric(name="tensor_cores_likely_active", value=True),
                ],
            ),
            _make_result(
                "DataLoader",
                metrics=[
                    BenchmarkMetric(name="best_num_workers", value=2),
                    BenchmarkMetric(name="best_pin_memory", value=False),
                ],
            ),
        ],
    )

    result = generate_advisor_result(suite)

    assert {
        recommendation.id for recommendation in result.recommendations
    } >= {
        "mixed_precision_enable",
        "tensor_core_friendly_workloads",
        "dataloader_tune_workers",
    }


def test_recommendations_are_deduplicated_by_id(monkeypatch) -> None:
    first = _make_recommendation(
        id="duplicate",
        title="First duplicate",
        impact="high",
    )
    second = _make_recommendation(
        id="duplicate",
        title="Second duplicate",
        impact="low",
    )

    monkeypatch.setattr(
        engine,
        "_ADVISOR_RULES",
        (
            lambda suite: [first],
            lambda suite: [second],
        ),
    )

    result = generate_advisor_result(_make_suite([]))

    assert len(result.recommendations) == 1
    assert result.recommendations[0].title == "First duplicate"


def test_priorities_are_assigned_starting_at_one() -> None:
    suite = _make_suite(
        [
            _make_result(
                "Mixed Precision",
                metrics=[BenchmarkMetric(name="amp_speedup_ratio", value=1.3)],
            ),
            _make_result(
                "DataLoader",
                metrics=[
                    BenchmarkMetric(name="best_num_workers", value=2),
                    BenchmarkMetric(name="best_pin_memory", value=False),
                ],
            ),
        ],
    )

    result = generate_advisor_result(suite)

    assert [recommendation.priority for recommendation in result.recommendations] == [
        1,
        2,
    ]


def test_recommendations_are_sorted_by_rank_score(monkeypatch) -> None:
    low_rank = _make_recommendation(
        id="low",
        title="Low rank",
        impact="low",
        confidence="low",
        effort="high",
    )
    high_rank = _make_recommendation(
        id="high",
        title="High rank",
        impact="high",
        confidence="high",
        effort="low",
    )

    monkeypatch.setattr(
        engine,
        "_ADVISOR_RULES",
        (lambda suite: [low_rank, high_rank],),
    )

    result = generate_advisor_result(_make_suite([]))

    assert [recommendation.id for recommendation in result.recommendations] == [
        "high",
        "low",
    ]


def test_empty_suite_returns_no_recommendations_and_warning() -> None:
    result = generate_advisor_result(_make_suite([]))

    assert result.recommendations == []
    assert result.warnings == [
        "No optimization recommendations could be generated from the available "
        "benchmark results.",
    ]


def test_rule_failure_is_caught_and_added_to_warnings(monkeypatch) -> None:
    def failing_rule(suite: BenchmarkSuiteResult) -> list[Recommendation]:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        engine,
        "_ADVISOR_RULES",
        (
            failing_rule,
            lambda suite: [
                _make_recommendation(
                    id="after-failure",
                    title="After failure",
                ),
            ],
        ),
    )

    result = generate_advisor_result(_make_suite([]))

    assert result.recommendations[0].id == "after-failure"
    assert result.warnings == ["Advisor rule failed: failing_rule: boom"]


def test_missing_benchmark_metrics_do_not_crash() -> None:
    suite = _make_suite(
        [
            _make_result("Mixed Precision", metrics=[]),
            _make_result("Matrix Multiplication", metrics=[]),
            _make_result("Batch Size Sweep", metrics=[]),
            _make_result("DataLoader", metrics=[]),
        ],
    )

    result = generate_advisor_result(suite)

    assert result.recommendations == []
    assert result.warnings == [
        "No optimization recommendations could be generated from the available "
        "benchmark results.",
    ]


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


def _make_recommendation(
    *,
    id: str,
    title: str,
    impact: str = "medium",
    confidence: str = "high",
    effort: str = "medium",
    estimated_speedup: float | None = None,
) -> Recommendation:
    return Recommendation(
        id=id,
        title=title,
        category="test",
        priority=0,
        impact=impact,
        confidence=confidence,
        effort=effort,
        estimated_speedup=estimated_speedup,
        summary="Test summary.",
        rationale="Test rationale.",
        suggested_action="Test action.",
        code_snippet=None,
    )
