"""Tests for Phase 3 advisor recommendation schemas."""

import json
from dataclasses import asdict
from datetime import datetime, timezone

from gpuboost.schemas.recommendation import (
    AdvisorResult,
    Recommendation,
    create_timestamp,
)


def test_recommendation_creation() -> None:
    recommendation = Recommendation(
        id="batch-size",
        title="Increase batch size",
        category="training",
        priority=1,
        impact="high",
        confidence="medium",
        effort="low",
        estimated_speedup=1.25,
        summary="The benchmark has spare memory headroom.",
        rationale="Larger batches may improve GPU occupancy.",
        suggested_action="Try the next larger stable batch size.",
        code_snippet="batch_size *= 2",
        related_metrics=["gpu_utilization", "memory_free_mb"],
        warnings=["Validate convergence after changing batch size."],
    )

    assert recommendation.id == "batch-size"
    assert recommendation.priority == 1
    assert recommendation.estimated_speedup == 1.25
    assert recommendation.related_metrics == ["gpu_utilization", "memory_free_mb"]


def test_advisor_result_creation() -> None:
    recommendation = _make_recommendation()
    result = AdvisorResult(
        generated_at="2026-01-01T00:00:00+00:00",
        recommendations=[recommendation],
        warnings=["Advisor output is informational."],
    )

    assert result.generated_at == "2026-01-01T00:00:00+00:00"
    assert result.recommendations == [recommendation]
    assert result.warnings == ["Advisor output is informational."]


def test_to_dict_output() -> None:
    result = AdvisorResult(
        generated_at="2026-01-01T00:00:00+00:00",
        recommendations=[_make_recommendation()],
    )

    data = result.to_dict()

    assert data["generated_at"] == "2026-01-01T00:00:00+00:00"
    assert data["recommendations"][0]["id"] == "mixed-precision"
    assert data["recommendations"][0]["code_snippet"] is None
    assert data["recommendations"][0]["related_metrics"] == ["fp32_throughput"]
    assert data["warnings"] == []


def test_json_serialization() -> None:
    result = AdvisorResult(
        generated_at="2026-01-01T00:00:00+00:00",
        recommendations=[_make_recommendation()],
    )

    serialized = json.dumps(result.to_dict())
    from_asdict = json.dumps(asdict(result))

    assert json.loads(serialized)["recommendations"][0]["title"] == "Use mixed precision"
    assert json.loads(from_asdict)["recommendations"][0]["estimated_speedup"] is None


def test_default_empty_lists_are_independent() -> None:
    first = AdvisorResult(generated_at="2026-01-01T00:00:00+00:00")
    second = AdvisorResult(generated_at="2026-01-01T00:00:01+00:00")

    first.recommendations.append(_make_recommendation())
    first.warnings.append("first only")

    assert len(first.recommendations) == 1
    assert second.recommendations == []
    assert second.warnings == []


def test_create_timestamp_returns_utc_iso_timestamp() -> None:
    timestamp = create_timestamp()
    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo == timezone.utc


def _make_recommendation() -> Recommendation:
    return Recommendation(
        id="mixed-precision",
        title="Use mixed precision",
        category="precision",
        priority=2,
        impact="medium",
        confidence="high",
        effort="medium",
        estimated_speedup=None,
        summary="Mixed precision benchmark results suggest a possible speedup.",
        rationale="Tensor core capable GPUs can benefit from lower precision math.",
        suggested_action="Enable automatic mixed precision in the training loop.",
        code_snippet=None,
        related_metrics=["fp32_throughput"],
    )
