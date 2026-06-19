"""Tests for Phase 3 advisor scoring helpers."""

from gpuboost.advisor.scoring import (
    confidence_from_signal,
    confidence_score,
    effort_score,
    impact_from_speedup,
    impact_score,
    rank_score,
    sort_and_prioritize,
)
from gpuboost.schemas.recommendation import Recommendation


def test_impact_from_speedup_boundaries() -> None:
    assert impact_from_speedup(None) == "low"
    assert impact_from_speedup(1.05) == "low"
    assert impact_from_speedup(1.0501) == "medium"
    assert impact_from_speedup(1.24) == "medium"
    assert impact_from_speedup(1.25) == "high"


def test_confidence_from_signal_behavior() -> None:
    assert (
        confidence_from_signal(
            metric_present=False,
            benchmark_status_ok=True,
            warnings=[],
        )
        == "low"
    )
    assert (
        confidence_from_signal(
            metric_present=True,
            benchmark_status_ok=False,
            warnings=[],
        )
        == "low"
    )
    assert (
        confidence_from_signal(
            metric_present=True,
            benchmark_status_ok=True,
            warnings=["partial signal"],
        )
        == "medium"
    )
    assert (
        confidence_from_signal(
            metric_present=True,
            benchmark_status_ok=True,
            warnings=[],
        )
        == "high"
    )


def test_score_mappings() -> None:
    assert effort_score("low") == 1
    assert effort_score("medium") == 2
    assert effort_score("high") == 3
    assert impact_score("low") == 1
    assert impact_score("medium") == 2
    assert impact_score("high") == 3
    assert confidence_score("low") == 1
    assert confidence_score("medium") == 2
    assert confidence_score("high") == 3


def test_unknown_impact_and_confidence_default_to_low() -> None:
    assert impact_score("urgent") == 1
    assert confidence_score("") == 1


def test_unknown_effort_defaults_to_medium_not_low() -> None:
    # Effort is the divisor in rank_score; defaulting an unknown label to low
    # (1) would inflate the score, so it must default to medium (2).
    assert effort_score("unknown") == 2


def test_rank_score_formula() -> None:
    recommendation = _make_recommendation(
        title="Ranked",
        impact="high",
        confidence="medium",
        effort="low",
    )

    assert rank_score(recommendation) == 6.0


def test_sort_and_prioritize_ordering() -> None:
    low_rank_fast = _make_recommendation(
        title="Fast but weak",
        impact="medium",
        confidence="medium",
        effort="medium",
        estimated_speedup=1.5,
    )
    high_rank_none = _make_recommendation(
        title="Alpha recommendation",
        impact="high",
        confidence="high",
        effort="low",
        estimated_speedup=None,
    )
    high_rank_speedup = _make_recommendation(
        title="Beta recommendation",
        impact="high",
        confidence="high",
        effort="low",
        estimated_speedup=1.3,
    )
    high_rank_same_speedup_alpha = _make_recommendation(
        title="Aardvark recommendation",
        impact="high",
        confidence="high",
        effort="low",
        estimated_speedup=1.3,
    )

    sorted_recommendations = sort_and_prioritize(
        [
            low_rank_fast,
            high_rank_none,
            high_rank_speedup,
            high_rank_same_speedup_alpha,
        ],
    )

    assert sorted_recommendations == [
        high_rank_same_speedup_alpha,
        high_rank_speedup,
        high_rank_none,
        low_rank_fast,
    ]


def test_sort_and_prioritize_assigns_priorities() -> None:
    recommendations = [
        _make_recommendation(title="Second", impact="low"),
        _make_recommendation(title="First", impact="high"),
    ]

    sorted_recommendations = sort_and_prioritize(recommendations)

    assert [recommendation.priority for recommendation in sorted_recommendations] == [
        1,
        2,
    ]
    assert sorted_recommendations[0].title == "First"


def _make_recommendation(
    *,
    title: str,
    impact: str = "medium",
    confidence: str = "high",
    effort: str = "medium",
    estimated_speedup: float | None = None,
) -> Recommendation:
    return Recommendation(
        id=title.lower().replace(" ", "-"),
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
