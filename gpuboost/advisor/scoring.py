"""Scoring helpers for GPUBoost advisor recommendations."""

from __future__ import annotations

from gpuboost.schemas.recommendation import Recommendation


_SCORE_BY_LABEL = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


def impact_from_speedup(speedup: float | None) -> str:
    """Infer impact label from an estimated speedup multiplier."""

    if speedup is None or speedup <= 1.05:
        return "low"
    if speedup < 1.25:
        return "medium"
    return "high"


def confidence_from_signal(
    metric_present: bool,
    benchmark_status_ok: bool,
    warnings: list[str],
) -> str:
    """Infer confidence label from benchmark signal quality."""

    if not metric_present:
        return "low"
    if not benchmark_status_ok:
        return "low"
    if warnings:
        return "medium"
    return "high"


def effort_score(effort: str) -> int:
    """Return numeric score for an effort label.

    Effort is the divisor in rank_score, so an unknown label defaults to medium
    rather than low to avoid silently inflating a recommendation's rank.
    """

    return _label_score(effort, default=2)


def impact_score(impact: str) -> int:
    """Return numeric score for an impact label."""

    return _label_score(impact, default=1)


def confidence_score(confidence: str) -> int:
    """Return numeric score for a confidence label."""

    return _label_score(confidence, default=1)


def rank_score(recommendation: Recommendation) -> float:
    """Return ranking score for a recommendation."""

    return (
        impact_score(recommendation.impact)
        * confidence_score(recommendation.confidence)
        / effort_score(recommendation.effort)
    )


def sort_and_prioritize(
    recommendations: list[Recommendation],
) -> list[Recommendation]:
    """Sort recommendations by rank and assign one-based priorities."""

    sorted_recommendations = sorted(
        recommendations,
        key=lambda recommendation: (
            -rank_score(recommendation),
            -(recommendation.estimated_speedup or 0),
            recommendation.title,
        ),
    )

    for index, recommendation in enumerate(sorted_recommendations, start=1):
        recommendation.priority = index

    return sorted_recommendations


def _label_score(label: str, default: int = 1) -> int:
    return _SCORE_BY_LABEL.get(label, default)
