"""Advisor engine for generating optimization recommendations."""

from __future__ import annotations

from collections.abc import Callable

from gpuboost.advisor.rules import (
    batch_size_rule,
    dataloader_rule,
    mixed_precision_rule,
    tensor_core_rule,
    warning_propagation_rule,
)
from gpuboost.advisor.scoring import sort_and_prioritize
from gpuboost.schemas.benchmark_result import BenchmarkSuiteResult
from gpuboost.schemas.recommendation import (
    AdvisorResult,
    Recommendation,
    create_timestamp,
)


RuleFunction = Callable[[BenchmarkSuiteResult], list[Recommendation]]

_NO_RECOMMENDATIONS_WARNING = (
    "No optimization recommendations could be generated from the available "
    "benchmark results."
)

_ADVISOR_RULES: tuple[RuleFunction, ...] = (
    mixed_precision_rule,
    tensor_core_rule,
    batch_size_rule,
    dataloader_rule,
    warning_propagation_rule,
)


def generate_advisor_result(suite: BenchmarkSuiteResult) -> AdvisorResult:
    """Generate advisor recommendations from benchmark suite results."""

    recommendations: list[Recommendation] = []
    warnings: list[str] = []

    for rule in _ADVISOR_RULES:
        try:
            recommendations.extend(rule(suite))
        except Exception as error:  # noqa: BLE001 - advisor must isolate rule failures.
            warnings.append(f"Advisor rule failed: {rule.__name__}: {error}")

    deduplicated = _deduplicate_recommendations(recommendations)
    prioritized = sort_and_prioritize(deduplicated)

    if not prioritized:
        warnings.append(_NO_RECOMMENDATIONS_WARNING)

    return AdvisorResult(
        generated_at=create_timestamp(),
        recommendations=prioritized,
        warnings=warnings,
    )


def _deduplicate_recommendations(
    recommendations: list[Recommendation],
) -> list[Recommendation]:
    seen_ids: set[str] = set()
    deduplicated: list[Recommendation] = []

    for recommendation in recommendations:
        if recommendation.id in seen_ids:
            continue
        seen_ids.add(recommendation.id)
        deduplicated.append(recommendation)

    return deduplicated
