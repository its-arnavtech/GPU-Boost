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
)
_WARNING_RULE: RuleFunction = warning_propagation_rule


def generate_advisor_result(suite: BenchmarkSuiteResult) -> AdvisorResult:
    """Generate advisor recommendations from benchmark suite results."""

    recommendations: list[Recommendation] = []
    warnings: list[str] = []

    for rule in _ADVISOR_RULES:
        try:
            recommendations.extend(rule(suite))
        except Exception as error:  # noqa: BLE001 - advisor must isolate rule failures.
            warnings.append(f"Advisor rule failed: {rule.__name__}: {error}")

    try:
        warning_recommendations = _WARNING_RULE(suite)
    except Exception as error:  # noqa: BLE001 - advisor must isolate rule failures.
        warnings.append(f"Advisor rule failed: {_WARNING_RULE.__name__}: {error}")
    else:
        recommendations.extend(
            _filter_redundant_warning_recommendations(
                recommendations,
                warning_recommendations,
            ),
        )

    deduplicated = _deduplicate_recommendations(recommendations)
    prioritized = sort_and_prioritize(deduplicated)

    if not prioritized:
        warnings.append(_NO_RECOMMENDATIONS_WARNING)

    return AdvisorResult(
        generated_at=create_timestamp(),
        recommendations=prioritized,
        warnings=warnings,
    )


def _filter_redundant_warning_recommendations(
    specific_recommendations: list[Recommendation],
    warning_recommendations: list[Recommendation],
) -> list[Recommendation]:
    covered_warnings = {
        warning
        for recommendation in specific_recommendations
        if recommendation.category != "warning"
        for warning in recommendation.warnings
    }

    return [
        recommendation
        for recommendation in warning_recommendations
        if not any(warning in covered_warnings for warning in recommendation.warnings)
    ]


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
