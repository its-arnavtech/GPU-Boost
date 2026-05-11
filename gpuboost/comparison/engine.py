"""Before/after benchmark comparison engine for GPUBoost Phase 8."""

from __future__ import annotations

from typing import Any

from gpuboost.comparison.metrics import (
    DEFAULT_COMPARISON_METRICS,
    iter_benchmark_metrics,
    metric_higher_is_better,
)
from gpuboost.schemas.comparison import (
    BenchmarkMetricDelta,
    ComparisonResult,
    ComparisonSection,
    ComparisonMetricValue,
    create_timestamp,
)


ComparableMetric = tuple[str, Any, Any, str | None]


def compare_benchmarks(
    baseline: dict,
    optimized: dict,
    metric_names: list[str] | None = None,
    baseline_label: str = "baseline",
    optimized_label: str = "optimized",
    tolerance_pct: float = 1.0,
) -> ComparisonResult:
    """Compare two benchmark result dictionaries."""

    names = metric_names or DEFAULT_COMPARISON_METRICS
    warnings = _missing_metric_warnings(baseline, optimized, names)
    grouped = group_comparable_metrics(baseline, optimized, names)

    sections = []
    for title, comparable_metrics in grouped.items():
        metrics = [
            calculate_metric_delta(
                metric_name=name,
                before=before,
                after=after,
                unit=unit,
                tolerance_pct=tolerance_pct,
            )
            for name, before, after, unit in comparable_metrics
        ]
        sections.append(
            ComparisonSection(
                title=title,
                metrics=metrics,
                verdict=section_verdict(metrics),
            )
        )

    if sections:
        status = "partial" if warnings else "ok"
        error = None
    else:
        status = "error"
        error = "No comparable metrics were found."
        warnings.append(error)

    return ComparisonResult(
        generated_at=create_timestamp(),
        status=status,
        baseline_label=baseline_label,
        optimized_label=optimized_label,
        sections=sections,
        overall_verdict=overall_verdict(sections),
        warnings=warnings,
        error=error,
    )


def calculate_metric_delta(
    metric_name: str,
    before: ComparisonMetricValue,
    after: ComparisonMetricValue,
    unit: str | None = None,
    tolerance_pct: float = 1.0,
) -> BenchmarkMetricDelta:
    """Calculate a structured before/after delta for one metric."""

    higher_is_better = metric_higher_is_better(metric_name)
    absolute_delta = None
    percent_delta = None

    if _is_numeric(before) and _is_numeric(after):
        absolute_delta = float(after) - float(before)
        if float(before) != 0.0:
            percent_delta = (absolute_delta / abs(float(before))) * 100.0

        direction = _numeric_direction(
            absolute_delta=absolute_delta,
            percent_delta=percent_delta,
            higher_is_better=higher_is_better,
            tolerance_pct=tolerance_pct,
        )
    elif before is None or after is None:
        direction = "unchanged" if before is None and after is None else "unknown"
    else:
        direction = "unchanged" if before == after else "unknown"

    return BenchmarkMetricDelta(
        name=metric_name,
        unit=unit,
        before=before,
        after=after,
        absolute_delta=absolute_delta,
        percent_delta=percent_delta,
        direction=direction,
        higher_is_better=higher_is_better,
        summary=_metric_summary(
            metric_name=metric_name,
            before=before,
            after=after,
            unit=unit,
            direction=direction,
            percent_delta=percent_delta,
        ),
    )


def section_verdict(metrics: list[BenchmarkMetricDelta]) -> str:
    """Return the verdict for a section of metric deltas."""

    if not metrics:
        return "unknown"
    return _verdict_from_directions([metric.direction for metric in metrics])


def overall_verdict(sections: list[ComparisonSection]) -> str:
    """Return the overall verdict for comparison sections."""

    if not sections:
        return "unknown"
    return _verdict_from_directions([section.verdict for section in sections])


def group_comparable_metrics(
    baseline: dict,
    optimized: dict,
    metric_names: list[str],
) -> dict[str, list[ComparableMetric]]:
    """Group comparable metric values by benchmark section title."""

    baseline_records = iter_benchmark_metrics(baseline)
    optimized_records = iter_benchmark_metrics(optimized)
    grouped: dict[str, list[ComparableMetric]] = {}

    for metric_name in metric_names:
        baseline_matches = _records_for_metric(baseline_records, metric_name)
        optimized_matches = _records_for_metric(optimized_records, metric_name)
        if not baseline_matches or not optimized_matches:
            continue

        optimized_used: set[int] = set()
        for baseline_record in baseline_matches:
            optimized_index = _find_optimized_match(
                baseline_record=baseline_record,
                optimized_matches=optimized_matches,
                optimized_used=optimized_used,
            )
            if optimized_index is None:
                continue

            optimized_used.add(optimized_index)
            optimized_record = optimized_matches[optimized_index]
            title = _section_title(baseline_record.get("benchmark_name"))
            unit = baseline_record.get("unit")
            if unit is None:
                unit = optimized_record.get("unit")
            if not isinstance(unit, str):
                unit = None

            grouped.setdefault(title, []).append(
                (
                    metric_name,
                    baseline_record.get("value"),
                    optimized_record.get("value"),
                    unit,
                )
            )

    return grouped


def _missing_metric_warnings(
    baseline: dict,
    optimized: dict,
    metric_names: list[str],
) -> list[str]:
    baseline_metrics = {
        record["metric_name"] for record in iter_benchmark_metrics(baseline)
    }
    optimized_metrics = {
        record["metric_name"] for record in iter_benchmark_metrics(optimized)
    }

    warnings = []
    for metric_name in metric_names:
        if metric_name not in baseline_metrics:
            warnings.append(f"Metric missing from baseline result: {metric_name}")
        if metric_name not in optimized_metrics:
            warnings.append(f"Metric missing from optimized result: {metric_name}")
    return warnings


def _records_for_metric(records: list[dict], metric_name: str) -> list[dict]:
    return [record for record in records if record["metric_name"] == metric_name]


def _find_optimized_match(
    baseline_record: dict,
    optimized_matches: list[dict],
    optimized_used: set[int],
) -> int | None:
    baseline_name = baseline_record.get("benchmark_name")

    for index, optimized_record in enumerate(optimized_matches):
        if index in optimized_used:
            continue
        if optimized_record.get("benchmark_name") == baseline_name:
            return index

    for index in range(len(optimized_matches)):
        if index not in optimized_used:
            return index

    return None


def _numeric_direction(
    absolute_delta: float,
    percent_delta: float | None,
    higher_is_better: bool,
    tolerance_pct: float,
) -> str:
    if percent_delta is not None:
        if abs(percent_delta) <= tolerance_pct:
            return "unchanged"
        improved = percent_delta > 0 if higher_is_better else percent_delta < 0
        return "improved" if improved else "regressed"

    if absolute_delta == 0:
        return "unchanged"

    improved = absolute_delta > 0 if higher_is_better else absolute_delta < 0
    return "improved" if improved else "regressed"


def _verdict_from_directions(directions: list[str]) -> str:
    has_improved = "improved" in directions
    has_regressed = "regressed" in directions

    if has_improved and not has_regressed:
        return "improved"
    if has_regressed and not has_improved:
        return "regressed"
    if has_improved and has_regressed:
        return "mixed"
    if directions and all(direction == "unchanged" for direction in directions):
        return "unchanged"
    return "unknown"


def _is_numeric(value: ComparisonMetricValue) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _section_title(benchmark_name: Any) -> str:
    if isinstance(benchmark_name, str) and benchmark_name:
        return benchmark_name
    return "Benchmark Metrics"


def _metric_summary(
    metric_name: str,
    before: ComparisonMetricValue,
    after: ComparisonMetricValue,
    unit: str | None,
    direction: str,
    percent_delta: float | None,
) -> str:
    before_text = _format_value(before, unit)
    after_text = _format_value(after, unit)

    if direction in {"improved", "regressed"}:
        percent_text = ""
        if percent_delta is not None:
            percent_text = f" ({percent_delta:+.2f}%)."
        else:
            percent_text = "."
        return f"{metric_name} {direction} from {before_text} to {after_text}{percent_text}"

    if direction == "unchanged":
        return f"{metric_name} was unchanged at {after_text}."

    return f"{metric_name} changed from {before_text} to {after_text}."


def _format_value(value: ComparisonMetricValue, unit: str | None) -> str:
    if isinstance(value, float):
        text = f"{value:.2f}"
    else:
        text = str(value)

    if unit:
        return f"{text}{unit}"
    return text
