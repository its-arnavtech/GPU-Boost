"""Tests for Phase 8.1 comparison schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from gpuboost.schemas.comparison import (
    BenchmarkMetricDelta,
    ComparisonResult,
    ComparisonSection,
    create_timestamp,
)


def test_benchmark_metric_delta_creation() -> None:
    metric = _make_metric()

    assert metric.name == "amp_speedup_ratio"
    assert metric.unit == "x"
    assert metric.before == 1.0
    assert metric.after == 1.25
    assert metric.absolute_delta == 0.25
    assert metric.percent_delta == 25.0
    assert metric.direction == "improved"
    assert metric.higher_is_better is True
    assert metric.summary == "AMP speedup improved."


def test_comparison_section_creation() -> None:
    metric = _make_metric()
    section = ComparisonSection(
        title="Mixed Precision",
        metrics=[metric],
        verdict="improved",
        warnings=["Synthetic result."],
    )

    assert section.title == "Mixed Precision"
    assert section.metrics == [metric]
    assert section.verdict == "improved"
    assert section.warnings == ["Synthetic result."]


def test_comparison_result_creation() -> None:
    section = ComparisonSection(title="Mixed Precision", verdict="improved")
    result = ComparisonResult(
        generated_at="2026-01-01T00:00:00+00:00",
        status="ok",
        baseline_label="before",
        optimized_label="after",
        sections=[section],
        overall_verdict="improved",
        warnings=[],
        error=None,
    )

    assert result.generated_at == "2026-01-01T00:00:00+00:00"
    assert result.status == "ok"
    assert result.baseline_label == "before"
    assert result.optimized_label == "after"
    assert result.sections == [section]
    assert result.overall_verdict == "improved"
    assert result.warnings == []
    assert result.error is None


def test_to_dict_nesting_works() -> None:
    result = ComparisonResult(
        generated_at="2026-01-01T00:00:00+00:00",
        status="ok",
        baseline_label="baseline",
        optimized_label="optimized",
        sections=[
            ComparisonSection(
                title="Mixed Precision",
                metrics=[_make_metric()],
                verdict="improved",
            )
        ],
        overall_verdict="improved",
    )

    data = result.to_dict()

    assert data["sections"][0]["title"] == "Mixed Precision"
    assert data["sections"][0]["metrics"][0]["name"] == "amp_speedup_ratio"
    assert data["sections"][0]["metrics"][0]["after"] == 1.25
    assert data["sections"][0]["warnings"] == []
    assert data["warnings"] == []
    assert data["error"] is None


def test_json_serialization_works() -> None:
    result = ComparisonResult(
        generated_at="2026-01-01T00:00:00+00:00",
        status="ok",
        baseline_label="baseline",
        optimized_label="optimized",
        sections=[
            ComparisonSection(
                title="Mixed Precision",
                metrics=[_make_metric()],
                verdict="improved",
            )
        ],
        overall_verdict="improved",
    )

    serialized = json.dumps(result.to_dict())
    deserialized = json.loads(serialized)

    assert deserialized["baseline_label"] == "baseline"
    assert deserialized["sections"][0]["metrics"][0]["summary"] == (
        "AMP speedup improved."
    )


def test_default_list_fields_are_isolated_between_instances() -> None:
    first_section = ComparisonSection(title="First", verdict="unknown")
    second_section = ComparisonSection(title="Second", verdict="unknown")
    first_result = ComparisonResult(
        generated_at="2026-01-01T00:00:00+00:00",
        status="partial",
        baseline_label="baseline",
        optimized_label="optimized",
        overall_verdict="unknown",
    )
    second_result = ComparisonResult(
        generated_at="2026-01-01T00:00:01+00:00",
        status="partial",
        baseline_label="baseline",
        optimized_label="optimized",
        overall_verdict="unknown",
    )

    first_section.metrics.append(_make_metric())
    first_section.warnings.append("First warning.")
    first_result.sections.append(first_section)
    first_result.warnings.append("Partial comparison.")

    assert len(first_section.metrics) == 1
    assert second_section.metrics == []
    assert first_section.warnings == ["First warning."]
    assert second_section.warnings == []
    assert first_result.sections == [first_section]
    assert second_result.sections == []
    assert first_result.warnings == ["Partial comparison."]
    assert second_result.warnings == []


def test_create_timestamp_returns_non_empty_utc_iso_string() -> None:
    timestamp = create_timestamp()
    parsed = datetime.fromisoformat(timestamp)

    assert timestamp
    assert parsed.tzinfo == timezone.utc


def test_has_regressions_returns_true_for_section_verdict() -> None:
    result = _make_result(
        ComparisonSection(title="Throughput", verdict="regressed"),
        overall_verdict="regressed",
    )

    assert result.has_regressions() is True


def test_has_regressions_returns_true_for_metric_direction() -> None:
    metric = BenchmarkMetricDelta(
        name="batch_1_median_ms",
        unit="ms",
        before=10.0,
        after=12.0,
        absolute_delta=2.0,
        percent_delta=20.0,
        direction="regressed",
        higher_is_better=False,
        summary="Step time regressed.",
    )
    result = _make_result(
        ComparisonSection(title="Latency", metrics=[metric], verdict="mixed"),
        overall_verdict="mixed",
    )

    assert result.has_regressions() is True


def test_has_regressions_returns_false_when_none_present() -> None:
    result = _make_result(
        ComparisonSection(title="Throughput", verdict="improved"),
        overall_verdict="improved",
    )

    assert result.has_regressions() is False


def test_has_improvements_returns_true_for_section_verdict() -> None:
    result = _make_result(
        ComparisonSection(title="Throughput", verdict="improved"),
        overall_verdict="improved",
    )

    assert result.has_improvements() is True


def test_has_improvements_returns_true_for_metric_direction() -> None:
    result = _make_result(
        ComparisonSection(title="Throughput", metrics=[_make_metric()], verdict="mixed"),
        overall_verdict="mixed",
    )

    assert result.has_improvements() is True


def test_has_improvements_returns_false_when_none_present() -> None:
    result = _make_result(
        ComparisonSection(title="Throughput", verdict="unchanged"),
        overall_verdict="unchanged",
    )

    assert result.has_improvements() is False


def test_error_result_can_carry_error_message() -> None:
    result = ComparisonResult(
        generated_at="2026-01-01T00:00:00+00:00",
        status="error",
        baseline_label="baseline",
        optimized_label="optimized",
        overall_verdict="unknown",
        error="Missing optimized benchmark.",
    )

    assert result.status == "error"
    assert result.error == "Missing optimized benchmark."
    assert result.to_dict()["error"] == "Missing optimized benchmark."


def _make_metric() -> BenchmarkMetricDelta:
    return BenchmarkMetricDelta(
        name="amp_speedup_ratio",
        unit="x",
        before=1.0,
        after=1.25,
        absolute_delta=0.25,
        percent_delta=25.0,
        direction="improved",
        higher_is_better=True,
        summary="AMP speedup improved.",
    )


def _make_result(
    section: ComparisonSection,
    overall_verdict: str,
) -> ComparisonResult:
    return ComparisonResult(
        generated_at="2026-01-01T00:00:00+00:00",
        status="ok",
        baseline_label="baseline",
        optimized_label="optimized",
        sections=[section],
        overall_verdict=overall_verdict,
    )
