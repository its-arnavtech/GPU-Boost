"""Tests for Phase 4 code analysis schemas."""

import json
from datetime import datetime, timezone

from gpuboost.schemas.code_analysis import (
    CodeAnalysisResult,
    CodeFinding,
    create_timestamp,
)


def test_code_finding_creation() -> None:
    finding = CodeFinding(
        id="dataloader-workers",
        title="Increase DataLoader workers",
        category="dataloader",
        severity="warning",
        confidence="high",
        filepath="train.py",
        line=12,
        column=20,
        end_line=12,
        end_column=32,
        summary="The DataLoader uses the default worker count.",
        rationale="Single-process data loading can starve the GPU.",
        suggested_action="Set num_workers based on CPU availability.",
        code_snippet="DataLoader(dataset)",
        related_recommendation_ids=["dataloader-num-workers"],
        tags=["pytorch", "input-pipeline"],
    )

    assert finding.id == "dataloader-workers"
    assert finding.category == "dataloader"
    assert finding.severity == "warning"
    assert finding.related_recommendation_ids == ["dataloader-num-workers"]
    assert finding.tags == ["pytorch", "input-pipeline"]


def test_code_analysis_result_creation() -> None:
    finding = _make_finding()
    result = CodeAnalysisResult(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        findings=[finding],
        warnings=["Analysis is experimental."],
        error=None,
    )

    assert result.generated_at == "2026-01-01T00:00:00+00:00"
    assert result.filepath == "train.py"
    assert result.status == "ok"
    assert result.findings == [finding]
    assert result.warnings == ["Analysis is experimental."]
    assert result.error is None


def test_to_dict_output() -> None:
    result = CodeAnalysisResult(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        findings=[_make_finding()],
    )

    data = result.to_dict()

    assert data["filepath"] == "train.py"
    assert data["status"] == "ok"
    assert data["findings"][0]["id"] == "mixed-precision-context"
    assert data["findings"][0]["line"] == 24
    assert data["findings"][0]["related_recommendation_ids"] == [
        "mixed-precision",
    ]
    assert data["warnings"] == []
    assert data["error"] is None


def test_json_serialization() -> None:
    result = CodeAnalysisResult(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        findings=[_make_finding()],
    )

    serialized = json.dumps(result.to_dict())
    deserialized = json.loads(serialized)

    assert deserialized["findings"][0]["title"] == "Use mixed precision"
    assert deserialized["findings"][0]["tags"] == ["amp", "training"]


def test_default_empty_lists_are_independent() -> None:
    first_finding = CodeFinding(
        id="first",
        title="First",
        category="general",
        severity="info",
        confidence="low",
        filepath="first.py",
        line=None,
        column=None,
        end_line=None,
        end_column=None,
        summary="First summary.",
        rationale="First rationale.",
        suggested_action="Review the code.",
        code_snippet=None,
    )
    second_finding = CodeFinding(
        id="second",
        title="Second",
        category="general",
        severity="info",
        confidence="low",
        filepath="second.py",
        line=None,
        column=None,
        end_line=None,
        end_column=None,
        summary="Second summary.",
        rationale="Second rationale.",
        suggested_action="Review the code.",
        code_snippet=None,
    )
    first_result = CodeAnalysisResult(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="first.py",
        status="ok",
    )
    second_result = CodeAnalysisResult(
        generated_at="2026-01-01T00:00:01+00:00",
        filepath="second.py",
        status="ok",
    )

    first_finding.related_recommendation_ids.append("first-only")
    first_finding.tags.append("first-only")
    first_result.findings.append(first_finding)
    first_result.warnings.append("first only")

    assert first_finding.related_recommendation_ids == ["first-only"]
    assert first_finding.tags == ["first-only"]
    assert second_finding.related_recommendation_ids == []
    assert second_finding.tags == []
    assert first_result.findings == [first_finding]
    assert first_result.warnings == ["first only"]
    assert second_result.findings == []
    assert second_result.warnings == []


def test_optional_line_column_fields_can_be_none() -> None:
    finding = CodeFinding(
        id="file-level",
        title="File-level finding",
        category="general",
        severity="info",
        confidence="medium",
        filepath="train.py",
        line=None,
        column=None,
        end_line=None,
        end_column=None,
        summary="This finding applies to the whole file.",
        rationale="No exact source span is available.",
        suggested_action="Review the file-level guidance.",
        code_snippet=None,
    )

    data = finding.to_dict()

    assert finding.line is None
    assert finding.column is None
    assert finding.end_line is None
    assert finding.end_column is None
    assert data["line"] is None
    assert data["column"] is None


def test_create_timestamp_returns_utc_iso_timestamp() -> None:
    timestamp = create_timestamp()
    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo == timezone.utc


def _make_finding() -> CodeFinding:
    return CodeFinding(
        id="mixed-precision-context",
        title="Use mixed precision",
        category="mixed_precision",
        severity="info",
        confidence="medium",
        filepath="train.py",
        line=24,
        column=8,
        end_line=24,
        end_column=32,
        summary="The training loop does not appear to use autocast.",
        rationale="Mixed precision can improve throughput on tensor core GPUs.",
        suggested_action="Wrap forward and loss computation in autocast.",
        code_snippet=None,
        related_recommendation_ids=["mixed-precision"],
        tags=["amp", "training"],
    )
