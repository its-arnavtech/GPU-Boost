"""Tests for Phase 11.1 dataset schemas."""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone

from gpuboost.schemas.dataset import (
    BenchmarkContextRow,
    DatasetLabel,
    DatasetManifest,
    DatasetPrivacyFlags,
    DatasetRow,
    DatasetSplitSummary,
    DatasetValidationIssue,
    DatasetValidationReport,
    create_timestamp,
)


def test_dataset_privacy_flags_defaults_safe() -> None:
    privacy = DatasetPrivacyFlags()

    assert privacy.contains_raw_source is False
    assert privacy.contains_raw_diff is False
    assert privacy.contains_stdout is False
    assert privacy.contains_stderr is False
    assert privacy.contains_sensitive_path is False
    assert privacy.notes == []
    assert privacy.is_safe_for_export() is True


def test_dataset_privacy_flags_unsafe_when_raw_flags_true() -> None:
    raw_source = DatasetPrivacyFlags(contains_raw_source=True)
    raw_diff = DatasetPrivacyFlags(contains_raw_diff=True)
    stdout = DatasetPrivacyFlags(contains_stdout=True)
    stderr = DatasetPrivacyFlags(contains_stderr=True)
    sensitive_path = DatasetPrivacyFlags(contains_sensitive_path=True)

    assert raw_source.is_safe_for_export() is False
    assert raw_diff.is_safe_for_export() is False
    assert stdout.is_safe_for_export() is False
    assert stderr.is_safe_for_export() is False
    assert sensitive_path.is_safe_for_export() is False


def test_dataset_label_known_and_unknown() -> None:
    known = DatasetLabel(value="improved", source="comparison", confidence=0.9)
    unknown = DatasetLabel(value="unknown", source="unknown")

    assert known.is_known() is True
    assert unknown.is_known() is False


def test_dataset_row_creation() -> None:
    row = _make_row()

    assert row.row_id == "row-001"
    assert row.created_at == "2026-01-01T00:00:00+00:00"
    assert row.schema_version == "dataset.row.v1"
    assert row.source == "gpuboost_history"
    assert row.row_type == "optimization_outcome"
    assert row.hardware == {"gpu_count": 1, "vram_gb": 24.0}
    assert row.workload == {"framework": "torch"}
    assert row.features == {"batch_size": 32}
    assert row.metrics == {"tokens_per_second": 4200.5}
    assert row.split == "train"
    assert row.quality_score == 0.95


def test_dataset_row_to_dict_nested_label_and_privacy() -> None:
    row = _make_row()

    data = row.to_dict()

    assert data["label"]["value"] == "improved"
    assert data["label"]["source"] == "comparison"
    assert data["privacy"]["contains_raw_source"] is False
    assert data["privacy"]["notes"] == []


def test_dataset_row_json_serialization() -> None:
    row = _make_row()

    serialized = json.dumps(row.to_dict())
    deserialized = json.loads(serialized)

    assert deserialized["row_id"] == "row-001"
    assert deserialized["label"]["confidence"] == 0.87
    assert deserialized["metrics"]["tokens_per_second"] == 4200.5


def test_dataset_row_is_labeled() -> None:
    assert _make_row(label=DatasetLabel(value="improved", source="comparison")).is_labeled()
    assert (
        _make_row(label=DatasetLabel(value="unknown", source="unknown")).is_labeled()
        is False
    )


def test_dataset_row_is_safe_for_export() -> None:
    safe = _make_row()
    unsafe = _make_row(privacy=DatasetPrivacyFlags(contains_raw_diff=True))

    assert safe.is_safe_for_export() is True
    assert unsafe.is_safe_for_export() is False


def test_dataset_row_has_split() -> None:
    with_split = _make_row(split="validation")
    without_split = _make_row(split=None)

    assert with_split.has_split() is True
    assert without_split.has_split() is False


def test_benchmark_context_row_creation_and_to_dict() -> None:
    row = BenchmarkContextRow(
        row_id="benchmark-001",
        created_at="2026-01-01T00:00:00+00:00",
        source="mlperf",
        benchmark_name="MLPerf Training",
        workload_name="resnet",
        hardware_name="NVIDIA H100",
        software_stack={"cuda": "12.4", "driver": "550"},
        metrics={"throughput": 123.4},
        units={"throughput": "samples/sec"},
        url="https://example.invalid/mlperf",
        notes="Schema-only context.",
        metadata={"phase": 11},
    )

    data = row.to_dict()

    assert row.schema_version == "dataset.benchmark_context.v1"
    assert data["benchmark_name"] == "MLPerf Training"
    assert data["software_stack"]["cuda"] == "12.4"
    assert data["units"]["throughput"] == "samples/sec"


def test_dataset_manifest_creation_and_to_dict() -> None:
    manifest = DatasetManifest(
        generated_at="2026-01-01T00:00:00+00:00",
        dataset_name="gpuboost-local",
        dataset_version="0.1.0",
        row_count=3,
        labeled_count=2,
        unlabeled_count=1,
        sources={"gpuboost_history": 3},
        splits={"train": 2, "validation": 1},
        privacy_safe=True,
        warnings=["Synthetic manifest."],
        metadata={"phase": 11},
    )

    data = manifest.to_dict()

    assert manifest.schema_version == "dataset.manifest.v1"
    assert data["dataset_name"] == "gpuboost-local"
    assert data["sources"]["gpuboost_history"] == 3
    assert data["privacy_safe"] is True


def test_dataset_validation_issue_creation_and_to_dict() -> None:
    issue = DatasetValidationIssue(
        severity="warning",
        code="missing_split",
        message="Row has no split assigned.",
        row_id="row-001",
        field="split",
    )

    data = issue.to_dict()

    assert data["severity"] == "warning"
    assert data["code"] == "missing_split"
    assert data["row_id"] == "row-001"
    assert data["field"] == "split"


def test_dataset_validation_report_has_errors_and_warnings() -> None:
    error_report = DatasetValidationReport(
        generated_at="2026-01-01T00:00:00+00:00",
        status="failed",
        row_count=1,
        valid_row_count=0,
        invalid_row_count=1,
        issues=[
            DatasetValidationIssue(
                severity="error",
                code="invalid_label",
                message="Label is invalid.",
            )
        ],
    )
    warning_report = DatasetValidationReport(
        generated_at="2026-01-01T00:00:00+00:00",
        status="warning",
        row_count=1,
        valid_row_count=1,
        invalid_row_count=0,
        issues=[
            DatasetValidationIssue(
                severity="warning",
                code="missing_split",
                message="Split is missing.",
            )
        ],
    )
    warning_list_report = DatasetValidationReport(
        generated_at="2026-01-01T00:00:00+00:00",
        status="warning",
        row_count=1,
        valid_row_count=1,
        invalid_row_count=0,
        warnings=["Dataset is small."],
    )
    clean_report = DatasetValidationReport(
        generated_at="2026-01-01T00:00:00+00:00",
        status="passed",
        row_count=1,
        valid_row_count=1,
        invalid_row_count=0,
    )

    assert error_report.has_errors() is True
    assert error_report.has_warnings() is False
    assert warning_report.has_errors() is False
    assert warning_report.has_warnings() is True
    assert warning_list_report.has_warnings() is True
    assert clean_report.has_errors() is False
    assert clean_report.has_warnings() is False


def test_dataset_split_summary_creation_and_to_dict() -> None:
    summary = DatasetSplitSummary(
        generated_at="2026-01-01T00:00:00+00:00",
        train_count=8,
        validation_count=1,
        test_count=1,
        unassigned_count=0,
        strategy="manual",
        seed=42,
        warnings=["Small validation split."],
        metadata={"phase": 11},
    )

    data = summary.to_dict()

    assert summary.schema_version == "dataset.split.v1"
    assert data["train_count"] == 8
    assert data["strategy"] == "manual"
    assert data["seed"] == 42


def test_default_dict_and_list_fields_are_isolated_between_instances() -> None:
    first_privacy = DatasetPrivacyFlags()
    second_privacy = DatasetPrivacyFlags()
    first_row = _make_minimal_row("row-001")
    second_row = _make_minimal_row("row-002")
    first_context = BenchmarkContextRow(
        row_id="context-001",
        created_at="2026-01-01T00:00:00+00:00",
        source="phoronix",
        benchmark_name="Phoronix Test Suite",
    )
    second_context = BenchmarkContextRow(
        row_id="context-002",
        created_at="2026-01-01T00:00:00+00:00",
        source="rodinia",
        benchmark_name="Rodinia",
    )
    first_report = DatasetValidationReport(
        generated_at="2026-01-01T00:00:00+00:00",
        status="passed",
        row_count=0,
        valid_row_count=0,
        invalid_row_count=0,
    )
    second_report = DatasetValidationReport(
        generated_at="2026-01-01T00:00:00+00:00",
        status="passed",
        row_count=0,
        valid_row_count=0,
        invalid_row_count=0,
    )

    first_privacy.notes.append("First note.")
    first_row.hardware["gpu_count"] = 1
    first_row.warnings.append("First warning.")
    first_context.software_stack["cuda"] = "12.4"
    first_context.units["throughput"] = "items/sec"
    first_report.issues.append(
        DatasetValidationIssue(
            severity="info",
            code="empty",
            message="No rows.",
        )
    )
    first_report.metadata["phase"] = 11

    assert second_privacy.notes == []
    assert second_row.hardware == {}
    assert second_row.warnings == []
    assert second_context.software_stack == {}
    assert second_context.units == {}
    assert second_report.issues == []
    assert second_report.metadata == {}


def test_timestamp_helper_returns_non_empty_utc_iso_string() -> None:
    timestamp = create_timestamp()
    parsed = datetime.fromisoformat(timestamp)

    assert timestamp
    assert parsed.tzinfo == timezone.utc


def test_dataset_schemas_do_not_require_raw_payload_fields() -> None:
    forbidden = {
        "raw_source",
        "source_code",
        "raw_diff",
        "diff",
        "stdout",
        "stderr",
    }
    schema_fields = set()
    for schema in (
        DatasetPrivacyFlags,
        DatasetLabel,
        DatasetRow,
        BenchmarkContextRow,
        DatasetManifest,
        DatasetValidationIssue,
        DatasetValidationReport,
        DatasetSplitSummary,
    ):
        schema_fields.update(field.name for field in fields(schema))

    assert schema_fields.isdisjoint(forbidden)


def _make_row(
    label: DatasetLabel | None = None,
    privacy: DatasetPrivacyFlags | None = None,
    split: str | None = "train",
) -> DatasetRow:
    return DatasetRow(
        row_id="row-001",
        created_at="2026-01-01T00:00:00+00:00",
        source="gpuboost_history",
        row_type="optimization_outcome",
        hardware={"gpu_count": 1, "vram_gb": 24.0},
        workload={"framework": "torch"},
        features={"batch_size": 32},
        metrics={"tokens_per_second": 4200.5},
        label=label
        or DatasetLabel(
            value="improved",
            source="comparison",
            confidence=0.87,
            notes="Benchmark improved.",
        ),
        privacy=privacy or DatasetPrivacyFlags(),
        split=split,
        quality_score=0.95,
        metadata={"phase": 11},
    )


def _make_minimal_row(row_id: str) -> DatasetRow:
    return DatasetRow(
        row_id=row_id,
        created_at="2026-01-01T00:00:00+00:00",
        source="manual",
        row_type="benchmark_context",
        label=DatasetLabel(value="unknown", source="unknown"),
        privacy=DatasetPrivacyFlags(),
    )
