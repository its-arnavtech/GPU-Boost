"""Tests for Phase 11.4 dataset validation and privacy checks."""

from __future__ import annotations

from gpuboost.dataset.validation import (
    is_hardware_specs_context,
    is_scalar_safe,
    validate_benchmark_context_rows,
    validate_dataset_rows,
    validate_no_raw_sensitive_fields,
)
from gpuboost.schemas.dataset import (
    BenchmarkContextRow,
    DatasetLabel,
    DatasetPrivacyFlags,
    DatasetRow,
)


def test_valid_dataset_row_passes() -> None:
    report = validate_dataset_rows([_make_row()])

    assert report.status == "passed"
    assert report.valid_row_count == 1
    assert report.invalid_row_count == 0
    assert report.issues == []


def test_unknown_label_creates_warning() -> None:
    row = _make_row(label=DatasetLabel(value="unknown", source="unknown"))

    report = validate_dataset_rows([row])

    assert report.status == "warning"
    assert _has_issue(report, "warning", "unknown_label")


def test_unsafe_privacy_flag_fails() -> None:
    row = _make_row(privacy=DatasetPrivacyFlags(contains_raw_source=True))

    report = validate_dataset_rows([row])

    assert report.status == "failed"
    assert report.invalid_row_count == 1
    assert _has_issue(report, "error", "unsafe_privacy")


def test_invalid_label_fails() -> None:
    row = _make_row(label=DatasetLabel(value="better", source="manual"))

    report = validate_dataset_rows([row])

    assert report.status == "failed"
    assert _has_issue(report, "error", "invalid_label")


def test_invalid_split_fails() -> None:
    row = _make_row(split="holdout")

    report = validate_dataset_rows([row])

    assert report.status == "failed"
    assert _has_issue(report, "error", "invalid_split")


def test_empty_row_id_fails() -> None:
    row = _make_row(row_id="")

    report = validate_dataset_rows([row])

    assert report.status == "failed"
    assert _has_issue(report, "error", "empty_row_id")


def test_nested_non_scalar_feature_fails() -> None:
    row = _make_row(features={"nested": {"bad": True}})

    report = validate_dataset_rows([row])

    assert report.status == "failed"
    assert _has_issue(report, "error", "non_scalar_value")


def test_low_quality_creates_warning() -> None:
    row = _make_row(quality_score=0.3)

    report = validate_dataset_rows([row])

    assert report.status == "warning"
    assert _has_issue(report, "warning", "low_quality_score")


def test_empty_metrics_warning_for_optimization_outcome() -> None:
    row = _make_row(metrics={})

    report = validate_dataset_rows([row])

    assert report.status == "warning"
    assert _has_issue(report, "warning", "empty_metrics")


def test_valid_benchmark_context_row_passes() -> None:
    report = validate_benchmark_context_rows([_make_context_row()])

    assert report.status == "passed"
    assert report.valid_row_count == 1
    assert report.invalid_row_count == 0


def test_benchmark_context_missing_optional_fields_warns() -> None:
    row = _make_context_row(
        hardware_name=None,
        workload_name=None,
        url=None,
    )

    report = validate_benchmark_context_rows([row])

    assert report.status == "warning"
    assert _has_issue(report, "warning", "missing_hardware_name")
    assert _has_issue(report, "warning", "missing_workload_name")
    assert _has_issue(report, "warning", "missing_url")


def test_benchmark_context_empty_metrics_fails() -> None:
    row = _make_context_row(metrics={})

    report = validate_benchmark_context_rows([row])

    assert report.status == "failed"
    assert _has_issue(report, "error", "empty_metrics")


def test_hardware_specs_context_without_metrics_passes_with_warning_only() -> None:
    row = _make_context_row(
        workload_name=None,
        metrics={},
        metadata={
            "context_type": "hardware_specs",
            "source_kind": "gpu_specs",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "memory_size_mb": 24576,
        },
    )

    report = validate_benchmark_context_rows([row])

    assert is_hardware_specs_context(row) is True
    assert report.status == "warning"
    assert not _has_issue(report, "error", "empty_metrics")
    assert _has_issue(report, "warning", "missing_workload_name")


def test_hardware_specs_context_without_hardware_name_fails() -> None:
    row = _make_context_row(
        hardware_name=None,
        metrics={},
        metadata={
            "context_type": "hardware_specs",
            "source_kind": "gpu_specs",
            "gpu_name": "NVIDIA GeForce RTX 4090",
        },
    )

    report = validate_benchmark_context_rows([row])

    assert report.status == "failed"
    assert _has_issue(report, "error", "missing_hardware_name")


def test_hardware_specs_context_without_useful_metadata_fails() -> None:
    row = _make_context_row(
        workload_name=None,
        metrics={},
        metadata={
            "context_type": "hardware_specs",
            "source_kind": "gpu_specs",
        },
    )

    report = validate_benchmark_context_rows([row])

    assert report.status == "failed"
    assert _has_issue(report, "error", "empty_hardware_specs_metadata")


def test_hardware_specs_context_with_non_scalar_metadata_fails() -> None:
    row = _make_context_row(
        metrics={},
        metadata={
            "context_type": "hardware_specs",
            "source_kind": "gpu_specs",
            "gpu_name": {"bad": True},
        },
    )

    report = validate_benchmark_context_rows([row])

    assert report.status == "failed"
    assert _has_issue(report, "error", "non_scalar_value")


def test_sensitive_keys_detected() -> None:
    issues = validate_no_raw_sensitive_fields(
        {"features": {"source_code": "def train(): pass"}},
        row_id="row-001",
    )

    assert any(issue.code == "sensitive_key" for issue in issues)


def test_raw_unified_diff_value_detected() -> None:
    issues = validate_no_raw_sensitive_fields(
        {"patch": "--- a/train.py\n+++ b/train.py\n@@ -1 +1 @@\n-pass\n+ok"},
        row_id="row-001",
    )

    assert any(issue.code == "raw_unified_diff" for issue in issues)


def test_stdout_and_stderr_keys_detected() -> None:
    issues = validate_no_raw_sensitive_fields(
        {"stdout": "output", "nested": {"stderr": "error"}},
        row_id="row-001",
    )

    assert sum(1 for issue in issues if issue.code == "sensitive_key") == 2


def test_report_status_passed_warning_failed() -> None:
    passed = validate_dataset_rows([_make_row()])
    warning = validate_dataset_rows([_make_row(label=DatasetLabel("unknown", "manual"))])
    failed = validate_dataset_rows([_make_row(row_id="")])

    assert passed.status == "passed"
    assert warning.status == "warning"
    assert failed.status == "failed"


def test_scalar_safety_helper() -> None:
    assert is_scalar_safe("x") is True
    assert is_scalar_safe(1) is True
    assert is_scalar_safe(1.5) is True
    assert is_scalar_safe(True) is True
    assert is_scalar_safe(None) is True
    assert is_scalar_safe({}) is False
    assert is_scalar_safe([]) is False
    assert is_scalar_safe(object()) is False


def _make_row(
    row_id: str = "row-001",
    label: DatasetLabel | None = None,
    privacy: DatasetPrivacyFlags | None = None,
    split: str | None = "train",
    quality_score: float | None = 0.9,
    features: dict | None = None,
    metrics: dict | None = None,
) -> DatasetRow:
    return DatasetRow(
        row_id=row_id,
        created_at="2026-01-01T00:00:00+00:00",
        source="gpuboost_history",
        row_type="optimization_outcome",
        hardware={"gpu_name": "NVIDIA A100"},
        workload={"command": "agent optimize"},
        features=features if features is not None else {"action_count": 1},
        metrics=metrics if metrics is not None else {"benchmark_median_ms": 9.5},
        label=label or DatasetLabel(value="improved", source="comparison"),
        privacy=privacy or DatasetPrivacyFlags(),
        split=split,
        quality_score=quality_score,
    )


def _make_context_row(
    hardware_name: str | None = "NVIDIA A100",
    workload_name: str | None = "resnet",
    url: str | None = "https://example.invalid/benchmark",
    metrics: dict | None = None,
    metadata: dict | None = None,
) -> BenchmarkContextRow:
    return BenchmarkContextRow(
        row_id="context-001",
        created_at="2026-01-01T00:00:00+00:00",
        source="mlperf",
        benchmark_name="MLPerf Training",
        workload_name=workload_name,
        hardware_name=hardware_name,
        software_stack={"cuda": "12.4"},
        metrics=metrics if metrics is not None else {"samples_per_sec": 123.4},
        units={"samples_per_sec": "samples/sec"},
        url=url,
        metadata=metadata if metadata is not None else {},
    )


def _has_issue(report, severity: str, code: str) -> bool:
    return any(
        issue.severity == severity and issue.code == code
        for issue in report.issues
    )
