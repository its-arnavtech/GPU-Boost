"""Validation and privacy checks for GPUBoost dataset rows."""

from __future__ import annotations

from typing import Any

from gpuboost.schemas.dataset import (
    BenchmarkContextRow,
    DatasetRow,
    DatasetValidationIssue,
    DatasetValidationReport,
    create_timestamp,
)


VALID_ROW_TYPES = {
    "optimization_outcome",
    "benchmark_context",
    "controlled_experiment",
}
VALID_LABEL_VALUES = {"improved", "regressed", "neutral", "failed", "unknown"}
VALID_SPLITS = {"train", "validation", "test"}
SENSITIVE_KEYS = {
    "raw_source",
    "source_code",
    "raw_diff",
    "unified_diff",
    "stdout",
    "stderr",
    "file_contents",
}


def validate_dataset_rows(rows: list[DatasetRow]) -> DatasetValidationReport:
    """Validate ML-ready dataset rows and return a structured report."""

    issues: list[DatasetValidationIssue] = []
    error_row_ids: set[str | None] = set()

    for row in rows:
        row_issues = _validate_dataset_row(row)
        issues.extend(row_issues)
        if any(issue.severity == "error" for issue in row_issues):
            error_row_ids.add(row.row_id or None)

    return _build_report(
        row_count=len(rows),
        invalid_row_count=len(error_row_ids),
        issues=issues,
    )


def validate_benchmark_context_rows(
    rows: list[BenchmarkContextRow],
) -> DatasetValidationReport:
    """Validate local benchmark context rows and return a structured report."""

    issues: list[DatasetValidationIssue] = []
    error_row_ids: set[str | None] = set()

    for row in rows:
        row_issues = _validate_benchmark_context_row(row)
        issues.extend(row_issues)
        if any(issue.severity == "error" for issue in row_issues):
            error_row_ids.add(row.row_id or None)

    return _build_report(
        row_count=len(rows),
        invalid_row_count=len(error_row_ids),
        issues=issues,
    )


def validate_no_raw_sensitive_fields(
    data: dict,
    row_id: str | None = None,
) -> list[DatasetValidationIssue]:
    """Recursively check data for obvious raw source, diff, or output leaks."""

    issues: list[DatasetValidationIssue] = []
    _inspect_raw_sensitive_value(data, row_id, None, issues)
    return issues


def is_scalar_safe(value: object) -> bool:
    """Return whether a value is safe for dataset scalar fields."""

    return isinstance(value, str | int | float | bool) or value is None


def is_hardware_specs_context(row: BenchmarkContextRow) -> bool:
    """Return whether a benchmark context row is a hardware-specification row."""

    return row.metadata.get("context_type") == "hardware_specs"


def _validate_dataset_row(row: DatasetRow) -> list[DatasetValidationIssue]:
    issues: list[DatasetValidationIssue] = []

    if not row.row_id:
        issues.append(_issue("error", "empty_row_id", "row_id must be non-empty.", row.row_id, "row_id"))
    if not row.source:
        issues.append(_issue("error", "empty_source", "source must be non-empty.", row.row_id, "source"))
    if row.row_type not in VALID_ROW_TYPES:
        issues.append(_issue("error", "invalid_row_type", "row_type is not supported.", row.row_id, "row_type"))
    if row.label.value not in VALID_LABEL_VALUES:
        issues.append(_issue("error", "invalid_label", "label.value is not supported.", row.row_id, "label.value"))
    if not row.label.source:
        issues.append(_issue("error", "empty_label_source", "label.source must be non-empty.", row.row_id, "label.source"))
    if row.split is not None and row.split not in VALID_SPLITS:
        issues.append(_issue("error", "invalid_split", "split is not supported.", row.row_id, "split"))
    if not row.privacy.is_safe_for_export():
        issues.append(_issue("error", "unsafe_privacy", "privacy flags are not safe for export.", row.row_id, "privacy"))

    for section_name, values in (
        ("hardware", row.hardware),
        ("features", row.features),
        ("metrics", row.metrics),
    ):
        issues.extend(_validate_scalar_mapping(values, row.row_id, section_name))
        issues.extend(validate_no_raw_sensitive_fields(values, row.row_id))

    if row.label.value == "unknown":
        issues.append(_issue("warning", "unknown_label", "label is unknown.", row.row_id, "label.value"))
    if row.quality_score is None:
        issues.append(_issue("warning", "missing_quality_score", "quality_score is missing.", row.row_id, "quality_score"))
    elif row.quality_score < 0.5:
        issues.append(_issue("warning", "low_quality_score", "quality_score is below 0.5.", row.row_id, "quality_score"))
    if row.row_type == "optimization_outcome" and not row.metrics:
        issues.append(_issue("warning", "empty_metrics", "optimization_outcome row has empty metrics.", row.row_id, "metrics"))
    if not row.features:
        issues.append(_issue("warning", "empty_features", "features are empty.", row.row_id, "features"))

    return issues


def _validate_benchmark_context_row(
    row: BenchmarkContextRow,
) -> list[DatasetValidationIssue]:
    issues: list[DatasetValidationIssue] = []
    is_hardware_specs = is_hardware_specs_context(row)

    if not row.row_id:
        issues.append(_issue("error", "empty_row_id", "row_id must be non-empty.", row.row_id, "row_id"))
    if not row.source:
        issues.append(_issue("error", "empty_source", "source must be non-empty.", row.row_id, "source"))
    if not row.benchmark_name:
        issues.append(_issue("error", "empty_benchmark_name", "benchmark_name must be non-empty.", row.row_id, "benchmark_name"))
    if not row.metrics and not is_hardware_specs:
        issues.append(_issue("error", "empty_metrics", "metrics must be non-empty.", row.row_id, "metrics"))
    if is_hardware_specs and not row.hardware_name:
        issues.append(_issue("error", "missing_hardware_name", "hardware_name must be non-empty for hardware_specs context.", row.row_id, "hardware_name"))
    if is_hardware_specs and not _has_useful_hardware_specs_metadata(row.metadata):
        issues.append(
            _issue(
                "error",
                "empty_hardware_specs_metadata",
                "hardware_specs context must include at least one useful spec field in metadata.",
                row.row_id,
                "metadata",
            )
        )

    issues.extend(_validate_scalar_mapping(row.metrics, row.row_id, "metrics"))
    issues.extend(_validate_scalar_mapping(row.software_stack, row.row_id, "software_stack"))
    issues.extend(_validate_scalar_mapping(row.metadata, row.row_id, "metadata"))
    issues.extend(validate_no_raw_sensitive_fields(row.metrics, row.row_id))
    issues.extend(validate_no_raw_sensitive_fields(row.software_stack, row.row_id))
    issues.extend(validate_no_raw_sensitive_fields(row.metadata, row.row_id))

    if not row.hardware_name and not is_hardware_specs:
        issues.append(_issue("warning", "missing_hardware_name", "hardware_name is missing.", row.row_id, "hardware_name"))
    if not row.workload_name:
        issues.append(_issue("warning", "missing_workload_name", "workload_name is missing.", row.row_id, "workload_name"))
    if not row.url:
        issues.append(_issue("warning", "missing_url", "url is missing.", row.row_id, "url"))

    return issues


def _validate_scalar_mapping(
    values: dict[str, Any],
    row_id: str | None,
    field_prefix: str,
) -> list[DatasetValidationIssue]:
    issues: list[DatasetValidationIssue] = []
    for key, value in values.items():
        if not is_scalar_safe(value):
            issues.append(
                _issue(
                    "error",
                    "non_scalar_value",
                    "dataset fields must contain only scalar-safe values.",
                    row_id,
                    f"{field_prefix}.{key}",
                )
            )
    return issues


def _has_useful_hardware_specs_metadata(
    metadata: dict[str, Any],
) -> bool:
    ignored_keys = {"context_type", "source_kind"}
    return any(
        key not in ignored_keys and is_scalar_safe(value) and value is not None
        for key, value in metadata.items()
    )


def _inspect_raw_sensitive_value(
    value: Any,
    row_id: str | None,
    field: str | None,
    issues: list[DatasetValidationIssue],
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            next_field = key_text if field is None else f"{field}.{key_text}"
            if key_text.lower() in SENSITIVE_KEYS:
                issues.append(
                    _issue(
                        "error",
                        "sensitive_key",
                        "data contains an unsafe raw or sensitive field key.",
                        row_id,
                        next_field,
                    )
                )
            _inspect_raw_sensitive_value(item, row_id, next_field, issues)
    elif isinstance(value, list | tuple | set):
        for index, item in enumerate(value):
            next_field = f"{field}[{index}]" if field else f"[{index}]"
            _inspect_raw_sensitive_value(item, row_id, next_field, issues)
    elif isinstance(value, str):
        if _looks_like_unified_diff(value):
            issues.append(
                _issue(
                    "error",
                    "raw_unified_diff",
                    "string value appears to contain a raw unified diff.",
                    row_id,
                    field,
                )
            )
        elif _looks_like_python_source(value):
            issues.append(
                _issue(
                    "error",
                    "raw_source_block",
                    "string value appears to contain a raw Python source block.",
                    row_id,
                    field,
                )
            )


def _looks_like_unified_diff(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith("--- ") and "\n+++ " in stripped


def _looks_like_python_source(value: str) -> bool:
    if len(value) < 120 or "\n" not in value:
        return False
    source_markers = ("def ", "class ", "import ", "from ")
    marker_count = sum(1 for marker in source_markers if marker in value)
    return marker_count >= 2 and ("    " in value or "\n\t" in value)


def _build_report(
    row_count: int,
    invalid_row_count: int,
    issues: list[DatasetValidationIssue],
) -> DatasetValidationReport:
    if any(issue.severity == "error" for issue in issues):
        status = "failed"
    elif any(issue.severity == "warning" for issue in issues):
        status = "warning"
    else:
        status = "passed"

    return DatasetValidationReport(
        generated_at=create_timestamp(),
        status=status,
        row_count=row_count,
        valid_row_count=row_count - invalid_row_count,
        invalid_row_count=invalid_row_count,
        issues=issues,
    )


def _issue(
    severity: str,
    code: str,
    message: str,
    row_id: str | None,
    field: str | None,
) -> DatasetValidationIssue:
    return DatasetValidationIssue(
        severity=severity,
        code=code,
        message=message,
        row_id=row_id,
        field=field,
    )
