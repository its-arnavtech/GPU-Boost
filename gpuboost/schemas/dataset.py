"""Dataclass schemas for GPUBoost Phase 11 ML-ready datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


DatasetValue = str | int | float | bool | None


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class DatasetPrivacyFlags:
    """Privacy flags describing whether a row contains raw or sensitive data."""

    contains_raw_source: bool = False
    contains_raw_diff: bool = False
    contains_stdout: bool = False
    contains_stderr: bool = False
    contains_sensitive_path: bool = False
    notes: list[str] = field(default_factory=list)

    def is_safe_for_export(self) -> bool:
        """Return whether the data is safe to export."""

        return not any(
            (
                self.contains_raw_source,
                self.contains_raw_diff,
                self.contains_stdout,
                self.contains_stderr,
                self.contains_sensitive_path,
            )
        )


@dataclass(slots=True)
class DatasetLabel:
    """Label describing the outcome represented by a dataset row."""

    value: str
    source: str
    confidence: float | None = None
    notes: str | None = None

    def is_known(self) -> bool:
        """Return whether this label has a known value."""

        return self.value != "unknown"


@dataclass(slots=True, kw_only=True)
class DatasetRow:
    """Schema-only ML-ready dataset row."""

    row_id: str
    created_at: str
    schema_version: str = "dataset.row.v1"
    source: str
    row_type: str
    hardware: dict[str, DatasetValue] = field(default_factory=dict)
    workload: dict[str, DatasetValue] = field(default_factory=dict)
    features: dict[str, DatasetValue] = field(default_factory=dict)
    metrics: dict[str, DatasetValue] = field(default_factory=dict)
    label: DatasetLabel
    privacy: DatasetPrivacyFlags
    split: str | None = None
    quality_score: float | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, DatasetValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the row as JSON-serializable data."""

        return asdict(self)

    def is_labeled(self) -> bool:
        """Return whether this row has a known label."""

        return self.label.is_known()

    def is_safe_for_export(self) -> bool:
        """Return whether this row is safe to export."""

        return self.privacy.is_safe_for_export()

    def has_split(self) -> bool:
        """Return whether this row has an assigned dataset split."""

        return self.split is not None


@dataclass(slots=True, kw_only=True)
class BenchmarkContextRow:
    """Third-party benchmark context for ML dataset construction."""

    row_id: str
    created_at: str
    schema_version: str = "dataset.benchmark_context.v1"
    source: str
    benchmark_name: str
    workload_name: str | None = None
    hardware_name: str | None = None
    software_stack: dict[str, DatasetValue] = field(default_factory=dict)
    metrics: dict[str, DatasetValue] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    notes: str | None = None
    metadata: dict[str, DatasetValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the context row as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True, kw_only=True)
class DatasetManifest:
    """Schema-only manifest for a generated dataset."""

    generated_at: str
    schema_version: str = "dataset.manifest.v1"
    dataset_name: str
    dataset_version: str
    row_count: int
    labeled_count: int
    unlabeled_count: int
    sources: dict[str, int] = field(default_factory=dict)
    splits: dict[str, int] = field(default_factory=dict)
    privacy_safe: bool
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, DatasetValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class DatasetValidationIssue:
    """One schema-level dataset validation issue."""

    severity: str
    code: str
    message: str
    row_id: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the validation issue as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True, kw_only=True)
class DatasetValidationReport:
    """Schema-only dataset validation report."""

    generated_at: str
    schema_version: str = "dataset.validation.v1"
    status: str
    row_count: int
    valid_row_count: int
    invalid_row_count: int
    issues: list[DatasetValidationIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, DatasetValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the validation report as JSON-serializable data."""

        return asdict(self)

    def has_errors(self) -> bool:
        """Return whether any validation issue is an error."""

        return any(issue.severity == "error" for issue in self.issues)

    def has_warnings(self) -> bool:
        """Return whether the report contains warnings."""

        return bool(self.warnings) or any(
            issue.severity == "warning" for issue in self.issues
        )


@dataclass(slots=True, kw_only=True)
class DatasetSplitSummary:
    """Schema-only summary of dataset split counts."""

    generated_at: str
    schema_version: str = "dataset.split.v1"
    train_count: int
    validation_count: int
    test_count: int
    unassigned_count: int
    strategy: str
    seed: int | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, DatasetValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the split summary as JSON-serializable data."""

        return asdict(self)
