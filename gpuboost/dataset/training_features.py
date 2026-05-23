"""Safe Phase 12 training feature extraction for dataset rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from gpuboost.schemas.dataset import DatasetRow, DatasetValue


TARGET_DERIVED_EXACT_NAMES = {
    "label",
    "label_value",
    "overall_verdict",
    "target",
    "target_label",
    "improved_metric_count",
    "regressed_metric_count",
    "unchanged_metric_count",
}
TARGET_DERIVED_PREFIXES = (
    "after_",
    "before_",
    "comparison_",
    "delta_",
    "percent_delta_",
)
UNSAFE_NAME_PARTS = (
    "api_key",
    "credential",
    "file_contents",
    "password",
    "raw_diff",
    "raw_source",
    "secret",
    "source_code",
    "stderr",
    "stdout",
    "token",
    "unified_diff",
)
IDENTIFIER_EXACT_NAMES = {
    "absolute_path",
    "created_at",
    "local_path",
    "row_id",
    "script_path",
    "split",
}


def is_target_derived_feature_name(name: str) -> bool:
    """Return whether a feature name is derived from the label target."""

    normalized = _normalize_name(name)
    basename = _basename(normalized)
    if basename in TARGET_DERIVED_EXACT_NAMES:
        return True
    if basename.endswith("_overall_verdict") or basename.endswith("_label"):
        return True
    return basename.startswith(TARGET_DERIVED_PREFIXES)


def is_safe_training_feature_name(name: str) -> bool:
    """Return whether a field name is safe for Phase 12 training inputs."""

    normalized = _normalize_name(name)
    basename = _basename(normalized)
    if not normalized or not basename:
        return False
    if is_target_derived_feature_name(normalized):
        return False
    if basename in IDENTIFIER_EXACT_NAMES:
        return False
    return not any(part in basename for part in UNSAFE_NAME_PARTS)


def extract_training_features_from_row(row: DatasetRow) -> dict[str, DatasetValue]:
    """Return safe scalar features for Phase 12 training without mutating row."""

    features: dict[str, DatasetValue] = {}
    for namespace, values in (
        ("hardware", row.hardware),
        ("workload", row.workload),
        ("features", row.features),
        ("metrics", row.metrics),
        ("metadata", row.metadata),
    ):
        for key, value in values.items():
            namespaced_key = f"{namespace}.{key}"
            if is_safe_training_feature_name(namespaced_key) and _is_scalar_safe(value):
                features[namespaced_key] = value
    return features


def extract_training_label_from_row(row: DatasetRow) -> str | None:
    """Return the known training label for a row, or None for unknown labels."""

    if not row.label.is_known():
        return None
    return row.label.value


def build_training_matrix(
    rows: list[DatasetRow],
) -> tuple[list[dict[str, DatasetValue]], list[str], list[str]]:
    """Build aligned feature, label, and row-id lists for known-label rows."""

    feature_rows: list[dict[str, DatasetValue]] = []
    labels: list[str] = []
    row_ids: list[str] = []
    for row in rows:
        label = extract_training_label_from_row(row)
        if label is None:
            continue
        feature_rows.append(extract_training_features_from_row(row))
        labels.append(label)
        row_ids.append(row.row_id)
    return feature_rows, labels, row_ids


def audit_training_feature_leakage(rows: list[DatasetRow]) -> dict[str, Any]:
    """Audit original rows and extracted features for Phase 12 leakage risk."""

    excluded_fields: dict[str, list[str]] = {}
    leaked_fields: dict[str, list[str]] = {}
    trainable_row_count = 0

    for row in rows:
        label = extract_training_label_from_row(row)
        if label is not None:
            trainable_row_count += 1

        row_excluded = _excluded_original_fields(row)
        if row_excluded:
            excluded_fields[row.row_id] = row_excluded

        leaked = [
            name
            for name in extract_training_features_from_row(row)
            if not is_safe_training_feature_name(name)
        ]
        if leaked:
            leaked_fields[row.row_id] = leaked

    leaked_feature_count = sum(len(fields) for fields in leaked_fields.values())
    return {
        "status": "passed" if leaked_feature_count == 0 else "failed",
        "row_count": len(rows),
        "trainable_row_count": trainable_row_count,
        "leaked_feature_count": leaked_feature_count,
        "leaked_fields": leaked_fields,
        "excluded_fields": excluded_fields,
    }


def _excluded_original_fields(row: DatasetRow) -> list[str]:
    fields: list[str] = []
    for namespace, values in (
        ("hardware", row.hardware),
        ("workload", row.workload),
        ("features", row.features),
        ("metrics", row.metrics),
        ("metadata", row.metadata),
    ):
        for key, value in values.items():
            namespaced_key = f"{namespace}.{key}"
            if (
                not is_safe_training_feature_name(namespaced_key)
                or not _is_scalar_safe(value)
            ):
                fields.append(namespaced_key)

    extra_exclusions = defaultdict(list)
    extra_exclusions[row.row_id].extend(
        ("row_id", "created_at", "label", "split", "privacy")
    )
    fields.extend(extra_exclusions[row.row_id])
    return sorted(set(fields))


def _is_scalar_safe(value: object) -> bool:
    return isinstance(value, str | int | float | bool) or value is None


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _basename(name: str) -> str:
    return name.rsplit(".", 1)[-1]
