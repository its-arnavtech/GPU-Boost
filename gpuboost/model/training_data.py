"""Training dataset loading and summaries for Phase 12.1."""

from __future__ import annotations

import json
from collections import Counter
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from gpuboost.dataset.training_features import (
    extract_training_features_from_row,
    extract_training_label_from_row,
)
from gpuboost.schemas.dataset import (
    DatasetLabel,
    DatasetPrivacyFlags,
    DatasetRow,
    DatasetValue,
)
from gpuboost.schemas.training import TrainingDatasetSummary


def load_training_rows_jsonl(path: str) -> list[DatasetRow]:
    """Load DatasetRows from a UTF-8 JSONL file."""

    rows: list[DatasetRow] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL in {path} at line {line_number}: {exc.msg}"
            ) from exc
        if isinstance(payload, dict):
            rows.append(_dataset_row_from_dict(payload))
    return rows


def summarize_training_rows(rows: list[DatasetRow]) -> TrainingDatasetSummary:
    """Summarize rows through the safe training feature extraction layer."""

    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    feature_names: set[str] = set()
    labeled_count = 0

    for row in rows:
        label = extract_training_label_from_row(row)
        split_counts[row.split or "unassigned"] += 1
        features = extract_training_features_from_row(row)
        feature_names.update(features)
        if label is None:
            continue
        labeled_count += 1
        label_counts[label] += 1

    warnings: list[str] = []
    if labeled_count == 0:
        warnings.append("No labeled training rows found.")
    if len(label_counts) < 2:
        warnings.append("Fewer than 2 known label classes are present.")
    for split_name in ("train", "validation", "test"):
        if split_counts.get(split_name, 0) == 0:
            warnings.append(f"No {split_name} split rows found.")

    return TrainingDatasetSummary(
        row_count=len(rows),
        labeled_count=labeled_count,
        skipped_count=len(rows) - labeled_count,
        feature_count=len(feature_names),
        label_counts=dict(sorted(label_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
        warnings=warnings,
        metadata={"source": "safe_training_features"},
    )


def load_default_training_rows(
    dataset_path: str = "data/gpuboost/generated/training_dataset.jsonl",
) -> list[DatasetRow]:
    """Load default generated training rows when present."""

    if not Path(dataset_path).exists():
        return []
    return load_training_rows_jsonl(dataset_path)


def _dataset_row_from_dict(data: dict[str, Any]) -> DatasetRow:
    label_data = data.get("label") if isinstance(data.get("label"), dict) else {}
    privacy_data = data.get("privacy") if isinstance(data.get("privacy"), dict) else {}

    return DatasetRow(
        row_id=str(data.get("row_id") or ""),
        created_at=str(data.get("created_at") or ""),
        schema_version=str(data.get("schema_version") or "dataset.row.v1"),
        source=str(data.get("source") or ""),
        row_type=str(data.get("row_type") or ""),
        hardware=_safe_dataset_scalar_dict(data.get("hardware")),
        workload=_safe_dataset_scalar_dict(data.get("workload")),
        features=_safe_dataset_scalar_dict(data.get("features")),
        metrics=_safe_dataset_scalar_dict(data.get("metrics")),
        label=DatasetLabel(
            value=str(label_data.get("value") or "unknown"),
            source=str(label_data.get("source") or "unknown"),
            confidence=(
                label_data.get("confidence")
                if isinstance(label_data.get("confidence"), int | float)
                else None
            ),
            notes=str(label_data.get("notes"))
            if label_data.get("notes") is not None
            else None,
        ),
        privacy=DatasetPrivacyFlags(
            contains_raw_source=bool(privacy_data.get("contains_raw_source", False)),
            contains_raw_diff=bool(privacy_data.get("contains_raw_diff", False)),
            contains_stdout=bool(privacy_data.get("contains_stdout", False)),
            contains_stderr=bool(privacy_data.get("contains_stderr", False)),
            contains_sensitive_path=bool(
                privacy_data.get("contains_sensitive_path", False)
            ),
            notes=[
                str(item)
                for item in privacy_data.get("notes", [])
                if isinstance(item, str)
            ],
        ),
        split=str(data.get("split")) if data.get("split") is not None else None,
        quality_score=data.get("quality_score")
        if isinstance(data.get("quality_score"), int | float)
        else None,
        warnings=[str(item) for item in data.get("warnings", []) if isinstance(item, str)],
        metadata=_safe_dataset_scalar_dict(data.get("metadata")),
    )


def _safe_dataset_scalar_dict(value: object) -> dict[str, DatasetValue]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, str | int | float | bool) or item is None
    }
