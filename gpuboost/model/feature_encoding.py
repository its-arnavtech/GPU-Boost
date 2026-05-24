"""Deterministic feature and label encoding for Phase 12.1."""

from __future__ import annotations

import hashlib
from typing import Any

from gpuboost.dataset.training_features import (
    build_training_matrix,
    extract_training_label_from_row,
)
from gpuboost.model.training_data import summarize_training_rows
from gpuboost.schemas.dataset import DatasetRow
from gpuboost.schemas.training import EncodedTrainingDataset, TrainingFeatureSpec

ScalarFeatureValue = str | int | float | bool | None


def infer_feature_spec(feature_dicts: list[dict]) -> TrainingFeatureSpec:
    """Infer a stable feature spec from scalar feature dictionaries."""

    values_by_name: dict[str, list[ScalarFeatureValue]] = {}
    for feature_dict in feature_dicts:
        for name, value in feature_dict.items():
            if _is_scalar_feature_value(value):
                values_by_name.setdefault(str(name), []).append(value)

    feature_names = sorted(values_by_name)
    boolean_features: list[str] = []
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for name in feature_names:
        non_null_values = [value for value in values_by_name[name] if value is not None]
        if non_null_values and all(isinstance(value, bool) for value in non_null_values):
            boolean_features.append(name)
        elif non_null_values and all(
            isinstance(value, int | float) and not isinstance(value, bool)
            for value in non_null_values
        ):
            numeric_features.append(name)
        else:
            categorical_features.append(name)

    return TrainingFeatureSpec(
        feature_names=feature_names,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        boolean_features=boolean_features,
    )


def encode_feature_dicts(
    feature_dicts: list[dict],
    feature_spec: TrainingFeatureSpec | None = None,
) -> tuple[list[list[float]], TrainingFeatureSpec]:
    """Encode safe feature dictionaries into a numeric matrix."""

    spec = feature_spec or infer_feature_spec(feature_dicts)
    categorical = set(spec.categorical_features)
    numeric = set(spec.numeric_features)
    boolean = set(spec.boolean_features)

    matrix: list[list[float]] = []
    for feature_dict in feature_dicts:
        row: list[float] = []
        for name in spec.feature_names:
            value = feature_dict.get(name)
            if name in boolean:
                row.append(1.0 if value is True else 0.0)
            elif name in numeric:
                row.append(_numeric_value(value))
            elif name in categorical:
                row.append(_category_bucket(value, spec.unknown_value))
            else:
                row.append(0.0)
        matrix.append(row)

    return matrix, spec


def encode_labels(labels: list[str]) -> tuple[list[int], dict[str, int]]:
    """Encode labels with stable sorted ordering."""

    label_to_index = {label: index for index, label in enumerate(sorted(set(labels)))}
    return [label_to_index[label] for label in labels], label_to_index


def build_encoded_training_dataset(
    rows: list[DatasetRow],
    feature_spec: TrainingFeatureSpec | None = None,
) -> EncodedTrainingDataset:
    """Build an encoded training dataset from DatasetRows using safe features."""

    feature_dicts, labels, row_ids = build_training_matrix(rows)
    split = [
        row.split or "unassigned"
        for row in rows
        if extract_training_label_from_row(row) is not None
    ]
    X, spec = encode_feature_dicts(feature_dicts, feature_spec)
    y, label_to_index = encode_labels(labels)
    summary = summarize_training_rows(rows)

    warnings = list(summary.warnings)
    if not rows:
        warnings.append("No training rows provided.")
    if not labels:
        warnings.append("No labeled training rows available for encoding.")
    if len(label_to_index) < 2:
        warnings.append("Fewer than 2 encoded classes are present.")
    if not spec.feature_names:
        warnings.append("No safe training features were encoded.")

    return EncodedTrainingDataset(
        row_ids=row_ids,
        feature_names=spec.feature_names,
        X=X,
        y=y,
        labels=sorted(label_to_index),
        label_to_index=label_to_index,
        split=split,
        summary=summary,
        feature_spec=spec,
        warnings=sorted(set(warnings)),
        metadata={
            "feature_schema_version": spec.schema_version,
            "source": "safe_training_features",
        },
    )


def _is_scalar_feature_value(value: Any) -> bool:
    return isinstance(value, str | int | float | bool) or value is None


def _numeric_value(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _category_bucket(value: object, unknown_value: str) -> float:
    category = unknown_value if value is None else str(value)
    digest = hashlib.sha256(category.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float((1 << 64) - 1)
