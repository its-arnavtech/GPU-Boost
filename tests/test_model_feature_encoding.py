"""Tests for Phase 12.1 feature encoding."""

from __future__ import annotations

from gpuboost.model.feature_encoding import (
    build_encoded_training_dataset,
    encode_feature_dicts,
    encode_labels,
    infer_feature_spec,
)
from gpuboost.schemas.dataset import DatasetLabel, DatasetPrivacyFlags, DatasetRow


def test_feature_spec_inference_is_deterministic() -> None:
    feature_dicts = [
        {"z_cat": "amp", "a_num": 1, "b_bool": True, "ignored": ["x"]},
        {"z_cat": "batch", "a_num": 2.5, "b_bool": False},
    ]

    first = infer_feature_spec(feature_dicts)
    second = infer_feature_spec(list(reversed(feature_dicts)))

    assert first.feature_names == ["a_num", "b_bool", "z_cat"]
    assert first.to_dict() == second.to_dict()
    assert first.numeric_features == ["a_num"]
    assert first.boolean_features == ["b_bool"]
    assert first.categorical_features == ["z_cat"]


def test_encoding_numeric_bool_categorical_and_missing_values() -> None:
    feature_dicts = [
        {"num": 2, "flag": True, "cat": "amp"},
        {"num": None, "flag": False, "cat": None},
        {"flag": True},
    ]

    X, spec = encode_feature_dicts(feature_dicts)
    repeated, _ = encode_feature_dicts(feature_dicts, spec)

    assert spec.feature_names == ["cat", "flag", "num"]
    assert X == repeated
    assert X[0][1:] == [1.0, 2.0]
    assert X[1][1:] == [0.0, 0.0]
    assert X[2][1:] == [1.0, 0.0]
    assert 0.0 <= X[0][0] <= 1.0
    assert 0.0 <= X[1][0] <= 1.0


def test_label_encoding_is_stable() -> None:
    encoded, mapping = encode_labels(["regressed", "improved", "improved"])

    assert mapping == {"improved": 0, "regressed": 1}
    assert encoded == [1, 0, 0]


def test_encoded_dataset_skips_unknown_labels_and_excludes_unsafe_fields() -> None:
    rows = [
        _make_row(
            "row-1",
            split="train",
            features={
                "workload_family": "amp",
                "overall_verdict": "improved",
                "raw_source": "print('secret')",
                "stdout": "noise",
            },
            metrics={
                "fp32_samples_per_sec": 100.0,
                "before_best_images_per_sec": 90.0,
                "after_best_images_per_sec": 110.0,
                "delta_best_images_per_sec": 20.0,
            },
        ),
        _make_row(
            "row-2",
            label=DatasetLabel(value="unknown", source="unknown"),
            split="test",
            features={"workload_family": "batch", "stderr": "noise"},
        ),
        _make_row(
            "row-3",
            label=DatasetLabel(value="regressed", source="comparison"),
            split="validation",
            features={"workload_family": "batch", "raw_diff": "--- a"},
        ),
    ]

    dataset = build_encoded_training_dataset(rows)

    assert dataset.row_ids == ["row-1", "row-3"]
    assert dataset.split == ["train", "validation"]
    assert dataset.labels == ["improved", "regressed"]
    forbidden_parts = (
        "overall_verdict",
        "raw_source",
        "raw_diff",
        "stdout",
        "stderr",
        "before_",
        "after_",
        "delta_",
    )
    assert all(
        not any(part in feature_name for part in forbidden_parts)
        for feature_name in dataset.feature_names
    )


def _make_row(
    row_id: str,
    *,
    label: DatasetLabel | None = None,
    split: str | None = None,
    features: dict | None = None,
    metrics: dict | None = None,
) -> DatasetRow:
    return DatasetRow(
        row_id=row_id,
        created_at="2026-01-01T00:00:00+00:00",
        source="controlled_experiment",
        row_type="optimization_outcome",
        hardware={"gpu_name": "Test GPU"},
        workload={"batch_size": 32},
        features=features or {},
        metrics=metrics or {},
        label=label or DatasetLabel(value="improved", source="comparison"),
        privacy=DatasetPrivacyFlags(),
        split=split,
    )
