"""Tests for encoded dataset split helpers."""

from __future__ import annotations

import pytest

from gpuboost.model.training_splits import get_eval_split, split_encoded_dataset
from gpuboost.schemas.training import EncodedTrainingDataset, TrainingDatasetSummary


def test_split_encoded_dataset_groups_rows_by_split() -> None:
    dataset = _encoded_dataset(["train", "validation", "test", "custom"])

    splits = split_encoded_dataset(dataset)

    assert splits["train"]["row_ids"] == ["row-0"]
    assert splits["validation"]["y"] == [1]
    assert splits["test"]["X"] == [[2.0]]
    assert splits["unassigned"]["row_ids"] == ["row-3"]


def test_get_eval_split_prefers_validation_then_test_then_train() -> None:
    splits = split_encoded_dataset(_encoded_dataset(["train", "test"]))

    assert get_eval_split(splits) == "test"
    assert get_eval_split(splits, preferred="train") == "train"


def test_get_eval_split_raises_when_all_splits_empty() -> None:
    splits = split_encoded_dataset(_encoded_dataset([]))

    with pytest.raises(ValueError, match="No non-empty"):
        get_eval_split(splits)


def _encoded_dataset(split: list[str]) -> EncodedTrainingDataset:
    y = [index % 2 for index in range(len(split))]
    return EncodedTrainingDataset(
        row_ids=[f"row-{index}" for index in range(len(split))],
        feature_names=["feature"],
        X=[[float(index)] for index in range(len(split))],
        y=y,
        labels=["improved", "regressed"],
        label_to_index={"improved": 0, "regressed": 1},
        split=split,
        summary=TrainingDatasetSummary(
            row_count=len(split),
            labeled_count=len(split),
            skipped_count=0,
            feature_count=1,
            label_counts={"improved": y.count(0), "regressed": y.count(1)},
            split_counts={name: split.count(name) for name in set(split)},
            warnings=[],
            metadata={},
        ),
        warnings=[],
        metadata={},
    )
