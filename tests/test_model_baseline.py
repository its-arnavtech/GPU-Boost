"""Tests for Phase 12.1 majority baseline scaffold."""

from __future__ import annotations

import pytest

from gpuboost.model.baseline import (
    MajorityClassBaseline,
    NearestCentroidBaseline,
    RandomBaseline,
    SimpleKNNBaseline,
    train_majority_baseline,
)
from gpuboost.schemas.training import EncodedTrainingDataset, TrainingDatasetSummary


def test_majority_baseline_fit_predict_and_tie_handling() -> None:
    model = MajorityClassBaseline().fit([[0.0], [1.0]], [1, 0])

    assert model.majority_class == 0
    assert model.predict([[0.0], [5.0], [9.0]]) == [0, 0, 0]
    assert model.to_dict()["majority_class"] == 0


def test_majority_baseline_fit_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="empty labels"):
        MajorityClassBaseline().fit([], [])


def test_random_baseline_predictions_are_seeded_and_repeatable() -> None:
    model = RandomBaseline(seed=7).fit([[0.0], [1.0], [2.0]], [1, 0, 1])

    first = model.predict([[10.0], [11.0], [12.0], [13.0]])
    second = model.predict([[10.0], [11.0], [12.0], [13.0]])

    assert first == second
    assert model.class_labels == [0, 1]
    assert model.to_dict()["seed"] == 7


def test_random_baseline_fit_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="empty labels"):
        RandomBaseline().fit([], [])


def test_nearest_centroid_baseline_learns_separable_data() -> None:
    model = NearestCentroidBaseline().fit(
        [[0.0, 0.0], [0.2, 0.0], [10.0, 10.0], [10.2, 10.0]],
        [0, 0, 1, 1],
    )

    assert model.predict([[0.1, 0.0], [10.1, 10.0]]) == [0, 1]
    assert model.to_dict()["class_labels"] == [0, 1]


def test_nearest_centroid_baseline_uses_lowest_label_tie_breaker() -> None:
    model = NearestCentroidBaseline().fit([[0.0], [2.0]], [0, 1])

    assert model.predict([[1.0]]) == [0]


def test_nearest_centroid_baseline_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="empty features"):
        NearestCentroidBaseline().fit([], [])


def test_simple_knn_baseline_predicts_by_nearest_votes() -> None:
    model = SimpleKNNBaseline(k=3).fit(
        [[0.0], [0.2], [8.0], [9.0], [10.0]],
        [0, 0, 1, 1, 1],
    )

    assert model.predict([[0.1], [9.2]]) == [0, 1]
    assert model.to_dict()["train_count"] == 5


def test_train_majority_baseline_evaluates_validation_split() -> None:
    dataset = _encoded_dataset(split=["train", "train", "validation"], y=[0, 0, 1])

    result = train_majority_baseline(dataset)

    assert result.status == "ok"
    assert result.metadata["eval_split"] == "validation"
    assert result.metadata["train_count"] == 2
    assert result.accuracy == 0.0


def test_train_majority_baseline_falls_back_to_test_split() -> None:
    dataset = _encoded_dataset(split=["train", "train", "test"], y=[0, 1, 0])

    result = train_majority_baseline(dataset)

    assert result.status == "ok"
    assert result.metadata["eval_split"] == "test"
    assert result.accuracy == 1.0


def test_train_majority_baseline_handles_missing_eval_split() -> None:
    dataset = _encoded_dataset(split=["train", "train"], y=[0, 1])

    result = train_majority_baseline(dataset)

    assert result.status == "error"
    assert "evaluation split" in result.warnings[0]


def _encoded_dataset(split: list[str], y: list[int]) -> EncodedTrainingDataset:
    return EncodedTrainingDataset(
        row_ids=[f"row-{index}" for index in range(len(y))],
        feature_names=["feature"],
        X=[[float(index)] for index in range(len(y))],
        y=y,
        labels=["improved", "regressed"],
        label_to_index={"improved": 0, "regressed": 1},
        split=split,
        summary=TrainingDatasetSummary(
            row_count=len(y),
            labeled_count=len(y),
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
