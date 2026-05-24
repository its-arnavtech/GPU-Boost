"""Dependency-free baseline models for Phase 12."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Any

from gpuboost.model.evaluation import evaluate_predictions
from gpuboost.schemas.training import EncodedTrainingDataset, TrainingEvaluationResult


@dataclass(slots=True)
class MajorityClassBaseline:
    """A deterministic majority-class sanity baseline."""

    majority_class: int | None = None

    def fit(self, X: list[list[float]], y: list[int]) -> MajorityClassBaseline:
        """Fit the baseline by storing the majority class."""

        if not y:
            raise ValueError("Cannot fit MajorityClassBaseline with empty labels.")
        counts = Counter(y)
        self.majority_class = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            0
        ][0]
        return self

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict the stored majority class for every row."""

        if self.majority_class is None:
            raise ValueError("MajorityClassBaseline must be fit before prediction.")
        return [self.majority_class for _ in X]

    def to_dict(self) -> dict[str, Any]:
        """Return the baseline state as JSON-serializable data."""

        return {
            "model_name": "majority_class_baseline",
            "majority_class": self.majority_class,
        }


@dataclass(slots=True)
class RandomBaseline:
    """A deterministic random-class baseline for sanity checks."""

    seed: int = 42
    class_labels: list[int] | None = None

    def fit(self, X: list[list[float]], y: list[int]) -> RandomBaseline:
        """Fit by storing the observed sorted class labels."""

        if not y:
            raise ValueError("Cannot fit RandomBaseline with empty labels.")
        self.class_labels = sorted(set(y))
        return self

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict a deterministic random label sequence."""

        if not self.class_labels:
            raise ValueError("RandomBaseline must be fit before prediction.")
        rng = random.Random(self.seed)
        return [rng.choice(self.class_labels) for _ in X]

    def to_dict(self) -> dict[str, Any]:
        """Return the baseline state as JSON-serializable data."""

        return {
            "model_name": "random_baseline",
            "seed": self.seed,
            "class_labels": self.class_labels or [],
        }


@dataclass(slots=True)
class NearestCentroidBaseline:
    """A nearest-class-centroid classifier with deterministic tie-breaking."""

    class_labels: list[int] | None = None
    centroids: dict[int, list[float]] | None = None

    def fit(self, X: list[list[float]], y: list[int]) -> NearestCentroidBaseline:
        """Compute one centroid per class."""

        _validate_xy_for_fit("NearestCentroidBaseline", X, y)
        dimensions = len(X[0])
        sums: dict[int, list[float]] = {}
        counts: Counter[int] = Counter()
        for row, label in zip(X, y, strict=False):
            if len(row) != dimensions:
                raise ValueError("NearestCentroidBaseline requires consistent rows.")
            counts[label] += 1
            sums.setdefault(label, [0.0 for _ in range(dimensions)])
            for index, value in enumerate(row):
                sums[label][index] += float(value)

        self.class_labels = sorted(counts)
        self.centroids = {
            label: [value / counts[label] for value in sums[label]]
            for label in self.class_labels
        }
        return self

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict the nearest centroid by Euclidean distance."""

        if not self.class_labels or not self.centroids:
            raise ValueError("NearestCentroidBaseline must be fit before prediction.")
        return [self._predict_one(row) for row in X]

    def to_dict(self) -> dict[str, Any]:
        """Return the baseline state as JSON-serializable data."""

        return {
            "model_name": "nearest_centroid_baseline",
            "class_labels": self.class_labels or [],
            "centroids": self.centroids or {},
        }

    def _predict_one(self, row: list[float]) -> int:
        assert self.class_labels is not None
        assert self.centroids is not None

        best_label = self.class_labels[0]
        best_distance = _squared_distance(row, self.centroids[best_label])
        for label in self.class_labels[1:]:
            distance = _squared_distance(row, self.centroids[label])
            if distance < best_distance:
                best_label = label
                best_distance = distance
        return best_label


@dataclass(slots=True)
class SimpleKNNBaseline:
    """A tiny k-nearest-neighbor baseline for small encoded datasets."""

    k: int = 3
    X_train: list[list[float]] | None = None
    y_train: list[int] | None = None
    class_labels: list[int] | None = None

    def fit(self, X: list[list[float]], y: list[int]) -> SimpleKNNBaseline:
        """Store training points for nearest-neighbor voting."""

        if self.k <= 0:
            raise ValueError("SimpleKNNBaseline requires k > 0.")
        _validate_xy_for_fit("SimpleKNNBaseline", X, y)
        dimensions = len(X[0])
        if any(len(row) != dimensions for row in X):
            raise ValueError("SimpleKNNBaseline requires consistent rows.")
        self.X_train = [[float(value) for value in row] for row in X]
        self.y_train = list(y)
        self.class_labels = sorted(set(y))
        return self

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict by majority vote among the nearest k stored points."""

        if not self.X_train or not self.y_train or not self.class_labels:
            raise ValueError("SimpleKNNBaseline must be fit before prediction.")
        return [self._predict_one(row) for row in X]

    def to_dict(self) -> dict[str, Any]:
        """Return metadata without storing the full training matrix."""

        return {
            "model_name": "simple_knn_baseline",
            "k": self.k,
            "train_count": len(self.y_train or []),
            "class_labels": self.class_labels or [],
        }

    def _predict_one(self, row: list[float]) -> int:
        assert self.X_train is not None
        assert self.y_train is not None

        distances = sorted(
            (
                (_squared_distance(row, train_row), label)
                for train_row, label in zip(self.X_train, self.y_train, strict=False)
            ),
            key=lambda item: (item[0], item[1]),
        )
        votes = Counter(label for _, label in distances[: self.k])
        return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def train_majority_baseline(
    dataset: EncodedTrainingDataset,
    eval_split: str = "validation",
) -> TrainingEvaluationResult:
    """Train and evaluate a dependency-free majority baseline."""

    train_indexes = [
        index for index, split_name in enumerate(dataset.split) if split_name == "train"
    ]
    eval_indexes = [
        index for index, split_name in enumerate(dataset.split) if split_name == eval_split
    ]
    used_eval_split = eval_split
    if not eval_indexes:
        eval_indexes = [
            index for index, split_name in enumerate(dataset.split) if split_name == "test"
        ]
        used_eval_split = "test"

    if not train_indexes:
        return _error_result(dataset, "No train split rows available for baseline fit.")
    if not eval_indexes:
        return _error_result(dataset, "No evaluation split rows available.")

    X_train = [dataset.X[index] for index in train_indexes]
    y_train = [dataset.y[index] for index in train_indexes]
    X_eval = [dataset.X[index] for index in eval_indexes]
    y_eval = [dataset.y[index] for index in eval_indexes]

    try:
        baseline = MajorityClassBaseline().fit(X_train, y_train)
        predictions = baseline.predict(X_eval)
    except ValueError as exc:
        return _error_result(dataset, str(exc))

    result = evaluate_predictions(
        y_eval,
        predictions,
        dataset.labels,
        "majority_class_baseline",
    )
    result.metadata.update(
        {
            "train_count": len(train_indexes),
            "eval_count": len(eval_indexes),
            "eval_split": used_eval_split,
            "majority_class": baseline.majority_class,
        }
    )
    return result


def _error_result(
    dataset: EncodedTrainingDataset,
    warning: str,
) -> TrainingEvaluationResult:
    return TrainingEvaluationResult(
        model_name="majority_class_baseline",
        status="error",
        accuracy=None,
        macro_f1=None,
        label_metrics={},
        confusion_matrix=[],
        labels=dataset.labels,
        warnings=[warning],
        metadata={},
    )


def _validate_xy_for_fit(
    model_name: str,
    X: list[list[float]],
    y: list[int],
) -> None:
    if not X:
        raise ValueError(f"Cannot fit {model_name} with empty features.")
    if not y:
        raise ValueError(f"Cannot fit {model_name} with empty labels.")
    if len(X) != len(y):
        raise ValueError(f"{model_name} feature and label counts do not match.")


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum(
        (float(left_value) - float(right_value)) ** 2
        for left_value, right_value in zip(left, right, strict=False)
    )
