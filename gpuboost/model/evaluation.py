"""Dependency-free evaluation helpers for Phase 12.1 baselines."""

from __future__ import annotations

from collections import Counter

from gpuboost.schemas.training import TrainingEvaluationResult


def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
    """Return exact-match accuracy."""

    if not y_true:
        return 0.0
    correct = sum(1 for truth, pred in zip(y_true, y_pred, strict=False) if truth == pred)
    return correct / len(y_true)


def macro_f1_score(y_true: list[int], y_pred: list[int], labels: list[int]) -> float:
    """Return unweighted mean F1 across labels."""

    if not labels:
        return 0.0
    return sum(_label_scores(y_true, y_pred, label)["f1"] for label in labels) / len(labels)


def confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    labels: list[int],
) -> list[list[int]]:
    """Return a confusion matrix with rows=true labels and columns=pred labels."""

    index_by_label = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, pred in zip(y_true, y_pred, strict=False):
        if truth in index_by_label and pred in index_by_label:
            matrix[index_by_label[truth]][index_by_label[pred]] += 1
    return matrix


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    model_name: str,
) -> TrainingEvaluationResult:
    """Evaluate encoded predictions without external ML dependencies."""

    if not y_true or not y_pred:
        return _error_result(model_name, label_names, "Evaluation inputs are empty.")
    if len(y_true) != len(y_pred):
        return _error_result(
            model_name,
            label_names,
            "Evaluation input lengths do not match.",
        )

    labels = list(range(len(label_names)))
    label_metrics: dict[str, dict[str, float]] = {}
    for label_index, label_name in enumerate(label_names):
        label_metrics[label_name] = _label_scores(y_true, y_pred, label_index)

    return TrainingEvaluationResult(
        model_name=model_name,
        status="ok",
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=macro_f1_score(y_true, y_pred, labels),
        label_metrics=label_metrics,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels),
        labels=label_names,
        warnings=[],
        metadata={"eval_count": len(y_true)},
    )


def majority_class_predictions(y_train: list[int], y_eval: list[int]) -> list[int]:
    """Predict the train majority class for each eval row."""

    if not y_train:
        return []
    counts = Counter(y_train)
    majority_class = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return [majority_class for _ in y_eval]


def _label_scores(
    y_true: list[int],
    y_pred: list[int],
    label: int,
) -> dict[str, float]:
    true_positive = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth == label and pred == label
    )
    false_positive = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth != label and pred == label
    )
    false_negative = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth == label and pred != label
    )
    support = sum(1 for truth in y_true if truth == label)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(support),
    }


def _error_result(
    model_name: str,
    label_names: list[str],
    warning: str,
) -> TrainingEvaluationResult:
    return TrainingEvaluationResult(
        model_name=model_name,
        status="error",
        accuracy=None,
        macro_f1=None,
        label_metrics={},
        confusion_matrix=[],
        labels=label_names,
        warnings=[warning],
        metadata={},
    )
