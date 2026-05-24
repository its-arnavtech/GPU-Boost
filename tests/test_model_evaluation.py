"""Tests for dependency-free Phase 12.1 evaluation helpers."""

from __future__ import annotations

from gpuboost.model.evaluation import (
    accuracy_score,
    confusion_matrix,
    evaluate_predictions,
    macro_f1_score,
    majority_class_predictions,
)


def test_evaluation_accuracy_macro_f1_and_confusion_matrix() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]

    assert accuracy_score(y_true, y_pred) == 0.5
    assert macro_f1_score(y_true, y_pred, [0, 1]) == 0.5
    assert confusion_matrix(y_true, y_pred, [0, 1]) == [[1, 1], [1, 1]]


def test_evaluate_predictions_returns_per_label_metrics() -> None:
    result = evaluate_predictions(
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        ["improved", "regressed"],
        "baseline",
    )

    assert result.status == "ok"
    assert result.accuracy == 0.75
    assert result.confusion_matrix == [[2, 0], [1, 1]]
    assert result.label_metrics["improved"]["precision"] == 2 / 3
    assert result.label_metrics["regressed"]["recall"] == 0.5


def test_evaluate_predictions_handles_invalid_inputs() -> None:
    empty = evaluate_predictions([], [], ["improved"], "baseline")
    mismatch = evaluate_predictions([0], [0, 1], ["improved"], "baseline")

    assert empty.status == "error"
    assert empty.accuracy is None
    assert mismatch.status == "error"
    assert "lengths" in mismatch.warnings[0]


def test_majority_class_predictions_uses_stable_lowest_label_tie_breaker() -> None:
    assert majority_class_predictions([1, 0], [9, 9, 9]) == [0, 0, 0]
