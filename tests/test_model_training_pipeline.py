"""Tests for Phase 12.2 baseline comparison pipeline."""

from __future__ import annotations

from pathlib import Path

from gpuboost.model.training_pipeline import (
    run_baseline_model_comparison,
    select_best_evaluation,
)
from gpuboost.schemas.training import (
    EncodedTrainingDataset,
    TrainingDatasetSummary,
    TrainingEvaluationResult,
)


def test_run_baseline_model_comparison_returns_successful_result() -> None:
    dataset = _encoded_dataset(
        X=[[0.0], [0.2], [8.0], [9.0], [0.1], [8.8]],
        y=[0, 0, 1, 1, 0, 1],
        split=["train", "train", "train", "train", "validation", "validation"],
    )

    result = run_baseline_model_comparison(dataset, seed=11)

    assert result["schema_version"] == "training.baseline_comparison.v1"
    assert result["status"] == "ok"
    assert result["eval_split_used"] == "validation"
    assert {model["model_name"] for model in result["models"]} == {
        "majority_class_baseline",
        "random_baseline",
        "nearest_centroid_baseline",
        "simple_knn_baseline",
    }
    assert result["best_model_name"] in {
        "nearest_centroid_baseline",
        "simple_knn_baseline",
    }
    assert isinstance(result["best_macro_f1"], float)


def test_run_baseline_model_comparison_returns_error_for_insufficient_data() -> None:
    dataset = _encoded_dataset(
        X=[[0.0], [1.0]],
        y=[0, 1],
        split=["validation", "validation"],
    )

    result = run_baseline_model_comparison(dataset)

    assert result["status"] == "error"
    assert result["models"] == []
    assert "No train split rows" in result["warnings"][0]


def test_select_best_evaluation_tie_breaks_by_accuracy_then_model_name() -> None:
    weak = _evaluation("z_model", macro_f1=0.7, accuracy=0.6)
    accurate = _evaluation("b_model", macro_f1=0.7, accuracy=0.9)
    alpha = _evaluation("a_model", macro_f1=0.7, accuracy=0.9)

    assert select_best_evaluation([weak, accurate, alpha]) is alpha


def test_select_best_evaluation_ignores_error_results() -> None:
    result = TrainingEvaluationResult(
        model_name="error",
        status="error",
        accuracy=None,
        macro_f1=None,
        label_metrics={},
        confusion_matrix=[],
        labels=["improved"],
        warnings=["boom"],
        metadata={},
    )

    assert select_best_evaluation([result]) is None


def test_training_baselines_do_not_depend_on_sklearn() -> None:
    sources = [
        Path("gpuboost/model/baseline.py").read_text(encoding="utf-8"),
        Path("gpuboost/model/training_pipeline.py").read_text(encoding="utf-8"),
    ]

    assert all("sklearn" not in source for source in sources)


def _evaluation(
    model_name: str,
    *,
    macro_f1: float,
    accuracy: float,
) -> TrainingEvaluationResult:
    return TrainingEvaluationResult(
        model_name=model_name,
        status="ok",
        accuracy=accuracy,
        macro_f1=macro_f1,
        label_metrics={},
        confusion_matrix=[],
        labels=["improved", "regressed"],
        warnings=[],
        metadata={},
    )


def _encoded_dataset(
    *,
    X: list[list[float]],
    y: list[int],
    split: list[str],
) -> EncodedTrainingDataset:
    return EncodedTrainingDataset(
        row_ids=[f"row-{index}" for index in range(len(y))],
        feature_names=["feature"],
        X=X,
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
