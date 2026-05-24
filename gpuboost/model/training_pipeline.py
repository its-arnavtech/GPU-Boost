"""Baseline model comparison pipeline for Phase 12.2."""

from __future__ import annotations

from typing import Any

from gpuboost.model.baseline import (
    MajorityClassBaseline,
    NearestCentroidBaseline,
    RandomBaseline,
    SimpleKNNBaseline,
)
from gpuboost.model.evaluation import evaluate_predictions
from gpuboost.model.training_splits import get_eval_split, split_encoded_dataset
from gpuboost.schemas.training import EncodedTrainingDataset, TrainingEvaluationResult

BASELINE_COMPARISON_SCHEMA_VERSION = "training.baseline_comparison.v1"


def run_baseline_model_comparison(
    dataset: EncodedTrainingDataset,
    eval_split: str = "validation",
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate dependency-free structured baselines."""

    warnings = list(dataset.warnings)
    try:
        splits = split_encoded_dataset(dataset)
        eval_split_used = get_eval_split(splits, preferred=eval_split)
    except ValueError as error:
        warnings.append(str(error))
        return _comparison_error(dataset, eval_split, warnings)

    train_split = splits["train"]
    evaluation_split = splits[eval_split_used]
    if not train_split["y"]:
        warnings.append("No train split rows available for baseline model fitting.")
    if not evaluation_split["y"]:
        warnings.append("No evaluation split rows available.")
    if not dataset.labels:
        warnings.append("No encoded labels are available.")

    if warnings and (
        not train_split["y"] or not evaluation_split["y"] or not dataset.labels
    ):
        return _comparison_error(dataset, eval_split_used, warnings)

    X_train = train_split["X"]
    y_train = train_split["y"]
    X_eval = evaluation_split["X"]
    y_eval = evaluation_split["y"]
    model_entries: list[dict[str, Any]] = []
    evaluations: list[TrainingEvaluationResult] = []

    baselines = [
        MajorityClassBaseline(),
        RandomBaseline(seed=seed),
        NearestCentroidBaseline(),
        SimpleKNNBaseline(k=3),
    ]
    for baseline in baselines:
        model_name = baseline.to_dict()["model_name"]
        try:
            baseline.fit(X_train, y_train)
            predictions = baseline.predict(X_eval)
            evaluation = evaluate_predictions(
                y_eval,
                predictions,
                dataset.labels,
                model_name,
            )
        except ValueError as error:
            evaluation = _model_error_result(dataset, model_name, str(error))

        evaluation.metadata.update(
            {
                "train_count": len(y_train),
                "eval_count": len(y_eval),
                "eval_split": eval_split_used,
            }
        )
        metadata = baseline.to_dict()
        model_entries.append(
            {
                "model_name": model_name,
                "evaluation": evaluation.to_dict(),
                "metadata": metadata,
            }
        )
        evaluations.append(evaluation)
        warnings.extend(evaluation.warnings)

    best = select_best_evaluation(evaluations)
    return {
        "schema_version": BASELINE_COMPARISON_SCHEMA_VERSION,
        "status": "ok" if best is not None else "error",
        "dataset_summary": _dataset_summary(dataset),
        "eval_split_used": eval_split_used,
        "models": model_entries,
        "best_model_name": best.model_name if best is not None else None,
        "best_macro_f1": best.macro_f1 if best is not None else None,
        "warnings": sorted(set(warnings)),
    }


def select_best_evaluation(
    results: list[TrainingEvaluationResult],
) -> TrainingEvaluationResult | None:
    """Select the strongest usable evaluation with deterministic tie-breaks."""

    usable = [
        result
        for result in results
        if result.status == "ok"
        and result.macro_f1 is not None
        and result.accuracy is not None
    ]
    if not usable:
        return None
    return sorted(
        usable,
        key=lambda result: (
            -(result.macro_f1 or 0.0),
            -(result.accuracy or 0.0),
            result.model_name,
        ),
    )[0]


def _comparison_error(
    dataset: EncodedTrainingDataset,
    eval_split_used: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_COMPARISON_SCHEMA_VERSION,
        "status": "error",
        "dataset_summary": _dataset_summary(dataset),
        "eval_split_used": eval_split_used,
        "models": [],
        "best_model_name": None,
        "best_macro_f1": None,
        "warnings": sorted(set(warnings)),
    }


def _model_error_result(
    dataset: EncodedTrainingDataset,
    model_name: str,
    warning: str,
) -> TrainingEvaluationResult:
    return TrainingEvaluationResult(
        model_name=model_name,
        status="error",
        accuracy=None,
        macro_f1=None,
        label_metrics={},
        confusion_matrix=[],
        labels=dataset.labels,
        warnings=[warning],
        metadata={},
    )


def _dataset_summary(dataset: EncodedTrainingDataset) -> dict[str, Any]:
    summary = dataset.summary.to_dict()
    summary.update(
        {
            "encoded_row_count": len(dataset.y),
            "encoded_feature_count": dataset.feature_count(),
            "encoded_class_count": dataset.class_count(),
        }
    )
    return summary
