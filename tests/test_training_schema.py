"""Tests for Phase 12.1 training schemas."""

from __future__ import annotations

from gpuboost.schemas.training import (
    EncodedTrainingDataset,
    ModelArtifactManifest,
    NeuralSearchResult,
    NeuralTrainingConfig,
    NeuralTrainingHistory,
    NeuralTrainingResult,
    TrainingDatasetSummary,
    TrainingEvaluationResult,
    TrainingFeatureSpec,
)


def test_training_schemas_serialize() -> None:
    summary = TrainingDatasetSummary(
        row_count=3,
        labeled_count=2,
        skipped_count=1,
        feature_count=4,
        label_counts={"improved": 1, "regressed": 1},
        split_counts={"train": 1, "validation": 1, "unassigned": 1},
        warnings=["small"],
        metadata={"phase": 12},
    )
    spec = TrainingFeatureSpec(
        feature_names=["a", "b"],
        categorical_features=["b"],
        numeric_features=["a"],
        boolean_features=[],
    )
    dataset = EncodedTrainingDataset(
        row_ids=["row-1", "row-2"],
        feature_names=spec.feature_names,
        X=[[1.0, 0.2], [2.0, 0.3]],
        y=[0, 1],
        labels=["improved", "regressed"],
        label_to_index={"improved": 0, "regressed": 1},
        split=["train", "validation"],
        summary=summary,
        warnings=[],
        metadata={"feature_schema_version": spec.schema_version},
    )
    evaluation = TrainingEvaluationResult(
        model_name="baseline",
        status="ok",
        accuracy=0.5,
        macro_f1=0.33,
        label_metrics={"improved": {"support": 1.0}},
        confusion_matrix=[[1, 0], [1, 0]],
        labels=["improved", "regressed"],
        warnings=[],
        metadata={"eval_count": 2},
    )

    assert summary.to_dict()["row_count"] == 3
    assert spec.to_dict()["schema_version"] == "training.feature_spec.v1"
    assert dataset.to_dict()["summary"]["labeled_count"] == 2
    assert dataset.class_count() == 2
    assert dataset.feature_count() == 2
    assert evaluation.to_dict()["accuracy"] == 0.5
    assert evaluation.is_usable() is True


def test_training_evaluation_result_is_not_usable_for_error() -> None:
    result = TrainingEvaluationResult(
        model_name="baseline",
        status="error",
        accuracy=None,
        macro_f1=None,
        label_metrics={},
        confusion_matrix=[],
        labels=[],
        warnings=["empty"],
        metadata={},
    )

    assert result.is_usable() is False


def test_neural_training_schemas_serialize_and_track_usability() -> None:
    config = NeuralTrainingConfig(max_epochs=5, hidden_sizes=[8])
    history = NeuralTrainingHistory(
        epochs_ran=3,
        best_epoch=2,
        train_loss=[1.0, 0.8, 0.7],
        validation_loss=[1.1, 0.9, 0.85],
        validation_macro_f1=[0.2, 0.5, 0.6],
        warnings=[],
        metadata={"device": "cpu"},
    )
    evaluation = TrainingEvaluationResult(
        model_name="mlp_classifier",
        status="ok",
        accuracy=1.0,
        macro_f1=1.0,
        label_metrics={},
        confusion_matrix=[],
        labels=["improved", "regressed"],
        warnings=[],
        metadata={},
    )
    result = NeuralTrainingResult(
        status="ok",
        config=config,
        history=history,
        validation_evaluation=evaluation,
        test_evaluation=None,
        baseline_comparison={
            "best_baseline_model_name": "nearest_centroid_baseline",
            "best_baseline_macro_f1": 0.7,
            "beats_baseline": True,
        },
        warnings=[],
        metadata={},
    )
    search = NeuralSearchResult(
        status="ok",
        best_result=result,
        candidates=[result],
        best_config=config,
        best_validation_macro_f1=1.0,
        best_test_macro_f1=None,
        baseline_macro_f1=0.7,
        target_met=True,
        beats_baseline=True,
        warnings=[],
        metadata={},
    )

    assert config.to_dict()["schema_version"] == "training.neural_config.v1"
    assert history.to_dict()["epochs_ran"] == 3
    assert result.to_dict()["validation_evaluation"]["macro_f1"] == 1.0
    assert result.is_usable() is True
    assert search.to_dict()["best_config"]["hidden_sizes"] == [8]
    assert search.is_usable() is True


def test_model_artifact_manifest_serializes_relative_file_names() -> None:
    manifest = ModelArtifactManifest(
        created_at="2026-01-01T00:00:00+00:00",
        model_name="mlp_classifier",
        model_file="model.pt",
        feature_spec_file="feature_spec.json",
        label_mapping_file="label_mapping.json",
        training_config_file="training_config.json",
        evaluation_report_file="evaluation_report.json",
        input_size=2,
        output_size=2,
        labels=["improved", "regressed"],
        feature_names=["features.batch_size", "features.workload_family"],
        validation_macro_f1=0.9,
        test_macro_f1=0.8,
        baseline_macro_f1=0.7,
        beats_baseline=True,
        target_macro_f1=0.85,
        target_met=False,
        warnings=["review"],
        metadata={"model_artifact_saved": True},
    )

    data = manifest.to_dict()

    assert data["schema_version"] == "training.model_artifact.v1"
    assert data["model_file"] == "model.pt"
    assert data["input_size"] == 2
