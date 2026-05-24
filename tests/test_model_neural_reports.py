"""Tests for Phase 12.3 neural report writing."""

from __future__ import annotations

import json

from gpuboost.model.neural_reports import write_neural_training_reports
from gpuboost.schemas.training import (
    NeuralSearchResult,
    NeuralTrainingConfig,
    NeuralTrainingHistory,
    NeuralTrainingResult,
    TrainingEvaluationResult,
)


def test_write_neural_training_reports_writes_json_and_markdown(tmp_path) -> None:
    config = NeuralTrainingConfig(hidden_sizes=[8], max_epochs=3)
    result = NeuralTrainingResult(
        status="ok",
        config=config,
        history=NeuralTrainingHistory(
            epochs_ran=3,
            best_epoch=2,
            train_loss=[1.0, 0.8],
            validation_loss=[1.1, 0.7],
            validation_macro_f1=[0.4, 0.9],
            warnings=[],
            metadata={},
        ),
        validation_evaluation=_evaluation(0.9),
        test_evaluation=_evaluation(0.82),
        baseline_comparison={
            "best_baseline_model_name": "nearest_centroid_baseline",
            "best_baseline_macro_f1": 0.7,
            "beats_baseline": True,
        },
        warnings=["review before integration"],
        metadata={},
    )
    search = NeuralSearchResult(
        status="ok",
        best_result=result,
        candidates=[result],
        best_config=config,
        best_validation_macro_f1=0.9,
        best_test_macro_f1=0.82,
        baseline_macro_f1=0.7,
        target_met=True,
        beats_baseline=True,
        warnings=[],
        metadata={},
    )

    output = write_neural_training_reports(search, str(tmp_path))

    json_report = tmp_path / "neural_training_report.json"
    markdown_report = tmp_path / "neural_training_report.md"
    assert output == {
        "json_report": str(json_report),
        "markdown_report": str(markdown_report),
    }
    assert json.loads(json_report.read_text(encoding="utf-8"))["status"] == "ok"
    markdown = markdown_report.read_text(encoding="utf-8")
    assert "GPUBoost Neural Model Training" in markdown
    assert "Target met: yes" in markdown
    assert "nearest_centroid_baseline" in markdown
    assert "not a production model artifact" in markdown


def _evaluation(macro_f1: float) -> TrainingEvaluationResult:
    return TrainingEvaluationResult(
        model_name="mlp_classifier",
        status="ok",
        accuracy=macro_f1,
        macro_f1=macro_f1,
        label_metrics={},
        confusion_matrix=[],
        labels=["improved", "regressed"],
        warnings=[],
        metadata={},
    )
