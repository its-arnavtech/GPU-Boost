"""Tests for Phase 12.3 neural training pipeline."""

from __future__ import annotations

import pytest

from gpuboost.model import neural, neural_training
from gpuboost.model.neural import torch_available
from gpuboost.model.neural_training import (
    OVERFIT_WARNING,
    TARGET_WARNING,
    _require_torch,
    build_default_neural_search_configs,
    run_neural_config_search,
    run_neural_hyperparameter_search,
    train_neural_classifier,
)
from gpuboost.schemas.training import (
    EncodedTrainingDataset,
    NeuralTrainingConfig,
    NeuralTrainingHistory,
    NeuralTrainingResult,
    TrainingDatasetSummary,
    TrainingEvaluationResult,
)


def test_require_torch_raises_runtime_error_when_unavailable(monkeypatch) -> None:
    # Explicit guard (not assert) so it still fires under `python -O`.
    monkeypatch.setattr(neural, "torch", None)

    with pytest.raises(RuntimeError, match="PyTorch is required"):
        _require_torch()


def test_require_torch_returns_module_when_available() -> None:
    if not torch_available():
        pytest.skip("PyTorch is unavailable.")

    assert _require_torch() is neural.torch


def test_train_neural_classifier_returns_usable_result_on_tiny_dataset() -> None:
    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    dataset = _separable_dataset(include_test=True)

    result = train_neural_classifier(
        dataset,
        NeuralTrainingConfig(
            hidden_sizes=[8],
            dropout=0.0,
            learning_rate=0.01,
            max_epochs=40,
            patience=8,
            batch_size=4,
            seed=3,
            device="cpu",
        ),
        baseline_macro_f1=0.4,
    )

    assert result.status == "ok"
    assert result.is_usable() is True
    assert result.validation_evaluation is not None
    assert result.validation_evaluation.macro_f1 is not None
    assert result.history.epochs_ran >= 1
    assert result.history.best_epoch is not None
    assert result.metadata["model_artifact_saved"] is False
    assert result.metadata["agent_integration_changed"] is False


def test_train_neural_classifier_handles_missing_torch_gracefully(monkeypatch) -> None:
    monkeypatch.setattr(neural_training.neural, "torch_available", lambda: False)

    result = train_neural_classifier(_separable_dataset())

    assert result.status == "error"
    assert "PyTorch is unavailable" in result.warnings[0]


def test_train_neural_classifier_rejects_insufficient_classes() -> None:
    result = train_neural_classifier(
        _encoded_dataset(
            X=[[0.0], [1.0], [2.0]],
            y=[0, 0, 0],
            split=["train", "train", "validation"],
            labels=["improved"],
        )
    )

    assert result.status == "error"
    assert any("Fewer than 2" in warning for warning in result.warnings)


def test_train_neural_classifier_adds_overfit_warning_for_validation_test_gap() -> None:
    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    dataset = _overfit_gap_dataset()

    result = train_neural_classifier(
        dataset,
        NeuralTrainingConfig(
            hidden_sizes=[12],
            dropout=0.0,
            learning_rate=0.02,
            max_epochs=60,
            patience=10,
            batch_size=4,
            seed=9,
            device="cpu",
            class_weighting=False,
        ),
    )

    assert result.validation_evaluation is not None
    assert result.test_evaluation is not None
    assert OVERFIT_WARNING in result.warnings


def test_default_neural_search_configs_are_deterministic_and_bounded() -> None:
    first = build_default_neural_search_configs(max_epochs=7, seed=5)
    second = build_default_neural_search_configs(max_epochs=7, seed=5)

    assert [config.to_dict() for config in first] == [
        config.to_dict() for config in second
    ]
    assert first[0].hidden_sizes == [16]
    assert first[0].max_epochs == 7
    assert {config.seed for config in first[:3]} == {5, 6, 7}


def test_hyperparameter_search_selects_by_validation_not_test(monkeypatch) -> None:
    configs = [
        NeuralTrainingConfig(hidden_sizes=[4], seed=1),
        NeuralTrainingConfig(hidden_sizes=[8], seed=2),
    ]

    def fake_train(dataset, config, baseline_macro_f1=None):
        if config.seed == 1:
            return _neural_result(config, validation_f1=0.8, test_f1=0.1)
        return _neural_result(config, validation_f1=0.7, test_f1=1.0)

    monkeypatch.setattr(neural_training, "train_neural_classifier", fake_train)

    result = run_neural_config_search(
        _separable_dataset(),
        configs,
        baseline_macro_f1=0.75,
        target_macro_f1=0.85,
    )

    assert result.best_config is configs[0]
    assert result.best_validation_macro_f1 == 0.8
    assert result.best_test_macro_f1 == 0.1
    assert result.beats_baseline is True
    assert result.target_met is False
    assert TARGET_WARNING in result.warnings


def test_target_met_logic_allows_small_test_lag(monkeypatch) -> None:
    config = NeuralTrainingConfig(seed=1)

    def fake_train(dataset, config, baseline_macro_f1=None):
        return _neural_result(config, validation_f1=0.86, test_f1=0.76)

    monkeypatch.setattr(neural_training, "train_neural_classifier", fake_train)

    result = run_neural_config_search(
        _separable_dataset(),
        [config],
        baseline_macro_f1=0.5,
        target_macro_f1=0.85,
    )

    assert result.target_met is True
    assert result.beats_baseline is True


def test_run_neural_hyperparameter_search_truncates_candidates(monkeypatch) -> None:
    def fake_train(dataset, config, baseline_macro_f1=None):
        return _neural_result(config, validation_f1=0.2, test_f1=None)

    monkeypatch.setattr(neural_training, "train_neural_classifier", fake_train)

    result = run_neural_hyperparameter_search(
        _separable_dataset(),
        max_epochs=3,
        max_candidates=2,
    )

    assert result.metadata["candidate_count"] == 2


def _neural_result(
    config: NeuralTrainingConfig,
    *,
    validation_f1: float,
    test_f1: float | None,
) -> NeuralTrainingResult:
    validation = _evaluation(config.model_name, validation_f1)
    test = _evaluation(config.model_name, test_f1) if test_f1 is not None else None
    return NeuralTrainingResult(
        status="ok",
        config=config,
        history=NeuralTrainingHistory(
            epochs_ran=2,
            best_epoch=1,
            train_loss=[1.0, 0.9],
            validation_loss=[1.0, 0.8],
            validation_macro_f1=[validation_f1],
            warnings=[],
            metadata={},
        ),
        validation_evaluation=validation,
        test_evaluation=test,
        baseline_comparison={},
        warnings=[],
        metadata={},
    )


def _evaluation(model_name: str, macro_f1: float) -> TrainingEvaluationResult:
    return TrainingEvaluationResult(
        model_name=model_name,
        status="ok",
        accuracy=macro_f1,
        macro_f1=macro_f1,
        label_metrics={},
        confusion_matrix=[],
        labels=["improved", "regressed"],
        warnings=[],
        metadata={},
    )


def _separable_dataset(*, include_test: bool = False) -> EncodedTrainingDataset:
    X = [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2], [0.05], [1.05]]
    y = [0, 0, 0, 1, 1, 1, 0, 1]
    split = [
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
    ]
    if include_test:
        X.extend([[0.15], [1.15]])
        y.extend([0, 1])
        split.extend(["test", "test"])
    return _encoded_dataset(X=X, y=y, split=split)


def _overfit_gap_dataset() -> EncodedTrainingDataset:
    return _encoded_dataset(
        X=[[0.0], [0.1], [1.0], [1.1], [0.05], [1.05], [0.02], [1.02]],
        y=[0, 0, 1, 1, 0, 1, 1, 0],
        split=[
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "test",
            "test",
        ],
    )


def _encoded_dataset(
    *,
    X: list[list[float]],
    y: list[int],
    split: list[str],
    labels: list[str] | None = None,
) -> EncodedTrainingDataset:
    labels = labels or ["improved", "regressed"]
    return EncodedTrainingDataset(
        row_ids=[f"row-{index}" for index in range(len(y))],
        feature_names=["signal"] if X and X[0] else [],
        X=X,
        y=y,
        labels=labels,
        label_to_index={label: index for index, label in enumerate(labels)},
        split=split,
        summary=TrainingDatasetSummary(
            row_count=len(y),
            labeled_count=len(y),
            skipped_count=0,
            feature_count=1 if X and X[0] else 0,
            label_counts={label: y.count(index) for index, label in enumerate(labels)},
            split_counts={name: split.count(name) for name in set(split)},
            warnings=[],
            metadata={},
        ),
        warnings=[],
        metadata={},
    )
