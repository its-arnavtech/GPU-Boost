"""Local small-neural-model training for Phase 12.3."""

from __future__ import annotations

import copy
import random
from typing import Any

from gpuboost.model import neural
from gpuboost.model.evaluation import evaluate_predictions
from gpuboost.model.training_splits import split_encoded_dataset
from gpuboost.schemas.training import (
    EncodedTrainingDataset,
    NeuralSearchResult,
    NeuralTrainingConfig,
    NeuralTrainingHistory,
    NeuralTrainingResult,
    TrainingFeatureSpec,
    TrainingEvaluationResult,
)

OVERFIT_WARNING = "Validation/test gap suggests possible overfitting."
TARGET_WARNING = "Target macro F1 was not reached; best result is reported honestly."


def _require_torch() -> Any:
    """Return the torch module, raising a clear error if it is unavailable.

    Uses an explicit check rather than ``assert`` so the guard is not stripped
    when Python runs with ``-O``/``-OO`` (optimized mode), which would otherwise
    surface as a confusing ``AttributeError`` deep inside tensor operations.
    """

    torch = neural.torch
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for neural model training. "
            "Install it with: pip install torch"
        )
    return torch


def train_neural_classifier(
    dataset: EncodedTrainingDataset,
    config: NeuralTrainingConfig | None = None,
    baseline_macro_f1: float | None = None,
) -> NeuralTrainingResult:
    """Train one small MLP classifier on safe encoded features."""

    _, result = _train_neural_classifier_with_model(
        dataset,
        config=config,
        baseline_macro_f1=baseline_macro_f1,
    )
    return result


def _train_neural_classifier_with_model(
    dataset: EncodedTrainingDataset,
    config: NeuralTrainingConfig | None = None,
    baseline_macro_f1: float | None = None,
) -> tuple[object | None, NeuralTrainingResult]:
    """Train one small MLP classifier and optionally return the model."""

    config = config or NeuralTrainingConfig()
    warnings = list(dataset.warnings)
    validation_error = _validate_dataset_for_training(dataset)
    if validation_error is not None:
        warnings.append(validation_error)
        return None, _error_result(config, warnings, baseline_macro_f1)

    splits = split_encoded_dataset(dataset)
    train_split = splits["train"]
    validation_split = splits["validation"]
    test_split = splits["test"]
    train_classes = set(train_split["y"])
    if len(train_classes) < 2:
        warnings.append("Fewer than 2 training classes are present.")
        return None, _error_result(config, warnings, baseline_macro_f1)
    if not validation_split["y"]:
        warnings.append("No validation split rows available for neural training.")
        return None, _error_result(config, warnings, baseline_macro_f1)

    if not neural.torch_available():
        warnings.append("PyTorch is unavailable; neural training cannot run.")
        return None, _error_result(config, warnings, baseline_macro_f1)

    try:
        device = neural.select_torch_device(config.device)
    except ValueError as error:
        warnings.append(str(error))
        return None, _error_result(config, warnings, baseline_macro_f1)

    torch = _require_torch()
    neural.set_training_seed(config.seed)

    X_train = torch.tensor(train_split["X"], dtype=torch.float32, device=device)
    y_train = torch.tensor(train_split["y"], dtype=torch.long, device=device)
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    X_train = (X_train - mean) / std

    model = neural.MLPClassifier(
        input_size=dataset.feature_count(),
        output_size=dataset.class_count(),
        hidden_sizes=list(config.hidden_sizes),
        dropout=config.dropout,
    ).to(device)
    setattr(model, "_gpuboost_feature_mean", mean.detach().cpu())
    setattr(model, "_gpuboost_feature_std", std.detach().cpu())
    setattr(model, "_gpuboost_device", device)

    criterion = torch.nn.CrossEntropyLoss(
        weight=_class_weights(train_split["y"], dataset.class_count(), device)
        if config.class_weighting
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = NeuralTrainingHistory(
        epochs_ran=0,
        best_epoch=None,
        train_loss=[],
        validation_loss=[],
        validation_macro_f1=[],
        warnings=[],
        metadata={"device": device},
    )

    best_state: dict[str, Any] | None = None
    best_macro_f1 = -1.0
    best_validation_loss = float("inf")
    patience_remaining = config.patience
    rng = random.Random(config.seed)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        indexes = list(range(len(train_split["y"])))
        rng.shuffle(indexes)
        total_loss = 0.0
        for batch_indexes in _batches(indexes, config.batch_size):
            batch_tensor = torch.tensor(batch_indexes, dtype=torch.long, device=device)
            optimizer.zero_grad()
            logits = model(X_train.index_select(0, batch_tensor))
            loss = criterion(logits, y_train.index_select(0, batch_tensor))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_indexes)

        validation_loss = _loss_for_split(
            model,
            validation_split["X"],
            validation_split["y"],
            mean,
            std,
            criterion,
            device,
        )
        validation_evaluation = evaluate_neural_model(
            model,
            validation_split["X"],
            validation_split["y"],
            dataset.labels,
            config.model_name,
        )
        macro_f1 = validation_evaluation.macro_f1 or 0.0
        history.train_loss.append(total_loss / len(train_split["y"]))
        history.validation_loss.append(validation_loss)
        history.validation_macro_f1.append(macro_f1)
        history.epochs_ran = epoch

        improved = macro_f1 > best_macro_f1 or (
            macro_f1 == best_macro_f1 and validation_loss < best_validation_loss
        )
        if improved:
            best_macro_f1 = macro_f1
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            patience_remaining = config.patience
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_evaluation = evaluate_neural_model(
        model,
        validation_split["X"],
        validation_split["y"],
        dataset.labels,
        config.model_name,
    )
    test_evaluation = None
    if test_split["y"]:
        test_evaluation = evaluate_neural_model(
            model,
            test_split["X"],
            test_split["y"],
            dataset.labels,
            config.model_name,
        )
        if (
            validation_evaluation.macro_f1 is not None
            and test_evaluation.macro_f1 is not None
            and validation_evaluation.macro_f1 - test_evaluation.macro_f1 >= 0.20
        ):
            warnings.append(OVERFIT_WARNING)

    beats_baseline = (
        baseline_macro_f1 is not None
        and validation_evaluation.macro_f1 is not None
        and validation_evaluation.macro_f1 > baseline_macro_f1
    )
    history.warnings = list(warnings)
    result = NeuralTrainingResult(
        status="ok" if validation_evaluation.is_usable() else "error",
        config=config,
        history=history,
        validation_evaluation=validation_evaluation,
        test_evaluation=test_evaluation,
        baseline_comparison={
            "best_baseline_macro_f1": baseline_macro_f1,
            "beats_baseline": beats_baseline,
        },
        warnings=sorted(set(warnings + validation_evaluation.warnings)),
        metadata={
            "train_count": len(train_split["y"]),
            "validation_count": len(validation_split["y"]),
            "test_count": len(test_split["y"]),
            "ready_for_integration": beats_baseline and OVERFIT_WARNING not in warnings,
            "model_artifact_saved": False,
            "agent_integration_changed": False,
        },
    )
    return model, result


def predict_neural_classifier(model: object, X: list[list[float]]) -> list[int]:
    """Predict integer class IDs from an in-memory MLP classifier."""

    if not neural.torch_available():
        raise ValueError("PyTorch is unavailable; neural prediction cannot run.")
    if not X:
        return []
    torch = _require_torch()

    device = getattr(model, "_gpuboost_device", "cpu")
    mean = getattr(model, "_gpuboost_feature_mean", None)
    std = getattr(model, "_gpuboost_feature_std", None)
    tensor = torch.tensor(X, dtype=torch.float32, device=device)
    if mean is not None and std is not None:
        tensor = (tensor - mean.to(device)) / std.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        return [int(value) for value in logits.argmax(dim=1).cpu().tolist()]


def evaluate_neural_model(
    model: object,
    X: list[list[float]],
    y: list[int],
    label_names: list[str],
    model_name: str = "mlp_classifier",
) -> TrainingEvaluationResult:
    """Evaluate an in-memory neural model using dependency-free metrics."""

    if not neural.torch_available():
        return _evaluation_error(
            model_name,
            label_names,
            "PyTorch is unavailable; neural evaluation cannot run.",
        )
    try:
        predictions = predict_neural_classifier(model, X)
    except ValueError as error:
        return _evaluation_error(model_name, label_names, str(error))
    return evaluate_predictions(y, predictions, label_names, model_name)


def build_default_neural_search_configs(
    max_epochs: int = 100,
    seed: int = 42,
) -> list[NeuralTrainingConfig]:
    """Build a deterministic, modest neural search space."""

    configs: list[NeuralTrainingConfig] = []
    for hidden_sizes in ([16], [32, 16], [64, 32]):
        for dropout in (0.0, 0.1, 0.2):
            for learning_rate in (0.001, 0.003):
                for weight_decay in (0.0, 0.0001):
                    for candidate_seed in (seed, seed + 1, seed + 2):
                        configs.append(
                            NeuralTrainingConfig(
                                hidden_sizes=list(hidden_sizes),
                                dropout=dropout,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                max_epochs=max_epochs,
                                seed=candidate_seed,
                            )
                        )
    return configs


def run_neural_hyperparameter_search(
    dataset: EncodedTrainingDataset,
    baseline_macro_f1: float | None = None,
    target_macro_f1: float = 0.85,
    max_epochs: int = 100,
    seed: int = 42,
    max_candidates: int | None = 12,
) -> NeuralSearchResult:
    """Run a modest validation-selected MLP hyperparameter search."""

    configs = build_default_neural_search_configs(max_epochs=max_epochs, seed=seed)
    if max_candidates is not None:
        configs = configs[:max_candidates]
    return run_neural_config_search(
        dataset=dataset,
        configs=configs,
        baseline_macro_f1=baseline_macro_f1,
        target_macro_f1=target_macro_f1,
    )


def train_best_neural_model_for_artifact(
    dataset: EncodedTrainingDataset,
    baseline_macro_f1: float | None = None,
    target_macro_f1: float = 0.85,
    max_epochs: int = 100,
    seed: int = 42,
    max_candidates: int | None = 12,
) -> tuple[object, TrainingFeatureSpec, dict[str, int], NeuralSearchResult]:
    """Train/search and return a loadable in-memory model for artifact saving."""

    search = run_neural_hyperparameter_search(
        dataset,
        baseline_macro_f1=baseline_macro_f1,
        target_macro_f1=target_macro_f1,
        max_epochs=max_epochs,
        seed=seed,
        max_candidates=max_candidates,
    )
    if search.best_config is None:
        raise RuntimeError("No usable neural model was trained for artifact saving.")
    model, retrained_result = _train_neural_classifier_with_model(
        dataset,
        config=search.best_config,
        baseline_macro_f1=baseline_macro_f1,
    )
    if model is None or not retrained_result.is_usable():
        raise RuntimeError("Best neural model could not be retrained for artifact.")
    feature_spec = dataset.feature_spec or _numeric_feature_spec(dataset)
    return model, feature_spec, dict(dataset.label_to_index), search


def train_neural_model_for_artifact_config(
    dataset: EncodedTrainingDataset,
    config: NeuralTrainingConfig,
    baseline_macro_f1: float | None = None,
) -> tuple[object, TrainingFeatureSpec, dict[str, int], NeuralTrainingResult]:
    """Train one supplied config and return a model ready for artifact saving."""

    model, result = _train_neural_classifier_with_model(
        dataset,
        config=config,
        baseline_macro_f1=baseline_macro_f1,
    )
    if model is None or not result.is_usable():
        raise RuntimeError("Neural model could not be trained for artifact.")
    feature_spec = dataset.feature_spec or _numeric_feature_spec(dataset)
    return model, feature_spec, dict(dataset.label_to_index), result


def run_neural_config_search(
    dataset: EncodedTrainingDataset,
    configs: list[NeuralTrainingConfig],
    baseline_macro_f1: float | None = None,
    target_macro_f1: float = 0.85,
) -> NeuralSearchResult:
    """Run a supplied config list and select by validation metrics only."""

    candidates = [
        train_neural_classifier(dataset, config, baseline_macro_f1)
        for config in configs
    ]
    best = _select_best_neural_result(candidates)
    best_validation_macro_f1 = _validation_macro_f1(best)
    best_test_macro_f1 = _test_macro_f1(best)
    beats_baseline = (
        baseline_macro_f1 is not None
        and best_validation_macro_f1 is not None
        and best_validation_macro_f1 > baseline_macro_f1
    )
    target_met = (
        best_validation_macro_f1 is not None
        and best_validation_macro_f1 >= target_macro_f1
        and (
            best_test_macro_f1 is None
            or best_test_macro_f1 >= target_macro_f1 - 0.10
        )
    )

    warnings: list[str] = []
    if best is None:
        for candidate in candidates:
            warnings.extend(candidate.warnings)
    else:
        warnings.extend(best.warnings)
    if not target_met:
        warnings.append(TARGET_WARNING)

    return NeuralSearchResult(
        status="ok" if best is not None else "error",
        best_result=best,
        candidates=candidates,
        best_config=best.config if best is not None else None,
        best_validation_macro_f1=best_validation_macro_f1,
        best_test_macro_f1=best_test_macro_f1,
        baseline_macro_f1=baseline_macro_f1,
        target_macro_f1=target_macro_f1,
        target_met=target_met,
        beats_baseline=beats_baseline,
        warnings=sorted(set(warnings)),
        metadata={
            "candidate_count": len(candidates),
            "selection_metric": "validation_macro_f1",
            "ready_for_integration": beats_baseline
            and best is not None
            and OVERFIT_WARNING not in best.warnings,
            "model_artifact_saved": False,
            "agent_integration_changed": False,
        },
    )


def _validate_dataset_for_training(dataset: EncodedTrainingDataset) -> str | None:
    if dataset.class_count() < 2:
        return "Fewer than 2 encoded classes are present."
    if dataset.feature_count() == 0:
        return "No encoded feature columns are available."
    if not dataset.X or not dataset.y:
        return "No encoded training rows are available."
    return None


def _numeric_feature_spec(dataset: EncodedTrainingDataset) -> TrainingFeatureSpec:
    return TrainingFeatureSpec(
        feature_names=list(dataset.feature_names),
        categorical_features=[],
        numeric_features=list(dataset.feature_names),
        boolean_features=[],
    )


def _error_result(
    config: NeuralTrainingConfig,
    warnings: list[str],
    baseline_macro_f1: float | None,
) -> NeuralTrainingResult:
    return NeuralTrainingResult(
        status="error",
        config=config,
        history=NeuralTrainingHistory(
            epochs_ran=0,
            best_epoch=None,
            train_loss=[],
            validation_loss=[],
            validation_macro_f1=[],
            warnings=warnings,
            metadata={},
        ),
        validation_evaluation=None,
        test_evaluation=None,
        baseline_comparison={
            "best_baseline_macro_f1": baseline_macro_f1,
            "beats_baseline": False,
        },
        warnings=sorted(set(warnings)),
        metadata={
            "model_artifact_saved": False,
            "agent_integration_changed": False,
        },
    )


def _evaluation_error(
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


def _class_weights(
    y_train: list[int],
    class_count: int,
    device: str,
) -> object:
    torch = _require_torch()
    counts = [max(y_train.count(label), 0) for label in range(class_count)]
    total = sum(counts)
    weights = [
        total / (class_count * count) if count else 0.0
        for count in counts
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _loss_for_split(
    model: object,
    X: list[list[float]],
    y: list[int],
    mean: object,
    std: object,
    criterion: object,
    device: str,
) -> float:
    if not X or not y:
        return 0.0
    torch = _require_torch()
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32, device=device)
        targets = torch.tensor(y, dtype=torch.long, device=device)
        inputs = (inputs - mean.to(device)) / std.to(device)
        loss = criterion(model(inputs), targets)
    return float(loss.detach().cpu())


def _batches(indexes: list[int], batch_size: int) -> list[list[int]]:
    safe_batch_size = max(1, batch_size)
    return [
        indexes[start : start + safe_batch_size]
        for start in range(0, len(indexes), safe_batch_size)
    ]


def _select_best_neural_result(
    results: list[NeuralTrainingResult],
) -> NeuralTrainingResult | None:
    usable = [
        result
        for result in results
        if result.validation_evaluation is not None
        and result.validation_evaluation.status == "ok"
        and result.validation_evaluation.macro_f1 is not None
        and result.validation_evaluation.accuracy is not None
    ]
    if not usable:
        return None
    return sorted(
        usable,
        key=lambda result: (
            -(result.validation_evaluation.macro_f1 or 0.0),
            -(result.validation_evaluation.accuracy or 0.0),
            tuple(result.config.hidden_sizes),
            result.config.dropout,
            result.config.learning_rate,
            result.config.weight_decay,
            result.config.seed,
        ),
    )[0]


def _validation_macro_f1(result: NeuralTrainingResult | None) -> float | None:
    if result is None or result.validation_evaluation is None:
        return None
    return result.validation_evaluation.macro_f1


def _test_macro_f1(result: NeuralTrainingResult | None) -> float | None:
    if result is None or result.test_evaluation is None:
        return None
    return result.test_evaluation.macro_f1
