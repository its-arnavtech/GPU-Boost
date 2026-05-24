"""Tests for Phase 10C model providers."""

from __future__ import annotations

import json
import socket
import sys
import types
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest


@dataclass(slots=True)
class ModelFeatureSet:
    schema_version: str = "model.features.v1"
    hardware: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    advisor: dict[str, Any] = field(default_factory=dict)
    code: dict[str, Any] = field(default_factory=dict)
    patches: dict[str, Any] = field(default_factory=dict)
    trial: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelInput:
    run_id: str = "run_001"
    features: ModelFeatureSet = field(default_factory=ModelFeatureSet)
    context: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelPrediction:
    id: str
    score: float
    target: str = "optimization_outcome"
    label: str = "unknown"
    confidence: float | None = None
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelDecision:
    id: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelInferenceResult:
    generated_at: str
    model_available: bool
    model_name: str | None
    model_version: str | None
    fallback_used: bool
    status: str
    predictions: list[ModelPrediction] = field(default_factory=list)
    decisions: list[ModelDecision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture(autouse=True)
def install_model_schema() -> None:
    previous_module = sys.modules.get("gpuboost.schemas.model")
    schema_module = types.ModuleType("gpuboost.schemas.model")
    schema_module.ModelFeatureSet = ModelFeatureSet
    schema_module.ModelFeatureValue = str | int | float | bool | None
    schema_module.ModelInput = ModelInput
    schema_module.ModelPrediction = ModelPrediction
    schema_module.ModelDecision = ModelDecision
    schema_module.ModelInferenceResult = ModelInferenceResult
    schema_module.create_timestamp = create_timestamp
    sys.modules["gpuboost.schemas.model"] = schema_module
    yield
    if previous_module is None:
        sys.modules.pop("gpuboost.schemas.model", None)
    else:
        sys.modules["gpuboost.schemas.model"] = previous_module


def test_base_model_provider_predict_raises() -> None:
    from gpuboost.model.provider import BaseModelProvider

    provider = BaseModelProvider()

    with pytest.raises(NotImplementedError):
        provider.predict(ModelInput())


def test_null_model_provider_unavailable() -> None:
    from gpuboost.model.provider import NullModelProvider

    provider = NullModelProvider()

    assert provider.is_available() is False
    assert provider.model_name() is None
    assert provider.model_version() is None


def test_null_model_provider_predict_returns_fallback_result() -> None:
    from gpuboost.model.provider import FALLBACK_WARNING, NullModelProvider

    result = NullModelProvider().predict(ModelInput())

    assert result.model_available is False
    assert result.model_name is None
    assert result.model_version is None
    assert result.fallback_used is True
    assert result.status == "fallback"
    assert result.predictions == []
    assert result.decisions == []
    assert result.warnings == [FALLBACK_WARNING]
    assert result.error is None
    assert result.metadata == {"provider": "null"}


def test_static_model_provider_available_returns_ok() -> None:
    from gpuboost.model.provider import StaticModelProvider

    provider = StaticModelProvider()
    result = provider.predict(ModelInput())

    assert provider.is_available() is True
    assert result.model_available is True
    assert result.model_name == "static"
    assert result.model_version == "test"
    assert result.fallback_used is False
    assert result.status == "ok"


def test_static_model_provider_unavailable_returns_fallback() -> None:
    from gpuboost.model.provider import StaticModelProvider

    provider = StaticModelProvider(available=False)
    result = provider.predict(ModelInput())

    assert provider.is_available() is False
    assert result.model_available is False
    assert result.fallback_used is True
    assert result.status == "fallback"
    assert result.metadata == {"provider": "static"}


def test_static_model_provider_carries_predictions_and_decisions() -> None:
    from gpuboost.model.provider import StaticModelProvider

    prediction = ModelPrediction(id="pred_001", score=0.95)
    decision = ModelDecision(id="decision_001", action="use_bigger_batch")
    result = StaticModelProvider(
        predictions=[prediction],
        decisions=[decision],
        name="fixture-model",
        version="v1",
    ).predict(ModelInput())

    assert result.predictions == [prediction]
    assert result.decisions == [decision]
    assert result.model_name == "fixture-model"
    assert result.model_version == "v1"


def test_failing_model_provider_raises() -> None:
    from gpuboost.model.provider import FailingModelProvider

    provider = FailingModelProvider(message="boom")

    with pytest.raises(RuntimeError, match="boom"):
        provider.predict(ModelInput())


@pytest.mark.parametrize(
    "provider",
    [
        pytest.param("null", id="null"),
        pytest.param("static", id="static"),
    ],
)
def test_provider_outputs_are_json_serializable_through_to_dict(provider: str) -> None:
    from gpuboost.model.provider import NullModelProvider, StaticModelProvider

    if provider == "null":
        result = NullModelProvider().predict(ModelInput())
    else:
        result = StaticModelProvider(
            predictions=[ModelPrediction(id="pred_001", score=0.95)],
            decisions=[ModelDecision(id="decision_001", action="keep")],
        ).predict(ModelInput())

    serialized = json.dumps(result.to_dict())
    deserialized = json.loads(serialized)

    assert deserialized["metadata"]["provider"] == provider


def test_trained_local_model_provider_loads_and_predicts(tmp_path) -> None:
    from gpuboost.model.neural import torch_available

    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    from gpuboost.model.provider import TrainedLocalModelProvider

    manifest_path = _write_trained_provider_artifact(tmp_path)
    provider = TrainedLocalModelProvider(str(manifest_path), device="cpu")
    model_input = ModelInput(
        context={"features": {"features.safe_signal": 1.0}},
    )

    result = provider.predict(model_input)

    assert provider.is_available() is True
    assert result.status == "ok"
    assert result.model_available is True
    assert result.predictions
    assert result.predictions[0].label in {"improved", "regressed"}
    assert result.predictions[0].confidence is not None
    assert result.metadata["provider"] == "trained_local_model"
    assert result.metadata["patch_application_allowed"] is False


def test_trained_local_model_provider_missing_artifact_returns_error() -> None:
    from gpuboost.model.provider import TrainedLocalModelProvider

    provider = TrainedLocalModelProvider("missing-manifest.json", device="cpu")
    result = provider.predict(ModelInput())

    assert provider.is_available() is False
    assert result.status == "error"
    assert result.fallback_used is True
    assert "missing-manifest.json" in result.error


def test_providers_do_not_make_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpuboost.model.provider import NullModelProvider, StaticModelProvider

    def fail_network_call(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network call attempted")

    monkeypatch.setattr(socket, "create_connection", fail_network_call)
    monkeypatch.setattr(socket.socket, "connect", fail_network_call)

    NullModelProvider().predict(ModelInput())
    StaticModelProvider().predict(ModelInput())


def _write_trained_provider_artifact(tmp_path):
    from gpuboost.model.artifacts import save_neural_model_artifact
    from gpuboost.model.neural_training import train_best_neural_model_for_artifact
    from gpuboost.schemas.training import (
        EncodedTrainingDataset,
        TrainingDatasetSummary,
        TrainingFeatureSpec,
    )

    feature_spec = TrainingFeatureSpec(
        feature_names=["features.safe_signal"],
        categorical_features=[],
        numeric_features=["features.safe_signal"],
        boolean_features=[],
    )
    X = [[0.0], [0.1], [1.0], [1.1], [0.05], [1.05]]
    y = [0, 0, 1, 1, 0, 1]
    split = ["train", "train", "train", "train", "validation", "validation"]
    dataset = EncodedTrainingDataset(
        row_ids=[f"row-{index}" for index in range(len(y))],
        feature_names=feature_spec.feature_names,
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
            label_counts={"improved": 3, "regressed": 3},
            split_counts={name: split.count(name) for name in set(split)},
            warnings=[],
            metadata={},
        ),
        feature_spec=feature_spec,
        warnings=[],
        metadata={},
    )
    model, feature_spec, label_mapping, result = train_best_neural_model_for_artifact(
        dataset,
        max_epochs=3,
        max_candidates=1,
    )
    save_neural_model_artifact(
        model,
        feature_spec,
        label_mapping,
        result.best_config,
        result,
        output_dir=str(tmp_path),
        artifact_name="provider",
    )
    return tmp_path / "provider" / "manifest.json"
