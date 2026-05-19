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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelPrediction:
    id: str
    score: float

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


def test_providers_do_not_make_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpuboost.model.provider import NullModelProvider, StaticModelProvider

    def fail_network_call(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network call attempted")

    monkeypatch.setattr(socket, "create_connection", fail_network_call)
    monkeypatch.setattr(socket.socket, "connect", fail_network_call)

    NullModelProvider().predict(ModelInput())
    StaticModelProvider().predict(ModelInput())
