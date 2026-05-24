"""Tests for Phase 10D model inference orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from gpuboost.model.inference import (
    ModelInferenceResult,
    ModelInput,
    build_model_input_from_agent_result,
    model_result_to_artifact,
    run_model_inference,
)
from gpuboost.model import inference as model_inference
from gpuboost.schemas.agent import (
    AgentAction,
    AgentEvent,
    AgentGoal,
    AgentPlan,
    AgentRunResult,
)


class StaticModelProvider:
    """Synthetic provider that returns a fixed prediction."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def model_name(self) -> str:
        return "static-test-model"

    def model_version(self) -> str:
        return "1.0"

    def predict(self, model_input: ModelInput) -> ModelInferenceResult:
        if not self.available:
            return ModelInferenceResult(
                model_available=False,
                model_name=self.model_name(),
                model_version=self.model_version(),
                fallback_used=True,
                status="fallback",
                predictions=[],
                decisions=[],
                warnings=list(model_input.warnings),
                metadata={"provider": self.__class__.__name__},
            )
        return ModelInferenceResult(
            model_available=True,
            model_name=self.model_name(),
            model_version=self.model_version(),
            fallback_used=False,
            status="ok",
            predictions=[
                {
                    "id": "prediction_001",
                    "label": "keep_deterministic_recommendations",
                    "score": 0.75,
                }
            ],
            decisions=[{"id": "decision_001", "action": "no_override"}],
            warnings=list(model_input.warnings),
            metadata={"provider": self.__class__.__name__},
        )


class FailingModelProvider(StaticModelProvider):
    """Synthetic provider that raises from predict."""

    def predict(self, model_input: ModelInput) -> ModelInferenceResult:
        raise RuntimeError("synthetic provider failure")


def test_build_model_input_from_agent_result_creates_model_input() -> None:
    result = _make_agent_result()

    model_input = build_model_input_from_agent_result(result)

    assert isinstance(model_input, ModelInput)
    assert model_input.generated_at
    assert model_input.goal == "agent optimize"
    assert model_input.context["status"] == "ok"
    assert model_input.context["command"] == "agent optimize"
    assert model_input.features["status"] == "ok"


def test_run_model_inference_with_none_uses_null_provider_fallback() -> None:
    inference = run_model_inference(_make_agent_result())

    assert inference.model_available is False
    assert inference.fallback_used is True
    assert inference.status == "fallback"
    assert inference.predictions == []
    assert inference.decisions == []


def test_run_model_inference_with_static_provider_returns_predictions() -> None:
    inference = run_model_inference(
        _make_agent_result(),
        provider=StaticModelProvider(),
    )

    assert inference.model_available is True
    assert inference.fallback_used is False
    assert inference.status == "ok"
    assert inference.predictions[0]["label"] == "keep_deterministic_recommendations"
    assert inference.decisions[0]["action"] == "no_override"


def test_run_model_inference_provider_unavailable_returns_fallback() -> None:
    inference = run_model_inference(
        _make_agent_result(),
        provider=StaticModelProvider(available=False),
    )

    assert inference.model_available is False
    assert inference.fallback_used is True
    assert inference.status == "fallback"
    assert inference.predictions == []


def test_run_model_inference_provider_failure_returns_safe_fallback() -> None:
    inference = run_model_inference(
        _make_agent_result(),
        provider=FailingModelProvider(),
    )

    assert inference.model_available is False
    assert inference.fallback_used is True
    assert inference.status == "fallback"
    assert inference.predictions == []
    assert inference.decisions == []
    assert "synthetic provider failure" in inference.error
    assert inference.metadata["provider"] == "FailingModelProvider"
    assert inference.metadata["failure"] == "synthetic provider failure"
    assert any("synthetic provider failure" in warning for warning in inference.warnings)


def test_run_model_inference_selects_trained_provider_for_artifact(monkeypatch) -> None:
    created = []

    class FakeTrainedProvider(StaticModelProvider):
        def __init__(self, manifest_path: str) -> None:
            super().__init__(available=True)
            created.append(manifest_path)

        def model_name(self) -> str:
            return "trained-fixture"

        def predict(self, model_input: ModelInput) -> ModelInferenceResult:
            result = super().predict(model_input)
            result.metadata["provider"] = "trained_local_model"
            result.metadata["patch_application_allowed"] = False
            return result

    monkeypatch.setattr(
        model_inference,
        "TrainedLocalModelProvider",
        FakeTrainedProvider,
    )
    result = _make_agent_result()
    result.goal.options["model_artifact_path"] = "artifact/manifest.json"

    inference = run_model_inference(result)

    assert created == ["artifact/manifest.json"]
    assert inference.model_name == "trained-fixture"
    assert inference.metadata["provider"] == "trained_local_model"
    assert inference.metadata["patch_application_allowed"] is False


def test_model_result_to_artifact_returns_dict() -> None:
    inference = run_model_inference(_make_agent_result())

    artifact = model_result_to_artifact(inference)

    assert isinstance(artifact, dict)
    assert artifact["fallback_used"] is True


def test_warnings_are_copied_to_input_and_result() -> None:
    result = _make_agent_result(warnings=["review patch before applying"])

    model_input = build_model_input_from_agent_result(result)
    inference = run_model_inference(result, provider=StaticModelProvider())

    assert model_input.warnings == ["review patch before applying"]
    assert inference.warnings == ["review patch before applying"]
    assert model_input.warnings is not result.warnings


def test_no_raw_diff_stdout_or_stderr_in_features_or_artifact() -> None:
    result = _make_agent_result(
        artifacts={
            "diff": "--- a/train.py\n+++ b/train.py\n@@ raw diff",
            "stdout": "very noisy stdout",
            "stderr": "very noisy stderr",
        },
    )

    model_input = build_model_input_from_agent_result(result)
    artifact = model_result_to_artifact(run_model_inference(result))

    feature_data = _to_jsonable(model_input.features)

    assert _does_not_contain_key(feature_data, "diff")
    assert _does_not_contain_key(feature_data, "stdout")
    assert _does_not_contain_key(feature_data, "stderr")
    assert _does_not_contain_value(feature_data, "raw diff")
    assert _does_not_contain_value(feature_data, "very noisy stdout")
    assert _does_not_contain_value(artifact, "very noisy stderr")


def test_json_serialization_works() -> None:
    result = _make_agent_result(warnings=["synthetic warning"])
    model_input = build_model_input_from_agent_result(result)
    inference = run_model_inference(result, provider=StaticModelProvider())

    serialized_input = json.dumps(_to_jsonable(model_input))
    serialized_result = json.dumps(model_result_to_artifact(inference))

    assert json.loads(serialized_input)["warnings"] == ["synthetic warning"]
    assert json.loads(serialized_result)["status"] == "ok"


def _make_agent_result(
    *,
    warnings: list[str] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> AgentRunResult:
    goal = AgentGoal(
        id="optimize_script",
        kind="optimize_script",
        description="Optimize train.py for NVIDIA GPU performance",
        script_path="train.py",
        options={"quick": True},
        constraints=["review_patches_only"],
    )
    action = AgentAction(
        id="generate_recommendations",
        name="generate_recommendations",
        description="Generate deterministic recommendations.",
        required=True,
        status="ok",
    )
    plan = AgentPlan(id="plan_001", goal=goal, actions=[action])
    event = AgentEvent(
        timestamp="2026-01-01T00:00:00+00:00",
        action_id=action.id,
        level="info",
        message="Recommendations generated.",
        data={"recommendation_count": 2},
    )
    return AgentRunResult(
        generated_at="2026-01-01T00:00:01+00:00",
        goal=goal,
        plan=plan,
        status="ok",
        events=[event],
        warnings=warnings or [],
        artifacts=artifacts or {},
    )


def _does_not_contain_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key not in value and all(
            _does_not_contain_key(child, key) for child in value.values()
        )
    if isinstance(value, list):
        return all(_does_not_contain_key(child, key) for child in value)
    return True


def _does_not_contain_value(value: Any, text: str) -> bool:
    if isinstance(value, dict):
        return all(_does_not_contain_value(child, text) for child in value.values())
    if isinstance(value, list):
        return all(_does_not_contain_value(child, text) for child in value)
    if isinstance(value, str):
        return text not in value
    return True


def _to_jsonable(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value
