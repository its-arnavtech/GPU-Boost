"""Model inference orchestration over safe GPUBoost feature inputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from gpuboost.model.features import extract_model_features_from_agent_result
from gpuboost.model.provider import BaseModelProvider, NullModelProvider
from gpuboost.schemas.agent import AgentRunResult, create_timestamp
from gpuboost.schemas.model import ModelFeatureSet, ModelInferenceResult, ModelInput


def build_model_input_from_agent_result(result: AgentRunResult) -> ModelInput:
    """Build a model input from safe, derived agent run features."""

    features = extract_model_features_from_agent_result(result)
    features.metadata.setdefault("status", result.status)
    return ModelInput(
        generated_at=create_timestamp(),
        goal="agent optimize",
        features=features,
        context={
            "goal_id": result.goal.id,
            "goal_kind": result.goal.kind,
            "status": result.status,
            "command": "agent optimize",
        },
        warnings=list(result.warnings),
    )


def build_model_input(agent_state: Any) -> ModelInput:
    """Build a model input from either an agent result or live agent state."""

    if isinstance(agent_state, AgentRunResult):
        return build_model_input_from_agent_result(agent_state)

    goal = getattr(agent_state, "goal", None)
    metadata = getattr(agent_state, "metadata", {})
    features = ModelFeatureSet(
        metadata={
            "status": "running",
            "has_gpu_profile": getattr(agent_state, "gpu_profile", None) is not None,
            "has_benchmark_result": (
                getattr(agent_state, "benchmark_result", None) is not None
            ),
            "has_advisor_result": (
                getattr(agent_state, "advisor_result", None) is not None
            ),
            "has_code_analysis": getattr(agent_state, "code_analysis", None)
            is not None,
            "has_patch_plan": getattr(agent_state, "patch_plan", None) is not None,
            "has_diff": bool(getattr(agent_state, "diff", None)),
            "has_trial": "trial_result" in metadata,
            "warning_count": len(getattr(agent_state, "warnings", [])),
        }
    )
    return ModelInput(
        generated_at=create_timestamp(),
        goal="agent optimize",
        features=features,
        context={
            "goal_id": getattr(goal, "id", None),
            "goal_kind": getattr(goal, "kind", None),
            "has_script_path": bool(getattr(goal, "script_path", None)),
        },
        warnings=list(getattr(agent_state, "warnings", [])),
    )


def run_model_inference(
    agent_result: Any,
    provider: BaseModelProvider | None = None,
) -> ModelInferenceResult:
    """Run local model inference or a deterministic fallback."""

    model_input = build_model_input(agent_result)
    selected_provider = provider or NullModelProvider()
    try:
        return selected_provider.predict(model_input)
    except Exception as error:  # noqa: BLE001 - provider failures are non-fatal
        provider_name = selected_provider.__class__.__name__
        return ModelInferenceResult(
            generated_at=create_timestamp(),
            model_available=False,
            model_name=_safe_call(selected_provider, "model_name"),
            model_version=_safe_call(selected_provider, "model_version"),
            fallback_used=True,
            status="fallback",
            predictions=[],
            decisions=[],
            warnings=[f"Model inference provider failed: {error}"],
            error=str(error),
            metadata={
                "provider": provider_name,
                "failure": str(error),
            },
        )


def model_result_to_artifact(result: ModelInferenceResult) -> dict[str, Any]:
    """Return a JSON-safe artifact dictionary for a model result."""

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    return dict(result)


def _safe_call(provider: BaseModelProvider, method_name: str) -> str | None:
    method = getattr(provider, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:  # noqa: BLE001 - best-effort metadata only
        return None
