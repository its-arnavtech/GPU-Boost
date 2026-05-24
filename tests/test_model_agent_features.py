"""Tests for safe agent-state model feature extraction."""

from __future__ import annotations

import json

from gpuboost.agent.state import AgentState
from gpuboost.model.agent_features import (
    build_model_input_features_from_agent_state,
    build_model_input_from_agent_state,
)
from gpuboost.schemas.agent import AgentGoal


def test_agent_features_exclude_raw_and_target_derived_fields() -> None:
    state = _state()
    state.diff = "--- a\n+++ b\n@@ raw diff"
    state.code_analysis = {
        "findings": [{"severity": "high", "raw_source": "def train(): pass"}],
        "stdout": "noise",
        "overall_verdict": "improved",
        "delta_best_images_per_sec": 1.0,
    }
    state.patch_plan = {
        "suggestions": [
            {
                "category": "dataloader",
                "raw_diff": "--- raw",
                "replacement_text": "patch text",
            }
        ]
    }
    state.metadata["trial_result"] = {
        "status": "passed",
        "stdout": "noisy",
        "stderr": "noisy",
        "syntax_check_status": "passed",
    }

    features = build_model_input_features_from_agent_state(state)
    serialized = json.dumps(features)

    assert "raw_source" not in serialized
    assert "raw_diff" not in serialized
    assert "stdout" not in serialized
    assert "stderr" not in serialized
    assert "overall_verdict" not in serialized
    assert "delta_best_images_per_sec" not in serialized
    assert features["metadata.has_diff"] is True
    assert features["metadata.code_finding_count"] == 1
    assert features["metadata.patch_suggestion_count"] == 1
    assert features["metadata.has_trial"] is True
    assert features["metadata.syntax_check_status"] == "passed"


def test_agent_model_input_handles_sparse_state_and_is_json_serializable() -> None:
    state = _state(script_path=None)

    model_input = build_model_input_from_agent_state(state)

    assert model_input.goal == "agent optimize"
    assert model_input.context["feature_source"] == "safe_agent_state"
    json.dumps(model_input.to_dict())


def _state(script_path: str | None = "train.py") -> AgentState:
    return AgentState(
        goal=AgentGoal(
            id="optimize_script",
            kind="optimize_script",
            description="Optimize train.py",
            script_path=script_path,
            options={"quick": True, "model": True, "trial": False},
            constraints=[],
        )
    )
