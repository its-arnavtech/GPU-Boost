"""Tests for Phase 10A local model schemas."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import datetime, timezone

from gpuboost.schemas.model import (
    ModelDecision,
    ModelFeatureSet,
    ModelInferenceResult,
    ModelInput,
    ModelPrediction,
    create_timestamp,
)


def test_model_feature_set_creation_with_defaults() -> None:
    features = ModelFeatureSet()

    assert features.schema_version == "model.features.v1"
    assert features.hardware == {}
    assert features.benchmarks == {}
    assert features.advisor == {}
    assert features.code == {}
    assert features.patches == {}
    assert features.trial == {}
    assert features.comparison == {}
    assert features.history == {}
    assert features.metadata == {}


def test_model_input_creation_with_defaults() -> None:
    model_input = ModelInput(goal="Improve dataloader throughput.")

    assert model_input.generated_at
    assert model_input.goal == "Improve dataloader throughput."
    assert isinstance(model_input.features, ModelFeatureSet)
    assert model_input.context == {}
    assert model_input.warnings == []


def test_model_prediction_creation() -> None:
    prediction = _make_prediction()

    assert prediction.id == "prediction_001"
    assert prediction.target == "mixed_precision"
    assert prediction.label == "positive"
    assert prediction.score == 0.82
    assert prediction.confidence == 0.74
    assert prediction.rationale == "Hardware and benchmark signals support AMP."
    assert prediction.metadata == {"source": "local_schema_test"}


def test_model_decision_creation() -> None:
    decision = _make_decision()

    assert decision.recommendation_id == "rec_001"
    assert decision.action == "prioritize"
    assert decision.model_score == 0.82
    assert decision.confidence == 0.74
    assert decision.reason == "Prediction is positive with usable confidence."
    assert decision.safety_notes == ["Review numerical stability."]


def test_model_inference_result_creation_with_defaults() -> None:
    result = ModelInferenceResult()

    assert result.generated_at
    assert result.schema_version == "model.inference.v1"
    assert result.model_available is False
    assert result.model_name is None
    assert result.model_version is None
    assert result.fallback_used is False
    assert result.status == "fallback"
    assert result.predictions == []
    assert result.decisions == []
    assert result.warnings == []
    assert result.error is None
    assert result.metadata == {}


def test_to_dict_nesting_works() -> None:
    features = ModelFeatureSet(
        hardware={"cuda_available": True, "vram_gb": 24.0},
        benchmarks={"tokens_per_second": 4200.5},
    )
    result = ModelInferenceResult(
        generated_at="2026-01-01T00:00:00+00:00",
        model_available=True,
        model_name="local-gpuboost-model",
        model_version="0.1.0",
        fallback_used=False,
        status="ok",
        predictions=[_make_prediction()],
        decisions=[_make_decision()],
        metadata={"feature_schema": features.schema_version},
    )

    feature_data = features.to_dict()
    result_data = result.to_dict()

    assert feature_data["hardware"]["cuda_available"] is True
    assert feature_data["benchmarks"]["tokens_per_second"] == 4200.5
    assert result_data["predictions"][0]["target"] == "mixed_precision"
    assert result_data["decisions"][0]["action"] == "prioritize"
    assert result_data["metadata"]["feature_schema"] == "model.features.v1"


def test_json_serialization_works() -> None:
    model_input = ModelInput(
        generated_at="2026-01-01T00:00:00+00:00",
        goal="Tune batch size.",
        features=ModelFeatureSet(hardware={"gpu_count": 1}),
        context={"phase": "10A"},
    )
    result = ModelInferenceResult(
        generated_at="2026-01-01T00:00:01+00:00",
        predictions=[_make_prediction()],
        decisions=[_make_decision()],
        metadata={"input": asdict(model_input)},
    )

    serialized = json.dumps(result.to_dict())
    deserialized = json.loads(serialized)

    assert deserialized["predictions"][0]["label"] == "positive"
    assert deserialized["decisions"][0]["recommendation_id"] == "rec_001"
    assert deserialized["metadata"]["input"]["features"]["hardware"] == {
        "gpu_count": 1
    }


def test_default_list_and_dict_fields_are_isolated_between_instances() -> None:
    first_features = ModelFeatureSet()
    second_features = ModelFeatureSet()
    first_input = ModelInput()
    second_input = ModelInput()
    first_prediction = ModelPrediction(
        id="first_prediction",
        target="batch_size",
        label="neutral",
        score=None,
        confidence=None,
        rationale="No signal.",
    )
    second_prediction = ModelPrediction(
        id="second_prediction",
        target="tensor_cores",
        label="unknown",
        score=None,
        confidence=None,
        rationale="Missing hardware signal.",
    )
    first_decision = ModelDecision(
        recommendation_id="first_rec",
        action="review",
        model_score=None,
        confidence=None,
        reason="Needs human review.",
    )
    second_decision = ModelDecision(
        recommendation_id="second_rec",
        action="keep",
        model_score=None,
        confidence=None,
        reason="No model signal.",
    )
    first_result = ModelInferenceResult()
    second_result = ModelInferenceResult()

    first_features.hardware["gpu_count"] = 1
    first_features.metadata["phase"] = "10A"
    first_input.context["dry_run"] = True
    first_input.warnings.append("First warning.")
    first_prediction.metadata["signal"] = "benchmark"
    first_decision.safety_notes.append("First note.")
    first_result.predictions.append(first_prediction)
    first_result.decisions.append(first_decision)
    first_result.warnings.append("Fallback unavailable.")
    first_result.metadata["status_source"] = "test"

    assert second_features.hardware == {}
    assert second_features.metadata == {}
    assert second_input.context == {}
    assert second_input.warnings == []
    assert second_prediction.metadata == {}
    assert second_decision.safety_notes == []
    assert second_result.predictions == []
    assert second_result.decisions == []
    assert second_result.warnings == []
    assert second_result.metadata == {}


def test_timestamp_helper_returns_non_empty_utc_iso_string() -> None:
    timestamp = create_timestamp()
    parsed = datetime.fromisoformat(timestamp)

    assert timestamp
    assert parsed.tzinfo == timezone.utc


def test_model_feature_set_is_empty_true_and_false() -> None:
    empty_features = ModelFeatureSet()
    populated_features = ModelFeatureSet(history={"run_count": 3})

    assert empty_features.is_empty() is True
    assert populated_features.is_empty() is False


def test_model_inference_result_fallback_case() -> None:
    result = ModelInferenceResult(
        model_available=False,
        fallback_used=True,
        status="fallback",
        warnings=["No local model configured."],
    )

    assert result.used_fallback() is True
    assert result.to_dict()["status"] == "fallback"
    assert result.to_dict()["warnings"] == ["No local model configured."]


def test_model_inference_result_error_case() -> None:
    result = ModelInferenceResult(
        model_available=False,
        fallback_used=False,
        status="error",
        error="Local model schema validation failed.",
    )

    assert result.status == "error"
    assert result.error == "Local model schema validation failed."
    assert result.to_dict()["error"] == "Local model schema validation failed."


def test_model_inference_helpers() -> None:
    result = ModelInferenceResult(
        predictions=[_make_prediction()],
        decisions=[_make_decision()],
        fallback_used=True,
    )

    assert result.has_predictions() is True
    assert result.has_decisions() is True
    assert result.used_fallback() is True
    assert ModelInferenceResult().has_predictions() is False
    assert ModelInferenceResult().has_decisions() is False
    assert ModelInferenceResult().used_fallback() is False


def test_model_schemas_do_not_require_raw_payload_fields() -> None:
    forbidden = {
        "raw_source",
        "source_code",
        "raw_diff",
        "diff",
        "stdout",
        "stderr",
    }
    schema_fields = set()
    for schema in (
        ModelFeatureSet,
        ModelInput,
        ModelPrediction,
        ModelDecision,
        ModelInferenceResult,
    ):
        schema_fields.update(field.name for field in fields(schema))

    assert schema_fields.isdisjoint(forbidden)


def _make_prediction() -> ModelPrediction:
    return ModelPrediction(
        id="prediction_001",
        target="mixed_precision",
        label="positive",
        score=0.82,
        confidence=0.74,
        rationale="Hardware and benchmark signals support AMP.",
        metadata={"source": "local_schema_test"},
    )


def _make_decision() -> ModelDecision:
    return ModelDecision(
        recommendation_id="rec_001",
        action="prioritize",
        model_score=0.82,
        confidence=0.74,
        reason="Prediction is positive with usable confidence.",
        safety_notes=["Review numerical stability."],
    )
