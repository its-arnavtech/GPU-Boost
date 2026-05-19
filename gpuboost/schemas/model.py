"""Dataclass schemas for safe local model features and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


ModelFeatureValue = str | int | float | bool | None
ModelValue = ModelFeatureValue


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ModelFeatureSet:
    """Safe, derived feature groups for local model inputs."""

    schema_version: str = "model.features.v1"
    hardware: dict[str, ModelFeatureValue] = field(default_factory=dict)
    benchmarks: dict[str, ModelFeatureValue] = field(default_factory=dict)
    advisor: dict[str, ModelFeatureValue] = field(default_factory=dict)
    code: dict[str, ModelFeatureValue] = field(default_factory=dict)
    patches: dict[str, ModelFeatureValue] = field(default_factory=dict)
    trial: dict[str, ModelFeatureValue] = field(default_factory=dict)
    comparison: dict[str, ModelFeatureValue] = field(default_factory=dict)
    history: dict[str, ModelFeatureValue] = field(default_factory=dict)
    metadata: dict[str, ModelFeatureValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the feature set as JSON-serializable data."""

        return asdict(self)

    def is_empty(self) -> bool:
        """Return whether all feature groups are empty."""

        return not any(
            (
                self.hardware,
                self.benchmarks,
                self.advisor,
                self.code,
                self.patches,
                self.trial,
                self.comparison,
                self.history,
                self.metadata,
            )
        )

    def __getitem__(self, key: str) -> ModelValue:
        """Return a feature value by searching all feature groups."""

        for group in (
            self.hardware,
            self.benchmarks,
            self.advisor,
            self.code,
            self.patches,
            self.trial,
            self.comparison,
            self.history,
            self.metadata,
        ):
            if key in group:
                return group[key]
        raise KeyError(key)


@dataclass(slots=True)
class ModelInput:
    """Input envelope for local model inference."""

    generated_at: str = field(default_factory=create_timestamp)
    goal: str = ""
    features: ModelFeatureSet = field(default_factory=ModelFeatureSet)
    context: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the model input as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class ModelPrediction:
    """One model prediction signal."""

    id: str
    target: str
    label: str
    score: float | None
    confidence: float | None
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the prediction as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class ModelDecision:
    """A model-assisted decision about a deterministic recommendation."""

    recommendation_id: str
    action: str
    model_score: float | None
    confidence: float | None
    reason: str
    safety_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the decision as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class ModelInferenceResult:
    """Result envelope for local model inference."""

    generated_at: str = field(default_factory=create_timestamp)
    schema_version: str = "model.inference.v1"
    model_available: bool = False
    model_name: str | None = None
    model_version: str | None = None
    fallback_used: bool = False
    status: str = "fallback"
    predictions: list[ModelPrediction] = field(default_factory=list)
    decisions: list[ModelDecision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the inference result as JSON-serializable data."""

        return asdict(self)

    def has_predictions(self) -> bool:
        """Return whether inference produced predictions."""

        return bool(self.predictions)

    def has_decisions(self) -> bool:
        """Return whether inference produced decisions."""

        return bool(self.decisions)

    def used_fallback(self) -> bool:
        """Return whether deterministic fallback was used."""

        return self.fallback_used
