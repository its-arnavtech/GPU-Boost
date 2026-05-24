"""Dataclass schemas for Phase 12.1 training scaffolding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

TrainingMetadataValue = str | int | float | bool | None


def create_timestamp() -> str:
    """Return the current UTC time as an ISO timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TrainingDatasetSummary:
    """Summary of rows available to the local structured trainer."""

    row_count: int
    labeled_count: int
    skipped_count: int
    feature_count: int
    label_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, TrainingMetadataValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class TrainingFeatureSpec:
    """Stable description of encoded training feature columns."""

    feature_names: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    boolean_features: list[str]
    unknown_value: str = "__unknown__"
    schema_version: str = "training.feature_spec.v1"

    def to_dict(self) -> dict[str, Any]:
        """Return the feature spec as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True, kw_only=True)
class EncodedTrainingDataset:
    """Encoded matrix and labels for local structured baseline training."""

    schema_version: str = "training.encoded_dataset.v1"
    row_ids: list[str]
    feature_names: list[str]
    X: list[list[float]]
    y: list[int]
    labels: list[str]
    label_to_index: dict[str, int]
    split: list[str]
    summary: TrainingDatasetSummary
    feature_spec: TrainingFeatureSpec | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, TrainingMetadataValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the encoded dataset as JSON-serializable data."""

        return asdict(self)

    def class_count(self) -> int:
        """Return the number of encoded label classes."""

        return len(self.labels)

    def feature_count(self) -> int:
        """Return the number of encoded feature columns."""

        return len(self.feature_names)


@dataclass(slots=True, kw_only=True)
class TrainingEvaluationResult:
    """Evaluation metrics for a Phase 12 local training scaffold."""

    schema_version: str = "training.evaluation.v1"
    model_name: str
    status: str
    accuracy: float | None
    macro_f1: float | None
    label_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, TrainingMetadataValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation result as JSON-serializable data."""

        return asdict(self)

    def is_usable(self) -> bool:
        """Return whether this result can be used as a valid evaluation."""

        return self.status == "ok" and self.accuracy is not None


@dataclass(slots=True)
class NeuralTrainingConfig:
    """Configuration for the small Phase 12.3 neural classifier."""

    schema_version: str = "training.neural_config.v1"
    model_name: str = "mlp_classifier"
    hidden_sizes: list[int] = field(default_factory=lambda: [32, 16])
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 16
    seed: int = 42
    device: str = "auto"
    class_weighting: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the config as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class NeuralTrainingHistory:
    """Training curves and early stopping metadata."""

    epochs_ran: int
    best_epoch: int | None
    train_loss: list[float]
    validation_loss: list[float]
    validation_macro_f1: list[float]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, TrainingMetadataValue] = field(default_factory=dict)
    schema_version: str = "training.neural_history.v1"

    def to_dict(self) -> dict[str, Any]:
        """Return the history as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True, kw_only=True)
class NeuralTrainingResult:
    """Result from one small neural training run."""

    schema_version: str = "training.neural_result.v1"
    status: str
    config: NeuralTrainingConfig
    history: NeuralTrainingHistory
    validation_evaluation: TrainingEvaluationResult | None
    test_evaluation: TrainingEvaluationResult | None
    baseline_comparison: dict[str, TrainingMetadataValue] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, TrainingMetadataValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as JSON-serializable data."""

        return asdict(self)

    def is_usable(self) -> bool:
        """Return whether the validation evaluation is usable."""

        return (
            self.status == "ok"
            and self.validation_evaluation is not None
            and self.validation_evaluation.is_usable()
        )


@dataclass(slots=True, kw_only=True)
class NeuralSearchResult:
    """Result from modest Phase 12.3 neural hyperparameter search."""

    schema_version: str = "training.neural_search_result.v1"
    status: str
    best_result: NeuralTrainingResult | None
    candidates: list[NeuralTrainingResult]
    best_config: NeuralTrainingConfig | None
    best_validation_macro_f1: float | None
    best_test_macro_f1: float | None
    baseline_macro_f1: float | None
    target_macro_f1: float = 0.85
    target_met: bool = False
    beats_baseline: bool = False
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, TrainingMetadataValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the search result as JSON-serializable data."""

        return asdict(self)

    def is_usable(self) -> bool:
        """Return whether a usable best neural result exists."""

        return (
            self.status == "ok"
            and self.best_result is not None
            and self.best_result.is_usable()
        )


@dataclass(slots=True, kw_only=True)
class ModelArtifactManifest:
    """Manifest for a saved local Phase 12 neural model artifact."""

    schema_version: str = "training.model_artifact.v1"
    artifact_type: str = "mlp_classifier"
    created_at: str
    model_name: str
    model_format: str = "torch_state_dict"
    model_file: str
    feature_spec_file: str
    label_mapping_file: str
    training_config_file: str
    evaluation_report_file: str | None
    input_size: int
    output_size: int
    labels: list[str]
    feature_names: list[str]
    validation_macro_f1: float | None
    test_macro_f1: float | None
    baseline_macro_f1: float | None
    beats_baseline: bool
    target_macro_f1: float | None
    target_met: bool
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, TrainingMetadataValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as JSON-serializable data."""

        return asdict(self)
