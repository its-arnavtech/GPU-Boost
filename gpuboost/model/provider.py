"""Local model provider interfaces and deterministic fallback providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpuboost.model.artifacts import load_neural_model_artifact
from gpuboost.model.feature_encoding import encode_feature_dicts
from gpuboost.model.neural import torch, torch_available

if TYPE_CHECKING:
    from gpuboost.schemas.model import (
        ModelDecision,
        ModelInferenceResult,
        ModelInput,
        ModelPrediction,
    )


FALLBACK_WARNING = "No local model provider configured; skipped model inference."


class BaseModelProvider:
    """Base interface for local model providers."""

    def is_available(self) -> bool:
        """Return whether this provider can perform local inference."""

        raise NotImplementedError

    def model_name(self) -> str | None:
        """Return the provider model name, if any."""

        raise NotImplementedError

    def model_version(self) -> str | None:
        """Return the provider model version, if any."""

        raise NotImplementedError

    def predict(self, model_input: ModelInput) -> ModelInferenceResult:
        """Return model inference for the supplied input."""

        raise NotImplementedError


class NullModelProvider(BaseModelProvider):
    """Provider used when no local model is configured."""

    def is_available(self) -> bool:
        """Return False because the null provider never performs inference."""

        return False

    def model_name(self) -> str | None:
        """Return no model name for the null provider."""

        return None

    def model_version(self) -> str | None:
        """Return no model version for the null provider."""

        return None

    def predict(self, model_input: ModelInput) -> ModelInferenceResult:
        """Return an explicit fallback result without external calls."""

        return _create_inference_result(
            model_available=False,
            model_name=self.model_name(),
            model_version=self.model_version(),
            fallback_used=True,
            status="fallback",
            predictions=[],
            decisions=[],
            warnings=[FALLBACK_WARNING],
            error=None,
            metadata={"provider": "null"},
        )


class StaticModelProvider(BaseModelProvider):
    """Fake provider for tests and development fixtures."""

    def __init__(
        self,
        predictions: list[ModelPrediction] | None = None,
        decisions: list[ModelDecision] | None = None,
        available: bool = True,
        name: str = "static",
        version: str = "test",
    ) -> None:
        self._predictions = list(predictions or [])
        self._decisions = list(decisions or [])
        self._available = available
        self._name = name
        self._version = version

    def is_available(self) -> bool:
        """Return the configured availability state."""

        return self._available

    def model_name(self) -> str | None:
        """Return the configured fake model name."""

        return self._name

    def model_version(self) -> str | None:
        """Return the configured fake model version."""

        return self._version

    def predict(self, model_input: ModelInput) -> ModelInferenceResult:
        """Return configured static outputs without external calls."""

        available = self.is_available()
        return _create_inference_result(
            model_available=available,
            model_name=self.model_name(),
            model_version=self.model_version(),
            fallback_used=not available,
            status="ok" if available else "fallback",
            predictions=list(self._predictions),
            decisions=list(self._decisions),
            warnings=[],
            error=None,
            metadata={"provider": "static"},
        )


class FailingModelProvider(BaseModelProvider):
    """Fake provider that raises during prediction."""

    def __init__(self, message: str = "model provider failed") -> None:
        self._message = message

    def is_available(self) -> bool:
        """Return True because this provider is present but fails at runtime."""

        return True

    def model_name(self) -> str | None:
        """Return a stable fake model name."""

        return "failing"

    def model_version(self) -> str | None:
        """Return a stable fake model version."""

        return "test"

    def predict(self, model_input: ModelInput) -> ModelInferenceResult:
        """Raise the configured failure."""

        raise RuntimeError(self._message)


class TrainedLocalModelProvider(BaseModelProvider):
    """Provider backed by a saved Phase 12.4 neural model artifact."""

    def __init__(self, manifest_path: str, device: str = "auto") -> None:
        self.manifest_path = manifest_path
        self.device = device
        self._loaded = False
        self._load_error: str | None = None
        self._model: object | None = None
        self._feature_spec: object | None = None
        self._label_to_index: dict[str, int] = {}
        self._manifest: object | None = None

    def is_available(self) -> bool:
        """Return whether the artifact can be loaded for local inference."""

        self._ensure_loaded()
        return self._loaded and self._load_error is None

    def model_name(self) -> str | None:
        """Return the loaded artifact model name."""

        self._ensure_loaded()
        return getattr(self._manifest, "model_name", None)

    def model_version(self) -> str | None:
        """Return the artifact schema as a conservative model version."""

        self._ensure_loaded()
        return getattr(self._manifest, "schema_version", None)

    def predict(self, model_input: ModelInput) -> ModelInferenceResult:
        """Predict a label from safe structured features."""

        self._ensure_loaded()
        if self._load_error is not None or not self._loaded:
            return _create_inference_result(
                model_available=False,
                model_name=self.model_name(),
                model_version=self.model_version(),
                fallback_used=True,
                status="error",
                predictions=[],
                decisions=[],
                warnings=[f"Trained local model unavailable: {self._load_error}"],
                error=self._load_error,
                metadata={
                    "provider": "trained_local_model",
                    "artifact_manifest_path": self.manifest_path,
                    "patch_application_allowed": False,
                },
            )

        try:
            encoded = encode_feature_dicts(
                [_flatten_model_input_features(model_input)],
                self._feature_spec,
            )[0][0]
            label, confidence, probabilities = self._predict_encoded(encoded)
        except Exception as error:  # noqa: BLE001 - provider failures are non-fatal
            message = str(error) or error.__class__.__name__
            return _create_inference_result(
                model_available=False,
                model_name=self.model_name(),
                model_version=self.model_version(),
                fallback_used=True,
                status="error",
                predictions=[],
                decisions=[],
                warnings=[f"Trained local model prediction failed: {message}"],
                error=message,
                metadata={
                    "provider": "trained_local_model",
                    "patch_application_allowed": False,
                },
            )

        prediction = _create_model_prediction(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
        )
        return _create_inference_result(
            model_available=True,
            model_name=self.model_name(),
            model_version=self.model_version(),
            fallback_used=False,
            status="ok",
            predictions=[prediction],
            decisions=[],
            warnings=list(getattr(model_input, "warnings", [])),
            error=None,
            metadata={
                "provider": "trained_local_model",
                "artifact_manifest_path": self.manifest_path,
                "patch_application_allowed": False,
            },
        )

    def _ensure_loaded(self) -> None:
        if self._loaded or self._load_error is not None:
            return
        try:
            (
                self._model,
                self._feature_spec,
                self._label_to_index,
                self._manifest,
            ) = load_neural_model_artifact(self.manifest_path, self.device)
            self._loaded = True
        except Exception as error:  # noqa: BLE001 - unavailable provider is safe
            self._load_error = str(error) or error.__class__.__name__

    def _predict_encoded(
        self,
        encoded: list[float],
    ) -> tuple[str, float, dict[str, float]]:
        if not torch_available() or torch is None:
            raise RuntimeError("PyTorch is unavailable.")
        if self._model is None:
            raise RuntimeError("Model artifact is not loaded.")
        device = getattr(self._model, "_gpuboost_device", "cpu")
        tensor = torch.tensor([encoded], dtype=torch.float32, device=device)
        mean = getattr(self._model, "_gpuboost_feature_mean", None)
        std = getattr(self._model, "_gpuboost_feature_std", None)
        if mean is not None and std is not None:
            tensor = (tensor - mean.to(device)) / std.to(device)
        self._model.eval()
        with torch.no_grad():
            probabilities_tensor = torch.softmax(self._model(tensor), dim=1)[0].cpu()
        probabilities = [
            float(value) for value in probabilities_tensor.tolist()
        ]
        predicted_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        labels = _labels_from_mapping(self._label_to_index)
        label = labels[predicted_index] if predicted_index < len(labels) else "unknown"
        return (
            label,
            probabilities[predicted_index],
            {
                labels[index]: probability
                for index, probability in enumerate(probabilities)
                if index < len(labels)
            },
        )


def _create_inference_result(**values: Any) -> ModelInferenceResult:
    """Build a model result using the Phase 10 schema when available."""

    from gpuboost.schemas.model import ModelInferenceResult, create_timestamp

    try:
        return ModelInferenceResult(generated_at=create_timestamp(), **values)
    except TypeError as exc:
        if "generated_at" not in str(exc):
            raise
        return ModelInferenceResult(**values)


def _create_model_prediction(
    *,
    label: str,
    confidence: float,
    probabilities: dict[str, float],
) -> ModelPrediction:
    from gpuboost.schemas.model import ModelPrediction

    try:
        return ModelPrediction(
            id="trained_local_prediction",
            target="optimization_outcome",
            label=label,
            score=confidence,
            confidence=confidence,
            rationale="Prediction from local trained GPUBoost artifact.",
            metadata={
                "provider": "trained_local_model",
                "probabilities": probabilities,
                "patch_application_allowed": False,
            },
        )
    except TypeError:
        return ModelPrediction(id="trained_local_prediction", score=confidence)


def _flatten_model_input_features(model_input: ModelInput) -> dict[str, object]:
    context = getattr(model_input, "context", {})
    if isinstance(context, dict) and isinstance(context.get("features"), dict):
        return dict(context["features"])

    flattened: dict[str, object] = {}
    features = getattr(model_input, "features", None)
    for group_name in (
        "hardware",
        "benchmarks",
        "advisor",
        "code",
        "patches",
        "trial",
        "comparison",
        "history",
        "metadata",
    ):
        group = getattr(features, group_name, None)
        if isinstance(group, dict):
            for key, value in group.items():
                flattened[f"{group_name}.{key}"] = value
                flattened.setdefault(str(key), value)
    return flattened


def _labels_from_mapping(label_to_index: dict[str, int]) -> list[str]:
    return [
        label
        for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])
    ]
