"""Local model provider interfaces and deterministic fallback providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _create_inference_result(**values: Any) -> ModelInferenceResult:
    """Build a model result using the Phase 10 schema when available."""

    from gpuboost.schemas.model import ModelInferenceResult, create_timestamp

    try:
        return ModelInferenceResult(generated_at=create_timestamp(), **values)
    except TypeError as exc:
        if "generated_at" not in str(exc):
            raise
        return ModelInferenceResult(**values)
