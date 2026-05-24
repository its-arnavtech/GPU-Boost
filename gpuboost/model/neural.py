"""Small PyTorch neural model primitives for Phase 12.3."""

from __future__ import annotations

import random

try:  # pragma: no cover - exercised when torch is installed.
    import torch
    from torch import nn
except Exception:  # pragma: no cover - exercised by environments without torch.
    torch = None
    nn = None


def torch_available() -> bool:
    """Return whether PyTorch is importable for local neural training."""

    return torch is not None and nn is not None


if torch_available():

    class MLPClassifier(nn.Module):
        """Tiny CPU-compatible MLP classifier for safe structured features."""

        def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_sizes: list[int] | None = None,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            hidden_sizes = hidden_sizes or [32, 16]
            layers: list[nn.Module] = []
            previous_size = input_size
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(previous_size, hidden_size))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                previous_size = hidden_size
            layers.append(nn.Linear(previous_size, output_size))
            self.network = nn.Sequential(*layers)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Return class logits."""

            return self.network(inputs)

else:

    class MLPClassifier:  # type: ignore[no-redef]
        """Placeholder that raises cleanly when PyTorch is unavailable."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("PyTorch is required for MLPClassifier.")


def select_torch_device(device: str) -> str:
    """Select a PyTorch device string from a user-facing config value."""

    if not torch_available():
        raise ValueError("PyTorch is unavailable.")
    normalized = device.lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise ValueError("CUDA was requested but is not available.")
    raise ValueError(f"Unsupported torch device: {device}")


def set_training_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible local training."""

    random.seed(seed)
    if not torch_available():
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
