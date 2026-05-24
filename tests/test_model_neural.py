"""Tests for small Phase 12.3 neural model primitives."""

from __future__ import annotations

import pytest

from gpuboost.model.neural import (
    MLPClassifier,
    select_torch_device,
    torch,
    torch_available,
)


def test_select_torch_device_cpu() -> None:
    if not torch_available():
        with pytest.raises(ValueError, match="PyTorch is unavailable"):
            select_torch_device("cpu")
        return

    assert select_torch_device("cpu") == "cpu"


def test_select_torch_device_cuda_unavailable_behavior() -> None:
    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    if torch.cuda.is_available():
        assert select_torch_device("cuda") == "cuda"
    else:
        with pytest.raises(ValueError, match="CUDA"):
            select_torch_device("cuda")


def test_select_torch_device_rejects_invalid_value() -> None:
    if not torch_available():
        pytest.skip("PyTorch is unavailable.")

    with pytest.raises(ValueError, match="Unsupported"):
        select_torch_device("tpu")


def test_mlp_classifier_forward_pass_on_cpu() -> None:
    if not torch_available():
        pytest.skip("PyTorch is unavailable.")

    model = MLPClassifier(input_size=3, output_size=2, hidden_sizes=[4], dropout=0.0)
    outputs = model(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))

    assert tuple(outputs.shape) == (1, 2)
