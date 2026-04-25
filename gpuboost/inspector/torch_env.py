"""PyTorch and CUDA environment inspection for GPUBoost Phase 1."""

from __future__ import annotations

from gpuboost.schemas.gpu_profile import TorchEnvironmentProfile


def collect_torch_environment(
    warnings: list[str] | None = None,
) -> TorchEnvironmentProfile:
    """Collect PyTorch and CUDA runtime information without requiring CUDA."""

    warning_sink = warnings if warnings is not None else []

    try:
        import torch
    except Exception as exc:
        warning_sink.append(f"PyTorch could not be imported: {exc}")
        return TorchEnvironmentProfile(torch_installed=False)

    cuda_available = False
    device_count = 0
    cudnn_version = None

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        warning_sink.append(f"torch.cuda.is_available() failed: {exc}")

    try:
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        warning_sink.append(f"torch.cuda.device_count() failed: {exc}")

    try:
        cudnn_raw = torch.backends.cudnn.version()
        cudnn_version = str(cudnn_raw) if cudnn_raw is not None else None
    except Exception as exc:
        warning_sink.append(f"cuDNN version detection failed: {exc}")

    return TorchEnvironmentProfile(
        torch_installed=True,
        torch_version=getattr(torch, "__version__", None),
        cuda_available=cuda_available,
        torch_cuda_version=getattr(torch.version, "cuda", None),
        cudnn_version=cudnn_version,
        device_count=device_count,
    )


def inspect_torch_env() -> dict[str, object | None]:
    """Backward-compatible dictionary view for older Phase 1 callers."""

    return collect_torch_environment().to_dict()

