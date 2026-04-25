"""Formatting helpers for human-readable CLI output."""

from __future__ import annotations

from gpuboost.schemas.gpu_profile import GPUBoostProfile


def _display(value: object | None, suffix: str = "") -> str:
    if value is None:
        return "Unknown"
    return f"{value}{suffix}"


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "Unknown"
    return "Yes" if value else "No"


def format_profile(profile: GPUBoostProfile) -> str:
    """Format a Phase 1 profile for `gpuboost info`."""

    lines = [
        "GPUBoost Phase 1 GPU Inspector",
        "",
        "System:",
        f"- OS: {_display(profile.system.os)}",
        f"- Python: {_display(profile.system.python_version)}",
        f"- CPU: {_display(profile.system.cpu_model)}",
        f"- CPU cores: physical {_display(profile.system.cpu_cores_physical)}, "
        f"logical {_display(profile.system.cpu_cores_logical)}",
        f"- RAM: {_display(profile.system.ram_total_gb, ' GB')}",
        "",
        "PyTorch/CUDA:",
        f"- PyTorch installed: {_yes_no(profile.torch_env.torch_installed)}",
        f"- PyTorch version: {_display(profile.torch_env.torch_version)}",
        f"- CUDA available: {_yes_no(profile.torch_env.cuda_available)}",
        f"- Torch CUDA version: {_display(profile.torch_env.torch_cuda_version)}",
        f"- cuDNN version: {_display(profile.torch_env.cudnn_version)}",
        f"- Device count: {_display(profile.torch_env.device_count)}",
        "",
        "GPUs:",
    ]

    if not profile.gpus:
        lines.append("- No NVIDIA GPUs detected.")
    else:
        for gpu in profile.gpus:
            lines.extend(
                [
                    f"- GPU {gpu.index}: {_display(gpu.name)}",
                    f"  UUID: {_display(gpu.uuid)}",
                    f"  Architecture: {_display(gpu.architecture)}",
                    "  VRAM: "
                    f"{_display(gpu.used_vram_mb)} used / "
                    f"{_display(gpu.free_vram_mb)} free / "
                    f"{_display(gpu.total_vram_mb)} total MB",
                    "  Compute capability: "
                    f"{_display(gpu.cuda_compute_capability)}",
                    f"  Tensor Cores: {_yes_no(gpu.tensor_cores_supported)}",
                    "  Utilization: "
                    f"GPU {_display(gpu.utilization_gpu_percent, '%')}, "
                    f"memory {_display(gpu.utilization_memory_percent, '%')}",
                    f"  Temperature: {_display(gpu.temperature_c, ' C')}",
                    "  Power: "
                    f"{_display(gpu.power_draw_w, ' W')} / "
                    f"{_display(gpu.power_limit_w, ' W')}",
                ]
            )

    if profile.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in profile.warnings)

    return "\n".join(lines)

