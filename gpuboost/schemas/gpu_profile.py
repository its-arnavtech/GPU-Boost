"""Dataclass schemas for GPUBoost Phase 1 inspection output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class GPUDeviceProfile:
    """Profile for one NVIDIA GPU device."""

    index: int
    name: str
    uuid: Optional[str] = None
    architecture: Optional[str] = None
    total_vram_mb: Optional[int] = None
    used_vram_mb: Optional[int] = None
    free_vram_mb: Optional[int] = None
    cuda_compute_capability: Optional[str] = None
    tensor_cores_supported: Optional[bool] = None
    utilization_gpu_percent: Optional[float] = None
    utilization_memory_percent: Optional[float] = None
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Return the device profile as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class SystemProfile:
    """Profile for host operating system and CPU/RAM details."""

    os: str
    python_version: str
    cpu_model: Optional[str] = None
    cpu_cores_physical: Optional[int] = None
    cpu_cores_logical: Optional[int] = None
    ram_total_gb: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Return the system profile as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class TorchEnvironmentProfile:
    """Profile for PyTorch and CUDA runtime availability."""

    torch_installed: bool
    torch_version: Optional[str] = None
    cuda_available: bool = False
    torch_cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    device_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the PyTorch environment profile as JSON-serializable data."""

        return asdict(self)


@dataclass(slots=True)
class GPUBoostProfile:
    """Complete Phase 1 inspection profile."""

    generated_at: str
    system: SystemProfile
    torch_env: TorchEnvironmentProfile
    gpus: list[GPUDeviceProfile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the full profile as JSON-serializable data."""

        return asdict(self)

