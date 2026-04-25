"""Host system inspection for GPUBoost Phase 1."""

from __future__ import annotations

import platform
from typing import Optional

import psutil

from gpuboost.schemas.gpu_profile import SystemProfile


def _cpu_model() -> Optional[str]:
    """Return the best available CPU model string for the current platform."""

    processor = platform.processor()
    if processor:
        return processor

    machine = platform.machine()
    return machine or None


def collect_system_profile(warnings: list[str] | None = None) -> SystemProfile:
    """Collect operating system, Python, CPU, and memory information."""

    del warnings
    ram_total_gb: Optional[float] = None
    try:
        ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        ram_total_gb = None

    return SystemProfile(
        os=platform.platform(),
        python_version=platform.python_version(),
        cpu_model=_cpu_model(),
        cpu_cores_physical=psutil.cpu_count(logical=False),
        cpu_cores_logical=psutil.cpu_count(logical=True),
        ram_total_gb=ram_total_gb,
    )


def inspect_system() -> dict[str, object | None]:
    """Backward-compatible dictionary view for older Phase 1 callers."""

    return collect_system_profile().to_dict()

