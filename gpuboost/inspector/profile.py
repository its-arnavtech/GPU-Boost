"""Profile assembly for Phase 1 inspection output."""

from __future__ import annotations

from datetime import datetime, timezone

from gpuboost.inspector.gpu import collect_gpu_profiles
from gpuboost.inspector.system import collect_system_profile
from gpuboost.inspector.torch_env import collect_torch_environment
from gpuboost.schemas.gpu_profile import GPUBoostProfile


def collect_profile() -> GPUBoostProfile:
    """Collect a complete GPUBoost Phase 1 profile."""

    warnings: list[str] = []
    system = collect_system_profile(warnings)
    torch_env = collect_torch_environment(warnings)
    gpus = collect_gpu_profiles(warnings)

    return GPUBoostProfile(
        generated_at=datetime.now(timezone.utc).isoformat(),
        system=system,
        torch_env=torch_env,
        gpus=gpus,
        warnings=warnings,
    )

