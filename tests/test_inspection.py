"""CPU-safe tests for Phase 1 inspection collectors."""

from gpuboost.inspector.profile import collect_profile
from gpuboost.inspector.system import collect_system_profile
from gpuboost.inspector.torch_env import collect_torch_environment
from gpuboost.schemas.gpu_profile import (
    GPUBoostProfile,
    SystemProfile,
    TorchEnvironmentProfile,
)


def test_collect_system_profile_returns_valid_fields() -> None:
    profile = collect_system_profile()

    assert isinstance(profile, SystemProfile)
    assert profile.os
    assert profile.python_version
    assert profile.cpu_cores_logical is None or profile.cpu_cores_logical >= 1


def test_collect_torch_environment_does_not_crash() -> None:
    profile = collect_torch_environment()

    assert isinstance(profile, TorchEnvironmentProfile)
    assert isinstance(profile.torch_installed, bool)
    assert isinstance(profile.cuda_available, bool)
    assert isinstance(profile.device_count, int)


def test_collect_profile_does_not_crash() -> None:
    profile = collect_profile()

    assert isinstance(profile, GPUBoostProfile)
    assert profile.generated_at
    assert profile.system.os
    assert isinstance(profile.gpus, list)
    assert isinstance(profile.warnings, list)

