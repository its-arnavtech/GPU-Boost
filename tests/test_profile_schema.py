"""Tests for Phase 1 profile dataclasses."""

from gpuboost.schemas.gpu_profile import (
    GPUBoostProfile,
    GPUDeviceProfile,
    SystemProfile,
    TorchEnvironmentProfile,
)


def test_profile_schema_construction() -> None:
    system = SystemProfile(
        os="Test OS",
        python_version="3.12",
        cpu_model="Test CPU",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        ram_total_gb=16.0,
    )
    torch_env = TorchEnvironmentProfile(
        torch_installed=True,
        torch_version="2.x",
        cuda_available=True,
        torch_cuda_version="12.x",
        cudnn_version="9000",
        device_count=1,
    )
    gpu = GPUDeviceProfile(
        index=0,
        name="NVIDIA Test GPU",
        total_vram_mb=8192,
        cuda_compute_capability="8.9",
        tensor_cores_supported=True,
    )
    profile = GPUBoostProfile(
        generated_at="2026-01-01T00:00:00+00:00",
        system=system,
        torch_env=torch_env,
        gpus=[gpu],
        warnings=["test warning"],
    )

    data = profile.to_dict()

    assert data["system"]["os"] == "Test OS"
    assert data["torch_env"]["cuda_available"] is True
    assert data["gpus"][0]["name"] == "NVIDIA Test GPU"
    assert data["warnings"] == ["test warning"]

