"""Data schemas for GPUBoost output."""

from gpuboost.schemas.gpu_profile import (
    GPUBoostProfile,
    GPUDeviceProfile,
    SystemProfile,
    TorchEnvironmentProfile,
)
from gpuboost.schemas.benchmark_result import (
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkSuiteResult,
)

__all__ = [
    "BenchmarkMetric",
    "BenchmarkResult",
    "BenchmarkSuiteResult",
    "GPUBoostProfile",
    "GPUDeviceProfile",
    "SystemProfile",
    "TorchEnvironmentProfile",
]
