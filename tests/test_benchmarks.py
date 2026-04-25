"""CPU-safe tests for Phase 2 benchmark commands."""

from gpuboost.benchmarks import batch_sweep, matmul, mixed_precision, runner
from gpuboost.schemas.benchmark_result import BenchmarkResult, BenchmarkSuiteResult
from gpuboost.schemas.gpu_profile import (
    GPUBoostProfile,
    SystemProfile,
    TorchEnvironmentProfile,
)


def _no_cuda(_device_index: int = 0) -> tuple[bool, str]:
    return False, "test CUDA unavailable"


def test_cuda_benchmarks_skip_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(matmul, "require_cuda", _no_cuda)
    monkeypatch.setattr(mixed_precision, "require_cuda", _no_cuda)
    monkeypatch.setattr(batch_sweep, "require_cuda", _no_cuda)

    results = [
        matmul.run_matmul_benchmark(),
        mixed_precision.run_mixed_precision_benchmark(),
        batch_sweep.run_batch_sweep_benchmark(),
    ]

    assert all(result.status == "skipped" for result in results)
    assert all("test CUDA unavailable" in result.warnings for result in results)


def test_runner_returns_suite_result(monkeypatch) -> None:
    def fake_profile() -> GPUBoostProfile:
        return GPUBoostProfile(
            generated_at="2026-01-01T00:00:00+00:00",
            system=SystemProfile(os="Test OS", python_version="3.12"),
            torch_env=TorchEnvironmentProfile(
                torch_installed=True,
                cuda_available=False,
                device_count=0,
            ),
            gpus=[],
            warnings=["profile warning"],
        )

    def fake_benchmark(_device_index: int = 0) -> BenchmarkResult:
        return BenchmarkResult(
            name="Fake",
            status="skipped",
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:00:00+00:00",
            duration_sec=0.0,
            metrics=[],
            warnings=["bench warning"],
            error=None,
        )

    monkeypatch.setattr(runner, "collect_profile", fake_profile)
    monkeypatch.setattr(runner, "run_matmul_benchmark", fake_benchmark)
    monkeypatch.setattr(runner, "run_mixed_precision_benchmark", fake_benchmark)
    monkeypatch.setattr(runner, "run_batch_sweep_benchmark", fake_benchmark)

    suite = runner.run_quick_benchmark()

    assert isinstance(suite, BenchmarkSuiteResult)
    assert suite.cuda_available is False
    assert suite.device_index is None
    assert len(suite.results) == 3
    assert suite.warnings == ["profile warning"]

