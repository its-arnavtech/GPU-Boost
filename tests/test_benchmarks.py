"""CPU-safe tests for Phase 2 benchmark commands."""

from gpuboost.benchmarks import batch_sweep, dataloader, matmul, mixed_precision, runner
from gpuboost.benchmarks.common import round_metric
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
    monkeypatch.setattr(runner, "run_quick_dataloader_benchmark", fake_benchmark)

    suite = runner.run_quick_benchmark()

    assert isinstance(suite, BenchmarkSuiteResult)
    assert suite.cuda_available is False
    assert suite.device_index is None
    assert len(suite.results) == 4
    assert suite.warnings == ["profile warning"]


def test_round_metric_preserves_non_numeric_values() -> None:
    assert round_metric(1.23456, digits=3) == 1.235
    assert round_metric(7, digits=3) == 7
    assert round_metric(True, digits=3) is True
    assert round_metric(None, digits=3) is None
    assert round_metric("TensorCoreMLP", digits=3) == "TensorCoreMLP"


def test_best_batch_size_prefers_smaller_on_throughput_tie() -> None:
    metrics = batch_sweep._build_batch_sweep_metrics(
        batch_sizes=[1, 2, 4],
        throughput_by_batch={1: 100.0, 2: 300.0, 4: 300.0},
        median_ms_by_batch={1: 10.0, 2: 5.0, 4: 5.0},
    )
    by_name = {item.name: item.value for item in metrics}

    assert by_name["best_batch_size"] == 2


def test_best_batch_size_ignores_near_peak_larger_batch() -> None:
    # batch 4 is within ~2% of peak but slower; it must not be selected.
    metrics = batch_sweep._build_batch_sweep_metrics(
        batch_sizes=[1, 2, 4],
        throughput_by_batch={1: 100.0, 2: 300.0, 4: 297.0},
        median_ms_by_batch={1: 10.0, 2: 5.0, 4: 5.1},
    )
    by_name = {item.name: item.value for item in metrics}

    assert by_name["best_batch_size"] == 2


def test_dataloader_benchmark_respects_worker_options() -> None:
    # num_workers=0 only: no worker subprocesses, fast and deterministic.
    result = dataloader.run_dataloader_benchmark(
        num_workers_options=[0], dataset_length=64, max_batches=2
    )

    assert result.name == "DataLoader"
    if result.status == "ok":
        by_name = {item.name: item.value for item in result.metrics}
        assert by_name.get("best_num_workers") == 0


def test_quick_benchmark_includes_dataloader_result(monkeypatch) -> None:
    # The quick suite must include a DataLoader benchmark so DataLoader advisor
    # recommendations can appear in the default agent/quick workflow.
    def fake_profile():
        return GPUBoostProfile(
            generated_at="2026-01-01T00:00:00+00:00",
            system=SystemProfile(os="Test OS", python_version="3.12"),
            torch_env=TorchEnvironmentProfile(
                torch_installed=True, cuda_available=False, device_count=0
            ),
            gpus=[],
            warnings=[],
        )

    def fake_benchmark(_device_index: int = 0) -> BenchmarkResult:
        return _named_skipped("Fake")

    def fake_dataloader(_device_index: int = 0) -> BenchmarkResult:
        return _named_skipped("DataLoader")

    monkeypatch.setattr(runner, "collect_profile", fake_profile)
    monkeypatch.setattr(runner, "run_matmul_benchmark", fake_benchmark)
    monkeypatch.setattr(runner, "run_mixed_precision_benchmark", fake_benchmark)
    monkeypatch.setattr(runner, "run_batch_sweep_benchmark", fake_benchmark)
    monkeypatch.setattr(runner, "run_quick_dataloader_benchmark", fake_dataloader)

    suite = runner.run_quick_benchmark()
    names = [result.name for result in suite.results]

    assert "DataLoader" in names


def _named_skipped(name: str) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        status="skipped",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:00+00:00",
        duration_sec=0.0,
        metrics=[],
        warnings=[],
        error=None,
    )


def test_batch_sweep_metric_names_are_unique() -> None:
    metrics = batch_sweep._build_batch_sweep_metrics(
        batch_sizes=[1, 2, 4],
        throughput_by_batch={1: 100.0, 2: 180.0, 4: 300.0},
        median_ms_by_batch={1: 10.0, 2: 11.0, 4: 13.0},
    )
    names = [item.name for item in metrics]

    assert len(names) == len(set(names))
    assert "batch_1_images_per_sec" in names


def test_benchmark_result_preserves_warnings() -> None:
    result = BenchmarkResult(
        name="Warn",
        status="ok",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        duration_sec=1.0,
        metrics=[],
        warnings=["important warning"],
        error=None,
    )

    assert result.to_dict()["warnings"] == ["important warning"]



def test_synthetic_image_dataset_is_deterministic_per_index() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    ds = dataloader.SyntheticImageDataset(length=8)

    first_a, _ = ds[0]
    first_b, _ = ds[0]
    second, _ = ds[1]

    assert torch.equal(first_a, first_b)  # same index -> identical sample
    assert not torch.equal(first_a, second)  # different index -> different sample
