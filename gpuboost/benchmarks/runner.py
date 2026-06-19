"""Benchmark suite runner for GPUBoost Phase 2."""

from __future__ import annotations

import time
from collections.abc import Callable

from gpuboost.benchmarks.batch_sweep import run_batch_sweep_benchmark
from gpuboost.benchmarks.common import make_error_result, utc_now
from gpuboost.benchmarks.dataloader import (
    run_dataloader_benchmark,
    run_quick_dataloader_benchmark,
)
from gpuboost.benchmarks.matmul import run_matmul_benchmark
from gpuboost.benchmarks.mixed_precision import run_mixed_precision_benchmark
from gpuboost.inspector.profile import collect_profile
from gpuboost.schemas.benchmark_result import BenchmarkResult, BenchmarkSuiteResult


BenchmarkFn = Callable[[int], BenchmarkResult]


def _safe_run(fn: BenchmarkFn, device_index: int) -> BenchmarkResult:
    started_at = utc_now()
    started = time.perf_counter()
    try:
        return fn(device_index)
    except Exception as exc:
        result = make_error_result(fn.__name__, started_at, exc)
        result.duration_sec = round(time.perf_counter() - started, 4)
        return result


def _suite_result(
    benchmark_functions: list[BenchmarkFn],
    device_index: int,
) -> BenchmarkSuiteResult:
    profile = collect_profile()
    gpu_name = None
    for gpu in profile.gpus:
        if gpu.index == device_index:
            gpu_name = gpu.name
            break
    if gpu_name is None and profile.gpus:
        gpu_name = profile.gpus[0].name

    return BenchmarkSuiteResult(
        generated_at=utc_now(),
        gpu_name=gpu_name,
        cuda_available=profile.torch_env.cuda_available,
        device_index=device_index if profile.torch_env.cuda_available else None,
        results=[_safe_run(fn, device_index) for fn in benchmark_functions],
        warnings=profile.warnings,
    )


def run_quick_benchmark(device_index: int = 0) -> BenchmarkSuiteResult:
    """Run the Phase 2 quick benchmark suite."""

    return _suite_result(
        [
            run_matmul_benchmark,
            run_mixed_precision_benchmark,
            run_batch_sweep_benchmark,
            run_quick_dataloader_benchmark,
        ],
        device_index,
    )


def run_full_benchmark(device_index: int = 0) -> BenchmarkSuiteResult:
    """Run the full Phase 2 benchmark suite."""

    return _suite_result(
        [
            run_matmul_benchmark,
            run_mixed_precision_benchmark,
            run_batch_sweep_benchmark,
            run_dataloader_benchmark,
        ],
        device_index,
    )

