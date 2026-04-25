"""Batch size sweep benchmark for GPUBoost Phase 2."""

from __future__ import annotations

import time

from gpuboost.benchmarks.common import (
    clear_cuda_memory,
    is_oom_error,
    make_error_result,
    make_skipped_result,
    metric,
    require_cuda,
    time_cuda_callable,
    utc_now,
)
from gpuboost.schemas.benchmark_result import BenchmarkResult


class SyntheticConvNet:
    """Factory for a heavier synthetic image inference model."""

    @staticmethod
    def build(torch):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 1000),
        )


def _build_batch_sweep_metrics(
    batch_sizes: list[int],
    throughput_by_batch: dict[int, float],
    median_ms_by_batch: dict[int, float],
):
    best_batch_size = None
    best_images_per_sec = None
    if throughput_by_batch:
        peak_images_per_sec = max(throughput_by_batch.values())
        near_peak_threshold = peak_images_per_sec * 0.98
        best_batch_size = max(
            batch_size
            for batch_size, images_per_sec in throughput_by_batch.items()
            if images_per_sec >= near_peak_threshold
        )
        best_images_per_sec = throughput_by_batch[best_batch_size]

    batch_1_images_per_sec = throughput_by_batch.get(1)
    speedup_vs_batch_1 = (
        best_images_per_sec / batch_1_images_per_sec
        if best_images_per_sec is not None
        and batch_1_images_per_sec not in (None, 0)
        else None
    )
    max_successful_batch_size = max(throughput_by_batch) if throughput_by_batch else None

    metrics = [
        metric("best_batch_size", best_batch_size),
        metric("best_images_per_sec", best_images_per_sec, "images/sec"),
        metric("speedup_vs_batch_1", speedup_vs_batch_1, "x"),
        metric("max_successful_batch_size", max_successful_batch_size),
    ]
    for batch_size in batch_sizes:
        metrics.append(
            metric(
                f"batch_{batch_size}_images_per_sec",
                throughput_by_batch.get(batch_size),
                "images/sec",
            )
        )
        metrics.append(
            metric(
                f"batch_{batch_size}_median_ms",
                median_ms_by_batch.get(batch_size),
                "ms",
            )
        )

    return metrics


def run_batch_sweep_benchmark(device_index: int = 0) -> BenchmarkResult:
    """Measure forward-pass throughput across synthetic image batch sizes."""

    name = "Batch Size Sweep"
    available, reason = require_cuda(device_index)
    if not available:
        return make_skipped_result(name, reason or "CUDA is unavailable.")

    started_at = utc_now()
    started = time.perf_counter()
    warnings: list[str] = []

    try:
        import torch

        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        torch.backends.cudnn.benchmark = True
        model = (
            SyntheticConvNet.build(torch)
            .to(device)
            .to(memory_format=torch.channels_last)
            .eval()
        )
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        inner_forwards = 4
        throughput_by_batch: dict[int, float] = {}
        median_ms_by_batch: dict[int, float] = {}

        with torch.inference_mode():
            for batch_size in batch_sizes:
                try:
                    x = torch.randn(
                        (batch_size, 3, 224, 224),
                        device=device,
                    ).to(memory_format=torch.channels_last)

                    def work(batch=x) -> None:
                        for _ in range(inner_forwards):
                            model(batch)

                    timing = time_cuda_callable(work, warmup=8, repeats=20)
                    median_ms = float(timing["median_ms"]) / inner_forwards
                    median_ms_by_batch[batch_size] = median_ms
                    throughput_by_batch[batch_size] = batch_size / (median_ms / 1000)
                except RuntimeError as exc:
                    if is_oom_error(exc):
                        warnings.append(
                            f"Stopped batch sweep at batch size {batch_size}: CUDA OOM."
                        )
                        clear_cuda_memory()
                        break
                    raise
                finally:
                    try:
                        del x
                    except UnboundLocalError:
                        pass
                    clear_cuda_memory()

        metrics = _build_batch_sweep_metrics(
            batch_sizes,
            throughput_by_batch,
            median_ms_by_batch,
        )
        metrics.append(metric("inner_forwards_per_timing_sample", inner_forwards))
        best_batch_size = next(
            item.value for item in metrics if item.name == "best_batch_size"
        )

        if best_batch_size is not None and best_batch_size <= 4:
            warnings.append(
                "Best batch size was very small; benchmark may be "
                "CPU/Python-overhead limited or the GPU may be power/thermal "
                "constrained."
            )

        ended_at = utc_now()
        return BenchmarkResult(
            name=name,
            status="ok",
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=round(time.perf_counter() - started, 4),
            metrics=metrics,
            warnings=warnings,
            error=None,
        )
    except Exception as exc:
        clear_cuda_memory()
        result = make_error_result(name, started_at, exc, warnings)
        result.duration_sec = round(time.perf_counter() - started, 4)
        return result
