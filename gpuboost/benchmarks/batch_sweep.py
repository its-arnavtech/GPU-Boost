"""Batch size sweep benchmark for GPUBoost Phase 2."""

from __future__ import annotations

import time

from gpuboost.benchmarks.common import (
    make_error_result,
    make_skipped_result,
    metric,
    require_cuda,
    safe_empty_cache,
    timed_cuda,
    utc_now,
)
from gpuboost.schemas.benchmark_result import BenchmarkResult


def _make_cnn(torch):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 1000),
    )


def run_batch_sweep_benchmark(device_index: int = 0) -> BenchmarkResult:
    """Measure forward-pass throughput across synthetic batch sizes."""

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
        model = _make_cnn(torch).to(device).eval()
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        throughput_by_batch: dict[int, float] = {}

        with torch.no_grad():
            for batch_size in batch_sizes:
                try:
                    x = torch.randn((batch_size, 3, 224, 224), device=device)

                    def work(batch=x) -> None:
                        model(batch)

                    elapsed_sec = timed_cuda(work, warmup=2, repeats=8)
                    throughput_by_batch[batch_size] = batch_size / elapsed_sec
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        warnings.append(
                            f"Stopped batch sweep at batch size {batch_size}: CUDA OOM."
                        )
                        safe_empty_cache()
                        break
                    raise
                finally:
                    try:
                        del x
                    except UnboundLocalError:
                        pass
                    safe_empty_cache()

        best_batch_size = None
        best_images_per_sec = None
        if throughput_by_batch:
            best_batch_size = max(
                throughput_by_batch,
                key=lambda batch_size: throughput_by_batch[batch_size],
            )
            best_images_per_sec = throughput_by_batch[best_batch_size]

        batch_1_images_per_sec = throughput_by_batch.get(1)
        speedup_vs_batch_1 = (
            best_images_per_sec / batch_1_images_per_sec
            if best_images_per_sec is not None
            and batch_1_images_per_sec not in (None, 0)
            else None
        )
        max_successful_batch_size = (
            max(throughput_by_batch) if throughput_by_batch else None
        )

        metrics = [
            metric("best_batch_size", best_batch_size),
            metric("best_images_per_sec", best_images_per_sec, "images/sec"),
            metric("batch_1_images_per_sec", batch_1_images_per_sec, "images/sec"),
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
        safe_empty_cache()
        result = make_error_result(name, started_at, exc, warnings)
        result.duration_sec = round(time.perf_counter() - started, 4)
        return result
