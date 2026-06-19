"""DataLoader throughput benchmark for GPUBoost Phase 2."""

from __future__ import annotations

import platform
import time
import warnings as py_warnings

from gpuboost.benchmarks.common import (
    make_error_result,
    make_skipped_result,
    metric,
    safe_empty_cache,
    synchronize_if_cuda,
    utc_now,
)
from gpuboost.schemas.benchmark_result import BenchmarkResult


class SyntheticImageDataset:
    """Picklable synthetic image dataset for DataLoader benchmarks."""

    def __init__(self, length: int = 2048, shape: tuple[int, int, int] = (3, 64, 64)):
        self.length = length
        self.shape = shape

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        import torch

        # Deterministic, index-stable samples: a per-index seeded generator keeps
        # the benchmark reproducible across runs and workers, and measures
        # DataLoader/transfer overhead rather than RNG variance.
        generator = torch.Generator().manual_seed(index % self.length)
        image = torch.randn(self.shape, generator=generator)
        return image, torch.tensor(0, dtype=torch.long)


def _loader_kwargs(num_workers: int) -> dict[str, int]:
    if num_workers <= 0:
        return {}
    return {"prefetch_factor": 2}


def run_dataloader_benchmark(
    device_index: int = 0,
    *,
    num_workers_options: list[int] | None = None,
    dataset_length: int = 2048,
    max_batches: int = 8,
) -> BenchmarkResult:
    """Measure synthetic DataLoader throughput across workers and pin_memory."""

    name = "DataLoader"
    started_at = utc_now()
    started = time.perf_counter()
    warnings: list[str] = []
    batch_size = 64
    if num_workers_options is None:
        num_workers_options = [0, 1, 2, 4, 8]

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        return make_skipped_result(name, f"PyTorch DataLoader unavailable: {exc}")

    cuda_available = False
    device = None
    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available and int(torch.cuda.device_count()) > device_index:
            torch.cuda.set_device(device_index)
            device = torch.device(f"cuda:{device_index}")
    except Exception as exc:
        warnings.append(f"CUDA transfer timing disabled: {exc}")
        cuda_available = False
        device = None

    dataset = SyntheticImageDataset(length=dataset_length)
    pin_memory_options = [False, True]
    throughput: dict[tuple[int, bool], float] = {}

    for num_workers in num_workers_options:
        for pin_memory in pin_memory_options:
            try:
                with py_warnings.catch_warnings():
                    py_warnings.filterwarnings(
                        "ignore",
                        message=".*pin_memory.*",
                        category=UserWarning,
                    )
                    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        **_loader_kwargs(num_workers),
                    )

                    samples = 0
                    synchronize_if_cuda()
                    case_started = time.perf_counter()
                    for batch_index, (x, _) in enumerate(loader):
                        if batch_index >= max_batches:
                            break
                        if device is not None:
                            x = x.to(
                                device,
                                non_blocking=bool(pin_memory and cuda_available),
                            )
                        samples += int(x.shape[0])
                    synchronize_if_cuda()

                elapsed_sec = time.perf_counter() - case_started
                if elapsed_sec > 0:
                    throughput[(num_workers, pin_memory)] = samples / elapsed_sec
            except Exception as exc:
                warnings.append(
                    "DataLoader case failed "
                    f"(num_workers={num_workers}, pin_memory={pin_memory}): {exc}"
                )
            finally:
                safe_empty_cache()

    if not throughput:
        result = make_error_result(
            name,
            started_at,
            "All DataLoader benchmark cases failed.",
            warnings,
        )
        result.duration_sec = round(time.perf_counter() - started, 4)
        return result

    best_key = max(throughput, key=lambda key: throughput[key])
    best_num_workers, best_pin_memory = best_key
    best_samples_per_sec = throughput[best_key]

    comparable_worker = best_num_workers
    if (comparable_worker, False) not in throughput or (
        comparable_worker,
        True,
    ) not in throughput:
        comparable_worker = 0
    best_without_pin = throughput.get((comparable_worker, False))
    best_with_pin = throughput.get((comparable_worker, True))
    pin_memory_speedup_ratio = (
        best_with_pin / best_without_pin
        if best_with_pin is not None and best_without_pin not in (None, 0)
        else None
    )

    if platform.system() == "Windows" and best_num_workers == 0:
        warnings.append(
            "num_workers=0 was fastest for this synthetic Windows benchmark; "
            "real image datasets may benefit from workers when preprocessing or "
            "disk I/O is significant."
        )

    metrics = [
        metric("best_num_workers", best_num_workers),
        metric("best_pin_memory", best_pin_memory),
        metric("best_samples_per_sec", best_samples_per_sec, "samples/sec"),
        metric(
            "num_workers_0_samples_per_sec",
            max(
                (
                    value
                    for (workers, _pin), value in throughput.items()
                    if workers == 0
                ),
                default=None,
            ),
            "samples/sec",
        ),
        metric("pin_memory_speedup_ratio", pin_memory_speedup_ratio, "x"),
        metric("pin_memory_comparison_num_workers", comparable_worker),
        metric("batch_size", batch_size),
        metric("batches_measured", max_batches),
    ]
    for num_workers in num_workers_options:
        for pin_memory in pin_memory_options:
            metrics.append(
                metric(
                    f"num_workers_{num_workers}_pin_memory_{pin_memory}_samples_per_sec",
                    throughput.get((num_workers, pin_memory)),
                    "samples/sec",
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


def run_quick_dataloader_benchmark(device_index: int = 0) -> BenchmarkResult:
    """Run a lightweight DataLoader benchmark for the quick suite.

    Tests only ``num_workers`` in ``{0, 2}`` on a small dataset so the quick
    suite stays fast while still producing ``best_num_workers``/``pin_memory``
    signals the advisor needs to recommend DataLoader settings.
    """

    return run_dataloader_benchmark(
        device_index,
        num_workers_options=[0, 2],
        dataset_length=256,
        max_batches=4,
    )
