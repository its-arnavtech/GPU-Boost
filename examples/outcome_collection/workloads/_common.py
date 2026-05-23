"""Shared helpers for controlled outcome workload scripts."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import warnings
from typing import Any, Callable

_TORCH_NUMPY_WARNING = r"Failed to initialize NumPy:.*"


def smoke_mode() -> bool:
    """Return whether workload scripts should run in tiny test mode."""

    return os.environ.get("GPUBOOST_OUTCOME_SMOKE") == "1"


def load_torch():
    """Import torch when available without making it a hard script requirement."""

    warnings.filterwarnings(
        "ignore",
        message=_TORCH_NUMPY_WARNING,
        category=UserWarning,
    )
    try:
        import torch
    except Exception:
        return None
    return torch


def parse_bool(value: str) -> bool:
    """Parse a simple true/false command-line value."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected true or false, got: {value}")


def positive_int(value: str) -> int:
    """Parse a positive integer command-line value."""

    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"Expected a positive integer, got: {value}")
    return parsed


def resolve_device(
    torch_module,
    requested_device: str = "auto",
) -> tuple[str, bool, str | None]:
    """Return device string, CUDA availability, and best-effort GPU name."""

    requested = requested_device.strip().lower()
    if (
        torch_module is not None
        and requested in {"auto", "cuda"}
        and torch_module.cuda.is_available()
    ):
        device_index = torch_module.cuda.current_device()
        return "cuda", True, torch_module.cuda.get_device_name(device_index)
    return "cpu", False, None


def synchronize(torch_module, cuda_available: bool) -> None:
    """Synchronize CUDA timing only when needed."""

    if torch_module is not None and cuda_available:
        torch_module.cuda.synchronize()


def timed_iterations(
    step: Callable[[], int],
    *,
    warmup_iterations: int,
    measured_iterations: int,
    torch_module,
    cuda_available: bool,
) -> tuple[float, float, int]:
    """Run warmups and measured iterations, returning throughput and median ms."""

    for _ in range(warmup_iterations):
        step()

    synchronize(torch_module, cuda_available)
    durations = []
    samples = 0
    for _ in range(measured_iterations):
        start = time.perf_counter()
        samples += step()
        synchronize(torch_module, cuda_available)
        durations.append(time.perf_counter() - start)

    total_seconds = sum(durations)
    samples_per_sec = samples / total_seconds if total_seconds > 0 else 0.0
    median_step_ms = statistics.median(durations) * 1000.0 if durations else 0.0
    return samples_per_sec, median_step_ms, samples


def emit_result(
    *,
    workload: str,
    variant: str,
    metrics: list[dict[str, Any]],
    cuda_available: bool,
    gpu_name: str | None,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Write a GPUBoost-compatible benchmark JSON object to stdout."""

    metadata: dict[str, Any] = {
        "workload": workload,
        "variant": variant,
        "cuda_available": cuda_available,
    }
    if gpu_name is not None:
        metadata["gpu_name"] = gpu_name
    if extra_metadata:
        metadata.update(extra_metadata)

    json.dump(
        {
            "results": [
                {
                    "name": f"Controlled {workload.title()} Workload",
                    "status": "ok",
                    "metrics": metrics,
                }
            ],
            "metadata": metadata,
        },
        sys.stdout,
        sort_keys=True,
    )
    sys.stdout.write("\n")


def fallback_measurement(*, batch_size: int, iterations: int) -> tuple[float, float, int]:
    """Small deterministic CPU fallback for environments without PyTorch."""

    durations = []
    samples = 0
    values = list(range(256))
    for _ in range(iterations):
        start = time.perf_counter()
        total = 0
        for value in values:
            total += (value * value) % 97
        if total < 0:
            raise RuntimeError("unreachable")
        samples += batch_size
        durations.append(time.perf_counter() - start)

    total_seconds = sum(durations)
    samples_per_sec = samples / total_seconds if total_seconds > 0 else 0.0
    median_step_ms = statistics.median(durations) * 1000.0 if durations else 0.0
    return samples_per_sec, median_step_ms, samples
