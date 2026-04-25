"""Shared helpers for GPUBoost Phase 2 benchmarks."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypeVar

from gpuboost.schemas.benchmark_result import BenchmarkMetric, BenchmarkResult

T = TypeVar("T")


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


def require_cuda(device_index: int = 0) -> tuple[bool, str | None]:
    """Return whether PyTorch CUDA is available for the requested device."""

    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch could not be imported: {exc}"

    try:
        if not torch.cuda.is_available():
            return False, "CUDA is not available in this PyTorch environment."
        device_count = int(torch.cuda.device_count())
        if device_count <= device_index:
            return (
                False,
                f"CUDA device {device_index} is unavailable; found {device_count}.",
            )
    except Exception as exc:
        return False, f"CUDA availability check failed: {exc}"

    return True, None


def get_default_device(device_index: int = 0):
    """Return the selected CUDA torch.device, or None when CUDA is unavailable."""

    available, _ = require_cuda(device_index)
    if not available:
        return None

    import torch

    return torch.device(f"cuda:{device_index}")


def synchronize_if_cuda() -> None:
    """Synchronize the current CUDA device when CUDA is available."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timed_cuda(fn: Callable[[], T], warmup: int = 3, repeats: int = 10) -> float:
    """Time a CUDA workload and return the median elapsed seconds."""

    timing = time_cuda_callable(fn, warmup=warmup, repeats=repeats)
    return float(timing["median_ms"]) / 1000


def time_cuda_callable(
    fn: Callable[[], T],
    warmup: int = 10,
    repeats: int = 30,
) -> dict[str, float | int]:
    """Time a CUDA callable with CUDA events and return millisecond stats."""

    import torch

    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    timings_ms: list[float] = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        timings_ms.append(float(start_event.elapsed_time(end_event)))

    return {
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "repeats": repeats,
    }


def is_oom_error(exc: BaseException) -> bool:
    """Return whether an exception looks like CUDA out-of-memory."""

    return "out of memory" in str(exc).lower()


def clear_cuda_memory() -> None:
    """Synchronize and release cached CUDA memory when possible."""

    synchronize_if_cuda()
    safe_empty_cache()


def round_metric(value: float | int | str | bool | None, digits: int = 3):
    """Round numeric metrics while preserving strings, booleans, and None."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    return round(float(value), digits)


def safe_empty_cache() -> None:
    """Release cached CUDA memory when possible."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def format_number(value: float | int | None, digits: int = 2) -> str:
    """Format a number for human output."""

    if value is None:
        return "Unknown"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def estimate_tensor_core_support_from_capability(
    compute_capability: str | None,
) -> bool | None:
    """Estimate Tensor Core support from CUDA compute capability."""

    if compute_capability is None:
        return None
    try:
        major = int(compute_capability.split(".", maxsplit=1)[0])
    except (ValueError, IndexError):
        return None
    return major >= 7


def make_skipped_result(name: str, reason: str) -> BenchmarkResult:
    """Create a skipped benchmark result."""

    now = utc_now()
    return BenchmarkResult(
        name=name,
        status="skipped",
        started_at=now,
        ended_at=now,
        duration_sec=0.0,
        metrics=[],
        warnings=[reason],
        error=None,
    )


def make_error_result(
    name: str,
    started_at: str,
    error: Exception | str,
    warnings: list[str] | None = None,
) -> BenchmarkResult:
    """Create an error benchmark result."""

    ended_at = utc_now()
    return BenchmarkResult(
        name=name,
        status="error",
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=0.0,
        metrics=[],
        warnings=warnings or [],
        error=str(error),
    )


def metric(name: str, value: float | int | str | bool | None, unit: str | None = None):
    """Create a benchmark metric."""

    return BenchmarkMetric(name=name, value=value, unit=unit)

