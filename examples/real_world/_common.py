"""Shared helpers for GPUBoost real-world demo workloads."""

from __future__ import annotations

import json
import statistics
import sys
import time
import warnings
from contextlib import nullcontext
from typing import Any, Callable


_TORCH_NUMPY_WARNING = r"Failed to initialize NumPy:.*"


def load_torch():
    """Import torch while suppressing a known optional NumPy warning."""

    warnings.filterwarnings(
        "ignore",
        message=_TORCH_NUMPY_WARNING,
        category=UserWarning,
    )
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only without torch.
        raise SystemExit("PyTorch is required to run this example.") from exc
    return torch


def positive_int(value: str) -> int:
    """Parse a positive integer command-line value."""

    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"Expected a positive integer, got: {value}")
    return parsed


def resolve_device(torch_module, requested_device: str) -> tuple[str, bool]:
    """Resolve auto/cpu/cuda into a torch device string and CUDA flag."""

    requested = requested_device.strip().lower()
    cuda_available = bool(torch_module.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    if requested in {"auto", "cuda"} and cuda_available:
        return "cuda", True
    return "cpu", False


def set_deterministic_seed(torch_module, seed: int) -> None:
    """Seed PyTorch and choose deterministic-friendly settings."""

    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def synchronize(torch_module, cuda_available: bool) -> None:
    """Synchronize CUDA only when CUDA is active."""

    if cuda_available:
        torch_module.cuda.synchronize()


def timed_steps(
    step: Callable[[], int],
    *,
    warmup_steps: int,
    measured_steps: int,
    torch_module,
    cuda_available: bool,
) -> tuple[float, float, int]:
    """Run warmup and measured steps, returning throughput and median latency."""

    for _ in range(warmup_steps):
        step()

    synchronize(torch_module, cuda_available)
    durations = []
    samples = 0
    for _ in range(measured_steps):
        start = time.perf_counter()
        samples += step()
        synchronize(torch_module, cuda_available)
        durations.append(time.perf_counter() - start)

    total_seconds = sum(durations)
    samples_per_sec = samples / total_seconds if total_seconds > 0 else 0.0
    median_step_ms = statistics.median(durations) * 1000.0 if durations else 0.0
    return samples_per_sec, median_step_ms, samples


def autocast_context(torch_module, cuda_available: bool, enabled: bool):
    """Return an AMP autocast context only when CUDA AMP is useful."""

    use_amp = bool(enabled and cuda_available)
    if not use_amp:
        return nullcontext()
    if hasattr(torch_module, "amp"):
        return torch_module.amp.autocast("cuda", enabled=True)
    return torch_module.cuda.amp.autocast(enabled=True)


def make_grad_scaler(torch_module, cuda_available: bool, enabled: bool):
    """Create a CUDA AMP GradScaler with compatibility for older torch builds."""

    use_amp = bool(enabled and cuda_available)
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "GradScaler"):
        return torch_module.amp.GradScaler("cuda", enabled=use_amp)
    return torch_module.cuda.amp.GradScaler(enabled=use_amp)


def benchmark_metrics(
    samples_per_sec: float,
    median_step_ms: float,
) -> list[dict[str, Any]]:
    """Return the benchmark metrics required by GPUBoost comparison tooling."""

    return [
        {
            "name": "samples_per_sec",
            "value": float(samples_per_sec),
            "unit": "samples/sec",
        },
        {"name": "median_step_ms", "value": float(median_step_ms), "unit": "ms"},
    ]


def emit_result(
    *,
    benchmark_json: bool,
    workload_family: str,
    variant: str,
    cuda_available: bool,
    samples_per_sec: float,
    median_step_ms: float,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Emit benchmark JSON or a short human-readable summary."""

    metrics = benchmark_metrics(samples_per_sec, median_step_ms)
    metadata: dict[str, Any] = {
        "example": "real_world",
        "workload_family": workload_family,
        "variant": variant,
        "cuda_available": cuda_available,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    payload = {
        "results": [
            {
                "name": f"{workload_family}:{variant}",
                "status": "ok",
                "metrics": metrics,
            }
        ],
        "metadata": metadata,
    }
    if benchmark_json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return

    print(
        f"{workload_family} {variant}: "
        f"{samples_per_sec:.2f} samples/sec, {median_step_ms:.2f} ms/step"
    )
