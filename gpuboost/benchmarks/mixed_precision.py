"""Mixed precision training benchmark for GPUBoost Phase 2."""

from __future__ import annotations

import time

from gpuboost.benchmarks.common import (
    make_error_result,
    make_skipped_result,
    metric,
    require_cuda,
    safe_empty_cache,
    synchronize_if_cuda,
    utc_now,
)
from gpuboost.schemas.benchmark_result import BenchmarkResult


def _make_model(torch):
    return torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 10),
    )


def _autocast(torch, enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def _grad_scaler(torch):
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _run_training_loop(
    torch,
    model,
    optimizer,
    criterion,
    x,
    y,
    iterations: int,
    use_amp: bool,
) -> float:
    scaler = None
    if use_amp:
        scaler = _grad_scaler(torch)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(torch, enabled=use_amp):
            output = model(x)
            loss = criterion(output, y)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    synchronize_if_cuda()
    started = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(torch, enabled=use_amp):
            output = model(x)
            loss = criterion(output, y)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    synchronize_if_cuda()
    return time.perf_counter() - started


def run_mixed_precision_benchmark(device_index: int = 0) -> BenchmarkResult:
    """Compare FP32 and AMP training throughput on a synthetic MLP."""

    name = "Mixed Precision"
    available, reason = require_cuda(device_index)
    if not available:
        return make_skipped_result(name, reason or "CUDA is unavailable.")

    started_at = utc_now()
    started = time.perf_counter()
    warnings: list[str] = []
    batch_size = 256
    iterations = 20

    try:
        import torch

        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        x = torch.randn((batch_size, 1024), device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        criterion = torch.nn.CrossEntropyLoss()

        fp32_model = _make_model(torch).to(device)
        fp32_optimizer = torch.optim.AdamW(fp32_model.parameters(), lr=1e-3)
        fp32_seconds = _run_training_loop(
            torch,
            fp32_model,
            fp32_optimizer,
            criterion,
            x,
            y,
            iterations,
            use_amp=False,
        )
        fp32_samples_per_sec = batch_size * iterations / fp32_seconds

        amp_samples_per_sec = None
        amp_speedup = None
        try:
            amp_model = _make_model(torch).to(device)
            amp_optimizer = torch.optim.AdamW(amp_model.parameters(), lr=1e-3)
            amp_seconds = _run_training_loop(
                torch,
                amp_model,
                amp_optimizer,
                criterion,
                x,
                y,
                iterations,
                use_amp=True,
            )
            amp_samples_per_sec = batch_size * iterations / amp_seconds
            amp_speedup = amp_samples_per_sec / fp32_samples_per_sec
        except Exception as exc:
            warnings.append(f"AMP benchmark failed: {exc}")

        ended_at = utc_now()
        return BenchmarkResult(
            name=name,
            status="ok",
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=round(time.perf_counter() - started, 4),
            metrics=[
                metric("fp32_samples_per_sec", fp32_samples_per_sec, "samples/sec"),
                metric("amp_samples_per_sec", amp_samples_per_sec, "samples/sec"),
                metric("amp_speedup_ratio", amp_speedup, "x"),
                metric("batch_size", batch_size),
                metric("iterations", iterations),
            ],
            warnings=warnings,
            error=None,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            warnings.append("Mixed precision benchmark hit CUDA OOM.")
        safe_empty_cache()
        result = make_error_result(name, started_at, exc, warnings)
        result.duration_sec = round(time.perf_counter() - started, 4)
        return result
    except Exception as exc:
        safe_empty_cache()
        result = make_error_result(name, started_at, exc, warnings)
        result.duration_sec = round(time.perf_counter() - started, 4)
        return result
    finally:
        safe_empty_cache()
