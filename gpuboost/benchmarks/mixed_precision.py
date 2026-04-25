"""Mixed precision training benchmark for GPUBoost Phase 2."""

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


class TensorCoreMLP:
    """Factory for a Tensor Core-friendly dense model."""

    input_dim = 4096
    hidden_dim = 4096
    output_dim = 4096

    @classmethod
    def build(cls, torch):
        return torch.nn.Sequential(
            torch.nn.Linear(cls.input_dim, cls.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(cls.hidden_dim, cls.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(cls.hidden_dim, cls.output_dim),
        )


def _autocast(torch):
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)


def _grad_scaler(torch):
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _make_training_state(torch, device, batch_size: int):
    torch.manual_seed(1337)
    fp32_model = TensorCoreMLP.build(torch).to(device)
    torch.manual_seed(1337)
    amp_model = TensorCoreMLP.build(torch).to(device)

    x = torch.randn((batch_size, TensorCoreMLP.input_dim), device=device)
    target = torch.randn((batch_size, TensorCoreMLP.output_dim), device=device)
    criterion = torch.nn.MSELoss()
    fp32_optimizer = torch.optim.SGD(fp32_model.parameters(), lr=1e-3)
    amp_optimizer = torch.optim.SGD(amp_model.parameters(), lr=1e-3)

    return (
        fp32_model,
        amp_model,
        fp32_optimizer,
        amp_optimizer,
        criterion,
        x,
        target,
    )


def _fp32_step(torch, model, optimizer, criterion, x, target) -> None:
    del torch
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()


def _amp_step(torch, model, optimizer, criterion, scaler, x, target) -> None:
    optimizer.zero_grad(set_to_none=True)
    with _autocast(torch):
        output = model(x)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def run_mixed_precision_benchmark(device_index: int = 0) -> BenchmarkResult:
    """Compare FP32 and AMP training on a Tensor Core-friendly synthetic MLP."""

    name = "Mixed Precision"
    available, reason = require_cuda(device_index)
    if not available:
        return make_skipped_result(name, reason or "CUDA is unavailable.")

    started_at = utc_now()
    started = time.perf_counter()
    warnings: list[str] = []
    batch_size_options = [256, 128, 64, 32, 16]
    selected_batch_size = None

    try:
        import torch

        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")

        state = None
        for batch_size in batch_size_options:
            try:
                state = _make_training_state(torch, device, batch_size)
                selected_batch_size = batch_size
                if batch_size != batch_size_options[0]:
                    warnings.append(
                        "Selected fallback batch size "
                        f"{batch_size} after CUDA OOM at larger sizes."
                    )
                break
            except RuntimeError as exc:
                if is_oom_error(exc):
                    warnings.append(
                        f"Mixed precision setup hit CUDA OOM at batch size {batch_size}."
                    )
                    clear_cuda_memory()
                    continue
                raise

        if state is None or selected_batch_size is None:
            return make_skipped_result(
                name,
                "Mixed precision benchmark could not allocate the model at any "
                "fallback batch size.",
            )

        (
            fp32_model,
            amp_model,
            fp32_optimizer,
            amp_optimizer,
            criterion,
            x,
            target,
        ) = state
        scaler = _grad_scaler(torch)

        def fp32_step() -> None:
            _fp32_step(torch, fp32_model, fp32_optimizer, criterion, x, target)

        def amp_step() -> None:
            _amp_step(torch, amp_model, amp_optimizer, criterion, scaler, x, target)

        fp32_timing = time_cuda_callable(fp32_step, warmup=6, repeats=20)
        amp_timing = time_cuda_callable(amp_step, warmup=6, repeats=20)

        median_fp32_step_ms = float(fp32_timing["median_ms"])
        median_amp_step_ms = float(amp_timing["median_ms"])
        fp32_samples_per_sec = selected_batch_size / (median_fp32_step_ms / 1000)
        amp_samples_per_sec = selected_batch_size / (median_amp_step_ms / 1000)
        amp_speedup = amp_samples_per_sec / fp32_samples_per_sec

        if amp_speedup < 1.0:
            warnings.append(
                "AMP was slower than FP32 for this synthetic workload; this can "
                "happen on small models, low batch sizes, or thermally constrained "
                "systems."
            )

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
                metric("batch_size", selected_batch_size),
                metric("iterations", int(fp32_timing["repeats"])),
                metric("repeats", int(fp32_timing["repeats"])),
                metric("median_fp32_step_ms", median_fp32_step_ms, "ms"),
                metric("median_amp_step_ms", median_amp_step_ms, "ms"),
                metric("model_type", "TensorCoreMLP"),
                metric("input_dim", TensorCoreMLP.input_dim),
                metric("hidden_dim", TensorCoreMLP.hidden_dim),
            ],
            warnings=warnings,
            error=None,
        )
    except RuntimeError as exc:
        if is_oom_error(exc):
            warnings.append("Mixed precision benchmark hit CUDA OOM.")
        clear_cuda_memory()
        result = make_error_result(name, started_at, exc, warnings)
        result.duration_sec = round(time.perf_counter() - started, 4)
        return result
    except Exception as exc:
        clear_cuda_memory()
        result = make_error_result(name, started_at, exc, warnings)
        result.duration_sec = round(time.perf_counter() - started, 4)
        return result
    finally:
        clear_cuda_memory()

