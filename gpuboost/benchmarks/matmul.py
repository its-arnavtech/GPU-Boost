"""Matrix multiplication benchmark for GPUBoost Phase 2."""

from __future__ import annotations

import time

from gpuboost.benchmarks.common import (
    estimate_tensor_core_support_from_capability,
    make_error_result,
    make_skipped_result,
    metric,
    require_cuda,
    safe_empty_cache,
    timed_cuda,
    utc_now,
)
from gpuboost.schemas.benchmark_result import BenchmarkResult


def _is_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def run_matmul_benchmark(device_index: int = 0) -> BenchmarkResult:
    """Benchmark CUDA matrix multiplication across sizes and dtypes."""

    name = "Matrix Multiplication"
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
        sizes = [512, 1024, 2048, 4096]
        dtype_specs = [
            ("fp32", torch.float32),
            ("fp16", torch.float16),
        ]

        try:
            major, minor = torch.cuda.get_device_capability(device_index)
            compute_capability = f"{major}.{minor}"
        except Exception as exc:
            compute_capability = None
            warnings.append(f"Compute capability detection failed: {exc}")

        best: dict[str, tuple[float, int]] = {}
        per_case: list[tuple[str, int, float]] = []

        for dtype_name, dtype in dtype_specs:
            for size in sizes:
                try:
                    a = torch.randn((size, size), device=device, dtype=dtype)
                    b = torch.randn((size, size), device=device, dtype=dtype)

                    def work(left=a, right=b) -> None:
                        torch.matmul(left, right)

                    elapsed_sec = timed_cuda(work, warmup=2, repeats=5)
                    tflops = (2 * size**3) / elapsed_sec / 1e12
                    per_case.append((dtype_name, size, tflops))

                    current_best = best.get(dtype_name)
                    if current_best is None or tflops > current_best[0]:
                        best[dtype_name] = (tflops, size)
                except RuntimeError as exc:
                    if _is_oom(exc):
                        warnings.append(
                            f"Skipped {dtype_name} matmul size {size}: CUDA OOM."
                        )
                        safe_empty_cache()
                        continue
                    raise
                finally:
                    try:
                        del a
                        del b
                    except UnboundLocalError:
                        pass
                    safe_empty_cache()

        best_fp32 = best.get("fp32")
        best_fp16 = best.get("fp16")
        fp32_tflops = best_fp32[0] if best_fp32 else None
        fp16_tflops = best_fp16[0] if best_fp16 else None
        fp16_speedup = (
            fp16_tflops / fp32_tflops
            if fp16_tflops is not None and fp32_tflops not in (None, 0)
            else None
        )
        tensor_cores_supported = estimate_tensor_core_support_from_capability(
            compute_capability
        )
        tensor_cores_likely_active = (
            bool(tensor_cores_supported and fp16_speedup and fp16_speedup > 1.2)
        )

        metrics = [
            metric("best_fp32_tflops", fp32_tflops, "TFLOPS"),
            metric("best_fp16_tflops", fp16_tflops, "TFLOPS"),
            metric("fp16_speedup_ratio", fp16_speedup, "x"),
            metric("best_size_fp32", best_fp32[1] if best_fp32 else None, "n"),
            metric("best_size_fp16", best_fp16[1] if best_fp16 else None, "n"),
            metric("tensor_cores_likely_active", tensor_cores_likely_active),
        ]
        for dtype_name, size, tflops in per_case:
            metrics.append(metric(f"{dtype_name}_size_{size}_tflops", tflops, "TFLOPS"))

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
