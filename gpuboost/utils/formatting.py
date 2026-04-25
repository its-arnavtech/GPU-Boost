"""Formatting helpers for human-readable CLI output."""

from __future__ import annotations

from gpuboost.schemas.benchmark_result import BenchmarkResult, BenchmarkSuiteResult
from gpuboost.schemas.gpu_profile import GPUBoostProfile


def _display(value: object | None, suffix: str = "") -> str:
    if value is None:
        return "Unknown"
    return f"{value}{suffix}"


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "Unknown"
    return "Yes" if value else "No"


def _metric_value(result: BenchmarkResult, name: str) -> object | None:
    for item in result.metrics:
        if item.name == name:
            return item.value
    return None


def format_profile(profile: GPUBoostProfile) -> str:
    """Format a Phase 1 profile for `gpuboost info`."""

    lines = [
        "GPUBoost Phase 1 GPU Inspector",
        "",
        "System:",
        f"- OS: {_display(profile.system.os)}",
        f"- Python: {_display(profile.system.python_version)}",
        f"- CPU: {_display(profile.system.cpu_model)}",
        f"- CPU cores: physical {_display(profile.system.cpu_cores_physical)}, "
        f"logical {_display(profile.system.cpu_cores_logical)}",
        f"- RAM: {_display(profile.system.ram_total_gb, ' GB')}",
        "",
        "PyTorch/CUDA:",
        f"- PyTorch installed: {_yes_no(profile.torch_env.torch_installed)}",
        f"- PyTorch version: {_display(profile.torch_env.torch_version)}",
        f"- CUDA available: {_yes_no(profile.torch_env.cuda_available)}",
        f"- Torch CUDA version: {_display(profile.torch_env.torch_cuda_version)}",
        f"- cuDNN version: {_display(profile.torch_env.cudnn_version)}",
        f"- Device count: {_display(profile.torch_env.device_count)}",
        "",
        "GPUs:",
    ]

    if not profile.gpus:
        lines.append("- No NVIDIA GPUs detected.")
    else:
        for gpu in profile.gpus:
            lines.extend(
                [
                    f"- GPU {gpu.index}: {_display(gpu.name)}",
                    f"  UUID: {_display(gpu.uuid)}",
                    f"  Architecture: {_display(gpu.architecture)}",
                    "  VRAM: "
                    f"{_display(gpu.used_vram_mb)} used / "
                    f"{_display(gpu.free_vram_mb)} free / "
                    f"{_display(gpu.total_vram_mb)} total MB",
                    "  Compute capability: "
                    f"{_display(gpu.cuda_compute_capability)}",
                    f"  Tensor Cores: {_yes_no(gpu.tensor_cores_supported)}",
                    "  Utilization: "
                    f"GPU {_display(gpu.utilization_gpu_percent, '%')}, "
                    f"memory {_display(gpu.utilization_memory_percent, '%')}",
                    f"  Temperature: {_display(gpu.temperature_c, ' C')}",
                    "  Power: "
                    f"{_display(gpu.power_draw_w, ' W')} / "
                    f"{_display(gpu.power_limit_w, ' W')}",
                ]
            )

    if profile.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in profile.warnings)

    return "\n".join(lines)


def format_benchmark_suite(suite: BenchmarkSuiteResult) -> str:
    """Format a Phase 2 benchmark suite for terminal output."""

    lines = [
        "GPUBoost Phase 2 Benchmark Suite",
        "",
        f"GPU: {_display(suite.gpu_name)}",
        f"CUDA available: {_yes_no(suite.cuda_available)}",
        f"Device index: {_display(suite.device_index)}",
        "",
    ]

    result_by_name = {result.name: result for result in suite.results}

    matmul = result_by_name.get("Matrix Multiplication")
    if matmul is not None:
        lines.extend(
            [
                "Matrix Multiplication:",
                f"- Status: {matmul.status}",
                "- Best FP32: "
                f"{_display(_metric_value(matmul, 'best_fp32_tflops'), ' TFLOPS')}",
                "- Best FP16: "
                f"{_display(_metric_value(matmul, 'best_fp16_tflops'), ' TFLOPS')}",
                "- FP16 speedup: "
                f"{_display(_metric_value(matmul, 'fp16_speedup_ratio'), 'x')}",
                "- Tensor Cores likely active: "
                f"{_yes_no(_metric_value(matmul, 'tensor_cores_likely_active'))}",
                "",
            ]
        )

    mixed = result_by_name.get("Mixed Precision")
    if mixed is not None:
        lines.extend(
            [
                "Mixed Precision:",
                f"- Status: {mixed.status}",
                "- FP32 throughput: "
                f"{_display(_metric_value(mixed, 'fp32_samples_per_sec'), ' samples/sec')}",
                "- AMP throughput: "
                f"{_display(_metric_value(mixed, 'amp_samples_per_sec'), ' samples/sec')}",
                "- AMP speedup: "
                f"{_display(_metric_value(mixed, 'amp_speedup_ratio'), 'x')}",
                "",
            ]
        )

    batch = result_by_name.get("Batch Size Sweep")
    if batch is not None:
        lines.extend(
            [
                "Batch Size Sweep:",
                f"- Status: {batch.status}",
                f"- Best batch size: {_display(_metric_value(batch, 'best_batch_size'))}",
                "- Best throughput: "
                f"{_display(_metric_value(batch, 'best_images_per_sec'), ' images/sec')}",
                "- Speedup vs batch=1: "
                f"{_display(_metric_value(batch, 'speedup_vs_batch_1'), 'x')}",
                "",
            ]
        )

    dataloader = result_by_name.get("DataLoader")
    if dataloader is not None:
        lines.extend(
            [
                "DataLoader:",
                f"- Status: {dataloader.status}",
                "- Best num_workers: "
                f"{_display(_metric_value(dataloader, 'best_num_workers'))}",
                "- Best pin_memory: "
                f"{_display(_metric_value(dataloader, 'best_pin_memory'))}",
                "- Best throughput: "
                f"{_display(_metric_value(dataloader, 'best_samples_per_sec'), ' samples/sec')}",
                "",
            ]
        )

    warnings = list(suite.warnings)
    for result in suite.results:
        warnings.extend(result.warnings)
        if result.error:
            warnings.append(f"{result.name} error: {result.error}")

    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines).rstrip()
