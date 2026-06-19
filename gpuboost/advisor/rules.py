"""Advisor rules for turning benchmark results into recommendations."""

from __future__ import annotations

import re

from gpuboost.advisor.scoring import confidence_from_signal, impact_from_speedup
from gpuboost.advisor.utils import format_speedup, get_metric, get_result_by_name
from gpuboost.schemas.benchmark_result import BenchmarkResult, BenchmarkSuiteResult
from gpuboost.schemas.recommendation import Recommendation


_MIXED_PRECISION_RESULT = "Mixed Precision"
_MATRIX_MULTIPLICATION_RESULT = "Matrix Multiplication"
_BATCH_SIZE_SWEEP_RESULT = "Batch Size Sweep"
_DATALOADER_RESULT = "DataLoader"

_MIXED_PRECISION_RELATED_METRICS = [
    "amp_speedup_ratio",
    "fp32_samples_per_sec",
    "amp_samples_per_sec",
    "median_fp32_step_ms",
    "median_amp_step_ms",
]

_TENSOR_CORE_RELATED_METRICS = [
    "fp16_speedup_ratio",
    "tensor_cores_likely_active",
    "best_fp16_tflops",
    "best_fp32_tflops",
]

_BATCH_SIZE_RELATED_METRICS = [
    "best_batch_size",
    "best_images_per_sec",
    "batch_1_images_per_sec",
    "speedup_vs_batch_1",
    "max_successful_batch_size",
]


_AMP_CODE_SNIPPET = """\
scaler = torch.amp.GradScaler("cuda")

for inputs, targets in dataloader:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
"""


def mixed_precision_rule(suite: BenchmarkSuiteResult) -> list[Recommendation]:
    """Recommend whether to use AMP based on the Mixed Precision benchmark."""

    result = get_result_by_name(suite, _MIXED_PRECISION_RESULT)
    if _result_unusable(result):
        return []

    amp_speedup_ratio = get_metric(result, "amp_speedup_ratio")
    if amp_speedup_ratio is None:
        return []

    if amp_speedup_ratio >= 1.10:
        return [_mixed_precision_enable_recommendation(result, amp_speedup_ratio)]
    if amp_speedup_ratio < 1.0:
        return [_mixed_precision_slow_recommendation(result, amp_speedup_ratio)]
    return [_mixed_precision_limited_recommendation(result, amp_speedup_ratio)]


def tensor_core_rule(suite: BenchmarkSuiteResult) -> list[Recommendation]:
    """Recommend Tensor Core-friendly workload choices from matmul results."""

    result = get_result_by_name(suite, _MATRIX_MULTIPLICATION_RESULT)
    if _result_unusable(result):
        return []

    fp16_speedup_ratio = get_metric(result, "fp16_speedup_ratio")
    if fp16_speedup_ratio is None:
        return []

    tensor_cores_likely_active = get_metric(result, "tensor_cores_likely_active")
    if tensor_cores_likely_active is True and fp16_speedup_ratio >= 2.0:
        return [_tensor_core_friendly_recommendation(result, fp16_speedup_ratio)]
    if fp16_speedup_ratio < 1.5:
        return [_tensor_core_weak_recommendation(result, fp16_speedup_ratio)]
    return []


def batch_size_rule(suite: BenchmarkSuiteResult) -> list[Recommendation]:
    """Recommend batch-size tuning actions from the batch sweep benchmark."""

    result = get_result_by_name(suite, _BATCH_SIZE_SWEEP_RESULT)
    if _result_unusable(result):
        return []

    recommendations: list[Recommendation] = []
    best_batch_size = get_metric(result, "best_batch_size")
    speedup_vs_batch_1 = get_metric(result, "speedup_vs_batch_1")

    increase_recommended = (
        speedup_vs_batch_1 is not None
        and best_batch_size is not None
        and speedup_vs_batch_1 >= 1.25
        and best_batch_size > 1
    )
    if increase_recommended:
        recommendations.append(
            _batch_size_increase_recommendation(
                result,
                best_batch_size,
                speedup_vs_batch_1,
            ),
        )

    # Only warn about limited scaling when we are NOT already recommending a
    # larger batch; otherwise the two recommendations contradict each other.
    if (
        not increase_recommended
        and best_batch_size is not None
        and best_batch_size <= 4
    ):
        recommendations.append(
            _batch_size_limited_recommendation(
                result,
                speedup_vs_batch_1,
            ),
        )

    return recommendations


def dataloader_rule(suite: BenchmarkSuiteResult) -> list[Recommendation]:
    """Recommend DataLoader worker and transfer settings."""

    result = get_result_by_name(suite, _DATALOADER_RESULT)
    if _result_unusable(result):
        return []

    recommendations: list[Recommendation] = []
    best_num_workers = get_metric(result, "best_num_workers")
    best_pin_memory = get_metric(result, "best_pin_memory")
    pin_memory_speedup_ratio = get_metric(result, "pin_memory_speedup_ratio")

    if best_num_workers is not None and best_num_workers > 0:
        recommendations.append(
            _dataloader_workers_recommendation(
                result,
                best_num_workers,
                best_pin_memory,
            ),
        )

    if (
        best_pin_memory is True
        and pin_memory_speedup_ratio is not None
        and pin_memory_speedup_ratio >= 1.10
    ):
        recommendations.append(
            _dataloader_pinned_memory_recommendation(
                result,
                pin_memory_speedup_ratio,
            ),
        )

    if best_num_workers == 0:
        recommendations.append(_dataloader_workers_limited_recommendation(result))

    return recommendations


def warning_propagation_rule(suite: BenchmarkSuiteResult) -> list[Recommendation]:
    """Promote benchmark warnings into advisor recommendations."""

    recommendations = []
    for result in suite.results:
        if result.warnings:
            recommendations.append(
                Recommendation(
                    id=f"warning_{_slugify(result.name)}",
                    title=f"Review benchmark warning: {result.name}",
                    category="warning",
                    priority=0,
                    impact="medium",
                    confidence="high",
                    effort="low",
                    estimated_speedup=None,
                    summary="; ".join(result.warnings),
                    rationale=(
                        "Benchmark warnings indicate results that may require "
                        "interpretation before applying recommendations."
                    ),
                    suggested_action=(
                        "Review the warning and validate the recommendation "
                        "against a real workload."
                    ),
                    code_snippet=None,
                    related_metrics=[],
                    warnings=list(result.warnings),
                ),
            )
    return recommendations


def _result_unusable(result: BenchmarkResult | None) -> bool:
    return result is None or result.status != "ok"


def _mixed_precision_enable_recommendation(
    result: BenchmarkResult,
    speedup: float,
) -> Recommendation:
    return Recommendation(
        id="mixed_precision_enable",
        title="Enable mixed precision",
        category="mixed_precision",
        priority=0,
        impact=impact_from_speedup(speedup),
        confidence=confidence_from_signal(True, result.status == "ok", result.warnings),
        effort="low",
        estimated_speedup=speedup,
        summary=f"AMP improved synthetic training throughput by {format_speedup(speedup)}.",
        rationale=(
            "AMP can use Tensor Core-friendly lower precision operations on "
            "supported NVIDIA GPUs, improving throughput for compatible training "
            "workloads while preserving FP32 where needed."
        ),
        suggested_action=(
            "Wrap the forward pass and loss computation with "
            "torch.amp.autocast('cuda') and use torch.amp.GradScaler('cuda') "
            "during training."
        ),
        code_snippet=_AMP_CODE_SNIPPET,
        related_metrics=list(_MIXED_PRECISION_RELATED_METRICS),
        warnings=list(result.warnings),
    )


def _mixed_precision_slow_recommendation(
    result: BenchmarkResult,
    speedup: float,
) -> Recommendation:
    return Recommendation(
        id="mixed_precision_do_not_enable_blindly",
        title="Do not enable mixed precision blindly",
        category="mixed_precision",
        priority=0,
        impact="low",
        confidence="medium",
        effort="low",
        estimated_speedup=speedup,
        summary="AMP was slower than FP32 in this benchmark.",
        rationale=(
            "AMP can be slower for small models, low batch sizes, workloads "
            "dominated by CPU or Python overhead, or laptops constrained by "
            "thermal and power limits."
        ),
        suggested_action="Profile your real workload before enabling AMP globally.",
        code_snippet=None,
        related_metrics=["amp_speedup_ratio"],
        warnings=list(result.warnings),
    )


def _mixed_precision_limited_recommendation(
    result: BenchmarkResult,
    speedup: float,
) -> Recommendation:
    return Recommendation(
        id="mixed_precision_limited_benefit",
        title="Mixed precision benefit is limited",
        category="mixed_precision",
        priority=0,
        impact="low",
        confidence="medium",
        effort="low",
        estimated_speedup=speedup,
        summary="AMP produced only a small throughput improvement in this benchmark.",
        rationale=(
            "AMP may still reduce memory usage, which can help with larger "
            "batches or models, but the throughput gain is small in this "
            "synthetic benchmark."
        ),
        suggested_action=(
            "Use AMP if memory savings matter, but validate throughput on your "
            "real workload."
        ),
        code_snippet=None,
        related_metrics=["amp_speedup_ratio"],
        warnings=list(result.warnings),
    )


def _tensor_core_friendly_recommendation(
    result: BenchmarkResult,
    speedup: float,
) -> Recommendation:
    return Recommendation(
        id="tensor_core_friendly_workloads",
        title="Prioritize Tensor Core-friendly workloads",
        category="tensor_cores",
        priority=0,
        impact="high",
        confidence=confidence_from_signal(True, result.status == "ok", result.warnings),
        effort="medium",
        estimated_speedup=speedup,
        summary=f"FP16 matrix multiplication was {format_speedup(speedup)} faster than FP32.",
        rationale=(
            "This GPU shows strong FP16 acceleration, so FP16 or BF16 "
            "matrix-heavy workloads are likely to benefit when they reach "
            "Tensor Core execution paths."
        ),
        suggested_action=(
            "Use dimensions divisible by 8 or 16, enable AMP, and avoid forcing "
            "FP32 for large Linear or Conv operations."
        ),
        code_snippet=None,
        related_metrics=list(_TENSOR_CORE_RELATED_METRICS),
        warnings=list(result.warnings),
    )


def _tensor_core_weak_recommendation(
    result: BenchmarkResult,
    speedup: float,
) -> Recommendation:
    return Recommendation(
        id="tensor_core_acceleration_weak",
        title="Tensor Core acceleration was weak",
        category="tensor_cores",
        priority=0,
        impact="medium",
        confidence="medium",
        effort="medium",
        estimated_speedup=speedup,
        summary="FP16 matrix multiplication did not significantly outperform FP32.",
        rationale=(
            "Weak FP16 acceleration may indicate CUDA build issues, a low GPU "
            "power state, input shapes that are not Tensor Core-friendly, or "
            "FP16 operations that are not engaging Tensor Cores."
        ),
        suggested_action=(
            "Check CUDA build, GPU power state, input shapes, and whether FP16 "
            "operations are reaching Tensor Cores."
        ),
        code_snippet=None,
        related_metrics=["fp16_speedup_ratio"],
        warnings=list(result.warnings),
    )


def _batch_size_increase_recommendation(
    result: BenchmarkResult,
    best_batch_size: int,
    speedup: float,
) -> Recommendation:
    return Recommendation(
        id="batch_size_increase",
        title="Use a larger batch size",
        category="batch_size",
        priority=0,
        impact=impact_from_speedup(speedup),
        confidence=confidence_from_signal(True, result.status == "ok", result.warnings),
        effort="low",
        estimated_speedup=speedup,
        summary=(
            f"Batch size {best_batch_size} achieved the best throughput in the sweep."
        ),
        rationale=(
            "Larger batches can improve GPU occupancy and reduce per-sample overhead."
        ),
        suggested_action=(
            f"Start with batch_size={best_batch_size}; if training is "
            "VRAM-limited, use gradient accumulation."
        ),
        code_snippet=f"DataLoader(dataset, batch_size={best_batch_size}, ...)",
        related_metrics=list(_BATCH_SIZE_RELATED_METRICS),
        warnings=list(result.warnings),
    )


def _batch_size_limited_recommendation(
    result: BenchmarkResult,
    speedup: float | None,
) -> Recommendation:
    return Recommendation(
        id="batch_size_scaling_limited",
        title="Batch scaling appears limited",
        category="batch_size",
        priority=0,
        impact="medium",
        confidence="medium",
        effort="medium",
        estimated_speedup=speedup,
        summary="The benchmark found a very small optimal batch size.",
        rationale=(
            "This can indicate CPU/Python overhead, memory bandwidth limits, "
            "laptop power limits, or a workload that is not compute-bound."
        ),
        suggested_action=(
            "Validate batch-size tuning on your real model before assuming "
            "bigger batches help."
        ),
        code_snippet=None,
        related_metrics=["best_batch_size", "speedup_vs_batch_1"],
        warnings=list(result.warnings),
    )


def _dataloader_workers_recommendation(
    result: BenchmarkResult,
    best_num_workers: int,
    best_pin_memory: bool | None,
) -> Recommendation:
    return Recommendation(
        id="dataloader_tune_workers",
        title="Tune DataLoader workers",
        category="dataloader",
        priority=0,
        impact="medium",
        confidence=confidence_from_signal(True, result.status == "ok", result.warnings),
        effort="low",
        estimated_speedup=None,
        summary=(
            f"num_workers={best_num_workers} was fastest in the synthetic "
            "DataLoader benchmark."
        ),
        rationale=(
            "DataLoader workers can reduce CPU input pipeline stalls when "
            "loading or preprocessing data."
        ),
        suggested_action=(
            f"Use num_workers={best_num_workers} as a starting point and "
            "re-test on your real dataset."
        ),
        code_snippet=(
            f"DataLoader(dataset, num_workers={best_num_workers}, "
            f"pin_memory={best_pin_memory})"
        ),
        related_metrics=["best_num_workers", "best_pin_memory", "best_samples_per_sec"],
        warnings=list(result.warnings),
    )


def _dataloader_pinned_memory_recommendation(
    result: BenchmarkResult,
    speedup: float,
) -> Recommendation:
    return Recommendation(
        id="dataloader_enable_pinned_memory",
        title="Enable pinned memory for faster GPU transfers",
        category="dataloader",
        priority=0,
        impact=impact_from_speedup(speedup),
        confidence=confidence_from_signal(True, result.status == "ok", result.warnings),
        effort="low",
        estimated_speedup=speedup,
        summary=f"Pinned memory improved transfer/input throughput by {format_speedup(speedup)}.",
        rationale=(
            "Pinned host memory can improve CPU-to-GPU transfer performance "
            "when using CUDA."
        ),
        suggested_action=(
            "Set pin_memory=True and use non_blocking=True when moving tensors "
            "to CUDA."
        ),
        code_snippet="batch = batch.to('cuda', non_blocking=True)",
        related_metrics=["best_pin_memory", "pin_memory_speedup_ratio"],
        warnings=list(result.warnings),
    )


def _dataloader_workers_limited_recommendation(
    result: BenchmarkResult,
) -> Recommendation:
    return Recommendation(
        id="dataloader_workers_may_not_help",
        title="DataLoader workers may not help this synthetic workload",
        category="dataloader",
        priority=0,
        impact="low",
        confidence="medium",
        effort="low",
        estimated_speedup=None,
        summary="num_workers=0 was fastest in this benchmark.",
        rationale=(
            "Synthetic datasets and Windows multiprocessing overhead can make "
            "worker processes slower than single-process loading."
        ),
        suggested_action="Re-test with your real dataset before deciding num_workers.",
        code_snippet=None,
        related_metrics=["best_num_workers"],
        warnings=list(result.warnings),
    )


def _slugify(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")
