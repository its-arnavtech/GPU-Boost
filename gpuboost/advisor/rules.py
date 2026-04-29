"""Advisor rules for turning benchmark results into recommendations."""

from __future__ import annotations

from gpuboost.advisor.scoring import confidence_from_signal, impact_from_speedup
from gpuboost.advisor.utils import format_speedup, get_metric, get_result_by_name
from gpuboost.schemas.benchmark_result import BenchmarkResult, BenchmarkSuiteResult
from gpuboost.schemas.recommendation import Recommendation


_MIXED_PRECISION_RESULT = "Mixed Precision"
_MATRIX_MULTIPLICATION_RESULT = "Matrix Multiplication"

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
