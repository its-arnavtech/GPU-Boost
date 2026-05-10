"""Static analysis findings for PyTorch DataLoader usage."""

from __future__ import annotations

import ast

from gpuboost.code_analysis.parser import parse_python_source
from gpuboost.code_analysis.visitors import BaseFindingVisitor, run_visitors
from gpuboost.schemas.code_analysis import CodeAnalysisResult


class DataLoaderFindingVisitor(BaseFindingVisitor):
    """Find potentially slow PyTorch DataLoader configuration."""

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if is_dataloader_call(node):
            self._check_num_workers(node)
            self._check_pin_memory(node)

        self.generic_visit(node)

    def _check_num_workers(self, node: ast.Call) -> None:
        keyword = get_keyword(node, "num_workers")
        if keyword is None:
            self.add_finding(
                id="dataloader_missing_num_workers",
                title="DataLoader is missing num_workers",
                category="dataloader",
                severity="warning",
                confidence="medium",
                summary="This DataLoader does not set num_workers.",
                rationale=(
                    "A single-process DataLoader can bottleneck the GPU when "
                    "loading or preprocessing data."
                ),
                suggested_action=(
                    "Set num_workers based on your CPU core count and benchmark "
                    "results."
                ),
                code_snippet="DataLoader(..., num_workers=4, ...)",
                related_recommendation_ids=["dataloader_tune_workers"],
                tags=["dataloader", "num_workers", "input_pipeline"],
                node=node,
            )
            return

        if literal_int_value(keyword.value) == 0:
            self.add_finding(
                id="dataloader_num_workers_zero",
                title="DataLoader uses num_workers=0",
                category="dataloader",
                severity="warning",
                confidence="high",
                summary="This DataLoader uses single-process data loading.",
                rationale=(
                    "num_workers=0 can leave the GPU waiting if dataset loading "
                    "or preprocessing is slow."
                ),
                suggested_action=(
                    "Try num_workers=2, 4, or the Advisor-recommended value and "
                    "benchmark again."
                ),
                code_snippet="DataLoader(..., num_workers=4, ...)",
                related_recommendation_ids=["dataloader_tune_workers"],
                tags=["dataloader", "num_workers", "input_pipeline"],
                node=node,
            )

    def _check_pin_memory(self, node: ast.Call) -> None:
        keyword = get_keyword(node, "pin_memory")
        if keyword is None:
            self.add_finding(
                id="dataloader_missing_pin_memory",
                title="DataLoader is missing pin_memory",
                category="dataloader",
                severity="info",
                confidence="medium",
                summary="This DataLoader does not set pin_memory.",
                rationale=(
                    "Pinned host memory can improve CPU-to-GPU transfer "
                    "performance when training on CUDA."
                ),
                suggested_action=(
                    "If using CUDA, try pin_memory=True and move tensors with "
                    "non_blocking=True."
                ),
                code_snippet="DataLoader(..., pin_memory=True, ...)",
                related_recommendation_ids=["dataloader_enable_pinned_memory"],
                tags=["dataloader", "pin_memory", "cuda_transfer"],
                node=node,
            )
            return

        if literal_bool_value(keyword.value) is False:
            self.add_finding(
                id="dataloader_pin_memory_false",
                title="DataLoader disables pin_memory",
                category="dataloader",
                severity="info",
                confidence="high",
                summary="This DataLoader explicitly disables pinned memory.",
                rationale=(
                    "Pinned memory can improve host-to-GPU transfers for CUDA "
                    "workloads."
                ),
                suggested_action=(
                    "If batches are moved to CUDA, benchmark pin_memory=True."
                ),
                code_snippet="DataLoader(..., pin_memory=True, ...)",
                related_recommendation_ids=["dataloader_enable_pinned_memory"],
                tags=["dataloader", "pin_memory", "cuda_transfer"],
                node=node,
            )


def analyze_dataloader_source(
    source: str,
    filepath: str = "<string>",
) -> CodeAnalysisResult:
    """Analyze Python source for DataLoader configuration findings."""

    tree, result = parse_python_source(source, filepath=filepath)
    if tree is None:
        return result

    result.findings = run_visitors(tree, filepath, [DataLoaderFindingVisitor])
    return result


def is_dataloader_call(node: ast.Call) -> bool:
    """Return whether a call expression appears to invoke a DataLoader."""

    return _call_name(node.func).endswith("DataLoader")


def get_keyword(node: ast.Call, name: str) -> ast.keyword | None:
    """Return a named keyword argument from a call, if present."""

    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword

    return None


def literal_int_value(node: ast.AST) -> int | None:
    """Return an integer literal value from an AST node."""

    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value

    return None


def literal_bool_value(node: ast.AST) -> bool | None:
    """Return a boolean literal value from an AST node."""

    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return node.value

    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr

    return ""
