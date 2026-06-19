"""Static analysis findings for synchronization-like calls in loops."""

from __future__ import annotations

import ast

from gpuboost.code_analysis.parser import parse_python_source
from gpuboost.code_analysis.visitors import BaseFindingVisitor, run_visitors
from gpuboost.schemas.code_analysis import CodeAnalysisResult


_NO_CUDA_CONTEXT_NOTE = (
    " No CUDA usage was detected in this file, so this may be a false positive "
    "on a CPU-only object."
)


class SyncCallFindingVisitor(BaseFindingVisitor):
    """Find calls that may synchronize or transfer tensors inside loops."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)
        self.loop_depth = 0
        self.cuda_used = False

    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        self.cuda_used = _module_uses_cuda(node)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
        self._visit_loop(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if self.loop_depth > 0 and isinstance(node.func, ast.Attribute):
            self._check_sync_call(node)

        self.generic_visit(node)

    def _visit_loop(self, node: ast.AST) -> None:
        self.loop_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self.loop_depth -= 1

    def _confidence(self, cuda_confidence: str) -> str:
        # Without any CUDA usage in the file the call is likely a harmless CPU
        # operation, so report low confidence instead of over-warning.
        return cuda_confidence if self.cuda_used else "low"

    def _rationale(self, rationale: str) -> str:
        return rationale if self.cuda_used else rationale + _NO_CUDA_CONTEXT_NOTE

    def _check_sync_call(self, node: ast.Call) -> None:
        call_name = node.func.attr
        if call_name == "item":
            self.add_finding(
                id="sync_call_item_in_loop",
                title=".item() call inside loop may force GPU synchronization",
                category="sync_call",
                severity="warning",
                confidence=self._confidence("high"),
                summary=(
                    "Calling .item() inside a loop can force CPU-GPU "
                    "synchronization."
                ),
                rationale=self._rationale(
                    ".item() transfers a scalar from GPU to CPU and can stall "
                    "the training loop when used every iteration."
                ),
                suggested_action=(
                    "Accumulate GPU tensors during the loop and convert to "
                    "Python scalars less frequently, such as once per logging "
                    "interval."
                ),
                code_snippet=None,
                related_recommendation_ids=[],
                tags=["sync_call", "item", "training_loop"],
                node=node,
            )
        elif call_name == "cpu":
            self.add_finding(
                id="sync_call_cpu_in_loop",
                title=".cpu() call inside loop may force GPU synchronization",
                category="sync_call",
                severity="warning",
                confidence=self._confidence("medium"),
                summary=(
                    "Calling .cpu() inside a loop can repeatedly transfer "
                    "tensors from GPU to CPU."
                ),
                rationale=self._rationale(
                    "Frequent GPU-to-CPU transfers can reduce throughput and "
                    "prevent compute/transfer overlap."
                ),
                suggested_action=(
                    "Move tensors to CPU outside the hot loop or only at "
                    "logging/checkpoint intervals."
                ),
                code_snippet=None,
                related_recommendation_ids=[],
                tags=["sync_call", "cpu", "transfer"],
                node=node,
            )
        elif call_name == "numpy":
            self.add_finding(
                id="sync_call_numpy_in_loop",
                title=".numpy() call inside loop may force CPU conversion",
                category="sync_call",
                severity="warning",
                confidence=self._confidence("medium"),
                summary=(
                    "Calling .numpy() inside a loop can force tensors onto CPU "
                    "and slow down GPU workloads."
                ),
                rationale=self._rationale(
                    "NumPy arrays live on CPU, so converting tensors inside "
                    "the training loop can create repeated synchronization and "
                    "transfer overhead."
                ),
                suggested_action=(
                    "Avoid NumPy conversion in the hot path; defer conversion "
                    "until after the loop or logging interval."
                ),
                code_snippet=None,
                related_recommendation_ids=[],
                tags=["sync_call", "numpy", "transfer"],
                node=node,
            )


def analyze_sync_calls_source(
    source: str,
    filepath: str = "<string>",
) -> CodeAnalysisResult:
    """Analyze Python source for synchronization-like calls inside loops."""

    tree, result = parse_python_source(source, filepath=filepath)
    if tree is None:
        return result

    result.findings = run_visitors(
        tree, filepath, [SyncCallFindingVisitor], warnings=result.warnings
    )
    return result


def _module_uses_cuda(tree: ast.AST) -> bool:
    """Heuristically detect whether the source uses CUDA/GPU devices at all."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "cuda":
            return True
        if isinstance(node, ast.Name) and node.id == "device":
            return True
        if isinstance(node, ast.keyword) and node.arg == "device":
            return True
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "cuda" in node.value.lower()
        ):
            return True
        if isinstance(node, ast.Attribute):
            name = _attribute_name(node).lower()
            if "cuda" in name or "device" in name:
                return True
    return False


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return ""
