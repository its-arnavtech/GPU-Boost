"""Static analysis findings for common PyTorch optimization patterns."""

from __future__ import annotations

import ast

from gpuboost.code_analysis.parser import parse_python_source
from gpuboost.code_analysis.visitors import BaseFindingVisitor, run_visitors
from gpuboost.schemas.code_analysis import CodeAnalysisResult


_FORWARD_LIKE_NAMES = {"model", "net", "network", "module", "forward", "predict"}


class OptimizationFindingVisitor(BaseFindingVisitor):
    """Find missing common PyTorch training and inference optimizations."""

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)
        self.cudnn_benchmark_enabled = False
        self.no_grad_context_depth = 0
        self.autocast_context_depth = 0

    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        self.generic_visit(node)
        if not self.cudnn_benchmark_enabled:
            self.add_finding(
                id="cudnn_benchmark_missing",
                title="cuDNN benchmark mode is not enabled",
                category="cudnn",
                severity="info",
                confidence="medium",
                summary=(
                    "This script does not appear to enable "
                    "torch.backends.cudnn.benchmark."
                ),
                rationale=(
                    "For fixed-size convolution workloads, cuDNN benchmark mode "
                    "can help PyTorch select faster convolution algorithms."
                ),
                suggested_action=(
                    "If your input shapes are stable, add "
                    "torch.backends.cudnn.benchmark = True near startup."
                ),
                code_snippet="torch.backends.cudnn.benchmark = True",
                related_recommendation_ids=[],
                tags=["cudnn", "convolution", "startup"],
            )

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if is_cudnn_benchmark_true_assignment(node):
            self.cudnn_benchmark_enabled = True

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if is_cudnn_benchmark_true_assignment(node):
            self.cudnn_benchmark_enabled = True

        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        self._visit_context_manager(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:  # noqa: N802
        self._visit_context_manager(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._check_loop(node)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self._check_loop(node)
        self.generic_visit(node)

    def _visit_context_manager(self, node: ast.With | ast.AsyncWith) -> None:
        has_no_grad = any(
            is_no_grad_context(item.context_expr)
            or is_inference_mode_context(item.context_expr)
            for item in node.items
        )
        has_autocast = any(
            is_autocast_context(item.context_expr) for item in node.items
        )

        if has_no_grad:
            self.no_grad_context_depth += 1
        if has_autocast:
            self.autocast_context_depth += 1

        try:
            self.generic_visit(node)
        finally:
            if has_autocast:
                self.autocast_context_depth -= 1
            if has_no_grad:
                self.no_grad_context_depth -= 1

    def _check_loop(self, node: ast.For | ast.While) -> None:
        contains_training_step = loop_contains_training_step(node)
        if contains_training_step:
            self._check_training_loop(node)
            return

        if (
            loop_contains_forward_like_call(node)
            and self.no_grad_context_depth == 0
            and not _loop_contains_no_grad_context(node)
        ):
            self.add_finding(
                id="inference_missing_no_grad",
                title="Inference loop may be missing no_grad or inference_mode",
                category="inference",
                severity="warning",
                confidence="medium",
                summary=(
                    "This loop appears to run forward passes without disabling "
                    "gradient tracking."
                ),
                rationale=(
                    "Disabling gradient tracking during evaluation or inference "
                    "reduces memory usage and overhead."
                ),
                suggested_action=(
                    "Wrap evaluation/inference loops with torch.no_grad() or "
                    "torch.inference_mode()."
                ),
                code_snippet=(
                    "with torch.inference_mode():\n"
                    "    outputs = model(inputs)"
                ),
                related_recommendation_ids=[],
                tags=["inference", "no_grad", "memory"],
                node=node,
            )

    def _check_training_loop(self, node: ast.For | ast.While) -> None:
        if self.autocast_context_depth > 0 or _loop_contains_autocast_context(node):
            return

        self.add_finding(
            id="mixed_precision_autocast_missing",
            title="Training loop may be missing mixed precision autocast",
            category="mixed_precision",
            severity="info",
            confidence="medium",
            summary="This training loop appears to run without AMP autocast.",
            rationale=(
                "On Tensor Core-capable NVIDIA GPUs, AMP can improve throughput "
                "for eligible Linear and Conv operations."
            ),
            suggested_action=(
                "If your model is AMP-safe, wrap the forward pass and loss "
                "computation with torch.amp.autocast('cuda')."
            ),
            code_snippet=(
                "with torch.amp.autocast('cuda'):\n"
                "    outputs = model(inputs)\n"
                "    loss = criterion(outputs, targets)"
            ),
            related_recommendation_ids=[
                "mixed_precision_enable",
                "mixed_precision_limited_benefit",
            ],
            tags=["mixed_precision", "autocast", "training_loop"],
            node=node,
        )


def analyze_optimization_source(
    source: str,
    filepath: str = "<string>",
) -> CodeAnalysisResult:
    """Analyze Python source for missing PyTorch optimization patterns."""

    tree, result = parse_python_source(source, filepath=filepath)
    if tree is None:
        return result

    result.findings = run_visitors(tree, filepath, [OptimizationFindingVisitor])
    return result


def is_no_grad_context(expr: ast.AST) -> bool:
    """Return whether an expression is a torch.no_grad context manager."""

    return _call_name_from_expr(expr) in {"torch.no_grad", "no_grad"}


def is_inference_mode_context(expr: ast.AST) -> bool:
    """Return whether an expression is a torch.inference_mode context manager."""

    return _call_name_from_expr(expr) in {
        "torch.inference_mode",
        "inference_mode",
    }


def is_autocast_context(expr: ast.AST) -> bool:
    """Return whether an expression is an AMP autocast context manager."""

    return _call_name_from_expr(expr) in {
        "torch.amp.autocast",
        "torch.cuda.amp.autocast",
        "autocast",
    }


def is_cudnn_benchmark_true_assignment(
    node: ast.Assign | ast.AnnAssign,
) -> bool:
    """Return whether an assignment enables torch.backends.cudnn.benchmark."""

    value = node.value
    if not (isinstance(value, ast.Constant) and value.value is True):
        return False

    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(
        _attribute_name(target) == "torch.backends.cudnn.benchmark"
        for target in targets
    )


def loop_contains_training_step(loop_node: ast.AST) -> bool:
    """Return whether a loop appears to contain a training update step."""

    return any(
        isinstance(node, ast.Call) and _is_training_step_call(node)
        for node in _walk_loop_body(loop_node)
    )


def loop_contains_forward_like_call(loop_node: ast.AST) -> bool:
    """Return whether a loop appears to contain a model forward call."""

    return any(
        isinstance(node, ast.Call) and _is_forward_like_call(node)
        for node in _walk_loop_body(loop_node)
    )


def _loop_contains_no_grad_context(loop_node: ast.AST) -> bool:
    return any(
        isinstance(node, (ast.With, ast.AsyncWith))
        and any(
            is_no_grad_context(item.context_expr)
            or is_inference_mode_context(item.context_expr)
            for item in node.items
        )
        for node in _walk_loop_body(loop_node)
    )


def _loop_contains_autocast_context(loop_node: ast.AST) -> bool:
    return any(
        isinstance(node, (ast.With, ast.AsyncWith))
        and any(is_autocast_context(item.context_expr) for item in node.items)
        for node in _walk_loop_body(loop_node)
    )


def _is_training_step_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False

    receiver = _attribute_name(node.func.value)
    return (receiver, node.func.attr) in {
        ("loss", "backward"),
        ("optimizer", "step"),
        ("scaler", "step"),
    }


def _is_forward_like_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id in _FORWARD_LIKE_NAMES
    if isinstance(node.func, ast.Attribute):
        return node.func.attr in _FORWARD_LIKE_NAMES

    return False


def _call_name_from_expr(expr: ast.AST) -> str:
    if isinstance(expr, ast.Call):
        return _attribute_name(expr.func)

    return _attribute_name(expr)


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr

    return ""


def _walk_loop_body(loop_node: ast.AST) -> list[ast.AST]:
    if not isinstance(loop_node, (ast.For, ast.While)):
        return []

    nodes: list[ast.AST] = []
    for statement in [*loop_node.body, *loop_node.orelse]:
        nodes.extend(ast.walk(statement))

    return nodes
