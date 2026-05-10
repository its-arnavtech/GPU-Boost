"""Combined runner for GPUBoost code analysis visitors."""

from __future__ import annotations

import ast

from gpuboost.code_analysis.dataloader import DataLoaderFindingVisitor
from gpuboost.code_analysis.optimizations import OptimizationFindingVisitor
from gpuboost.code_analysis.parser import parse_python_file, parse_python_source
from gpuboost.code_analysis.sync_calls import SyncCallFindingVisitor
from gpuboost.code_analysis.visitors import BaseFindingVisitor
from gpuboost.schemas.code_analysis import (
    CodeAnalysisResult,
    CodeFinding,
    create_timestamp,
)


_VISITORS: list[type[BaseFindingVisitor]] = [
    DataLoaderFindingVisitor,
    SyncCallFindingVisitor,
    OptimizationFindingVisitor,
]

_SEVERITY_ORDER = {
    "error": 0,
    "warning": 1,
    "info": 2,
}


def analyze_python_file(filepath: str) -> CodeAnalysisResult:
    """Analyze a Python file with all current code analysis visitors."""

    tree, parse_result = parse_python_file(filepath)
    if parse_result.status == "error" or tree is None:
        return parse_result

    return _run_analysis(tree, filepath)


def analyze_python_source(
    source: str,
    filepath: str = "<string>",
) -> CodeAnalysisResult:
    """Analyze Python source text with all current code analysis visitors."""

    tree, parse_result = parse_python_source(source, filepath=filepath)
    if parse_result.status == "error" or tree is None:
        return parse_result

    return _run_analysis(tree, filepath)


def sort_findings(findings: list[CodeFinding]) -> list[CodeFinding]:
    """Return findings in a stable location and severity order."""

    return sorted(
        findings,
        key=lambda finding: (
            finding.filepath,
            finding.line if finding.line is not None else 1_000_000_000,
            finding.column if finding.column is not None else 1_000_000_000,
            _SEVERITY_ORDER.get(finding.severity, 99),
            finding.title,
        ),
    )


def _run_analysis(tree: ast.AST, filepath: str) -> CodeAnalysisResult:
    findings: list[CodeFinding] = []
    warnings: list[str] = []

    for visitor_type in _VISITORS:
        try:
            visitor = visitor_type(filepath)
            visitor.visit(tree)
        except Exception as exc:
            warnings.append(
                f"Code analysis visitor failed: {visitor_type.__name__}: {exc}"
            )
            continue

        findings.extend(visitor.findings)

    return CodeAnalysisResult(
        generated_at=create_timestamp(),
        filepath=filepath,
        status="ok",
        findings=sort_findings(findings),
        warnings=warnings,
        error=None,
    )
