"""Visitor framework for GPUBoost code analysis findings."""

from __future__ import annotations

import ast

from gpuboost.schemas.code_analysis import CodeFinding


class BaseFindingVisitor(ast.NodeVisitor):
    """Base AST visitor that records code analysis findings."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.findings: list[CodeFinding] = []

    def add_finding(
        self,
        *,
        id: str,
        title: str,
        category: str,
        severity: str,
        confidence: str,
        summary: str,
        rationale: str,
        suggested_action: str,
        code_snippet: str | None = None,
        related_recommendation_ids: list[str] | None = None,
        tags: list[str] | None = None,
        node: ast.AST | None = None,
    ) -> None:
        """Create and store a finding, optionally using AST node source spans."""

        self.findings.append(
            CodeFinding(
                id=id,
                title=title,
                category=category,
                severity=severity,
                confidence=confidence,
                filepath=self.filepath,
                line=getattr(node, "lineno", None),
                column=getattr(node, "col_offset", None),
                end_line=getattr(node, "end_lineno", None),
                end_column=getattr(node, "end_col_offset", None),
                summary=summary,
                rationale=rationale,
                suggested_action=suggested_action,
                code_snippet=code_snippet,
                related_recommendation_ids=related_recommendation_ids or [],
                tags=tags or [],
            )
        )


def run_visitors(
    tree: ast.AST,
    filepath: str,
    visitors: list[type[BaseFindingVisitor]],
    warnings: list[str] | None = None,
) -> list[CodeFinding]:
    """Run finding visitors over a parsed AST and collect their findings.

    If a visitor raises (for example on an unexpected AST node shape), it is
    skipped instead of aborting the whole analysis. When a ``warnings`` list is
    provided, a message is appended for each failed visitor so callers can
    surface partial-failure rather than silently returning fewer findings.
    """

    findings: list[CodeFinding] = []
    for visitor_type in visitors:
        try:
            visitor = visitor_type(filepath)
            visitor.visit(tree)
        except Exception as exc:  # noqa: BLE001 - a visitor must not crash analysis
            if warnings is not None:
                warnings.append(
                    f"Code analysis visitor failed: {visitor_type.__name__}: {exc}"
                )
            continue

        findings.extend(visitor.findings)

    return findings
