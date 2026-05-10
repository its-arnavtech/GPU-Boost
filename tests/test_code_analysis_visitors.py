"""Tests for Phase 4 code analysis visitor framework."""

import ast

from gpuboost.code_analysis.visitors import BaseFindingVisitor, run_visitors


def test_add_finding_populates_filepath_and_node_location_fields() -> None:
    tree = ast.parse("x = torch.cuda.synchronize()\n")
    node = tree.body[0]
    visitor = BaseFindingVisitor(filepath="train.py")

    visitor.add_finding(
        id="sync-call",
        title="Avoid synchronization calls",
        category="sync_call",
        severity="warning",
        confidence="high",
        summary="A CUDA synchronization call was found.",
        rationale="Synchronization can hide asynchronous execution benefits.",
        suggested_action="Remove unnecessary synchronization from hot paths.",
        node=node,
    )

    finding = visitor.findings[0]
    assert finding.filepath == "train.py"
    assert finding.line == 1
    assert finding.column == 0
    assert finding.end_line == 1
    assert finding.end_column == 28


def test_add_finding_works_without_node() -> None:
    visitor = BaseFindingVisitor(filepath="train.py")

    visitor.add_finding(
        id="general-note",
        title="General note",
        category="general",
        severity="info",
        confidence="low",
        summary="This finding has no exact location.",
        rationale="The visitor emitted file-level guidance.",
        suggested_action="Review the file.",
        related_recommendation_ids=["general"],
        tags=["file-level"],
    )

    finding = visitor.findings[0]
    assert finding.filepath == "train.py"
    assert finding.line is None
    assert finding.column is None
    assert finding.end_line is None
    assert finding.end_column is None
    assert finding.related_recommendation_ids == ["general"]
    assert finding.tags == ["file-level"]


def test_run_visitors_collects_findings_from_multiple_visitors() -> None:
    tree = ast.parse("x = 1\n")

    findings = run_visitors(
        tree,
        filepath="train.py",
        visitors=[FirstDummyVisitor, SecondDummyVisitor],
    )

    assert [finding.id for finding in findings] == ["first", "second"]
    assert all(finding.filepath == "train.py" for finding in findings)


def test_run_visitors_continues_if_one_visitor_raises() -> None:
    tree = ast.parse("x = 1\n")

    findings = run_visitors(
        tree,
        filepath="train.py",
        visitors=[RaisingDummyVisitor, FirstDummyVisitor],
    )

    assert [finding.id for finding in findings] == ["first"]


class FirstDummyVisitor(BaseFindingVisitor):
    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        self.add_finding(
            id="first",
            title="First finding",
            category="general",
            severity="info",
            confidence="high",
            summary="First visitor found something.",
            rationale="This is emitted by the first test visitor.",
            suggested_action="Keep testing.",
            node=node,
        )


class SecondDummyVisitor(BaseFindingVisitor):
    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        self.add_finding(
            id="second",
            title="Second finding",
            category="general",
            severity="info",
            confidence="high",
            summary="Second visitor found something.",
            rationale="This is emitted by the second test visitor.",
            suggested_action="Keep testing.",
            node=node,
        )


class RaisingDummyVisitor(BaseFindingVisitor):
    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        raise RuntimeError("visitor failed")
