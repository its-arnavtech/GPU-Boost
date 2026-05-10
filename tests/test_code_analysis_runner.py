"""Tests for the combined Phase 4 code analysis runner."""

from __future__ import annotations

from gpuboost.code_analysis import runner
from gpuboost.code_analysis.dataloader import DataLoaderFindingVisitor
from gpuboost.code_analysis.runner import (
    analyze_python_file,
    analyze_python_source,
    sort_findings,
)
from gpuboost.code_analysis.visitors import BaseFindingVisitor
from gpuboost.schemas.code_analysis import CodeFinding


def test_analyze_python_source_combines_current_findings() -> None:
    result = analyze_python_source(
        "loader = DataLoader(dataset, num_workers=0, pin_memory=False)\n"
        "for batch in loader:\n"
        "    outputs = model(batch)\n"
        "    scalar = loss.item()\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
    )

    assert result.status == "ok"
    assert result.error is None
    assert result.warnings == []
    ids = _finding_ids(result)
    assert "dataloader_num_workers_zero" in ids
    assert "dataloader_pin_memory_false" in ids
    assert "sync_call_item_in_loop" in ids
    assert "mixed_precision_autocast_missing" in ids
    assert "cudnn_benchmark_missing" in ids


def test_analyze_python_file_works_with_tmp_path(tmp_path) -> None:
    filepath = tmp_path / "train.py"
    filepath.write_text(
        "torch.backends.cudnn.benchmark = True\n"
        "for batch in loader:\n"
        "    outputs = model(batch)\n",
        encoding="utf-8",
    )

    result = analyze_python_file(str(filepath))

    assert result.status == "ok"
    assert result.filepath == str(filepath)
    assert _finding_ids(result) == ["inference_missing_no_grad"]


def test_parse_error_returns_status_error() -> None:
    result = analyze_python_source("for batch in loader\n")

    assert result.status == "error"
    assert result.findings == []
    assert result.error is not None


def test_sort_findings_stable_ordering() -> None:
    findings = [
        _finding("b.py", None, None, "warning", "File level"),
        _finding("a.py", 2, 5, "info", "Later info"),
        _finding("a.py", 2, 5, "warning", "Earlier warning"),
        _finding("a.py", 1, 10, "info", "First line"),
    ]

    sorted_titles = [finding.title for finding in sort_findings(findings)]

    assert sorted_titles == [
        "First line",
        "Earlier warning",
        "Later info",
        "File level",
    ]


def test_visitor_failure_becomes_warning_and_does_not_crash(monkeypatch) -> None:
    class FailingVisitor(BaseFindingVisitor):
        def visit_Module(self, node) -> None:  # noqa: N802, ANN001
            msg = "boom"
            raise RuntimeError(msg)

    monkeypatch.setattr(
        runner,
        "_VISITORS",
        [FailingVisitor, DataLoaderFindingVisitor],
    )

    result = analyze_python_source("loader = DataLoader(dataset)\n")

    assert result.status == "ok"
    assert result.warnings == [
        "Code analysis visitor failed: FailingVisitor: boom"
    ]
    assert _finding_ids(result) == [
        "dataloader_missing_num_workers",
        "dataloader_missing_pin_memory",
    ]


def _finding(
    filepath: str,
    line: int | None,
    column: int | None,
    severity: str,
    title: str,
) -> CodeFinding:
    return CodeFinding(
        id=title.lower().replace(" ", "_"),
        title=title,
        category="test",
        severity=severity,
        confidence="medium",
        filepath=filepath,
        line=line,
        column=column,
        end_line=None,
        end_column=None,
        summary="summary",
        rationale="rationale",
        suggested_action="action",
        code_snippet=None,
    )


def _finding_ids(result) -> list[str]:
    return [finding.id for finding in result.findings]
