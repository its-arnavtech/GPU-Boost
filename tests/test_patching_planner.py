"""Tests for Phase 4 patch planning from code analysis findings."""

from __future__ import annotations

from gpuboost.patching.diff import generate_patch_plan_diff
from gpuboost.patching.planner import (
    create_patch_plan_from_analysis,
    find_import_block,
    get_source_line,
    insert_kwarg_before_closing_paren,
    replace_on_line,
)
from gpuboost.schemas.code_analysis import CodeAnalysisResult, CodeFinding


def test_creates_patch_for_num_workers_zero() -> None:
    source = "loader = DataLoader(dataset, batch_size=32, num_workers=0)\n"
    analysis = _analysis([_finding("dataloader_num_workers_zero", line=1)])

    plan = create_patch_plan_from_analysis(source, analysis)

    suggestion = plan.suggestions[0]
    edit = suggestion.edits[0]
    assert plan.status == "ok"
    assert plan.warnings == []
    assert suggestion.id == "patch_dataloader_num_workers_zero_line_1"
    assert suggestion.title == "Set DataLoader num_workers to 4"
    assert suggestion.finding_ids == ["dataloader_num_workers_zero"]
    assert edit.original_text == source
    assert edit.replacement_text == (
        "loader = DataLoader(dataset, batch_size=32, num_workers=4)\n"
    )


def test_creates_patch_for_pin_memory_false() -> None:
    source = "loader = DataLoader(dataset, pin_memory=False)\n"
    analysis = _analysis([_finding("dataloader_pin_memory_false", line=1)])

    plan = create_patch_plan_from_analysis(source, analysis)

    suggestion = plan.suggestions[0]
    assert suggestion.id == "patch_dataloader_pin_memory_false_line_1"
    assert suggestion.edits[0].replacement_text == (
        "loader = DataLoader(dataset, pin_memory=True)\n"
    )


def test_creates_patch_for_missing_pin_memory_single_line_dataloader() -> None:
    source = "loader = DataLoader(dataset, batch_size=32, num_workers=4)\n"
    analysis = _analysis([_finding("dataloader_missing_pin_memory", line=1)])

    plan = create_patch_plan_from_analysis(source, analysis)

    suggestion = plan.suggestions[0]
    assert suggestion.id == "patch_dataloader_missing_pin_memory_line_1"
    assert suggestion.edits[0].replacement_text == (
        "loader = DataLoader("
        "dataset, batch_size=32, num_workers=4, pin_memory=True)\n"
    )


def test_creates_patch_for_missing_num_workers_single_line_dataloader() -> None:
    source = "loader = DataLoader(dataset, batch_size=32, pin_memory=True)\n"
    analysis = _analysis([_finding("dataloader_missing_num_workers", line=1)])

    plan = create_patch_plan_from_analysis(source, analysis)

    suggestion = plan.suggestions[0]
    assert suggestion.id == "patch_dataloader_missing_num_workers_line_1"
    assert suggestion.edits[0].replacement_text == (
        "loader = DataLoader("
        "dataset, batch_size=32, pin_memory=True, num_workers=4)\n"
    )


def test_does_not_patch_multiline_dataloader_missing_kwarg() -> None:
    source = (
        "loader = DataLoader(\n"
        "    dataset,\n"
        "    batch_size=32,\n"
        ")\n"
    )
    analysis = _analysis([_finding("dataloader_missing_pin_memory", line=1)])

    plan = create_patch_plan_from_analysis(source, analysis)

    assert plan.suggestions == []
    assert plan.warnings == [
        "No safe automatic patch suggestions were generated."
    ]


def test_combines_num_workers_zero_and_missing_pin_memory_into_one_edit() -> None:
    source = "loader = DataLoader(dataset, batch_size=32, num_workers=0)\n"
    analysis = _analysis(
        [
            _finding(
                "dataloader_num_workers_zero",
                line=1,
                severity="info",
                confidence="medium",
            ),
            _finding(
                "dataloader_missing_pin_memory",
                line=1,
                severity="warning",
                confidence="high",
            ),
        ]
    )

    plan = create_patch_plan_from_analysis(source, analysis)

    assert len(plan.suggestions) == 1
    suggestion = plan.suggestions[0]
    edit = suggestion.edits[0]
    assert suggestion.id == "patch_dataloader_combined_line_1"
    assert suggestion.title == "Tune DataLoader workers and pinned memory"
    assert suggestion.category == "dataloader"
    assert suggestion.severity == "warning"
    assert suggestion.confidence == "high"
    assert suggestion.filepath == "train.py"
    assert suggestion.finding_ids == [
        "dataloader_num_workers_zero",
        "dataloader_missing_pin_memory",
    ]
    assert "num_workers" in suggestion.summary
    assert "pin_memory" in suggestion.summary
    assert "input pipeline stalls" in suggestion.rationale
    assert "CPU-to-GPU transfers" in suggestion.rationale
    assert len(suggestion.edits) == 1
    assert suggestion.warnings == []
    assert edit.replacement_text == (
        "loader = DataLoader("
        "dataset, batch_size=32, num_workers=4, pin_memory=True)\n"
    )


def test_combines_missing_num_workers_and_missing_pin_memory_into_one_edit() -> None:
    source = "loader = DataLoader(dataset, batch_size=32)\n"
    analysis = _analysis(
        [
            _finding("dataloader_missing_num_workers", line=1),
            _finding("dataloader_missing_pin_memory", line=1),
        ]
    )

    plan = create_patch_plan_from_analysis(source, analysis)

    assert len(plan.suggestions) == 1
    assert plan.suggestions[0].edits[0].replacement_text == (
        "loader = DataLoader("
        "dataset, batch_size=32, num_workers=4, pin_memory=True)\n"
    )


def test_combines_num_workers_zero_and_pin_memory_false_into_one_edit() -> None:
    source = (
        "loader = DataLoader("
        "dataset, batch_size=32, num_workers=0, pin_memory=False)\n"
    )
    analysis = _analysis(
        [
            _finding("dataloader_num_workers_zero", line=1),
            _finding("dataloader_pin_memory_false", line=1),
        ]
    )

    plan = create_patch_plan_from_analysis(source, analysis)

    assert len(plan.suggestions) == 1
    assert plan.suggestions[0].edits[0].replacement_text == (
        "loader = DataLoader("
        "dataset, batch_size=32, num_workers=4, pin_memory=True)\n"
    )


def test_does_not_create_separate_overlapping_suggestions_when_combined() -> None:
    source = "loader = DataLoader(dataset, batch_size=32, num_workers=0)\n"
    analysis = _analysis(
        [
            _finding("dataloader_num_workers_zero", line=1),
            _finding("dataloader_missing_pin_memory", line=1),
        ]
    )

    plan = create_patch_plan_from_analysis(source, analysis)

    assert [suggestion.id for suggestion in plan.suggestions] == [
        "patch_dataloader_combined_line_1"
    ]


def test_generated_combined_patch_plan_diff_has_no_overlap_warning() -> None:
    source = "loader = DataLoader(dataset, batch_size=32, num_workers=0)\n"
    analysis = _analysis(
        [
            _finding("dataloader_num_workers_zero", line=1),
            _finding("dataloader_missing_pin_memory", line=1),
        ]
    )

    plan = create_patch_plan_from_analysis(source, analysis)
    diff, warnings = generate_patch_plan_diff(source, plan)

    assert warnings == []
    assert "-loader = DataLoader(dataset, batch_size=32, num_workers=0)" in diff
    assert (
        "+loader = DataLoader("
        "dataset, batch_size=32, num_workers=4, pin_memory=True)"
    ) in diff


def test_combined_first_line_dataloader_diff_skips_overlapping_cudnn_patch() -> None:
    source = "loader = DataLoader(dataset, batch_size=32, num_workers=0)\n"
    analysis = _analysis(
        [
            _finding("dataloader_num_workers_zero", line=1),
            _finding("dataloader_missing_pin_memory", line=1),
            _finding("cudnn_benchmark_missing", line=None),
        ]
    )

    plan = create_patch_plan_from_analysis(source, analysis)
    diff, warnings = generate_patch_plan_diff(source, plan)

    assert [suggestion.id for suggestion in plan.suggestions] == [
        "patch_dataloader_combined_line_1"
    ]
    assert warnings == []
    assert (
        "+loader = DataLoader("
        "dataset, batch_size=32, num_workers=4, pin_memory=True)"
    ) in diff


def test_creates_cudnn_benchmark_insertion_after_imports() -> None:
    source = (
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "\n"
        "loader = DataLoader(dataset)\n"
    )
    analysis = _analysis([_finding("cudnn_benchmark_missing", line=None)])

    plan = create_patch_plan_from_analysis(source, analysis)

    edit = plan.suggestions[0].edits[0]
    assert plan.suggestions[0].id == "patch_cudnn_benchmark_missing"
    assert edit.start_line == 1
    assert edit.end_line == 2
    assert edit.original_text == (
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
    )
    assert "torch.backends.cudnn.benchmark = True" in edit.replacement_text


def test_does_not_create_cudnn_patch_if_already_present() -> None:
    source = "torch.backends.cudnn.benchmark = True\n"
    analysis = _analysis([_finding("cudnn_benchmark_missing", line=None)])

    plan = create_patch_plan_from_analysis(source, analysis)

    assert plan.suggestions == []
    assert plan.warnings == [
        "No safe automatic patch suggestions were generated."
    ]


def test_does_not_patch_sync_call_findings() -> None:
    source = "for batch in loader:\n    print(loss.item())\n"
    analysis = _analysis([_finding("sync_call_item_in_loop", line=2)])

    plan = create_patch_plan_from_analysis(source, analysis)

    assert plan.suggestions == []
    assert plan.warnings == [
        "No safe automatic patch suggestions were generated."
    ]


def test_failed_analysis_returns_error_patch_plan() -> None:
    analysis = CodeAnalysisResult(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="broken.py",
        status="error",
        error="Syntax error.",
    )

    plan = create_patch_plan_from_analysis("for batch in loader\n", analysis)

    assert plan.status == "error"
    assert plan.filepath == "broken.py"
    assert plan.suggestions == []
    assert plan.error == "Cannot create patch plan from failed analysis."


def test_no_patchable_findings_returns_ok_plan_with_warning() -> None:
    analysis = _analysis([_finding("mixed_precision_autocast_missing", line=1)])

    plan = create_patch_plan_from_analysis("for batch in loader:\n", analysis)

    assert plan.status == "ok"
    assert plan.suggestions == []
    assert plan.warnings == [
        "No safe automatic patch suggestions were generated."
    ]


def test_generated_patch_plan_produces_expected_diff() -> None:
    source = "loader = DataLoader(dataset, num_workers=0, pin_memory=False)\n"
    analysis = _analysis([_finding("dataloader_pin_memory_false", line=1)])

    plan = create_patch_plan_from_analysis(source, analysis)
    diff, warnings = generate_patch_plan_diff(source, plan)

    assert warnings == []
    assert "-loader = DataLoader(dataset, num_workers=0, pin_memory=False)" in diff
    assert "+loader = DataLoader(dataset, num_workers=0, pin_memory=True)" in diff


def test_helpers_cover_line_replace_kwarg_and_import_block() -> None:
    source = "import torch\n\nloader = DataLoader(dataset)\n"
    edit = replace_on_line(
        source,
        3,
        "DataLoader(dataset)",
        "DataLoader(dataset, num_workers=4)",
        "add workers",
    )

    assert get_source_line(source, 1) == "import torch\n"
    assert get_source_line(source, 99) is None
    assert edit is not None
    assert edit.replacement_text == "loader = DataLoader(dataset, num_workers=4)\n"
    assert insert_kwarg_before_closing_paren(
        "loader = DataLoader(dataset)\n",
        "pin_memory=True",
    ) == "loader = DataLoader(dataset, pin_memory=True)\n"
    assert find_import_block(source) == (1, 1, "import torch\n")


def _analysis(findings: list[CodeFinding]) -> CodeAnalysisResult:
    return CodeAnalysisResult(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        findings=findings,
    )


def _finding(
    finding_id: str,
    line: int | None,
    severity: str = "warning",
    confidence: str = "medium",
) -> CodeFinding:
    return CodeFinding(
        id=finding_id,
        title="Finding title",
        category=_category_for_finding(finding_id),
        severity=severity,
        confidence=confidence,
        filepath="train.py",
        line=line,
        column=0 if line is not None else None,
        end_line=line,
        end_column=None,
        summary="Finding summary.",
        rationale="Finding rationale.",
        suggested_action="Finding action.",
        code_snippet=None,
    )


def _category_for_finding(finding_id: str) -> str:
    if finding_id.startswith("dataloader"):
        return "dataloader"
    if finding_id.startswith("cudnn"):
        return "cudnn"
    if finding_id.startswith("sync_call"):
        return "sync_call"
    if finding_id.startswith("mixed_precision"):
        return "mixed_precision"
    return "general"
