"""Create safe patch plans from code analysis findings."""

from __future__ import annotations

from gpuboost.schemas.code_analysis import CodeAnalysisResult, CodeFinding
from gpuboost.schemas.patch_plan import (
    PatchEdit,
    PatchPlan,
    PatchSuggestion,
    create_timestamp,
)


_NO_SAFE_SUGGESTIONS_WARNING = (
    "No safe automatic patch suggestions were generated."
)
_CUDNN_BENCHMARK_LINE = "torch.backends.cudnn.benchmark = True"


def create_patch_plan_from_analysis(
    source_text: str,
    analysis: CodeAnalysisResult,
) -> PatchPlan:
    """Create a safe patch plan from selected code analysis findings."""

    if analysis.status != "ok":
        return PatchPlan(
            generated_at=create_timestamp(),
            filepath=analysis.filepath,
            status="error",
            suggestions=[],
            error="Cannot create patch plan from failed analysis.",
        )

    suggestions: list[PatchSuggestion] = []
    for finding in analysis.findings:
        suggestion = _create_suggestion_for_finding(source_text, finding)
        if suggestion is not None:
            suggestions.append(suggestion)

    warnings = [] if suggestions else [_NO_SAFE_SUGGESTIONS_WARNING]
    return PatchPlan(
        generated_at=create_timestamp(),
        filepath=analysis.filepath,
        status="ok",
        suggestions=suggestions,
        warnings=warnings,
        error=None,
    )


def get_source_line(source_text: str, line: int | None) -> str | None:
    """Return a 1-based source line with its newline, if available."""

    if line is None or line < 1:
        return None

    lines = source_text.splitlines(keepends=True)
    if line > len(lines):
        return None

    return lines[line - 1]


def replace_on_line(
    source_text: str,
    line: int | None,
    old: str,
    new: str,
    description: str,
) -> PatchEdit | None:
    """Create a single-line replacement edit when exact text is present."""

    source_line = get_source_line(source_text, line)
    if source_line is None or old not in source_line or line is None:
        return None

    return PatchEdit(
        filepath="",
        start_line=line,
        end_line=line,
        original_text=source_line,
        replacement_text=source_line.replace(old, new, 1),
        description=description,
    )


def insert_kwarg_before_closing_paren(
    line_text: str,
    kwarg_text: str,
) -> str | None:
    """Insert a keyword argument before the final closing parenthesis."""

    close_index = line_text.rfind(")")
    open_index = line_text.find("DataLoader(")
    if open_index == -1 or close_index == -1 or close_index < open_index:
        return None

    return f"{line_text[:close_index]}, {kwarg_text}{line_text[close_index:]}"


def find_import_block(source_text: str) -> tuple[int, int, str] | None:
    """Return the initial contiguous import block, if one exists."""

    lines = source_text.splitlines(keepends=True)
    start_line: int | None = None
    end_line: int | None = None

    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped and start_line is None:
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            if start_line is None:
                start_line = index
            end_line = index
            continue
        break

    if start_line is None or end_line is None:
        return None

    block_text = "".join(lines[start_line - 1 : end_line])
    return start_line, end_line, block_text


def _create_suggestion_for_finding(
    source_text: str,
    finding: CodeFinding,
) -> PatchSuggestion | None:
    if finding.id == "dataloader_num_workers_zero":
        edit = replace_on_line(
            source_text,
            finding.line,
            "num_workers=0",
            "num_workers=4",
            "Set DataLoader num_workers to 4.",
        )
        return _line_replacement_suggestion(
            finding,
            edit,
            "Set DataLoader num_workers to 4",
            f"patch_dataloader_num_workers_zero_line_{finding.line}",
        )

    if finding.id == "dataloader_pin_memory_false":
        edit = replace_on_line(
            source_text,
            finding.line,
            "pin_memory=False",
            "pin_memory=True",
            "Enable DataLoader pinned memory.",
        )
        return _line_replacement_suggestion(
            finding,
            edit,
            "Enable DataLoader pin_memory",
            f"patch_dataloader_pin_memory_false_line_{finding.line}",
        )

    if finding.id == "dataloader_missing_pin_memory":
        edit = _create_missing_kwarg_edit(
            source_text,
            finding,
            "pin_memory",
            "pin_memory=True",
            "Add pin_memory=True to DataLoader.",
        )
        return _line_replacement_suggestion(
            finding,
            edit,
            "Add DataLoader pin_memory=True",
            f"patch_dataloader_missing_pin_memory_line_{finding.line}",
        )

    if finding.id == "dataloader_missing_num_workers":
        edit = _create_missing_kwarg_edit(
            source_text,
            finding,
            "num_workers",
            "num_workers=4",
            "Add num_workers=4 to DataLoader.",
        )
        return _line_replacement_suggestion(
            finding,
            edit,
            "Add DataLoader num_workers=4",
            f"patch_dataloader_missing_num_workers_line_{finding.line}",
        )

    if finding.id == "cudnn_benchmark_missing":
        edit = _create_cudnn_benchmark_edit(source_text, finding.filepath)
        return _line_replacement_suggestion(
            finding,
            edit,
            "Enable cuDNN benchmark mode",
            "patch_cudnn_benchmark_missing",
        )

    return None


def _create_missing_kwarg_edit(
    source_text: str,
    finding: CodeFinding,
    existing_kwarg: str,
    inserted_kwarg: str,
    description: str,
) -> PatchEdit | None:
    source_line = get_source_line(source_text, finding.line)
    if (
        source_line is None
        or finding.line is None
        or "DataLoader(" not in source_line
        or existing_kwarg in source_line
    ):
        return None

    replacement = insert_kwarg_before_closing_paren(source_line, inserted_kwarg)
    if replacement is None:
        return None

    return PatchEdit(
        filepath=finding.filepath,
        start_line=finding.line,
        end_line=finding.line,
        original_text=source_line,
        replacement_text=replacement,
        description=description,
    )


def _create_cudnn_benchmark_edit(
    source_text: str,
    filepath: str,
) -> PatchEdit | None:
    if _CUDNN_BENCHMARK_LINE in source_text:
        return None

    import_block = find_import_block(source_text)
    if import_block is not None:
        start_line, end_line, original_text = import_block
        replacement_text = (
            f"{original_text.rstrip()}\n\n{_CUDNN_BENCHMARK_LINE}\n"
        )
        return PatchEdit(
            filepath=filepath,
            start_line=start_line,
            end_line=end_line,
            original_text=original_text,
            replacement_text=replacement_text,
            description="Enable cuDNN benchmark mode near startup.",
        )

    first_line = get_source_line(source_text, 1)
    if first_line is None:
        return None

    return PatchEdit(
        filepath=filepath,
        start_line=1,
        end_line=1,
        original_text=first_line,
        replacement_text=f"{_CUDNN_BENCHMARK_LINE}\n\n{first_line}",
        description="Enable cuDNN benchmark mode near startup.",
    )


def _line_replacement_suggestion(
    finding: CodeFinding,
    edit: PatchEdit | None,
    title: str,
    suggestion_id: str,
) -> PatchSuggestion | None:
    if edit is None:
        return None

    if not edit.filepath:
        edit.filepath = finding.filepath

    return PatchSuggestion(
        id=suggestion_id,
        title=title,
        category=finding.category,
        severity=finding.severity,
        confidence=finding.confidence,
        filepath=finding.filepath,
        finding_ids=[finding.id],
        summary=finding.summary,
        rationale=finding.rationale,
        edits=[edit],
        warnings=[],
    )
