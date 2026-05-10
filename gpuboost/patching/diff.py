"""Unified diff generation for GPUBoost patch plans."""

from __future__ import annotations

import difflib

from gpuboost.schemas.patch_plan import PatchEdit, PatchPlan


def apply_patch_edits_to_text(
    source_text: str,
    edits: list[PatchEdit],
) -> tuple[str, list[str]]:
    """Apply patch edits to source text without mutating files."""

    lines = source_text.splitlines(keepends=True)
    warnings: list[str] = []
    applied_ranges: list[tuple[int, int]] = []

    for edit in sorted(edits, key=lambda item: item.start_line, reverse=True):
        start_index = edit.start_line - 1
        end_index = edit.end_line

        if not _is_valid_line_range(edit, len(lines)):
            warnings.append(
                f"Skipped edit {edit.description}: invalid line range."
            )
            continue

        if any(
            _ranges_overlap(start_index, end_index, applied_start, applied_end)
            for applied_start, applied_end in applied_ranges
        ):
            warnings.append(
                f"Skipped edit {edit.description}: edit overlaps with another edit."
            )
            continue

        original_text = "".join(lines[start_index:end_index])
        if original_text != edit.original_text:
            warnings.append(
                "Skipped edit "
                f"{edit.description}: original text did not match source."
            )
            continue

        lines[start_index:end_index] = edit.replacement_text.splitlines(
            keepends=True
        )
        applied_ranges.append((start_index, end_index))

    return "".join(lines), warnings


def generate_unified_diff(
    original_text: str,
    modified_text: str,
    filepath: str,
    context_lines: int = 3,
) -> str:
    """Generate a unified diff for modified text."""

    if original_text == modified_text:
        return ""

    diff_lines = difflib.unified_diff(
        original_text.splitlines(),
        modified_text.splitlines(),
        fromfile=filepath,
        tofile=f"{filepath} (GPUBoost suggested)",
        n=context_lines,
        lineterm="",
    )
    return "\n".join(diff_lines)


def generate_patch_plan_diff(
    source_text: str,
    plan: PatchPlan,
) -> tuple[str, list[str]]:
    """Generate a unified diff string from a patch plan."""

    warnings = list(plan.warnings)
    if plan.status != "ok":
        warnings.append("Patch plan status is not ok.")
        return "", warnings

    edits = [
        edit
        for suggestion in plan.suggestions
        for edit in suggestion.edits
    ]
    if not edits:
        warnings.append("Patch plan contains no edits.")
        return "", warnings

    modified_text, apply_warnings = apply_patch_edits_to_text(source_text, edits)
    warnings.extend(apply_warnings)
    diff = generate_unified_diff(source_text, modified_text, plan.filepath)
    return diff, warnings


def _is_valid_line_range(edit: PatchEdit, line_count: int) -> bool:
    return (
        edit.start_line >= 1
        and edit.end_line >= edit.start_line
        and edit.start_line <= line_count
        and edit.end_line <= line_count
    )


def _ranges_overlap(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> bool:
    return first_start < second_end and second_start < first_end
