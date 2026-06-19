"""Tests for Phase 4 patch-plan unified diff generation."""

from __future__ import annotations

from gpuboost.patching.diff import (
    apply_patch_edits_to_text,
    generate_patch_plan_diff,
    generate_unified_diff,
)
from gpuboost.schemas.patch_plan import PatchEdit, PatchPlan, PatchSuggestion


def test_single_line_replacement() -> None:
    source = "a = 1\nb = 2\n"
    edit = _edit(
        start_line=2,
        end_line=2,
        original_text="b = 2\n",
        replacement_text="b = 3\n",
        description="update b",
    )

    modified, warnings = apply_patch_edits_to_text(source, [edit])

    assert modified == "a = 1\nb = 3\n"
    assert warnings == []


def test_multi_line_replacement() -> None:
    source = "before()\nloss = model(batch)\nloss.backward()\nafter()\n"
    edit = _edit(
        start_line=2,
        end_line=3,
        original_text="loss = model(batch)\nloss.backward()\n",
        replacement_text=(
            "with torch.amp.autocast('cuda'):\n"
            "    loss = model(batch)\n"
            "loss.backward()\n"
        ),
        description="wrap forward pass",
    )

    modified, warnings = apply_patch_edits_to_text(source, [edit])

    assert modified == (
        "before()\n"
        "with torch.amp.autocast('cuda'):\n"
        "    loss = model(batch)\n"
        "loss.backward()\n"
        "after()\n"
    )
    assert warnings == []


def test_multiple_edits_applied_bottom_to_top() -> None:
    source = "first\nsecond\nthird\n"
    edits = [
        _edit(1, 1, "first\n", "FIRST\n", "replace first"),
        _edit(3, 3, "third\n", "THIRD\n", "replace third"),
    ]

    modified, warnings = apply_patch_edits_to_text(source, edits)

    assert modified == "FIRST\nsecond\nTHIRD\n"
    assert warnings == []


def test_invalid_line_range_is_skipped_with_warning() -> None:
    source = "only\n"
    edit = _edit(2, 3, "", "replacement\n", "invalid edit")

    modified, warnings = apply_patch_edits_to_text(source, [edit])

    assert modified == source
    assert warnings == ["Skipped edit invalid edit: invalid line range."]


def test_original_text_mismatch_is_skipped_with_warning() -> None:
    source = "value = 1\n"
    edit = _edit(
        1,
        1,
        "value = 2\n",
        "value = 3\n",
        "replace value",
    )

    modified, warnings = apply_patch_edits_to_text(source, [edit])

    assert modified == source
    assert warnings == [
        "Skipped edit replace value: original text did not match source."
    ]


def test_overlapping_edits_are_skipped_with_warning() -> None:
    source = "one\ntwo\nthree\nfour\n"
    edits = [
        _edit(2, 3, "two\nthree\n", "TWO_THREE\n", "replace middle"),
        _edit(3, 4, "three\nfour\n", "THREE_FOUR\n", "replace bottom"),
    ]

    modified, warnings = apply_patch_edits_to_text(source, edits)

    assert modified == "one\ntwo\nTHREE_FOUR\n"
    assert warnings == [
        "Skipped edit replace middle: edit overlaps with another edit."
    ]


def test_non_overlapping_edits_apply_regardless_of_input_order() -> None:
    # Supplied top-to-bottom; the function must sort and apply bottom-up so the
    # earlier edit's replacement does not shift the later edit's line numbers.
    source = "one\ntwo\nthree\nfour\n"
    edits = [
        _edit(1, 1, "one\n", "ONE\n", "replace first"),
        _edit(4, 4, "four\n", "FOUR\n", "replace last"),
    ]

    modified, warnings = apply_patch_edits_to_text(source, edits)

    assert modified == "ONE\ntwo\nthree\nFOUR\n"
    assert warnings == []


def test_unified_diff_includes_fromfile_and_tofile_labels() -> None:
    diff = generate_unified_diff(
        "a = 1\n",
        "a = 2\n",
        "train.py",
    )

    assert "--- train.py" in diff
    assert "+++ train.py (GPUBoost suggested)" in diff
    assert "-a = 1" in diff
    assert "+a = 2" in diff


def test_empty_diff_returned_when_no_changes() -> None:
    diff = generate_unified_diff("same\n", "same\n", "train.py")

    assert diff == ""


def test_patch_plan_with_no_edits_returns_warning() -> None:
    plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        suggestions=[_suggestion(edits=[])],
    )

    diff, warnings = generate_patch_plan_diff("source\n", plan)

    assert diff == ""
    assert warnings == ["Patch plan contains no edits."]


def test_patch_plan_status_error_returns_warning() -> None:
    plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="error",
        error="Cannot plan patch.",
    )

    diff, warnings = generate_patch_plan_diff("source\n", plan)

    assert diff == ""
    assert warnings == ["Patch plan status is not ok."]


def test_generate_patch_plan_diff_collects_all_suggestion_edits() -> None:
    source = "a = 1\nb = 2\n"
    plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        suggestions=[
            _suggestion(edits=[_edit(1, 1, "a = 1\n", "a = 10\n", "edit a")]),
            _suggestion(edits=[_edit(2, 2, "b = 2\n", "b = 20\n", "edit b")]),
        ],
    )

    diff, warnings = generate_patch_plan_diff(source, plan)

    assert warnings == []
    assert "-a = 1" in diff
    assert "+a = 10" in diff
    assert "-b = 2" in diff
    assert "+b = 20" in diff


def test_final_newline_is_handled_reasonably() -> None:
    source = "first\nsecond"
    edit = _edit(
        start_line=2,
        end_line=2,
        original_text="second",
        replacement_text="second\n",
        description="add final newline",
    )

    modified, warnings = apply_patch_edits_to_text(source, [edit])

    assert modified == "first\nsecond\n"
    assert warnings == []


def _edit(
    start_line: int,
    end_line: int,
    original_text: str,
    replacement_text: str,
    description: str,
) -> PatchEdit:
    return PatchEdit(
        filepath="train.py",
        start_line=start_line,
        end_line=end_line,
        original_text=original_text,
        replacement_text=replacement_text,
        description=description,
    )


def _suggestion(edits: list[PatchEdit]) -> PatchSuggestion:
    return PatchSuggestion(
        id="suggestion",
        title="Suggestion",
        category="general",
        severity="info",
        confidence="medium",
        filepath="train.py",
        finding_ids=["finding"],
        summary="Summary.",
        rationale="Rationale.",
        edits=edits,
    )


def test_unified_diff_marks_missing_final_newline() -> None:
    # Neither side has a trailing newline; standard diff output notes it.
    diff = generate_unified_diff("a = 1", "a = 2", "train.py")

    assert "No newline at end of file" in diff
