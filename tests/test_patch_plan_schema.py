"""Tests for Phase 4 patch planning schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from gpuboost.schemas.patch_plan import (
    PatchEdit,
    PatchPlan,
    PatchSuggestion,
    create_timestamp,
)


def test_patch_edit_creation() -> None:
    edit = _make_edit()

    assert edit.filepath == "train.py"
    assert edit.start_line == 10
    assert edit.end_line == 12
    assert edit.original_text == "loss = model(batch)\n"
    assert edit.replacement_text == "with torch.amp.autocast('cuda'):\n"
    assert edit.description == "Wrap forward pass with autocast."


def test_patch_suggestion_creation() -> None:
    edit = _make_edit()
    suggestion = PatchSuggestion(
        id="add-autocast",
        title="Add AMP autocast",
        category="mixed_precision",
        severity="info",
        confidence="medium",
        filepath="train.py",
        finding_ids=["mixed_precision_autocast_missing"],
        summary="Training loop does not use autocast.",
        rationale="AMP can improve eligible GPU workloads.",
        edits=[edit],
        warnings=["Review numerical stability."],
    )

    assert suggestion.id == "add-autocast"
    assert suggestion.finding_ids == ["mixed_precision_autocast_missing"]
    assert suggestion.edits == [edit]
    assert suggestion.warnings == ["Review numerical stability."]


def test_patch_plan_creation() -> None:
    suggestion = _make_suggestion()
    plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        suggestions=[suggestion],
        warnings=["Patch planning is experimental."],
        error=None,
    )

    assert plan.generated_at == "2026-01-01T00:00:00+00:00"
    assert plan.filepath == "train.py"
    assert plan.status == "ok"
    assert plan.suggestions == [suggestion]
    assert plan.warnings == ["Patch planning is experimental."]
    assert plan.error is None


def test_to_dict_output() -> None:
    plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        suggestions=[_make_suggestion()],
    )

    data = plan.to_dict()

    assert data["filepath"] == "train.py"
    assert data["status"] == "ok"
    assert data["suggestions"][0]["id"] == "add-autocast"
    assert data["suggestions"][0]["edits"][0]["start_line"] == 10
    assert data["warnings"] == []
    assert data["error"] is None


def test_json_serialization() -> None:
    plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="train.py",
        status="ok",
        suggestions=[_make_suggestion()],
    )

    serialized = json.dumps(plan.to_dict())
    deserialized = json.loads(serialized)

    assert deserialized["suggestions"][0]["title"] == "Add AMP autocast"
    assert deserialized["suggestions"][0]["edits"][0]["description"] == (
        "Wrap forward pass with autocast."
    )


def test_default_empty_lists_are_independent() -> None:
    first_suggestion = PatchSuggestion(
        id="first",
        title="First",
        category="general",
        severity="info",
        confidence="low",
        filepath="first.py",
    )
    second_suggestion = PatchSuggestion(
        id="second",
        title="Second",
        category="general",
        severity="info",
        confidence="low",
        filepath="second.py",
    )
    first_plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="first.py",
        status="ok",
    )
    second_plan = PatchPlan(
        generated_at="2026-01-01T00:00:01+00:00",
        filepath="second.py",
        status="ok",
    )

    first_suggestion.finding_ids.append("first-finding")
    first_suggestion.edits.append(_make_edit())
    first_suggestion.warnings.append("first warning")
    first_plan.suggestions.append(first_suggestion)
    first_plan.warnings.append("plan warning")

    assert first_suggestion.finding_ids == ["first-finding"]
    assert len(first_suggestion.edits) == 1
    assert first_suggestion.warnings == ["first warning"]
    assert second_suggestion.finding_ids == []
    assert second_suggestion.edits == []
    assert second_suggestion.warnings == []
    assert first_plan.suggestions == [first_suggestion]
    assert first_plan.warnings == ["plan warning"]
    assert second_plan.suggestions == []
    assert second_plan.warnings == []


def test_status_error_plan_can_carry_error_message() -> None:
    plan = PatchPlan(
        generated_at="2026-01-01T00:00:00+00:00",
        filepath="broken.py",
        status="error",
        error="Unable to plan patches for parse error.",
    )

    data = plan.to_dict()

    assert plan.status == "error"
    assert plan.error == "Unable to plan patches for parse error."
    assert data["error"] == "Unable to plan patches for parse error."


def test_patch_suggestion_can_contain_multiple_patch_edits() -> None:
    first_edit = _make_edit()
    second_edit = PatchEdit(
        filepath="train.py",
        start_line=20,
        end_line=20,
        original_text="optimizer.step()\n",
        replacement_text="scaler.step(optimizer)\n",
        description="Use the AMP scaler for optimizer step.",
    )
    suggestion = _make_suggestion(edits=[first_edit, second_edit])

    assert suggestion.edits == [first_edit, second_edit]
    assert suggestion.to_dict()["edits"][1]["start_line"] == 20


def test_patch_suggestion_can_reference_multiple_finding_ids() -> None:
    suggestion = _make_suggestion(
        finding_ids=[
            "mixed_precision_autocast_missing",
            "cudnn_benchmark_missing",
        ],
    )

    assert suggestion.finding_ids == [
        "mixed_precision_autocast_missing",
        "cudnn_benchmark_missing",
    ]
    assert suggestion.to_dict()["finding_ids"] == [
        "mixed_precision_autocast_missing",
        "cudnn_benchmark_missing",
    ]


def test_create_timestamp_returns_utc_iso_timestamp() -> None:
    timestamp = create_timestamp()
    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo == timezone.utc


def _make_edit() -> PatchEdit:
    return PatchEdit(
        filepath="train.py",
        start_line=10,
        end_line=12,
        original_text="loss = model(batch)\n",
        replacement_text="with torch.amp.autocast('cuda'):\n",
        description="Wrap forward pass with autocast.",
    )


def _make_suggestion(
    *,
    edits: list[PatchEdit] | None = None,
    finding_ids: list[str] | None = None,
) -> PatchSuggestion:
    return PatchSuggestion(
        id="add-autocast",
        title="Add AMP autocast",
        category="mixed_precision",
        severity="info",
        confidence="medium",
        filepath="train.py",
        finding_ids=finding_ids or ["mixed_precision_autocast_missing"],
        summary="Training loop does not use autocast.",
        rationale="AMP can improve eligible GPU workloads.",
        edits=edits or [_make_edit()],
    )
