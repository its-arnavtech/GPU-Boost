"""Tests for Phase 9.3 history run record builder."""

from __future__ import annotations

from gpuboost.history.builder import (
    build_history_run_record,
    hash_file_if_exists,
    hash_text,
)
from gpuboost.schemas.agent import AgentAction, AgentGoal, AgentPlan, AgentRunResult


def test_hash_text_stable() -> None:
    assert hash_text("value = 1\n") == hash_text("value = 1\n")
    assert len(hash_text("value = 1\n")) == 64


def test_hash_file_if_exists_returns_hash_for_existing_file(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text("value = 1\n", encoding="utf-8", newline="\n")

    assert hash_file_if_exists(str(script)) == hash_text("value = 1\n")


def test_hash_file_if_exists_returns_none_for_missing_or_none(tmp_path) -> None:
    assert hash_file_if_exists(None) is None
    assert hash_file_if_exists(str(tmp_path / "missing.py")) is None


def test_build_history_run_record_basic_fields(tmp_path) -> None:
    script = tmp_path / "train.py"
    script.write_text("value = 1\n", encoding="utf-8", newline="\n")
    result = _make_result(script_path=str(script))

    record = build_history_run_record(result, command="agent optimize", run_id="run-1")

    assert record.run_id == "run-1"
    assert record.status == "ok"
    assert record.command == "agent optimize"
    assert record.schema_version == "history.run.v1"
    assert record.goal_kind == "optimize_script"
    assert record.goal_description == "Optimize script."
    assert record.script_path == str(script)
    assert record.script_sha256 == hash_text("value = 1\n")


def test_provided_run_id_is_used() -> None:
    record = build_history_run_record(_make_result(), run_id="provided")

    assert record.run_id == "provided"


def test_action_statuses_are_captured() -> None:
    result = _make_result(
        actions=[
            _make_action("inspect", "completed"),
            _make_action("benchmark", "failed"),
        ],
    )

    record = build_history_run_record(result, run_id="run-1")

    assert record.action_statuses == {
        "inspect": "completed",
        "benchmark": "failed",
    }


def test_script_hash_stored_without_raw_source(tmp_path) -> None:
    script = tmp_path / "train.py"
    source = "secret_model_code = True\n"
    script.write_text(source, encoding="utf-8", newline="\n")

    record = build_history_run_record(
        _make_result(script_path=str(script)),
        run_id="run-1",
    )
    data = record.to_dict()

    assert record.script_sha256 == hash_text(source)
    assert source not in str(data)
    assert "source_code" not in data
    assert "raw_source" not in data


def test_trial_summary_extracted_from_artifacts() -> None:
    result = _make_result(
        artifacts={
            "trial": {
                "status": "ok",
                "patch_applied": True,
                "syntax_check_status": "passed",
                "test_status": "passed",
                "original_file_unchanged": True,
            }
        }
    )

    record = build_history_run_record(result, run_id="run-1")

    assert record.trial_summary == {
        "status": "ok",
        "patch_applied": True,
        "syntax_check_status": "passed",
        "test_status": "passed",
        "original_file_unchanged": True,
    }


def test_comparison_summary_extracted_from_artifacts() -> None:
    result = _make_result(
        artifacts={
            "comparison": {
                "status": "ok",
                "overall_verdict": "improved",
                "sections": [{"title": "Throughput"}],
            }
        }
    )

    record = build_history_run_record(result, run_id="run-1")

    assert record.comparison_summary == {
        "status": "ok",
        "overall_verdict": "improved",
    }


def test_patch_summary_has_diff_true_when_diff_artifact_exists() -> None:
    record = build_history_run_record(
        _make_result(artifacts={"diff": "--- a\n+++ b\n"}),
        run_id="run-1",
    )

    assert record.patch_summary == {"has_diff": True}


def test_metadata_counts_are_correct() -> None:
    result = _make_result(
        actions=[
            _make_action("inspect", "completed"),
            _make_action("benchmark", "failed"),
            _make_action("summary", "pending"),
        ],
        artifacts={
            "diff": "--- a\n+++ b\n",
            "trial": {"status": "ok"},
            "comparison": {"status": "ok"},
        },
    )

    record = build_history_run_record(result, run_id="run-1")

    assert record.metadata == {
        "event_count": 0,
        "action_count": 3,
        "completed_action_count": 1,
        "failed_action_count": 1,
        "has_diff": True,
        "has_trial": True,
        "has_comparison": True,
    }


def test_warnings_and_error_copied() -> None:
    result = _make_result(status="error", warnings=["Careful."], error="Failed.")

    record = build_history_run_record(result, run_id="run-1")

    assert record.status == "error"
    assert record.warnings == ["Careful."]
    assert record.error == "Failed."


def test_missing_optional_artifacts_produce_empty_summaries() -> None:
    record = build_history_run_record(_make_result(artifacts={}), run_id="run-1")

    assert record.benchmark_summary == {}
    assert record.advisor_summary == {}
    assert record.code_summary == {}
    assert record.patch_summary == {}
    assert record.trial_summary == {}
    assert record.comparison_summary == {}


def test_no_raw_diff_stored_in_record() -> None:
    raw_diff = "--- train.py\n+++ train.py\n-value = 1\n+value = 2"
    record = build_history_run_record(
        _make_result(artifacts={"diff": raw_diff}),
        run_id="run-1",
    )

    assert raw_diff not in str(record.to_dict())
    assert record.metadata["has_diff"] is True


def test_no_stdout_or_stderr_stored_when_trial_artifact_has_them() -> None:
    record = build_history_run_record(
        _make_result(
            artifacts={
                "trial": {
                    "status": "ok",
                    "stdout": "full stdout",
                    "stderr": "full stderr",
                }
            }
        ),
        run_id="run-1",
    )

    data = record.to_dict()

    assert "stdout" not in str(data)
    assert "stderr" not in str(data)
    assert record.trial_summary == {"status": "ok"}


def _make_result(
    *,
    script_path: str | None = None,
    status: str = "ok",
    warnings: list[str] | None = None,
    error: str | None = None,
    actions: list[AgentAction] | None = None,
    artifacts: dict[str, object] | None = None,
) -> AgentRunResult:
    goal = AgentGoal(
        id="goal",
        kind="optimize_script",
        description="Optimize script.",
        script_path=script_path,
    )
    resolved_actions = actions if actions is not None else [_make_action("inspect")]
    plan = AgentPlan(id="plan", goal=goal, actions=resolved_actions)
    return AgentRunResult(
        generated_at="2026-01-01T00:00:00+00:00",
        goal=goal,
        plan=plan,
        status=status,
        warnings=warnings or [],
        error=error,
        artifacts=artifacts if artifacts is not None else {},
    )


def _make_action(action_id: str, status: str = "completed") -> AgentAction:
    return AgentAction(
        id=action_id,
        name=action_id,
        description=f"Run {action_id}.",
        required=True,
        status=status,
    )
