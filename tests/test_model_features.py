"""Tests for Phase 10B safe model feature extraction."""

from __future__ import annotations

from gpuboost.model.features import (
    extract_model_features_from_agent_result,
    extract_model_features_from_history_record,
    safe_count,
    sanitize_feature_value,
)
from gpuboost.schemas.agent import (
    AgentAction,
    AgentEvent,
    AgentGoal,
    AgentPlan,
    AgentRunResult,
)
from gpuboost.schemas.history import HistoryRunRecord


def test_extracts_features_from_result_with_trial_artifact() -> None:
    result = _make_result(
        actions=[
            _make_action("inspect", "completed"),
            _make_action("analyze_code", "completed"),
            _make_action("trial", "failed"),
        ],
        warnings=["Review patch."],
        plan_warnings=["Plan warning."],
        events=[_make_event()],
        artifacts={
            "gpu": {"gpu_name": "Test GPU", "cuda_available": True},
            "trial": {
                "status": "ok",
                "patch_applied": True,
                "syntax_check_status": "passed",
                "test_status": "passed",
                "original_file_unchanged": True,
                "steps": [{"name": "syntax"}, {"name": "tests"}],
                "stdout": "secret stdout",
                "stderr": "secret stderr",
            },
        },
    )

    features = extract_model_features_from_agent_result(result)
    data = features.to_dict()

    assert features.hardware == {"gpu_name": "Test GPU", "cuda_available": True}
    assert features.trial == {
        "status": "ok",
        "patch_applied": True,
        "syntax_check_status": "passed",
        "test_status": "passed",
        "original_file_unchanged": True,
        "step_count": 2,
    }
    assert features.metadata == {
        "action_count": 3,
        "completed_action_count": 2,
        "failed_action_count": 1,
        "warning_count": 2,
        "event_count": 1,
        "has_diff": False,
        "has_trial": True,
        "has_comparison": False,
    }
    assert "secret stdout" not in str(data)
    assert "secret stderr" not in str(data)
    assert not features.is_empty()


def test_extracts_features_from_result_with_comparison_artifact() -> None:
    result = _make_result(
        artifacts={
            "comparison": {
                "status": "ok",
                "overall_verdict": "improved",
                "sections": [{"title": "throughput"}],
            }
        }
    )

    features = extract_model_features_from_agent_result(result)

    assert features.comparison == {
        "status": "ok",
        "overall_verdict": "improved",
    }
    assert features.metadata["has_comparison"] is True


def test_has_diff_true_and_false_without_raw_diff() -> None:
    raw_diff = "--- train.py\n+++ train.py\n-value = 1\n+value = 2"
    with_diff = extract_model_features_from_agent_result(
        _make_result(artifacts={"diff": raw_diff})
    )
    without_diff = extract_model_features_from_agent_result(_make_result())

    assert with_diff.patches == {"has_diff": True}
    assert with_diff.metadata["has_diff"] is True
    assert raw_diff not in str(with_diff.to_dict())
    assert without_diff.patches == {"has_diff": False}
    assert without_diff.metadata["has_diff"] is False


def test_counts_result_summaries_when_available() -> None:
    result = _make_result(
        artifacts={
            "advisor": {
                "recommendations": [{"id": "amp"}, {"id": "batch"}],
                "warnings": ["careful"],
            },
            "code_analysis": {"findings": [{"id": "sync"}], "warnings": []},
            "patch_plan": {"suggestions": [{"id": "patch"}]},
            "benchmark": {
                "status": "ok",
                "metrics": [{"name": "throughput", "value": 123.4}],
            },
        }
    )

    features = extract_model_features_from_agent_result(result)

    assert features.advisor == {"recommendation_count": 2, "warning_count": 1}
    assert features.code["finding_count"] == 1
    assert features.patches["patch_suggestion_count"] == 1
    assert features.benchmarks["metric_count"] == 1
    assert features.benchmarks["metric_throughput"] == 123.4


def test_history_record_extraction_copies_safe_summaries() -> None:
    record = _make_record(
        benchmark_summary={"metric_count": 2, "status": "ok"},
        advisor_summary={"recommendation_count": 3},
        code_summary={"finding_count": 1},
        patch_summary={"has_diff": True, "patch_suggestion_count": 1},
        trial_summary={"status": "ok", "step_count": 2},
        comparison_summary={"status": "ok", "overall_verdict": "neutral"},
    )

    features = extract_model_features_from_history_record(record)

    assert features.hardware == {"gpu_name": "Test GPU", "cuda_available": True}
    assert features.benchmarks == {"metric_count": 2, "status": "ok"}
    assert features.advisor == {"recommendation_count": 3}
    assert features.code == {"finding_count": 1}
    assert features.patches == {"has_diff": True, "patch_suggestion_count": 1}
    assert features.trial == {"status": "ok", "step_count": 2}
    assert features.comparison == {"status": "ok", "overall_verdict": "neutral"}
    assert features.history == {
        "status": "ok",
        "command": "agent optimize",
        "goal_kind": "optimize_script",
        "has_trial": True,
        "has_comparison": True,
    }


def test_history_record_extraction_skips_raw_values() -> None:
    raw_source = "def secret():\n    return 1"
    raw_diff = "--- train.py\n+++ train.py\n-secret\n+safe"
    record = _make_record(
        code_summary={"raw_source": raw_source, "finding_count": 1},
        patch_summary={"raw_diff": raw_diff, "has_diff": True},
        trial_summary={
            "stdout": "secret stdout",
            "stderr": "secret stderr",
            "status": "ok",
        },
    )

    features = extract_model_features_from_history_record(record)
    data = str(features.to_dict())

    assert features.code == {"finding_count": 1}
    assert features.patches == {"has_diff": True}
    assert features.trial == {"status": "ok"}
    assert raw_source not in data
    assert raw_diff not in data
    assert "secret stdout" not in data
    assert "secret stderr" not in data


def test_sanitize_feature_value_handles_primitives() -> None:
    assert sanitize_feature_value("ok") == "ok"
    assert sanitize_feature_value(1) == 1
    assert sanitize_feature_value(1.5) == 1.5
    assert sanitize_feature_value(True) is True
    assert sanitize_feature_value(None) is None


def test_sanitize_feature_value_rejects_nested_values() -> None:
    assert sanitize_feature_value({"value": 1}) is None
    assert sanitize_feature_value([1, 2, 3]) is None


def test_sanitize_feature_value_truncates_or_omits_long_strings() -> None:
    value = "x" * 600
    sanitized = sanitize_feature_value(value)

    assert isinstance(sanitized, str)
    assert len(sanitized) == 500


def test_sanitize_feature_value_keeps_prose_that_mentions_code_words() -> None:
    # Multi-line descriptions that merely mention "import"/"class" must NOT be
    # dropped as raw content.
    prose = "Remember to import the new dataset first.\nThen run the class demo."
    assert sanitize_feature_value(prose) == prose

    note = "GPU notes: driver 535 --- update pending\nCUDA 12.1 detected"
    assert sanitize_feature_value(note) == note


def test_sanitize_feature_value_drops_real_source_code() -> None:
    assert sanitize_feature_value("def secret():\n    return 1") is None
    assert sanitize_feature_value("import torch\nimport os") is None
    assert (
        sanitize_feature_value("from torch import nn\nmodel = nn.Linear(1, 1)")
        is None
    )


def test_sanitize_feature_value_drops_diffs_and_code_fences() -> None:
    assert (
        sanitize_feature_value("--- a.py\n+++ b.py\n-x = 1\n+x = 2") is None
    )
    assert (
        sanitize_feature_value("@@ -1,2 +1,3 @@ def f():\n    pass") is None
    )
    assert sanitize_feature_value("Example:\n```\ncode\n```") is None


def test_safe_count_counts_only_supported_values() -> None:
    assert safe_count([1, 2]) == 2
    assert safe_count({"a": 1}) == 1
    assert safe_count("abc") == 3
    assert safe_count(10) is None


def _make_result(
    *,
    actions: list[AgentAction] | None = None,
    warnings: list[str] | None = None,
    plan_warnings: list[str] | None = None,
    events: list[AgentEvent] | None = None,
    artifacts: dict[str, object] | None = None,
) -> AgentRunResult:
    goal = AgentGoal(
        id="goal",
        kind="optimize_script",
        description="Optimize script.",
        script_path="train.py",
    )
    plan = AgentPlan(
        id="plan",
        goal=goal,
        actions=actions if actions is not None else [_make_action("inspect")],
        warnings=plan_warnings or [],
    )
    return AgentRunResult(
        generated_at="2026-01-01T00:00:00+00:00",
        goal=goal,
        plan=plan,
        status="ok",
        events=events or [],
        warnings=warnings or [],
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


def _make_event() -> AgentEvent:
    return AgentEvent(
        timestamp="2026-01-01T00:00:00+00:00",
        action_id="inspect",
        level="info",
        message="Inspected.",
    )


def _make_record(
    *,
    benchmark_summary: dict[str, object] | None = None,
    advisor_summary: dict[str, object] | None = None,
    code_summary: dict[str, object] | None = None,
    patch_summary: dict[str, object] | None = None,
    trial_summary: dict[str, object] | None = None,
    comparison_summary: dict[str, object] | None = None,
) -> HistoryRunRecord:
    return HistoryRunRecord(
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        status="ok",
        command="agent optimize",
        schema_version="history.run.v1",
        goal_kind="optimize_script",
        goal_description="Optimize script.",
        gpu_name="Test GPU",
        cuda_available=True,
        benchmark_summary=benchmark_summary or {},
        advisor_summary=advisor_summary or {},
        code_summary=code_summary or {},
        patch_summary=patch_summary or {},
        trial_summary=trial_summary or {},
        comparison_summary=comparison_summary or {},
    )


def test_sanitize_keeps_text_with_dash_and_at_separators() -> None:
    # Regression for substring-based raw-content detection: "---" used as a
    # visual separator (not a diff header) must not cause the value to be dropped.
    value = "Section A\n--- details ---\nSection B @@ note"
    assert sanitize_feature_value(value) == value
