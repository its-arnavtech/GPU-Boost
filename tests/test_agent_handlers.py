"""Tests for Phase 5.6 real agent action handlers."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from gpuboost.agent import handlers
from gpuboost.agent.actions import (
    ANALYZE_CODE,
    CREATE_PATCH_PLAN,
    GENERATE_DIFF,
    GENERATE_RECOMMENDATIONS,
    INSPECT_SYSTEM,
    RUN_QUICK_BENCHMARK,
    SUMMARIZE_RESULTS,
)
from gpuboost.agent.state import AgentState
from gpuboost.schemas.agent import AgentAction, AgentGoal


@dataclass(slots=True)
class FakeArtifact:
    data: dict[str, object]
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    filepath: str = "train.py"

    def to_dict(self) -> dict[str, object]:
        return dict(self.data)


def test_inspect_handler_stores_gpu_profile_dict(monkeypatch) -> None:
    profile = FakeArtifact(
        data={"gpus": [{"name": "NVIDIA Test GPU"}]},
        warnings=["Profile warning."],
    )
    monkeypatch.setattr(handlers.profile_module, "collect_profile", lambda: profile)
    state = _make_state()

    handlers.handle_inspect_system(state, _make_action(INSPECT_SYSTEM))

    assert state.gpu_profile == {"gpus": [{"name": "NVIDIA Test GPU"}]}
    assert state.warnings == ["Profile warning."]
    assert state.events[-1].message == "System inspection completed."


def test_quick_benchmark_handler_stores_result_dict_and_raw_suite(
    monkeypatch,
) -> None:
    suite = FakeArtifact(
        data={"results": [{"name": "Quick"}]},
        warnings=["Benchmark warning."],
    )
    captured: dict[str, int] = {}

    def fake_run_quick_benchmark(device_index: int = 0) -> FakeArtifact:
        captured["device_index"] = device_index
        return suite

    monkeypatch.setattr(
        handlers.benchmark_runner,
        "run_quick_benchmark",
        fake_run_quick_benchmark,
    )
    state = _make_state()

    handlers.handle_run_quick_benchmark(
        state,
        _make_action(RUN_QUICK_BENCHMARK, inputs={"device_index": 1}),
    )

    assert captured == {"device_index": 1}
    assert state.benchmark_result == {"results": [{"name": "Quick"}]}
    assert state.metadata["_benchmark_suite"] is suite
    assert state.warnings == ["Benchmark warning."]
    assert state.events[-1].message == "Quick benchmark completed."


def test_recommendations_handler_uses_raw_suite_and_stores_advisor_result(
    monkeypatch,
) -> None:
    suite = FakeArtifact(data={"results": []})
    advisor = FakeArtifact(
        data={"recommendations": [{"id": "rec_001"}]},
        warnings=["Advisor warning."],
    )
    captured: dict[str, object] = {}

    def fake_generate_advisor_result(received_suite: FakeArtifact) -> FakeArtifact:
        captured["suite"] = received_suite
        return advisor

    monkeypatch.setattr(
        handlers.advisor_engine,
        "generate_advisor_result",
        fake_generate_advisor_result,
    )
    state = _make_state()
    state.benchmark_result = suite.to_dict()
    state.metadata["_benchmark_suite"] = suite

    handlers.handle_generate_recommendations(
        state,
        _make_action(GENERATE_RECOMMENDATIONS),
    )

    assert captured == {"suite": suite}
    assert state.advisor_result == {"recommendations": [{"id": "rec_001"}]}
    assert state.warnings == ["Advisor warning."]
    assert state.events[-1].message == "Recommendations generated."


def test_recommendations_handler_fails_clearly_if_benchmark_missing() -> None:
    state = _make_state()

    with pytest.raises(
        ValueError,
        match="Benchmark result is required before generating recommendations.",
    ):
        handlers.handle_generate_recommendations(
            state,
            _make_action(GENERATE_RECOMMENDATIONS),
        )


def test_analyze_code_handler_stores_code_analysis_and_raw_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    script_path = tmp_path / "train.py"
    script_path.write_text("print('train')\n", encoding="utf-8")
    analysis = FakeArtifact(
        data={"status": "ok", "findings": [{"id": "finding_001"}]},
        warnings=["Analysis warning."],
        status="ok",
        filepath=str(script_path),
    )
    captured: dict[str, str] = {}

    def fake_analyze_python_file(filepath: str) -> FakeArtifact:
        captured["filepath"] = filepath
        return analysis

    monkeypatch.setattr(
        handlers.code_analysis_runner,
        "analyze_python_file",
        fake_analyze_python_file,
    )
    state = _make_state(script_path=str(script_path))

    handlers.handle_analyze_code(state, _make_action(ANALYZE_CODE))

    assert captured == {"filepath": str(script_path)}
    assert state.code_analysis == {"status": "ok", "findings": [{"id": "finding_001"}]}
    assert state.metadata["_code_analysis"] is analysis
    assert state.warnings == ["Analysis warning."]
    assert state.events[-1].message == "Code analysis completed."


def test_analyze_code_handler_fails_on_missing_script_path() -> None:
    state = _make_state(script_path=None)

    with pytest.raises(ValueError, match="script_path is required for code analysis."):
        handlers.handle_analyze_code(state, _make_action(ANALYZE_CODE))


def test_create_patch_plan_handler_reads_source_and_stores_artifacts(
    monkeypatch,
    tmp_path,
) -> None:
    script_path = tmp_path / "train.py"
    script_path.write_text("print('train')\n", encoding="utf-8")
    analysis = FakeArtifact(data={"status": "ok"}, filepath=str(script_path))
    patch_plan = FakeArtifact(
        data={"status": "ok", "suggestions": [{"id": "patch_001"}]},
        warnings=["Patch warning."],
    )
    captured: dict[str, object] = {}

    def fake_create_patch_plan_from_analysis(
        source_text: str,
        received_analysis: FakeArtifact,
    ) -> FakeArtifact:
        captured["source_text"] = source_text
        captured["analysis"] = received_analysis
        return patch_plan

    monkeypatch.setattr(
        handlers.patch_planner,
        "create_patch_plan_from_analysis",
        fake_create_patch_plan_from_analysis,
    )
    state = _make_state(script_path=str(script_path))
    state.metadata["_code_analysis"] = analysis

    handlers.handle_create_patch_plan(
        state,
        _make_action(CREATE_PATCH_PLAN),
    )

    assert captured == {
        "source_text": "print('train')\n",
        "analysis": analysis,
    }
    assert state.patch_plan == {"status": "ok", "suggestions": [{"id": "patch_001"}]}
    assert state.metadata["_patch_plan"] is patch_plan
    assert state.metadata["_source_text"] == "print('train')\n"
    assert state.warnings == ["Patch warning."]
    assert state.events[-1].message == "Patch plan created."


def test_generate_diff_handler_stores_diff_and_warnings(monkeypatch) -> None:
    patch_plan = FakeArtifact(data={"status": "ok"})
    captured: dict[str, object] = {}

    def fake_generate_patch_plan_diff(
        source_text: str,
        received_patch_plan: FakeArtifact,
    ) -> tuple[str, list[str]]:
        captured["source_text"] = source_text
        captured["patch_plan"] = received_patch_plan
        return "--- train.py\n+++ train.py\n", ["Diff warning."]

    monkeypatch.setattr(
        handlers.patch_diff,
        "generate_patch_plan_diff",
        fake_generate_patch_plan_diff,
    )
    state = _make_state()
    state.metadata["_patch_plan"] = patch_plan
    state.metadata["_source_text"] = "print('train')\n"

    handlers.handle_generate_diff(state, _make_action(GENERATE_DIFF))

    assert captured == {
        "source_text": "print('train')\n",
        "patch_plan": patch_plan,
    }
    assert state.diff == "--- train.py\n+++ train.py\n"
    assert state.warnings == ["Diff warning."]
    assert state.events[-1].message == "Diff generated."


def test_summarize_results_produces_correct_counts() -> None:
    state = _make_state()
    state.gpu_profile = {"gpus": []}
    state.benchmark_result = {"results": []}
    state.advisor_result = {"recommendations": [{"id": "rec_001"}]}
    state.code_analysis = {"findings": [{"id": "finding_001"}, {"id": "finding_002"}]}
    state.patch_plan = {"suggestions": [{"id": "patch_001"}]}
    state.diff = "--- train.py\n+++ train.py\n"
    state.warnings.extend(["First warning.", "Second warning."])
    state.failed_actions.append("failed_action")

    handlers.handle_summarize_results(state, _make_action(SUMMARIZE_RESULTS))

    assert state.metadata["summary"] == {
        "has_gpu_profile": True,
        "has_benchmark_result": True,
        "recommendation_count": 1,
        "code_finding_count": 2,
        "patch_suggestion_count": 1,
        "has_diff": True,
        "warning_count": 2,
        "failed_action_count": 1,
    }
    assert state.events[-1].message == "Results summarized."


def test_default_handlers_contains_all_action_names() -> None:
    mapping = handlers.default_handlers()

    assert set(mapping) == {
        INSPECT_SYSTEM,
        RUN_QUICK_BENCHMARK,
        GENERATE_RECOMMENDATIONS,
        ANALYZE_CODE,
        CREATE_PATCH_PLAN,
        GENERATE_DIFF,
        SUMMARIZE_RESULTS,
    }


def test_handlers_do_not_apply_patches_or_write_files(monkeypatch, tmp_path) -> None:
    script_path = tmp_path / "train.py"
    original_source = "value = 1\n"
    script_path.write_text(original_source, encoding="utf-8")
    analysis = FakeArtifact(data={"status": "ok"}, filepath=str(script_path))
    patch_plan = FakeArtifact(data={"status": "ok", "suggestions": []})

    monkeypatch.setattr(
        handlers.patch_planner,
        "create_patch_plan_from_analysis",
        lambda source_text, received_analysis: patch_plan,
    )
    monkeypatch.setattr(
        handlers.patch_diff,
        "generate_patch_plan_diff",
        lambda source_text, received_patch_plan: (
            "--- train.py\n+++ train.py\n-value = 1\n+value = 2",
            [],
        ),
    )
    state = _make_state(script_path=str(script_path))
    state.metadata["_code_analysis"] = analysis

    handlers.handle_create_patch_plan(state, _make_action(CREATE_PATCH_PLAN))
    handlers.handle_generate_diff(state, _make_action(GENERATE_DIFF))

    assert script_path.read_text(encoding="utf-8") == original_source
    assert state.diff == "--- train.py\n+++ train.py\n-value = 1\n+value = 2"


def _make_state(script_path: str | None = "train.py") -> AgentState:
    return AgentState(
        goal=AgentGoal(
            id="goal_001",
            kind="optimize_script",
            description="Synthetic handler test goal.",
            script_path=script_path,
        ),
    )


def _make_action(
    name: str,
    *,
    inputs: dict[str, str | int | float | bool | None] | None = None,
) -> AgentAction:
    return AgentAction(
        id=name,
        name=name,
        description=f"Synthetic action: {name}",
        required=True,
        inputs=inputs or {},
    )
