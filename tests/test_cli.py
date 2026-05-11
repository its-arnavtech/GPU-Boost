"""Tests for the Phase 1 CLI."""

import json

from gpuboost.agent.report import AgentReport, AgentReportSection
from gpuboost.cli import main as cli_main
from gpuboost.schemas.agent import AgentAction, AgentGoal, AgentPlan, AgentRunResult
from gpuboost.schemas.benchmark_result import BenchmarkSuiteResult
from gpuboost.schemas.gpu_profile import (
    GPUBoostProfile,
    SystemProfile,
    TorchEnvironmentProfile,
)
from gpuboost.schemas.recommendation import AdvisorResult, Recommendation


_FAKE_DIFF = "\n".join(
    [
        "--- train.py",
        "+++ train.py",
        "@@ -1 +1 @@",
        "-loader = DataLoader(dataset, num_workers=0)",
        "+loader = DataLoader(dataset, num_workers=4)",
    ]
)


def test_cli_info_json_outputs_valid_json(monkeypatch, capsys) -> None:
    def fake_collect_profile() -> GPUBoostProfile:
        return GPUBoostProfile(
            generated_at="2026-01-01T00:00:00+00:00",
            system=SystemProfile(os="Test OS", python_version="3.12"),
            torch_env=TorchEnvironmentProfile(
                torch_installed=False,
                cuda_available=False,
                device_count=0,
            ),
            gpus=[],
            warnings=["no gpu"],
        )

    monkeypatch.setattr(cli_main, "collect_profile", fake_collect_profile)

    exit_code = cli_main.main(["info", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["system"]["os"] == "Test OS"
    assert data["torch_env"]["device_count"] == 0
    assert data["warnings"] == ["no gpu"]
    assert captured.err == ""


def test_cli_benchmark_json_outputs_valid_json(monkeypatch, capsys) -> None:
    def fake_run_quick_benchmark(device_index: int = 0) -> BenchmarkSuiteResult:
        return BenchmarkSuiteResult(
            generated_at="2026-01-01T00:00:00+00:00",
            gpu_name="NVIDIA Test GPU",
            cuda_available=True,
            device_index=device_index,
            results=[],
            warnings=[],
        )

    monkeypatch.setattr(cli_main, "run_quick_benchmark", fake_run_quick_benchmark)

    exit_code = cli_main.main(["benchmark", "--quick", "--json", "--device", "0"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["gpu_name"] == "NVIDIA Test GPU"
    assert data["cuda_available"] is True
    assert data["device_index"] == 0
    assert data["results"] == []
    assert "benchmark" not in data
    assert "advisor" not in data
    assert captured.err == ""


def test_cli_benchmark_json_recommend_outputs_benchmark_and_advisor(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli_main, "run_quick_benchmark", _fake_run_quick_benchmark)
    monkeypatch.setattr(cli_main, "generate_advisor_result", _fake_advisor_result)

    exit_code = cli_main.main(["benchmark", "--quick", "--json", "--recommend"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert sorted(data) == ["advisor", "benchmark"]
    assert data["benchmark"]["gpu_name"] == "NVIDIA Test GPU"
    assert data["advisor"]["recommendations"][0]["title"] == "Enable mixed precision"
    assert captured.err == ""


def test_cli_benchmark_quick_recommend_human_output_includes_recommendations(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli_main, "run_quick_benchmark", _fake_run_quick_benchmark)
    monkeypatch.setattr(cli_main, "generate_advisor_result", _fake_advisor_result)

    exit_code = cli_main.main(["benchmark", "--quick", "--recommend"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Phase 2 Benchmark Suite" in captured.out
    assert "Recommendations:" in captured.out
    assert "[1] Enable mixed precision" in captured.out
    assert "Estimated speedup: 1.26x" in captured.out
    assert captured.err == ""


def test_cli_benchmark_quick_without_recommend_still_works(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli_main, "run_quick_benchmark", _fake_run_quick_benchmark)

    exit_code = cli_main.main(["benchmark", "--quick"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Phase 2 Benchmark Suite" in captured.out
    assert "Run with --recommend to generate optimization advice." in captured.out
    assert "Recommendations:" not in captured.out
    assert captured.err == ""


def test_cli_analyze_json_outputs_valid_json(tmp_path, capsys) -> None:
    filepath = tmp_path / "train.py"
    filepath.write_text(
        "loader = DataLoader(dataset, num_workers=0, pin_memory=False)\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(["analyze", str(filepath), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["status"] == "ok"
    assert data["filepath"] == str(filepath)
    assert any(
        finding["id"] == "dataloader_num_workers_zero"
        for finding in data["findings"]
    )
    assert captured.err == ""


def test_cli_analyze_json_remains_analysis_only_without_patch(tmp_path, capsys) -> None:
    filepath = tmp_path / "train.py"
    filepath.write_text(
        "loader = DataLoader(dataset, num_workers=0, pin_memory=False)\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(["analyze", str(filepath), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["status"] == "ok"
    assert "analysis" not in data
    assert "patch_plan" not in data
    assert "diff" not in data
    assert "patch_warnings" not in data
    assert captured.err == ""


def test_cli_analyze_json_patch_outputs_patch_payload(tmp_path, capsys) -> None:
    filepath = tmp_path / "train.py"
    filepath.write_text(
        "loader = DataLoader(dataset, num_workers=0)\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(["analyze", str(filepath), "--json", "--patch"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert sorted(data) == ["analysis", "diff", "patch_plan", "patch_warnings"]
    assert data["analysis"]["status"] == "ok"
    assert data["patch_plan"]["status"] == "ok"
    assert data["patch_plan"]["suggestions"]
    assert "--- " in data["diff"]
    assert "num_workers=4" in data["diff"]
    assert isinstance(data["patch_warnings"], list)
    assert captured.err == ""


def test_cli_analyze_human_output_includes_title(tmp_path, capsys) -> None:
    filepath = tmp_path / "eval.py"
    filepath.write_text(
        "torch.backends.cudnn.benchmark = True\n"
        "for batch in loader:\n"
        "    outputs = model(batch)\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(["analyze", str(filepath)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Code Analysis" in captured.out
    assert "Inference loop may be missing no_grad or inference_mode" in captured.out
    assert "Status: ok" in captured.out
    assert captured.err == ""


def test_cli_analyze_patch_human_output_includes_patch_section(tmp_path, capsys) -> None:
    filepath = tmp_path / "train.py"
    filepath.write_text(
        "loader = DataLoader(dataset, num_workers=0)\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(["analyze", str(filepath), "--patch"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Code Analysis" in captured.out
    assert "Patch Suggestions:" in captured.out
    assert "--- " in captured.out
    assert "num_workers=4" in captured.out
    assert captured.err == ""


def test_cli_analyze_patch_human_output_includes_safety_language(
    tmp_path,
    capsys,
) -> None:
    filepath = tmp_path / "train.py"
    filepath.write_text(
        "loader = DataLoader(dataset, pin_memory=False)\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(["analyze", str(filepath), "--patch"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (
        "GPUBoost does not apply patches automatically. "
        "Review the diff before applying changes."
    ) in captured.out
    assert captured.err == ""


def test_cli_analyze_patch_no_safe_suggestions_prints_message(
    tmp_path,
    capsys,
) -> None:
    filepath = tmp_path / "eval.py"
    filepath.write_text(
        "torch.backends.cudnn.benchmark = True\n"
        "for batch in loader:\n"
        "    value = loss.item()\n",
        encoding="utf-8",
    )

    exit_code = cli_main.main(["analyze", str(filepath), "--patch"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Patch Suggestions:" in captured.out
    assert "No safe automatic patch suggestions were generated." in captured.out
    assert "Patch Warnings:" in captured.out
    assert captured.err == ""


def test_cli_analyze_missing_file_exits_nonzero(capsys) -> None:
    exit_code = cli_main.main(["analyze", "missing-file.py"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "GPUBoost Code Analysis" in captured.out
    assert "Status: error" in captured.out
    assert "Error:" in captured.out
    assert captured.err == ""


def test_cli_analyze_patch_missing_file_exits_nonzero(capsys) -> None:
    exit_code = cli_main.main(["analyze", "missing-file.py", "--patch"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "GPUBoost Code Analysis" in captured.out
    assert "Status: error" in captured.out
    assert "Error:" in captured.out
    assert "Patch Suggestions:" not in captured.out
    assert captured.err == ""


def test_cli_agent_command_without_subcommand_still_works(capsys) -> None:
    exit_code = cli_main.main(["agent"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Agent" in captured.out
    assert "Available commands: optimize" in captured.out
    assert captured.err == ""


def test_agent_status_to_exit_code_maps_known_statuses() -> None:
    assert cli_main.agent_status_to_exit_code("ok") == 0
    assert cli_main.agent_status_to_exit_code("partial") == 0
    assert cli_main.agent_status_to_exit_code("error") == 1
    assert cli_main.agent_status_to_exit_code("weird") == 1


def test_cli_agent_optimize_human_output_calls_workflow(
    monkeypatch,
    capsys,
) -> None:
    calls = []

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append({"script_path": script_path, "quick": quick})
        return _fake_agent_result_and_report(script_path=script_path, quick=quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [{"script_path": None, "quick": True}]
    assert "GPUBoost Agent" in captured.out
    assert "Command: optimize" in captured.out
    assert "Status: ok" in captured.out
    assert "Script: none" in captured.out
    assert "Summary:" in captured.out
    assert "Synthetic report summary." in captured.out
    assert "Plan:" in captured.out
    assert "- inspect_system: completed" in captured.out
    assert "Report:" in captured.out
    assert "Goal" in captured.out
    assert "- Script: none" in captured.out
    assert "Safety:" in captured.out
    assert (
        "GPUBoost does not apply patches automatically. "
        "Review generated diffs before applying changes."
    ) in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_passes_script_path(monkeypatch, capsys) -> None:
    calls = []

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append({"script_path": script_path, "quick": quick})
        return _fake_agent_result_and_report(script_path=script_path, quick=quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Script: train.py" in captured.out
    assert calls == [{"script_path": "train.py", "quick": True}]
    assert captured.err == ""


def test_cli_agent_optimize_passes_quick(monkeypatch, capsys) -> None:
    calls = []

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append({"script_path": script_path, "quick": quick})
        return _fake_agent_result_and_report(script_path=script_path, quick=quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "--quick"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [{"script_path": None, "quick": True}]
    assert captured.err == ""


def test_cli_agent_optimize_human_output_includes_report_sections(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Goal" in captured.out
    assert "- Kind: optimize_script" in captured.out
    assert "- Description: Synthetic optimize script goal." in captured.out
    assert "Results" in captured.out
    assert "- completed: 2" in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_human_output_includes_diff_when_present(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            diff=_FAKE_DIFF,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Reviewable Patch Diff:" in captured.out
    assert (
        "GPUBoost does not apply patches automatically. "
        "Review the diff before applying changes."
    ) in captured.out
    assert "--- train.py" in captured.out
    assert "+loader = DataLoader(dataset, num_workers=4)" in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_human_output_omits_diff_section_without_diff(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Reviewable Patch Diff:" not in captured.out
    assert "Safety:" in captured.out
    assert "Review generated diffs before applying changes." in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_partial_human_output_includes_diff_when_present(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            status="partial",
            diff=_FAKE_DIFF,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Status: partial" in captured.out
    assert "Reviewable Patch Diff:" in captured.out
    assert _FAKE_DIFF in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_error_human_output_without_diff_omits_diff_section(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            status="error",
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Status: error" in captured.out
    assert "Reviewable Patch Diff:" not in captured.out
    assert "Error:" in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_human_output_limits_recent_events(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Recent Events:" in captured.out
    assert "event 1" not in captured.out
    assert "event 2" in captured.out
    assert "event 6" in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_error_human_output_includes_error_section(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            status="error",
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Status: error" in captured.out
    assert "Error:" in captured.out
    assert "- synthetic failure" in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_partial_human_output_includes_partial_status(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            status="partial",
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Status: partial" in captured.out
    assert "Warnings:" in captured.out
    assert "- synthetic warning" in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_partial_human_output_surfaces_failed_action(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_missing_script_result_and_report(script_path, quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "missing.py"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Status: partial" in captured.out
    assert "- analyze_code: failed" in captured.out
    assert "Error:" in captured.out
    assert "Unable to read file: missing.py" in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_final_smoke_matrix(monkeypatch, capsys) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        diff = _FAKE_DIFF if script_path else None
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            diff=diff,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize"])
    human_no_script = capsys.readouterr()
    assert exit_code == 0
    assert "GPUBoost Agent" in human_no_script.out
    assert "Script: none" in human_no_script.out
    assert "Safety:" in human_no_script.out
    assert "Reviewable Patch Diff:" not in human_no_script.out
    assert human_no_script.err == ""

    exit_code = cli_main.main(["agent", "optimize", "train.py"])
    human_script = capsys.readouterr()
    assert exit_code == 0
    assert "Script: train.py" in human_script.out
    assert "Reviewable Patch Diff:" in human_script.out
    assert _FAKE_DIFF in human_script.out
    assert human_script.err == ""

    exit_code = cli_main.main(["agent", "optimize", "--json"])
    json_no_script = capsys.readouterr()
    no_script_data = json.loads(json_no_script.out)
    assert exit_code == 0
    assert no_script_data["schema_version"] == "agent.optimize.v1"
    assert no_script_data["result"]["goal"]["script_path"] is None
    assert no_script_data["artifacts"]["diff"] is None
    assert json_no_script.err == ""

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--json"])
    json_script = capsys.readouterr()
    script_data = json.loads(json_script.out)
    assert exit_code == 0
    assert script_data["schema_version"] == "agent.optimize.v1"
    assert script_data["result"]["goal"]["script_path"] == "train.py"
    assert script_data["artifacts"]["diff"] == _FAKE_DIFF
    assert "Reviewable Patch Diff:" not in json_script.out
    assert json_script.err == ""


def test_cli_agent_optimize_unexpected_exception_human_output_is_clean(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "GPUBoost Agent" in captured.out
    assert "Command: optimize" in captured.out
    assert "Status: error" in captured.out
    assert "Error:" in captured.out
    assert "boom" in captured.out
    assert "Traceback" not in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_json_outputs_result_and_report(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert sorted(data) == [
        "artifacts",
        "command",
        "report",
        "result",
        "schema_version",
    ]
    assert data["schema_version"] == "agent.optimize.v1"
    assert data["command"] == "agent optimize"
    assert data["artifacts"] == {"diff": None}
    assert data["result"]["status"] == "ok"
    assert data["report"]["status"] == "ok"
    assert data["report"]["summary"] == "Synthetic report summary."
    assert captured.err == ""


def test_cli_agent_optimize_json_does_not_print_human_safety_text(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert sorted(data) == [
        "artifacts",
        "command",
        "report",
        "result",
        "schema_version",
    ]
    assert data["schema_version"] == "agent.optimize.v1"
    assert data["command"] == "agent optimize"
    assert "Safety:" not in captured.out
    assert "GPUBoost does not apply patches automatically" not in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_json_includes_diff_artifact_when_present(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            diff=_FAKE_DIFF,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["artifacts"]["diff"] == _FAKE_DIFF
    assert data["result"]["artifacts"]["diff"] == _FAKE_DIFF
    assert "Reviewable Patch Diff:" not in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_json_includes_null_diff_without_artifact(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["artifacts"]["diff"] is None
    assert data["result"]["artifacts"]["diff"] is None
    assert captured.err == ""


def test_cli_agent_optimize_json_no_longer_returns_not_implemented(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert sorted(data) == [
        "artifacts",
        "command",
        "report",
        "result",
        "schema_version",
    ]
    assert "status" not in data
    assert data["result"]["status"] != "not_implemented"
    assert data["report"]["status"] != "not_implemented"
    assert captured.err == ""


def test_cli_agent_optimize_script_json_outputs_script_path(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["result"]["goal"]["script_path"] == "train.py"
    assert data["result"]["goal"]["options"]["quick"] is True
    assert "Script: train.py" in data["report"]["sections"][0]["items"]
    assert captured.err == ""


def test_cli_agent_optimize_quick_is_reflected_in_json(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--quick", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["result"]["goal"]["script_path"] == "train.py"
    assert data["result"]["goal"]["options"]["quick"] is True
    assert "Quick: true" in data["report"]["sections"][0]["items"]
    assert captured.err == ""


def test_cli_agent_optimize_exits_nonzero_when_result_status_is_error(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            status="error",
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert sorted(data) == [
        "artifacts",
        "command",
        "report",
        "result",
        "schema_version",
    ]
    assert data["schema_version"] == "agent.optimize.v1"
    assert data["command"] == "agent optimize"
    assert data["result"]["status"] == "error"
    assert data["report"]["status"] == "error"
    assert captured.err == ""


def test_cli_agent_optimize_unexpected_exception_json_output_is_valid(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        raise RuntimeError("workflow exploded")

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert sorted(data) == [
        "artifacts",
        "command",
        "error",
        "report",
        "result",
        "schema_version",
    ]
    assert data["schema_version"] == "agent.optimize.v1"
    assert data["command"] == "agent optimize"
    assert data["result"] is None
    assert data["report"] is None
    assert data["artifacts"] == {"diff": None}
    assert data["error"] == "workflow exploded"
    assert "GPUBoost Agent" not in captured.out
    assert "Safety:" not in captured.out
    assert "Traceback" not in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_json_without_script_path_sets_null_script(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["result"]["goal"]["script_path"] is None
    assert data["result"]["goal"]["options"]["quick"] is True
    assert "Script: none" in data["report"]["sections"][0]["items"]
    assert captured.err == ""


def test_cli_agent_optimize_partial_json_keeps_stable_shape(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            status="partial",
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert sorted(data) == [
        "artifacts",
        "command",
        "report",
        "result",
        "schema_version",
    ]
    assert data["schema_version"] == "agent.optimize.v1"
    assert data["command"] == "agent optimize"
    assert data["result"]["status"] == "partial"
    assert data["report"]["status"] == "partial"
    assert captured.err == ""


def test_build_agent_optimize_json_payload_returns_stable_dict() -> None:
    result, report = _fake_agent_result_and_report(script_path="train.py", quick=True)

    payload = cli_main.build_agent_optimize_json_payload(result, report)

    assert sorted(payload) == [
        "artifacts",
        "command",
        "report",
        "result",
        "schema_version",
    ]
    assert payload["schema_version"] == "agent.optimize.v1"
    assert payload["command"] == "agent optimize"
    assert payload["artifacts"] == {"diff": None}
    assert payload["result"]["goal"]["script_path"] == "train.py"
    assert payload["result"]["goal"]["options"]["quick"] is True
    assert payload["report"]["status"] == "ok"


def _fake_run_optimize_script_workflow(
    script_path: str | None = None,
    quick: bool = True,
) -> tuple[AgentRunResult, AgentReport]:
    return _fake_agent_result_and_report(script_path=script_path, quick=quick)


def _fake_missing_script_result_and_report(
    script_path: str | None,
    quick: bool,
) -> tuple[AgentRunResult, AgentReport]:
    script_display = script_path or "missing.py"
    goal = AgentGoal(
        id="optimize_script",
        kind="optimize_script",
        description=f"Optimize {script_display} for NVIDIA GPU performance",
        script_path=script_path,
        options={"quick": quick},
        constraints=["do_not_modify_original_file"],
    )
    plan = AgentPlan(
        id="plan_optimize_script",
        goal=goal,
        actions=[
            AgentAction(
                id="inspect_system",
                name="inspect_system",
                description="Collect system information.",
                required=True,
                status="completed",
            ),
            AgentAction(
                id="analyze_code",
                name="analyze_code",
                description="Analyze code.",
                required=False,
                status="failed",
                error=f"Unable to read file: {script_display}",
            ),
        ],
    )
    result = AgentRunResult(
        generated_at="2026-01-01T00:00:00+00:00",
        goal=goal,
        plan=plan,
        status="partial",
        warnings=["Skipped because dependency failed: analyze_code"],
        artifacts={"diff": None},
    )
    report = AgentReport(
        title="GPUBoost Agent Report",
        status="partial",
        summary="The agent workflow completed with some non-fatal failures.",
        sections=[
            AgentReportSection(
                title="Goal",
                items=[
                    "Kind: optimize_script",
                    f"Script: {script_display}",
                ],
            ),
            AgentReportSection(
                title="Errors",
                items=[f"analyze_code: Unable to read file: {script_display}"],
            ),
        ],
        warnings=["Skipped because dependency failed: analyze_code"],
    )
    return result, report


def _fake_agent_result_and_report(
    script_path: str | None,
    quick: bool,
    status: str = "ok",
    diff: str | None = None,
) -> tuple[AgentRunResult, AgentReport]:
    goal = AgentGoal(
        id="optimize_script",
        kind="optimize_script",
        description="Synthetic optimize script goal.",
        script_path=script_path,
        options={"quick": quick},
        constraints=["do_not_modify_original_file"],
    )
    plan = AgentPlan(
        id="plan_optimize_script",
        goal=goal,
        actions=[
            AgentAction(
                id="inspect_system",
                name="inspect_system",
                description="Collect system information.",
                required=True,
                status="completed" if status != "error" else "failed",
                error="synthetic failure" if status == "error" else None,
            ),
            AgentAction(
                id="run_quick_benchmark",
                name="run_quick_benchmark",
                description="Run quick benchmark.",
                required=True,
                depends_on=["inspect_system"],
                status="completed" if status != "error" else "skipped",
                error="synthetic failure" if status == "error" else None,
            ),
        ],
    )
    result = AgentRunResult(
        generated_at="2026-01-01T00:00:00+00:00",
        goal=goal,
        plan=plan,
        status=status,
        events=[],
        warnings=["synthetic warning"] if status == "partial" else [],
        error="synthetic failure" if status == "error" else None,
        artifacts={"diff": diff},
    )
    script_display = script_path if script_path is not None else "none"
    report = AgentReport(
        title="GPUBoost Agent Report",
        status=status,
        summary="Synthetic report summary.",
        sections=[
            AgentReportSection(
                title="Goal",
                items=[
                    "Kind: optimize_script",
                    "Description: Synthetic optimize script goal.",
                    f"Script: {script_display}",
                    f"Quick: {str(quick).lower()}",
                ],
            ),
            AgentReportSection(
                title="Results",
                items=["completed: 2"],
            ),
            AgentReportSection(
                title="Warnings",
                items=["synthetic warning"] if status == "partial" else [],
            ),
            AgentReportSection(
                title="Errors",
                items=["inspect_system: synthetic failure"]
                if status == "error"
                else [],
            ),
            AgentReportSection(
                title="Events",
                items=[
                    "event 1",
                    "event 2",
                    "event 3",
                    "event 4",
                    "event 5",
                    "event 6",
                ],
            ),
            AgentReportSection(
                title="Plan",
                items=[
                    "inspect_system: completed",
                    "run_quick_benchmark: completed",
                ],
            ),
        ],
        warnings=["synthetic warning"] if status == "partial" else [],
        error="synthetic failure" if status == "error" else None,
    )
    return result, report


def _fake_run_quick_benchmark(device_index: int = 0) -> BenchmarkSuiteResult:
    return BenchmarkSuiteResult(
        generated_at="2026-01-01T00:00:00+00:00",
        gpu_name="NVIDIA Test GPU",
        cuda_available=True,
        device_index=device_index,
        results=[],
        warnings=[],
    )


def _fake_advisor_result(suite: BenchmarkSuiteResult) -> AdvisorResult:
    return AdvisorResult(
        generated_at="2026-01-01T00:00:01+00:00",
        recommendations=[
            Recommendation(
                id="mixed_precision_enable",
                title="Enable mixed precision",
                category="mixed_precision",
                priority=1,
                impact="medium",
                confidence="high",
                effort="low",
                estimated_speedup=1.26,
                summary="AMP improved synthetic training throughput by 1.26x.",
                rationale="Test rationale.",
                suggested_action="Use AMP.",
                code_snippet=None,
            ),
        ],
        warnings=[],
    )
