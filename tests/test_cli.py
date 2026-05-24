"""Tests for the Phase 1 CLI."""

import json

import pytest

from gpuboost.agent.report import AgentReport, AgentReportSection
from gpuboost.cli import main as cli_main
from gpuboost.history.store import insert_history_run
from gpuboost.schemas.agent import AgentAction, AgentGoal, AgentPlan, AgentRunResult
from gpuboost.schemas.benchmark_result import BenchmarkSuiteResult
from gpuboost.schemas.gpu_profile import (
    GPUBoostProfile,
    SystemProfile,
    TorchEnvironmentProfile,
)
from gpuboost.schemas.history import HistoryRunRecord
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


def test_cli_analyze_patch_human_output_includes_patch_section(
    tmp_path,
    capsys,
) -> None:
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
        model: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append({"script_path": script_path, "quick": quick, "model": model})
        return _fake_agent_result_and_report(script_path=script_path, quick=quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [{"script_path": None, "quick": True, "model": False}]
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
        model: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append({"script_path": script_path, "quick": quick, "model": model})
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
    assert calls == [{"script_path": "train.py", "quick": True, "model": False}]
    assert captured.err == ""


def test_cli_agent_optimize_passes_quick(monkeypatch, capsys) -> None:
    calls = []

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append({"script_path": script_path, "quick": quick, "model": model})
        return _fake_agent_result_and_report(script_path=script_path, quick=quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "--quick"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [{"script_path": None, "quick": True, "model": False}]
    assert captured.err == ""


def test_cli_agent_optimize_model_flag_passes_model_true(monkeypatch, capsys) -> None:
    calls = []

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append({"script_path": script_path, "quick": quick, "model": model})
        return _fake_agent_result_and_report(script_path=script_path, quick=quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--model"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [{"script_path": "train.py", "quick": True, "model": True}]
    assert captured.err == ""


def test_cli_agent_optimize_model_artifact_auto_enables_model(
    monkeypatch,
    capsys,
) -> None:
    calls = []

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        model_artifact_path: str | None = None,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append(
            {
                "script_path": script_path,
                "quick": quick,
                "model": model,
                "model_artifact_path": model_artifact_path,
            }
        )
        return _fake_agent_result_and_report(script_path=script_path, quick=quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(
        ["agent", "optimize", "train.py", "--model-artifact", "artifact/manifest.json"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [
        {
            "script_path": "train.py",
            "quick": True,
            "model": True,
            "model_artifact_path": "artifact/manifest.json",
        }
    ]
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
        model: bool = False,
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
    assert "Comparison:" not in captured.out
    assert "Safety:" in captured.out
    assert "Review generated diffs before applying changes." in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_human_output_shows_model_when_present(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            model_artifact=_fake_model_artifact(),
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--model"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Model:" in captured.out
    assert "- Status: fallback" in captured.out
    assert "- Available: no" in captured.out
    assert "- Fallback used: yes" in captured.out
    assert "- Model: none" in captured.out
    assert "- Patch application allowed: no" in captured.out
    assert "model prediction is advisory only" in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_human_output_omits_model_without_artifact(
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
    assert "Model:" not in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_human_output_includes_comparison_when_present(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            comparison_artifact={
                "status": "ok",
                "overall_verdict": "improved",
            },
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Comparison:" in captured.out
    assert "- Status: ok" in captured.out
    assert "- Overall verdict: improved" in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_partial_human_output_includes_diff_when_present(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
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
        model: bool = False,
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
        model: bool = False,
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
        model: bool = False,
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
        model: bool = False,
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
        model: bool = False,
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
    assert no_script_data["artifacts"]["comparison"] is None
    assert json_no_script.err == ""

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--json"])
    json_script = capsys.readouterr()
    script_data = json.loads(json_script.out)
    assert exit_code == 0
    assert script_data["schema_version"] == "agent.optimize.v1"
    assert script_data["result"]["goal"]["script_path"] == "train.py"
    assert script_data["artifacts"]["diff"] is None
    assert script_data["artifacts"]["diff_redacted"] is True
    assert script_data["result"]["artifacts"]["diff"] is None
    assert script_data["result"]["artifacts"]["diff_redacted"] is True
    assert script_data["artifacts"]["comparison"] is None
    assert _FAKE_DIFF not in json_script.out
    assert "Reviewable Patch Diff:" not in json_script.out
    assert json_script.err == ""


def test_cli_agent_optimize_unexpected_exception_human_output_is_clean(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
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
    assert data["artifacts"] == {
        "comparison": None,
        "diff": None,
        "diff_redacted": False,
        "history_run_id": None,
        "model": None,
        "raw_artifacts_included": False,
        "trial": None,
    }
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


def test_cli_agent_optimize_json_redacts_diff_artifact_when_present(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
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
    assert data["artifacts"]["diff"] is None
    assert data["artifacts"]["diff_redacted"] is True
    assert data["result"]["artifacts"]["diff"] is None
    assert data["result"]["artifacts"]["diff_redacted"] is True
    assert _FAKE_DIFF not in captured.out
    assert "Reviewable Patch Diff:" not in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_json_raw_artifacts_opt_in_includes_diff(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
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

    exit_code = cli_main.main(
        ["agent", "optimize", "train.py", "--json", "--include-raw-artifacts"]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["artifacts"]["diff"] == _FAKE_DIFF
    assert data["artifacts"]["diff_redacted"] is False
    assert data["result"]["artifacts"]["diff"] == _FAKE_DIFF


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
    assert data["artifacts"]["diff_redacted"] is False
    assert captured.err == ""


def test_cli_agent_optimize_json_without_model_includes_null_model(
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
    assert data["artifacts"]["model"] is None
    assert captured.err == ""


def test_cli_agent_optimize_json_with_model_includes_model_artifact(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        assert model is True
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            model_artifact=_fake_model_artifact(),
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--model", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["artifacts"]["model"]["status"] == "fallback"
    assert data["artifacts"]["model"]["patch_application_allowed"] is False
    assert data["artifacts"]["diff"] is None
    assert data["artifacts"]["trial"] is None
    assert data["artifacts"]["comparison"] is None
    assert data["artifacts"]["history_run_id"] is None
    assert "Model:" not in captured.out
    assert captured.err == ""


def test_cli_agent_optimize_json_with_trained_model_has_stable_advisory_shape(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        model_artifact_path: str | None = None,
    ) -> tuple[AgentRunResult, AgentReport]:
        assert model is True
        assert model_artifact_path == "artifact/manifest.json"
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            model_artifact=_fake_trained_model_artifact(),
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(
        [
            "agent",
            "optimize",
            "train.py",
            "--model",
            "--model-artifact",
            "artifact/manifest.json",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    model = data["artifacts"]["model"]
    assert exit_code == 0
    assert model["status"] == "ok"
    assert model["provider"] == "trained_local_model"
    assert model["prediction"] == {"label": "improved", "confidence": 0.91}
    assert model["probabilities"] == {"improved": 0.91, "regressed": 0.09}
    assert model["patch_application_allowed"] is False
    assert "raw diff" not in captured.out
    assert "stdout" not in captured.out
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
        model: bool = False,
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
        model: bool = False,
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
    assert data["artifacts"] == {
        "comparison": None,
        "diff": None,
        "history_run_id": None,
        "model": None,
        "trial": None,
    }
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
    assert data["artifacts"]["comparison"] is None
    assert captured.err == ""


def test_cli_agent_optimize_partial_json_keeps_stable_shape(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
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
    assert payload["artifacts"] == {
        "comparison": None,
        "diff": None,
        "diff_redacted": False,
        "history_run_id": None,
        "model": None,
        "raw_artifacts_included": False,
        "trial": None,
    }
    assert payload["result"]["goal"]["script_path"] == "train.py"
    assert payload["result"]["goal"]["options"]["quick"] is True
    assert payload["report"]["status"] == "ok"


def test_cli_agent_optimize_script_json_includes_null_comparison_artifact(
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
    assert data["artifacts"]["comparison"] is None
    assert data["result"]["goal"]["script_path"] == "train.py"
    assert captured.err == ""


def test_cli_trial_passes_trial_true_to_workflow(monkeypatch, capsys) -> None:
    calls = []

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        trial: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append(
            {"script_path": script_path, "quick": quick, "model": model, "trial": trial}
        )
        return _fake_agent_result_and_report(script_path=script_path, quick=quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--trial"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [
        {"script_path": "train.py", "quick": True, "model": False, "trial": True}
    ]
    assert captured.err == ""


def test_cli_model_works_with_trial_argument_parsing(monkeypatch, capsys) -> None:
    calls = []

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        trial: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append({"script_path": script_path, "model": model, "trial": trial})
        return _fake_agent_result_and_report(script_path=script_path, quick=quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--model", "--trial"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [{"script_path": "train.py", "model": True, "trial": True}]
    assert captured.err == ""


def test_cli_trial_without_script_path_fails_cleanly(capsys) -> None:
    exit_code = cli_main.main(["agent", "optimize", "--trial"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--trial requires a script_path." in captured.out
    assert captured.err == ""


def test_cli_trial_human_output_includes_trial_section(monkeypatch, capsys) -> None:
    trial_artifact = _fake_trial_artifact(status="passed")

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        trial: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            trial_artifact=trial_artifact,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--trial"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Trial Workspace:" in captured.out
    assert "Status: passed" in captured.out
    assert (
        "Trial mode applies patches only to a temporary copy. "
        "The original file is not modified."
    ) in captured.out


def test_cli_trial_json_includes_artifacts_trial(monkeypatch, capsys) -> None:
    trial_artifact = _fake_trial_artifact(status="passed")

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        trial: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            trial_artifact=trial_artifact,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--trial", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["artifacts"]["trial"]["status"] == "passed"
    assert data["artifacts"]["trial"]["patch_applied"] is True
    assert data["artifacts"]["trial"]["steps"][0]["stdout_redacted"] is False
    assert data["artifacts"]["trial"]["steps"][0]["stderr_redacted"] is False


def test_cli_no_trial_json_includes_null_trial(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        _fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["artifacts"]["trial"] is None


def test_cli_trial_failure_result_still_valid_json(monkeypatch, capsys) -> None:
    trial_artifact = _fake_trial_artifact(status="failed", error="trial failed")

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        trial: bool = False,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            status="partial",
            trial_artifact=trial_artifact,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "train.py", "--trial", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["result"]["status"] == "partial"
    assert data["artifacts"]["trial"]["error"] == "trial failed"


def test_cli_test_without_trial_fails(capsys) -> None:
    exit_code = cli_main.main(["agent", "optimize", "train.py", "--test", "echo ok"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--test requires --trial." in captured.out


def test_cli_test_with_trial_passes_test_command(monkeypatch, capsys) -> None:
    calls = []

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        trial: bool = False,
        test_command: str | None = None,
    ) -> tuple[AgentRunResult, AgentReport]:
        calls.append(
            {
                "script_path": script_path,
                "trial": trial,
                "test_command": test_command,
            }
        )
        return _fake_agent_result_and_report(script_path=script_path, quick=quick)

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(
        ["agent", "optimize", "train.py", "--trial", "--test", "python -c pass"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [
        {
            "script_path": "train.py",
            "trial": True,
            "test_command": "python -c pass",
        }
    ]
    assert captured.err == ""


def test_cli_test_with_trial_json_includes_test_command(monkeypatch, capsys) -> None:
    trial_artifact = _fake_trial_artifact(
        test_command="python -c pass",
        test_status="passed",
    )

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        trial: bool = False,
        test_command: str | None = None,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            trial_artifact=trial_artifact,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(
        [
            "agent",
            "optimize",
            "train.py",
            "--trial",
            "--test",
            "python -c pass",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["artifacts"]["trial"]["test_command"] == "python -c pass"
    assert data["artifacts"]["trial"]["test_status"] == "passed"


def test_cli_trial_json_redacts_stdout_and_stderr_by_default(
    monkeypatch,
    capsys,
) -> None:
    trial_artifact = _fake_trial_artifact(
        test_command="python -c pass",
        test_status="passed",
        stdout="secret stdout",
        stderr="secret stderr",
    )

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        trial: bool = False,
        test_command: str | None = None,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            trial_artifact=trial_artifact,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(
        [
            "agent",
            "optimize",
            "train.py",
            "--trial",
            "--test",
            "python -c pass",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    step = data["artifacts"]["trial"]["steps"][0]

    assert exit_code == 0
    assert "secret stdout" not in captured.out
    assert "secret stderr" not in captured.out
    assert step["stdout"] is None
    assert step["stderr"] is None
    assert step["stdout_redacted"] is True
    assert step["stderr_redacted"] is True


def test_cli_trial_json_raw_artifacts_opt_in_includes_stdout_and_stderr(
    monkeypatch,
    capsys,
) -> None:
    trial_artifact = _fake_trial_artifact(
        test_command="python -c pass",
        test_status="passed",
        stdout="raw stdout",
        stderr="raw stderr",
    )

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        trial: bool = False,
        test_command: str | None = None,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            trial_artifact=trial_artifact,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(
        [
            "agent",
            "optimize",
            "train.py",
            "--trial",
            "--test",
            "python -c pass",
            "--json",
            "--include-raw-artifacts",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["artifacts"]["trial"]["steps"][0]["stdout"] == "raw stdout"
    assert data["artifacts"]["trial"]["steps"][0]["stderr"] == "raw stderr"


def test_cli_test_human_output_displays_test_command(monkeypatch, capsys) -> None:
    trial_artifact = _fake_trial_artifact(
        test_command="python -c pass",
        test_status="passed",
    )

    def fake_run_optimize_script_workflow(
        script_path: str | None = None,
        quick: bool = True,
        model: bool = False,
        trial: bool = False,
        test_command: str | None = None,
    ) -> tuple[AgentRunResult, AgentReport]:
        return _fake_agent_result_and_report(
            script_path=script_path,
            quick=quick,
            trial_artifact=trial_artifact,
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(
        ["agent", "optimize", "train.py", "--trial", "--test", "python -c pass"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "- Test command: python -c pass" in captured.out
    assert "- Test status: passed" in captured.out


def test_cli_test_json_error_output_contains_no_human_text(capsys) -> None:
    exit_code = cli_main.main(
        ["agent", "optimize", "train.py", "--test", "echo ok", "--json"]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["error"] == "--test requires --trial."
    assert "GPUBoost Agent" not in captured.out


def test_cli_compare_human_output_includes_title_and_paths(tmp_path, capsys) -> None:
    baseline, optimized = _write_compare_files(tmp_path)

    exit_code = cli_main.main(["compare", str(baseline), str(optimized)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Comparison" in captured.out
    assert f"Baseline: {baseline}" in captured.out
    assert f"Optimized: {optimized}" in captured.out
    assert captured.err == ""


def test_cli_compare_human_output_includes_overall_verdict(tmp_path, capsys) -> None:
    baseline, optimized = _write_compare_files(tmp_path)

    exit_code = cli_main.main(["compare", str(baseline), str(optimized)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Overall verdict: improved" in captured.out


def test_cli_compare_human_output_includes_metric_delta_line(tmp_path, capsys) -> None:
    baseline, optimized = _write_compare_files(tmp_path)

    exit_code = cli_main.main(["compare", str(baseline), str(optimized)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "- best_fp16_tflops: 30.0 -> 33.0 TFLOPS (+10.00%) [improved]" in (
        captured.out
    )


def test_cli_compare_json_outputs_valid_json(tmp_path, capsys) -> None:
    baseline, optimized = _write_compare_files(tmp_path)

    exit_code = cli_main.main(["compare", str(baseline), str(optimized), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["schema_version"] == "comparison.v1"
    assert data["command"] == "compare"
    assert data["comparison"]["status"] == "ok"
    assert captured.err == ""


def test_cli_compare_json_includes_comparison(tmp_path, capsys) -> None:
    baseline, optimized = _write_compare_files(tmp_path)

    exit_code = cli_main.main(["compare", str(baseline), str(optimized), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["comparison"]["baseline_label"] == str(baseline)
    assert data["comparison"]["optimized_label"] == str(optimized)
    metric_names = [
        metric["name"]
        for section in data["comparison"]["sections"]
        for metric in section["metrics"]
    ]
    assert "best_fp16_tflops" in metric_names


def test_cli_compare_missing_file_human_exits_nonzero_no_traceback(
    tmp_path,
    capsys,
) -> None:
    baseline = tmp_path / "missing.json"
    optimized = tmp_path / "optimized.json"
    optimized.write_text(json.dumps(_compare_benchmark(33.0)), encoding="utf-8")

    exit_code = cli_main.main(["compare", str(baseline), str(optimized)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "GPUBoost Comparison" in captured.out
    assert "File not found:" in captured.out
    assert "Traceback" not in captured.out
    assert captured.err == ""


def test_cli_compare_missing_file_json_exits_nonzero_valid_json(
    tmp_path,
    capsys,
) -> None:
    baseline = tmp_path / "missing.json"
    optimized = tmp_path / "optimized.json"
    optimized.write_text(json.dumps(_compare_benchmark(33.0)), encoding="utf-8")

    exit_code = cli_main.main(["compare", str(baseline), str(optimized), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["schema_version"] == "comparison.v1"
    assert data["command"] == "compare"
    assert data["comparison"] is None
    assert "File not found:" in data["error"]
    assert "GPUBoost Comparison" not in captured.out
    assert "Traceback" not in captured.out
    assert captured.err == ""


def test_cli_compare_invalid_json_human_exits_nonzero_no_traceback(
    tmp_path,
    capsys,
) -> None:
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    baseline.write_text("{not json", encoding="utf-8")
    optimized.write_text(json.dumps(_compare_benchmark(33.0)), encoding="utf-8")

    exit_code = cli_main.main(["compare", str(baseline), str(optimized)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Invalid JSON:" in captured.out
    assert "Traceback" not in captured.out
    assert captured.err == ""


def test_cli_compare_invalid_json_json_exits_nonzero_valid_json(
    tmp_path,
    capsys,
) -> None:
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    baseline.write_text("{not json", encoding="utf-8")
    optimized.write_text(json.dumps(_compare_benchmark(33.0)), encoding="utf-8")

    exit_code = cli_main.main(["compare", str(baseline), str(optimized), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["comparison"] is None
    assert "Invalid JSON:" in data["error"]
    assert "GPUBoost Comparison" not in captured.out
    assert "Traceback" not in captured.out
    assert captured.err == ""


def test_cli_compare_status_error_exits_nonzero(tmp_path, capsys) -> None:
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    baseline.write_text(json.dumps({"results": []}), encoding="utf-8")
    optimized.write_text(json.dumps({"results": []}), encoding="utf-8")

    exit_code = cli_main.main(["compare", str(baseline), str(optimized)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Status: error" in captured.out
    assert "No comparable metrics were found." in captured.out


def test_cli_compare_partial_status_exits_zero(tmp_path, capsys) -> None:
    baseline, optimized = _write_compare_files(tmp_path, include_partial_only=True)

    exit_code = cli_main.main(["compare", str(baseline), str(optimized)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Status: partial" in captured.out
    assert "Warnings:" in captured.out


def test_cli_compare_json_mode_contains_no_human_text(tmp_path, capsys) -> None:
    baseline, optimized = _write_compare_files(tmp_path)

    exit_code = cli_main.main(["compare", str(baseline), str(optimized), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Comparison" not in captured.out
    assert "Overall verdict:" not in captured.out
    json.loads(captured.out)


def test_cli_dataset_collect_outcomes_human_output(tmp_path, capsys) -> None:
    pairs_path = _write_outcome_pairs_files(tmp_path)
    output_dir = tmp_path / "outcomes"

    exit_code = cli_main.main(
        [
            "dataset",
            "collect-outcomes",
            str(pairs_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Outcome Collection" in captured.out
    assert "Pairs: 1" in captured.out
    assert "Collected rows: 1" in captured.out
    assert "Validation: passed" in captured.out
    assert "- improved: 1" in captured.out
    assert "outcome_dataset_jsonl" in captured.out
    assert captured.err == ""


def test_cli_dataset_collect_outcomes_json_output(tmp_path, capsys) -> None:
    pairs_path = _write_outcome_pairs_files(tmp_path)
    output_dir = tmp_path / "outcomes"

    exit_code = cli_main.main(
        [
            "dataset",
            "collect-outcomes",
            str(pairs_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["schema_version"] == "dataset.outcome_collection.v1"
    assert data["command"] == "dataset collect-outcomes"
    assert data["result"]["collected_row_count"] == 1
    assert data["result"]["label_counts"] == {"improved": 1}
    assert "GPUBoost Outcome Collection" not in captured.out
    assert captured.err == ""


def test_cli_dataset_collect_outcomes_missing_file_error_json(capsys) -> None:
    exit_code = cli_main.main(
        [
            "dataset",
            "collect-outcomes",
            "missing-pairs.json",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["schema_version"] == "dataset.outcome_collection.v1"
    assert data["result"] is None
    assert "missing-pairs.json" in data["error"]
    assert "GPUBoost Outcome Collection" not in captured.out
    assert captured.err == ""


def test_cli_model_evaluate_baselines_human_output(tmp_path, capsys) -> None:
    dataset_path = _write_training_dataset_jsonl(tmp_path)
    output_dir = tmp_path / "model_training"

    exit_code = cli_main.main(
        [
            "model",
            "evaluate-baselines",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Baseline Model Evaluation" in captured.out
    assert "Rows:" in captured.out
    assert "Labels:" in captured.out
    assert "Eval split: validation" in captured.out
    assert "Best model:" in captured.out
    assert "Model scores:" in captured.out
    assert (output_dir / "baseline_comparison_report.json").exists()
    assert (output_dir / "baseline_comparison_report.md").exists()
    assert not list(output_dir.glob("*.pt"))
    assert not list(output_dir.glob("*.pkl"))
    assert not list(output_dir.glob("*.safetensors"))
    assert captured.err == ""


def test_cli_model_evaluate_baselines_json_output(tmp_path, capsys) -> None:
    dataset_path = _write_training_dataset_jsonl(tmp_path)
    output_dir = tmp_path / "model_training"

    exit_code = cli_main.main(
        [
            "model",
            "evaluate-baselines",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["schema_version"] == "training.baseline_comparison.v1"
    assert data["command"] == "model evaluate-baselines"
    assert data["result"]["status"] == "ok"
    assert data["result"]["output_files"]["json_report"].endswith(
        "baseline_comparison_report.json"
    )
    assert "GPUBoost Baseline Model Evaluation" not in captured.out
    assert captured.err == ""


def test_cli_model_evaluate_baselines_missing_dataset_error(capsys) -> None:
    exit_code = cli_main.main(
        [
            "model",
            "evaluate-baselines",
            "--dataset",
            "missing-training-dataset.jsonl",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["schema_version"] == "training.baseline_comparison.v1"
    assert data["result"] is None
    assert "missing-training-dataset.jsonl" in data["error"]
    assert captured.err == ""


def test_cli_model_train_neural_human_output(tmp_path, capsys) -> None:
    from gpuboost.model.neural import torch_available

    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    dataset_path = _write_training_dataset_jsonl(tmp_path)
    output_dir = tmp_path / "model_training"
    artifact_dir = tmp_path / "artifacts"

    exit_code = cli_main.main(
        [
            "model",
            "train-neural",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--max-epochs",
            "3",
            "--hidden-sizes",
            "8",
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Neural Model Training" in captured.out
    assert "Status:" in captured.out
    assert "Dataset rows:" in captured.out
    assert "Classes:" in captured.out
    assert "Best validation macro F1:" in captured.out
    assert "Best baseline macro F1:" in captured.out
    assert "Target met:" in captured.out
    assert (output_dir / "neural_training_report.json").exists()
    assert (output_dir / "neural_training_report.md").exists()
    assert not artifact_dir.exists()
    assert not list(output_dir.glob("*.pt"))
    assert not list(output_dir.glob("*.pth"))
    assert not list(output_dir.glob("*.pkl"))
    assert not list(output_dir.glob("*.safetensors"))
    assert captured.err == ""


def test_cli_model_train_neural_json_output(tmp_path, capsys) -> None:
    from gpuboost.model.neural import torch_available

    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    dataset_path = _write_training_dataset_jsonl(tmp_path)
    output_dir = tmp_path / "model_training"

    exit_code = cli_main.main(
        [
            "model",
            "train-neural",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--max-epochs",
            "3",
            "--hidden-sizes",
            "8",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["schema_version"] == "training.neural_search_result.v1"
    assert data["command"] == "model train-neural"
    assert data["result"]["metadata"]["candidate_count"] == 1
    assert data["result"]["output_files"]["json_report"].endswith(
        "neural_training_report.json"
    )
    assert "baseline_comparison" in data["result"]
    assert "GPUBoost Neural Model Training" not in captured.out
    assert captured.err == ""


def test_cli_model_train_neural_missing_dataset_error(capsys) -> None:
    exit_code = cli_main.main(
        [
            "model",
            "train-neural",
            "--dataset",
            "missing-training-dataset.jsonl",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["schema_version"] == "training.neural_search_result.v1"
    assert data["result"] is None
    assert "missing-training-dataset.jsonl" in data["error"]
    assert captured.err == ""


def test_cli_model_train_neural_save_artifact_and_validate_predict(
    tmp_path,
    capsys,
) -> None:
    from gpuboost.model.neural import torch_available

    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    dataset_path = _write_training_dataset_jsonl(tmp_path)
    output_dir = tmp_path / "model_training"
    artifact_dir = tmp_path / "artifacts"

    exit_code = cli_main.main(
        [
            "model",
            "train-neural",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--artifact-name",
            "fixture",
            "--max-epochs",
            "3",
            "--hidden-sizes",
            "8",
            "--save-artifact",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    manifest_path = data["result"]["artifact_manifest"]

    assert exit_code == 0
    assert manifest_path.endswith("manifest.json")
    assert data["result"]["artifact_manifest_path"] == manifest_path
    assert data["result"]["artifact_validation_status"] == "ok"
    assert data["result"]["patch_application_allowed"] is False
    assert "state_dict" not in captured.out
    assert (artifact_dir / "fixture" / "manifest.json").exists()

    validate_exit = cli_main.main(
        ["model", "validate-artifact", manifest_path, "--json"]
    )
    validate_output = json.loads(capsys.readouterr().out)

    assert validate_exit == 0
    assert validate_output["result"]["status"] == "ok"

    predict_exit = cli_main.main(
        [
            "model",
            "predict-artifact",
            manifest_path,
            "--features-json",
            '{"features.safe_signal": 1.0}',
            "--json",
        ]
    )
    predict_output = json.loads(capsys.readouterr().out)

    assert predict_exit == 0
    assert predict_output["result"]["status"] == "ok"
    assert predict_output["result"]["predictions"][0]["label"]
    assert predict_output["result"]["predictions"][0]["confidence"] is not None


def test_cli_model_train_neural_save_artifact_human_output_includes_next_steps(
    tmp_path,
    capsys,
) -> None:
    from gpuboost.model.neural import torch_available

    if not torch_available():
        pytest.skip("PyTorch is unavailable.")
    dataset_path = _write_training_dataset_jsonl(tmp_path)
    output_dir = tmp_path / "model_training"
    artifact_dir = tmp_path / "artifacts"

    exit_code = cli_main.main(
        [
            "model",
            "train-neural",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--artifact-name",
            "fixture-human",
            "--max-epochs",
            "3",
            "--hidden-sizes",
            "8",
            "--save-artifact",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Artifact:" in captured.out
    assert "fixture-human" in captured.out
    assert "python -m gpuboost model validate-artifact" in captured.out
    assert "python -m gpuboost agent optimize <script> --model-artifact" in captured.out
    assert "- Patch application allowed: no" in captured.out


def test_cli_model_list_artifacts_human_output(tmp_path, capsys) -> None:
    _write_cli_manifest_fixture(tmp_path / "fixture")

    exit_code = cli_main.main(
        ["model", "list-artifacts", "--artifacts-dir", str(tmp_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Model Artifacts" in captured.out
    assert "Found: 1" in captured.out
    assert "validation macro F1: 0.8000" in captured.out
    assert "validation status: ok" in captured.out
    assert "state_dict" not in captured.out


def test_cli_model_list_artifacts_json_output(tmp_path, capsys) -> None:
    _write_cli_manifest_fixture(tmp_path / "fixture")

    exit_code = cli_main.main(
        ["model", "list-artifacts", "--artifacts-dir", str(tmp_path), "--json"]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["schema_version"] == "training.model_artifacts.list.v1"
    assert data["command"] == "model list-artifacts"
    assert data["result"]["artifact_count"] == 1
    assert data["result"]["artifacts"][0]["validation_status"] == "ok"
    assert "state_dict" not in captured.out
    assert "model.pt" not in captured.out


def test_cli_model_show_artifact_human_output(tmp_path, capsys) -> None:
    manifest_path = _write_cli_manifest_fixture(tmp_path / "fixture")

    exit_code = cli_main.main(["model", "show-artifact", str(manifest_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost Model Artifact" in captured.out
    assert "Status: ok" in captured.out
    assert "Labels: improved, regressed" in captured.out
    assert "Feature count: 1" in captured.out


def test_cli_model_show_artifact_json_output(tmp_path, capsys) -> None:
    manifest_path = _write_cli_manifest_fixture(tmp_path / "fixture")

    exit_code = cli_main.main(
        ["model", "show-artifact", str(manifest_path), "--json"]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["schema_version"] == "training.model_artifact.show.v1"
    assert data["result"]["model_name"] == "mlp_classifier"
    assert data["result"]["validation_status"] == "ok"
    assert str(tmp_path) not in captured.out


def test_cli_model_check_artifact_pass_and_fail(tmp_path, capsys) -> None:
    manifest_path = _write_cli_manifest_fixture(tmp_path / "fixture")

    passed = cli_main.main(
        [
            "model",
            "check-artifact",
            str(manifest_path),
            "--min-test-macro-f1",
            "0.75",
            "--require-beats-baseline",
            "--json",
        ]
    )
    pass_output = json.loads(capsys.readouterr().out)

    failed = cli_main.main(
        [
            "model",
            "check-artifact",
            str(manifest_path),
            "--min-test-macro-f1",
            "0.90",
            "--json",
        ]
    )
    fail_output = json.loads(capsys.readouterr().out)

    assert passed == 0
    assert pass_output["schema_version"] == "training.model_artifact.check.v1"
    assert pass_output["result"]["status"] == "passed"
    assert failed == 1
    assert fail_output["result"]["status"] == "failed"


def test_cli_model_validate_artifact_missing_manifest_json(capsys) -> None:
    exit_code = cli_main.main(
        [
            "model",
            "validate-artifact",
            "missing-manifest.json",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["result"]["status"] == "error"
    assert "missing-manifest.json" in data["result"]["errors"][0]


def test_cli_model_predict_artifact_missing_manifest_json(capsys) -> None:
    exit_code = cli_main.main(
        [
            "model",
            "predict-artifact",
            "missing-manifest.json",
            "--features-json",
            "{}",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["result"]["status"] == "error"
    assert "missing-manifest.json" in data["result"]["error"]


def test_cli_history_list_human_output_with_no_runs(tmp_path, capsys) -> None:
    db_path = tmp_path / "history.db"

    exit_code = cli_main.main(["history", "list", "--db-path", str(db_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost History" in captured.out
    assert "Total runs: 0" in captured.out
    assert "No history runs found." in captured.out


def test_cli_history_list_human_output_with_runs(tmp_path, capsys) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_history_record("older", created_at="2026-01-01T00:00:00+00:00"), db_path=db_path)
    insert_history_run(_history_record("newer", created_at="2026-01-02T00:00:00+00:00"), db_path=db_path)

    exit_code = cli_main.main(["history", "list", "--db-path", str(db_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.index("newer") < captured.out.index("older")
    assert "script=train.py" in captured.out
    assert "trial=passed" in captured.out


def test_cli_history_list_json_valid_json_and_schema(tmp_path, capsys) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_history_record("run-001"), db_path=db_path)

    exit_code = cli_main.main(["history", "list", "--db-path", str(db_path), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["schema_version"] == "history.list.v1"
    assert data["command"] == "history list"
    assert data["history"]["runs"][0]["run_id"] == "run-001"


def test_cli_history_show_human_output_for_existing_run(tmp_path, capsys) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_history_record("run-001"), db_path=db_path)

    exit_code = cli_main.main(["history", "show", "run-001", "--db-path", str(db_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost History Run" in captured.out
    assert "Run ID: run-001" in captured.out
    assert "Script SHA256: abc123" in captured.out
    assert "- inspect_system: completed" in captured.out
    assert "- trial: status=passed" in captured.out


def test_cli_history_show_json_valid_json_and_schema(tmp_path, capsys) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_history_record("run-001"), db_path=db_path)

    exit_code = cli_main.main(
        ["history", "show", "run-001", "--db-path", str(db_path), "--json"]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["schema_version"] == "history.show.v1"
    assert data["command"] == "history show"
    assert data["run"]["run_id"] == "run-001"


def test_cli_history_show_missing_run_human_exits_nonzero(tmp_path, capsys) -> None:
    exit_code = cli_main.main(
        ["history", "show", "missing", "--db-path", str(tmp_path / "history.db")]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "History run not found: missing" in captured.out


def test_cli_history_show_missing_run_json_exits_nonzero(tmp_path, capsys) -> None:
    exit_code = cli_main.main(
        [
            "history",
            "show",
            "missing",
            "--db-path",
            str(tmp_path / "history.db"),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["schema_version"] == "history.show.v1"
    assert data["run"] is None
    assert data["error"] == "History run not found: missing"


def test_cli_history_db_path_is_respected(tmp_path, capsys) -> None:
    first_db = tmp_path / "first.db"
    second_db = tmp_path / "second.db"
    insert_history_run(_history_record("first"), db_path=first_db)
    insert_history_run(_history_record("second"), db_path=second_db)

    exit_code = cli_main.main(["history", "list", "--db-path", str(second_db)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "second" in captured.out
    assert "first" not in captured.out


def test_cli_history_output_omits_raw_source(tmp_path, capsys) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_history_record("run-001"), db_path=db_path)

    exit_code = cli_main.main(
        ["history", "show", "run-001", "--db-path", str(db_path), "--json"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "secret_training_value" not in captured.out
    assert "source_code" not in captured.out
    assert "raw_source" not in captured.out


def test_cli_agent_optimize_save_history_passes_workflow_args(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    calls = []

    def fake_run_optimize_script_workflow(**kwargs) -> tuple[AgentRunResult, AgentReport]:
        calls.append(kwargs)
        return _fake_agent_result_and_report(
            script_path=kwargs.get("script_path"),
            quick=kwargs.get("quick", True),
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )
    db_path = tmp_path / "history.db"

    exit_code = cli_main.main(
        [
            "agent",
            "optimize",
            "train.py",
            "--save-history",
            "--history-db-path",
            str(db_path),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert calls[0]["save_history"] is True
    assert calls[0]["history_db_path"] == str(db_path)


def test_cli_agent_optimize_json_includes_history_run_id(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(**kwargs) -> tuple[AgentRunResult, AgentReport]:
        result, report = _fake_agent_result_and_report(
            script_path=kwargs.get("script_path"),
            quick=kwargs.get("quick", True),
        )
        result.artifacts["history_run_id"] = "run-123"
        return result, report

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "--save-history", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["artifacts"]["history_run_id"] == "run-123"


def test_cli_agent_optimize_json_without_save_history_has_null_history_id(
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
    assert data["artifacts"]["history_run_id"] is None


def test_cli_agent_optimize_human_output_shows_saved_run(
    monkeypatch,
    capsys,
) -> None:
    def fake_run_optimize_script_workflow(**kwargs) -> tuple[AgentRunResult, AgentReport]:
        result, report = _fake_agent_result_and_report(
            script_path=kwargs.get("script_path"),
            quick=kwargs.get("quick", True),
        )
        result.artifacts["history_run_id"] = "run-123"
        return result, report

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(["agent", "optimize", "--save-history"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "History:" in captured.out
    assert "- Saved run: run-123" in captured.out


def test_cli_agent_optimize_history_db_path_without_save_history_is_accepted(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    calls = []

    def fake_run_optimize_script_workflow(**kwargs) -> tuple[AgentRunResult, AgentReport]:
        calls.append(kwargs)
        return _fake_agent_result_and_report(
            script_path=kwargs.get("script_path"),
            quick=kwargs.get("quick", True),
        )

    monkeypatch.setattr(
        cli_main,
        "run_optimize_script_workflow",
        fake_run_optimize_script_workflow,
    )

    exit_code = cli_main.main(
        ["agent", "optimize", "--history-db-path", str(tmp_path / "history.db")]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "history_db_path" not in calls[0]
    assert captured.err == ""


def _fake_run_optimize_script_workflow(
    script_path: str | None = None,
    quick: bool = True,
    model: bool = False,
) -> tuple[AgentRunResult, AgentReport]:
    return _fake_agent_result_and_report(script_path=script_path, quick=quick)


def _history_record(
    run_id: str,
    *,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> HistoryRunRecord:
    return HistoryRunRecord(
        run_id=run_id,
        created_at=created_at,
        status="ok",
        command="agent optimize",
        schema_version="history.run.v1",
        goal_kind="optimize_script",
        goal_description="Optimize train.py",
        script_path="train.py",
        script_sha256="abc123",
        gpu_name="NVIDIA Test GPU",
        cuda_available=True,
        trial_summary={"status": "passed"},
        action_statuses={"inspect_system": "completed"},
        metadata={"has_diff": False, "has_trial": True, "has_comparison": False},
    )


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
        artifacts={
            "diff": None,
            "trial": None,
            "comparison": None,
            "history_run_id": None,
            "model": None,
        },
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
    trial_artifact: dict[str, object] | None = None,
    model_artifact: dict[str, object] | None = None,
    comparison_artifact: dict[str, object] | None = None,
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
        artifacts={
            "diff": diff,
            "trial": trial_artifact,
            "comparison": comparison_artifact,
            "history_run_id": None,
            "model": model_artifact,
        },
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


def _fake_trial_artifact(
    *,
    status: str = "passed",
    test_command: str | None = None,
    test_status: str | None = "skipped",
    error: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, object]:
    return {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "status": status,
        "workspace": None,
        "steps": [
            {
                "name": "run_test_command",
                "status": test_status or "skipped",
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": None,
            }
        ],
        "patch_applied": True,
        "syntax_check_status": "passed",
        "test_command": test_command,
        "test_status": test_status,
        "original_file_unchanged": True,
        "warnings": [],
        "error": error,
    }


def _fake_model_artifact() -> dict[str, object]:
    return {
        "model_available": False,
        "fallback_used": True,
        "status": "fallback",
        "model": None,
        "version": None,
    }


def _fake_trained_model_artifact() -> dict[str, object]:
    return {
        "model_available": True,
        "model_name": "mlp_classifier",
        "model_version": "training.model_artifact.v1",
        "fallback_used": False,
        "status": "ok",
        "predictions": [
            {
                "id": "trained_local_prediction",
                "target": "optimization_outcome",
                "label": "improved",
                "score": 0.91,
                "confidence": 0.91,
                "rationale": "Prediction from local trained GPUBoost artifact.",
                "metadata": {
                    "provider": "trained_local_model",
                    "probabilities": {"improved": 0.91, "regressed": 0.09},
                    "patch_application_allowed": False,
                },
            }
        ],
        "decisions": [],
        "warnings": [],
        "metadata": {
            "provider": "trained_local_model",
            "artifact_manifest_path": "artifact/manifest.json",
            "patch_application_allowed": False,
        },
    }


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


def _write_compare_files(
    tmp_path,
    *,
    include_partial_only: bool = False,
):
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    if include_partial_only:
        baseline.write_text(json.dumps(_compare_benchmark(30.0)), encoding="utf-8")
        optimized.write_text(json.dumps(_compare_benchmark(33.0)), encoding="utf-8")
        return baseline, optimized

    baseline.write_text(
        json.dumps(_complete_compare_benchmark(best_fp16_tflops=30.0)),
        encoding="utf-8",
    )
    optimized.write_text(
        json.dumps(_complete_compare_benchmark(best_fp16_tflops=33.0)),
        encoding="utf-8",
    )
    return baseline, optimized


def _write_outcome_pairs_files(tmp_path):
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    pairs = tmp_path / "pairs.json"
    baseline.write_text(json.dumps(_compare_benchmark(30.0)), encoding="utf-8")
    optimized.write_text(json.dumps(_compare_benchmark(33.0)), encoding="utf-8")
    pairs.write_text(
        json.dumps(
            [
                {
                    "row_id": "outcome-001",
                    "workload_name": "tiny_cli_fixture",
                    "baseline_json_path": "baseline.json",
                    "optimized_json_path": "optimized.json",
                    "hardware": {"gpu_name": "NVIDIA Test GPU"},
                }
            ]
        ),
        encoding="utf-8",
    )
    return pairs


def _write_cli_manifest_fixture(artifact_dir):
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.pt").write_bytes(b"not inspected by cli summaries")
    (artifact_dir / "feature_spec.json").write_text(
        json.dumps(
            {
                "feature_names": ["features.safe_signal"],
                "categorical_features": [],
                "numeric_features": ["features.safe_signal"],
                "boolean_features": [],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "label_mapping.json").write_text(
        json.dumps({"improved": 0, "regressed": 1}),
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "evaluation_report.json").write_text("{}", encoding="utf-8")
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "training.model_artifact.v1",
                "artifact_type": "mlp_classifier",
                "created_at": "2026-01-01T00:00:00+00:00",
                "model_name": "mlp_classifier",
                "model_format": "torch_state_dict",
                "model_file": "model.pt",
                "feature_spec_file": "feature_spec.json",
                "label_mapping_file": "label_mapping.json",
                "training_config_file": "training_config.json",
                "evaluation_report_file": "evaluation_report.json",
                "input_size": 1,
                "output_size": 2,
                "labels": ["improved", "regressed"],
                "feature_names": ["features.safe_signal"],
                "validation_macro_f1": 0.8,
                "test_macro_f1": 0.76,
                "baseline_macro_f1": 0.7,
                "beats_baseline": True,
                "target_macro_f1": 0.85,
                "target_met": False,
                "warnings": [],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_training_dataset_jsonl(tmp_path):
    dataset_path = tmp_path / "training_dataset.jsonl"
    rows = [
        _training_row("row-1", "train", "improved", 0.0),
        _training_row("row-2", "train", "improved", 0.2),
        _training_row("row-3", "train", "regressed", 9.0),
        _training_row("row-4", "train", "regressed", 9.2),
        _training_row("row-5", "validation", "improved", 0.1),
        _training_row("row-6", "validation", "regressed", 9.1),
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return dataset_path


def _training_row(row_id: str, split: str, label: str, signal: float) -> dict:
    return {
        "row_id": row_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "schema_version": "dataset.row.v1",
        "source": "cli_test",
        "row_type": "optimization_outcome",
        "hardware": {"gpu_name": "NVIDIA Test GPU"},
        "workload": {"batch_size": 32},
        "features": {
            "workload_family": "cli_fixture",
            "safe_signal": signal,
        },
        "metrics": {"fp32_samples_per_sec": 100.0},
        "label": {"value": label, "source": "comparison"},
        "privacy": {
            "contains_raw_source": False,
            "contains_raw_diff": False,
            "contains_stdout": False,
            "contains_stderr": False,
            "contains_sensitive_path": False,
            "notes": [],
        },
        "split": split,
        "quality_score": 1.0,
        "warnings": [],
        "metadata": {},
    }


def _complete_compare_benchmark(best_fp16_tflops: float) -> dict:
    return {
        "results": [
            {
                "name": "Matrix Multiplication",
                "metrics": [
                    {"name": "best_fp32_tflops", "value": 20.0, "unit": "TFLOPS"},
                    {
                        "name": "best_fp16_tflops",
                        "value": best_fp16_tflops,
                        "unit": "TFLOPS",
                    },
                    {"name": "fp16_speedup_ratio", "value": 4.0, "unit": "x"},
                    {
                        "name": "fp32_samples_per_sec",
                        "value": 100.0,
                        "unit": "samples/sec",
                    },
                    {
                        "name": "amp_samples_per_sec",
                        "value": 120.0,
                        "unit": "samples/sec",
                    },
                    {"name": "amp_speedup_ratio", "value": 1.2, "unit": "x"},
                    {
                        "name": "best_images_per_sec",
                        "value": 200.0,
                        "unit": "images/sec",
                    },
                    {"name": "speedup_vs_batch_1", "value": 2.0, "unit": "x"},
                    {"name": "best_batch_size", "value": 32, "unit": None},
                    {
                        "name": "max_successful_batch_size",
                        "value": 64,
                        "unit": None,
                    },
                ],
            }
        ]
    }


def _compare_benchmark(
    best_fp16_tflops: float,
    *,
    fp16_speedup_ratio: float | None = None,
) -> dict:
    metrics = [
        {
            "name": "best_fp16_tflops",
            "value": best_fp16_tflops,
            "unit": "TFLOPS",
        },
    ]
    if fp16_speedup_ratio is not None:
        metrics.append(
            {
                "name": "fp16_speedup_ratio",
                "value": fp16_speedup_ratio,
                "unit": "x",
            }
        )

    return {
        "results": [
            {
                "name": "Matrix Multiplication",
                "metrics": metrics,
            }
        ]
    }
