"""Tests for the Phase 1 CLI."""

import json

from gpuboost.cli import main as cli_main
from gpuboost.schemas.benchmark_result import BenchmarkSuiteResult
from gpuboost.schemas.gpu_profile import (
    GPUBoostProfile,
    SystemProfile,
    TorchEnvironmentProfile,
)
from gpuboost.schemas.recommendation import AdvisorResult, Recommendation


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
