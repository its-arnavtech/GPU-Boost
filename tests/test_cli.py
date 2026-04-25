"""Tests for the Phase 1 CLI."""

import json

from gpuboost.cli import main as cli_main
from gpuboost.schemas.benchmark_result import BenchmarkSuiteResult
from gpuboost.schemas.gpu_profile import (
    GPUBoostProfile,
    SystemProfile,
    TorchEnvironmentProfile,
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
    assert captured.err == ""
