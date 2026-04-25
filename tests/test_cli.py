"""Tests for the Phase 1 CLI."""

import json

from gpuboost.cli import main as cli_main
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

