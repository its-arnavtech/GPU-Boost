"""Tests for the Phase 14 real-world demo benchmark pipeline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from gpuboost.dataset.outcome_collection import load_outcome_pairs_file
from gpuboost.demo.real_world import (
    DEFAULT_OUTPUT_ROOT,
    build_real_world_demo_pairs,
    write_real_world_pairs_file,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_real_world_demo_benchmarks.ps1"
GENERATED_ROOT = Path(DEFAULT_OUTPUT_ROOT)
EXPECTED_ROW_IDS = [
    "cnn_real_world",
    "transformer_toy_real_world",
    "dataloader_real_world",
]


def test_build_real_world_demo_pairs_returns_expected_pairs() -> None:
    pairs = build_real_world_demo_pairs()

    assert [pair["row_id"] for pair in pairs] == EXPECTED_ROW_IDS
    assert [pair["workload_name"] for pair in pairs] == EXPECTED_ROW_IDS

    for pair in pairs:
        assert pair["baseline_script"].startswith("examples/real_world/")
        assert pair["optimized_script"].startswith("examples/real_world/")
        assert pair["baseline_args"] == ["--quick", "--benchmark-json"]
        assert pair["optimized_args"] == ["--quick", "--benchmark-json"]
        assert pair["baseline_json_path"].endswith("/baseline.json")
        assert pair["optimized_json_path"].endswith("/optimized.json")
        assert pair["hardware"] == {"gpu_name": "auto"}
        assert pair["features"]["real_world_demo"] is True
        assert pair["features"]["uses_synthetic_data"] is True
        assert pair["metadata"]["example"] == "real_world"
        assert pair["metadata"]["no_downloads"] is True
        assert pair["metadata"]["no_external_apis"] is True


def test_real_world_demo_pair_paths_are_under_generated_demo_root() -> None:
    pairs = build_real_world_demo_pairs()

    for pair in pairs:
        for key in ("baseline_json_path", "optimized_json_path"):
            path = Path(pair[key])
            assert path.parts[: len(GENERATED_ROOT.parts)] == GENERATED_ROOT.parts
            assert "demo_real_world" in path.parts


def test_write_real_world_pairs_file_is_collect_outcomes_compatible(tmp_path) -> None:
    output_root = tmp_path / "data" / "gpuboost" / "generated" / "demo_real_world"
    pairs_path = output_root / "pairs.json"
    pairs = build_real_world_demo_pairs(str(output_root))

    written_path = write_real_world_pairs_file(pairs, str(pairs_path))

    assert written_path == str(pairs_path)
    payload = json.loads(pairs_path.read_text(encoding="utf-8"))
    normalized = load_outcome_pairs_file(str(pairs_path))
    assert [pair["row_id"] for pair in normalized] == EXPECTED_ROW_IDS

    for pair in payload:
        assert set(pair) == {
            "baseline_json_path",
            "features",
            "hardware",
            "metadata",
            "optimized_json_path",
            "row_id",
            "workload_name",
        }
        assert not Path(pair["baseline_json_path"]).is_absolute()
        assert not Path(pair["optimized_json_path"]).is_absolute()
        assert pair["baseline_json_path"].endswith("baseline.json")
        assert pair["optimized_json_path"].endswith("optimized.json")


def test_real_world_demo_runner_script_exists_and_is_safe() -> None:
    assert RUNNER_PATH.exists()

    script = RUNNER_PATH.read_text(encoding="utf-8")
    assert '$ErrorActionPreference = "Stop"' in script
    assert "data\\gpuboost\\generated\\demo_real_world" in script
    assert "--quick" in script
    assert "--benchmark-json" in script
    assert "ConvertFrom-Json" in script
    assert "[System.IO.File]::WriteAllText" in script
    assert "Write-Utf8Json" in script
    assert "python -m gpuboost dataset collect-outcomes" in script
    assert "python -m gpuboost compare" in script
    assert ">" not in script


def test_no_generated_demo_outputs_are_tracked() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "data/gpuboost/generated/demo_real_world"],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert completed.stdout == ""
