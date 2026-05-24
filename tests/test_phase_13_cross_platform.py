"""Phase 13B cross-platform script and path hardening checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gpuboost.dataset.outcome_collection import collect_outcomes_from_pairs_file
from gpuboost.dataset.outcome_grid import write_grid_pairs_file
from gpuboost.model.artifacts import summarize_model_artifact


SCRIPT_PATHS = [
    Path("scripts/run_outcome_experiments.ps1"),
    Path("scripts/run_outcome_experiment_grid.ps1"),
    Path("scripts/smoke_phase_12_model_workflow.ps1"),
]


def test_phase_13_powershell_scripts_exist() -> None:
    for script_path in SCRIPT_PATHS:
        assert script_path.exists()


def test_phase_13_powershell_scripts_avoid_plain_json_redirection() -> None:
    plain_json_redirect = re.compile(r"(?m)(?<!2)>\s*[^`\r\n]*\.json\b")

    for script_path in SCRIPT_PATHS:
        text = script_path.read_text(encoding="utf-8")

        assert plain_json_redirect.search(text) is None


def test_phase_13_powershell_scripts_use_utf8_json_safeguards() -> None:
    for script_path in SCRIPT_PATHS:
        text = script_path.read_text(encoding="utf-8")

        assert "$ErrorActionPreference = \"Stop\"" in text
        assert "UTF8" in text or "Write-Utf8JsonFile" in text
        if "run_outcome" in script_path.name:
            assert "ConvertFrom-Json -ErrorAction Stop" in text
            assert "[System.IO.File]::WriteAllText" in text


def test_phase_13_artifact_summary_hides_private_absolute_paths(tmp_path) -> None:
    manifest_path = tmp_path / "private" / "artifact" / "manifest.json"

    summary = summarize_model_artifact(str(manifest_path))
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["validation_status"] == "error"
    assert not Path(str(summary["manifest_path"])).is_absolute()
    assert str(tmp_path) not in serialized


def test_phase_13_pair_paths_accept_backslashes_and_forward_slashes(tmp_path) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    baseline_path = fixture_dir / "baseline.json"
    optimized_path = fixture_dir / "optimized.json"
    baseline_path.write_text(json.dumps(_benchmark(10.0)), encoding="utf-8")
    optimized_path.write_text(json.dumps(_benchmark(12.0)), encoding="utf-8")
    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text(
        json.dumps(
            [
                {
                    "row_id": "mixed-separators",
                    "baseline_json_path": "fixtures\\baseline.json",
                    "optimized_json_path": "fixtures/optimized.json",
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = collect_outcomes_from_pairs_file(
        str(pairs_path),
        output_dir=str(tmp_path / "out"),
    )

    assert summary["collected_row_count"] == 1
    assert summary["errors"] == []


def test_phase_13_grid_pairs_are_written_with_posix_separators(tmp_path) -> None:
    pairs = [
        {
            "row_id": "row-001",
            "workload_name": "fixture",
            "baseline_json_path": str(tmp_path / "grid\\row-001\\baseline.json"),
            "optimized_json_path": str(tmp_path / "grid/row-001/optimized.json"),
            "hardware": {},
            "features": {},
            "metadata": {},
        }
    ]
    output_path = tmp_path / "grid_pairs.json"

    write_grid_pairs_file(pairs, output_path=str(output_path))
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload[0]["baseline_json_path"] == "grid/row-001/baseline.json"
    assert payload[0]["optimized_json_path"] == "grid/row-001/optimized.json"


def test_phase_13_missing_artifact_path_returns_clean_error(tmp_path) -> None:
    summary = summarize_model_artifact(str(tmp_path / "missing" / "manifest.json"))

    assert summary["validation_status"] == "error"
    assert summary["validation_errors"]
    assert all("Traceback" not in error for error in summary["validation_errors"])


def _benchmark(value: float) -> dict[str, object]:
    return {
        "results": [
            {
                "name": "Matrix Multiply",
                "metrics": [
                    {
                        "name": "best_fp32_tflops",
                        "value": value,
                        "unit": "TFLOPS",
                    }
                ],
            }
        ]
    }
