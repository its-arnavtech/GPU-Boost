"""Tests for Phase 14 demo validation report generation."""

from __future__ import annotations

import json

from gpuboost.demo.reporting import create_demo_validation_report


def test_create_demo_validation_report_writes_json_and_markdown(tmp_path) -> None:
    report = create_demo_validation_report(
        comparison_results=[_comparison_result()],
        model_results=[_model_result()],
        output_dir=str(tmp_path),
    )

    json_path = tmp_path / "demo_validation_report.json"
    md_path = tmp_path / "demo_validation_report.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert payload["summary"]["workload_count"] == 1
    assert payload["workloads"][0]["name"] == "cnn_real_world"
    assert payload["workloads"][0]["overall_verdict"] == "improved"
    assert payload["workloads"][0]["metric_deltas"][0]["name"] == "samples_per_sec"
    assert payload["model_advisory_predictions"][0]["recommendation"] == "Enable AMP"
    assert "cnn_real_world" in markdown
    assert "Enable AMP" in markdown
    assert report["output_files"]["json"].endswith("demo_validation_report.json")


def test_demo_validation_report_includes_safety_notes(tmp_path) -> None:
    report = create_demo_validation_report([_comparison_result()], output_dir=str(tmp_path))

    assert report["safety_notes"]["model_advisory_only"] is True
    assert report["safety_notes"]["patch_application_allowed"] is False
    assert report["safety_notes"]["deterministic_checks_authoritative"] is True
    assert report["safety_notes"]["automatic_patch_application"] is False


def test_demo_validation_report_excludes_unsafe_payloads(tmp_path) -> None:
    unsafe = _comparison_result()
    unsafe.update(
        {
            "raw_source": "def train():\n    pass",
            "diff": "--- a/train.py\n+++ b/train.py",
            "stdout": "benchmark logs",
            "stderr": "traceback logs",
        }
    )
    report = create_demo_validation_report(
        [unsafe],
        model_results=[
            {
                **_model_result(),
                "source_code": "class Model: pass",
                "model_weights": "secret weights",
            }
        ],
        output_dir=str(tmp_path),
    )

    serialized = json.dumps(report)
    assert "def train" not in serialized
    assert "--- a/train.py" not in serialized
    assert "benchmark logs" not in serialized
    assert "traceback logs" not in serialized
    assert "secret weights" not in serialized
    assert "class Model" not in serialized


def test_demo_validation_report_avoids_private_absolute_paths(tmp_path) -> None:
    result = _comparison_result()
    result["baseline_label"] = r"C:\Users\TestUser\private\baseline.json"
    result["optimized_label"] = "/home/testuser/private/optimized.json"
    result["sections"][0]["metrics"][0]["summary"] = (
        r"C:\Users\TestUser\private\baseline.json improved"
    )

    report = create_demo_validation_report([result], output_dir=str(tmp_path))
    serialized = json.dumps(report)

    assert r"C:\Users\TestUser" not in serialized
    assert "/home/testuser" not in serialized
    assert "[path]" in serialized


def test_demo_validation_report_handles_empty_results_gracefully(tmp_path) -> None:
    report = create_demo_validation_report([], output_dir=str(tmp_path))

    assert report["summary"]["workload_count"] == 0
    assert report["workloads"] == []
    assert report["warnings"] == ["No comparison results were provided."]
    markdown = (tmp_path / "demo_validation_report.md").read_text(encoding="utf-8")
    assert "No comparison results were provided." in markdown


def test_demo_validation_report_is_json_serializable(tmp_path) -> None:
    report = create_demo_validation_report(
        [_comparison_result()],
        model_results=[_model_result()],
        output_dir=str(tmp_path),
    )

    encoded = json.dumps(report, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == "demo.validation_report.v1"


def _comparison_result() -> dict:
    return {
        "generated_at": "2026-05-24T00:00:00+00:00",
        "status": "ok",
        "workload_name": "cnn_real_world",
        "baseline_label": "baseline",
        "optimized_label": "optimized",
        "overall_verdict": "improved",
        "sections": [
            {
                "title": "cnn_real_world",
                "verdict": "improved",
                "metrics": [
                    {
                        "name": "samples_per_sec",
                        "unit": "samples/sec",
                        "before": 100.0,
                        "after": 150.0,
                        "absolute_delta": 50.0,
                        "percent_delta": 50.0,
                        "direction": "improved",
                        "summary": "samples_per_sec improved by 50%.",
                    }
                ],
            }
        ],
        "warnings": [],
        "error": None,
    }


def _model_result() -> dict:
    return {
        "name": "cnn_advisor",
        "recommendation": "Enable AMP",
        "confidence": 0.91,
        "estimated_speedup": "1.5x",
        "impact": "high",
    }
