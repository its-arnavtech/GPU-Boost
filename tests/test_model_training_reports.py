"""Tests for Phase 12.2 baseline report writing."""

from __future__ import annotations

import json

from gpuboost.model.training_reports import write_baseline_comparison_reports


def test_write_baseline_comparison_reports_writes_json_and_markdown(tmp_path) -> None:
    comparison = {
        "schema_version": "training.baseline_comparison.v1",
        "status": "ok",
        "dataset_summary": {
            "row_count": 4,
            "encoded_row_count": 4,
            "encoded_feature_count": 2,
            "encoded_class_count": 2,
            "label_counts": {"improved": 2, "regressed": 2},
            "split_counts": {"train": 2, "validation": 2},
        },
        "eval_split_used": "validation",
        "models": [
            {
                "model_name": "nearest_centroid_baseline",
                "evaluation": {
                    "status": "ok",
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                },
                "metadata": {},
            }
        ],
        "best_model_name": "nearest_centroid_baseline",
        "best_macro_f1": 1.0,
        "warnings": ["review dataset size"],
    }

    output = write_baseline_comparison_reports(comparison, str(tmp_path))

    json_report = tmp_path / "baseline_comparison_report.json"
    markdown_report = tmp_path / "baseline_comparison_report.md"
    assert output == {
        "json_report": str(json_report),
        "markdown_report": str(markdown_report),
    }
    assert json.loads(json_report.read_text(encoding="utf-8"))["status"] == "ok"
    markdown = markdown_report.read_text(encoding="utf-8")
    assert "GPUBoost Baseline Model Evaluation" in markdown
    assert "nearest_centroid_baseline" in markdown
    assert "not production model integration" in markdown
