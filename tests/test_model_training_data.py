"""Tests for Phase 12.1 training data loading and summaries."""

from __future__ import annotations

import json

import pytest

from gpuboost.model.training_data import (
    load_default_training_rows,
    load_training_rows_jsonl,
    summarize_training_rows,
)
from gpuboost.schemas.dataset import DatasetLabel, DatasetPrivacyFlags, DatasetRow


def test_loading_jsonl_dataset_rows(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        _make_row("row-1", split="train"),
        _make_row(
            "row-2",
            label=DatasetLabel(value="regressed", source="comparison"),
            split="validation",
        ),
    ]
    path.write_text(
        "\n".join(json.dumps(row.to_dict()) for row in rows) + "\n\n",
        encoding="utf-8",
    )

    loaded = load_training_rows_jsonl(str(path))

    assert [row.row_id for row in loaded] == ["row-1", "row-2"]
    assert loaded[1].label.value == "regressed"


def test_invalid_jsonl_raises_clean_error(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"row_id": "ok"}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL"):
        load_training_rows_jsonl(str(path))


def test_training_row_summary_counts_labels_splits_and_features() -> None:
    rows = [
        _make_row("row-1", split="train", features={"workload_family": "amp"}),
        _make_row(
            "row-2",
            label=DatasetLabel(value="regressed", source="comparison"),
            split="validation",
            features={"workload_family": "batch", "overall_verdict": "regressed"},
        ),
        _make_row(
            "row-3",
            label=DatasetLabel(value="unknown", source="unknown"),
            split="test",
            features={"workload_family": "neutral"},
        ),
    ]

    summary = summarize_training_rows(rows)

    assert summary.row_count == 3
    assert summary.labeled_count == 2
    assert summary.skipped_count == 1
    assert summary.label_counts == {"improved": 1, "regressed": 1}
    assert summary.split_counts == {"test": 1, "train": 1, "validation": 1}
    assert summary.feature_count == 5
    assert summary.warnings == []


def test_missing_default_training_rows_returns_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert load_default_training_rows() == []


def _make_row(
    row_id: str,
    *,
    label: DatasetLabel | None = None,
    split: str | None = None,
    features: dict | None = None,
) -> DatasetRow:
    return DatasetRow(
        row_id=row_id,
        created_at="2026-01-01T00:00:00+00:00",
        source="controlled_experiment",
        row_type="optimization_outcome",
        hardware={"gpu_name": "Test GPU"},
        workload={"batch_size": 32},
        features=features or {},
        metrics={"fp32_samples_per_sec": 100.0},
        label=label or DatasetLabel(value="improved", source="comparison"),
        privacy=DatasetPrivacyFlags(),
        split=split,
        metadata={"phase": 12},
    )
