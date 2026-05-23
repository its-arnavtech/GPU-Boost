"""Tests for Phase 11.6 dataset split assignment."""

from __future__ import annotations

import pytest

from gpuboost.dataset.splitting import (
    assign_dataset_splits,
    split_counts,
    validate_split_ratios,
)
from gpuboost.schemas.dataset import DatasetLabel, DatasetPrivacyFlags, DatasetRow


def test_ratios_validate() -> None:
    validate_split_ratios(0.8, 0.1, 0.1)
    validate_split_ratios(1.0, 0.0, 0.0)


def test_invalid_ratios_raise() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        validate_split_ratios(-0.1, 0.6, 0.5)
    with pytest.raises(ValueError, match="sum"):
        validate_split_ratios(0.8, 0.2, 0.2)


def test_deterministic_splits_with_seed() -> None:
    rows = [_make_row(f"row-{index}") for index in range(10)]

    first, _ = assign_dataset_splits(rows, seed=123)
    second, _ = assign_dataset_splits(rows, seed=123)

    assert [(row.row_id, row.split) for row in first] == [
        (row.row_id, row.split) for row in second
    ]


def test_preserves_existing_splits_when_overwrite_false() -> None:
    rows = [
        _make_row("row-001", split="test"),
        _make_row("row-002", split=None),
    ]

    assigned, _ = assign_dataset_splits(rows, overwrite=False, seed=1)

    assert assigned[0].split == "test"
    assert assigned[1].split == "train"


def test_overwrite_true_reassigns() -> None:
    rows = [
        _make_row("row-001", split="test"),
        _make_row("row-002", split="test"),
    ]

    assigned, _ = assign_dataset_splits(
        rows,
        train_ratio=1.0,
        validation_ratio=0.0,
        test_ratio=0.0,
        overwrite=True,
    )

    assert [row.split for row in assigned] == ["train", "train"]


def test_split_counts_works() -> None:
    rows = [
        _make_row("row-001", split="train"),
        _make_row("row-002", split="validation"),
        _make_row("row-003", split="test"),
        _make_row("row-004", split=None),
    ]

    assert split_counts(rows) == {
        "train": 1,
        "validation": 1,
        "test": 1,
        "unassigned": 1,
    }


def test_empty_rows_returns_empty_summary() -> None:
    assigned, summary = assign_dataset_splits([])

    assert assigned == []
    assert summary.train_count == 0
    assert summary.validation_count == 0
    assert summary.test_count == 0
    assert summary.unassigned_count == 0


def test_small_dataset_gets_train_split() -> None:
    assigned, summary = assign_dataset_splits([_make_row("row-001")])

    assert assigned[0].split == "train"
    assert summary.train_count == 1


def test_input_rows_are_not_mutated() -> None:
    rows = [_make_row("row-001", split=None)]

    assigned, _ = assign_dataset_splits(rows)

    assert rows[0].split is None
    assert assigned[0].split == "train"
    assert assigned[0] is not rows[0]


def test_summary_counts_are_correct() -> None:
    rows = [_make_row(f"row-{index}") for index in range(10)]

    assigned, summary = assign_dataset_splits(rows, seed=42)
    counts = split_counts(assigned)

    assert summary.train_count == counts["train"]
    assert summary.validation_count == counts["validation"]
    assert summary.test_count == counts["test"]
    assert summary.unassigned_count == counts["unassigned"]
    assert counts == {"train": 8, "validation": 1, "test": 1, "unassigned": 0}


def test_split_values_valid() -> None:
    rows = [_make_row(f"row-{index}") for index in range(20)]

    assigned, _ = assign_dataset_splits(rows)

    assert {row.split for row in assigned} <= {"train", "validation", "test"}


def _make_row(row_id: str, split: str | None = None) -> DatasetRow:
    return DatasetRow(
        row_id=row_id,
        created_at="2026-01-01T00:00:00+00:00",
        source="gpuboost_history",
        row_type="optimization_outcome",
        hardware={"gpu_name": "NVIDIA A100"},
        workload={"command": "agent optimize"},
        features={"action_count": 1},
        metrics={"benchmark_median_ms": 9.5},
        label=DatasetLabel(value="improved", source="comparison"),
        privacy=DatasetPrivacyFlags(),
        split=split,
        quality_score=0.9,
    )
