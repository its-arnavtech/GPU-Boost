"""Tests for Phase 4 DataLoader static analysis findings."""

import ast

from gpuboost.code_analysis.dataloader import (
    analyze_dataloader_source,
    get_keyword,
    is_dataloader_call,
    literal_bool_value,
    literal_int_value,
)


def test_detects_dataloader_missing_num_workers() -> None:
    result = analyze_dataloader_source("loader = DataLoader(dataset, batch_size=32)\n")

    assert "dataloader_missing_num_workers" in _finding_ids(result)


def test_detects_dataloader_num_workers_zero() -> None:
    result = analyze_dataloader_source(
        "loader = DataLoader(dataset, num_workers=0, pin_memory=True)\n"
    )

    assert "dataloader_num_workers_zero" in _finding_ids(result)
    assert "dataloader_missing_num_workers" not in _finding_ids(result)


def test_does_not_warn_for_num_workers_greater_than_zero() -> None:
    result = analyze_dataloader_source(
        "loader = DataLoader(dataset, num_workers=4, pin_memory=True)\n"
    )

    assert "dataloader_num_workers_zero" not in _finding_ids(result)
    assert "dataloader_missing_num_workers" not in _finding_ids(result)


def test_detects_missing_pin_memory() -> None:
    result = analyze_dataloader_source("loader = DataLoader(dataset, num_workers=4)\n")

    assert "dataloader_missing_pin_memory" in _finding_ids(result)


def test_detects_pin_memory_false() -> None:
    result = analyze_dataloader_source(
        "loader = DataLoader(dataset, num_workers=4, pin_memory=False)\n"
    )

    assert "dataloader_pin_memory_false" in _finding_ids(result)
    assert "dataloader_missing_pin_memory" not in _finding_ids(result)


def test_does_not_warn_for_pin_memory_true() -> None:
    result = analyze_dataloader_source(
        "loader = DataLoader(dataset, num_workers=4, pin_memory=True)\n"
    )

    assert result.status == "ok"
    assert result.findings == []


def test_supports_torch_utils_data_dataloader_call() -> None:
    result = analyze_dataloader_source(
        "loader = torch.utils.data.DataLoader(dataset, pin_memory=True)\n"
    )

    assert "dataloader_missing_num_workers" in _finding_ids(result)


def test_supports_alias_style_dataloader_call() -> None:
    result = analyze_dataloader_source(
        "loader = data.DataLoader(dataset, num_workers=0, pin_memory=True)\n"
    )

    assert "dataloader_num_workers_zero" in _finding_ids(result)


def test_finding_contains_line_and_column_location() -> None:
    result = analyze_dataloader_source(
        "def make_loader():\n"
        "    return DataLoader(dataset, batch_size=32, pin_memory=True)\n",
        filepath="train.py",
    )

    finding = result.findings[0]
    assert finding.filepath == "train.py"
    assert finding.line == 2
    assert finding.column == 11
    assert finding.end_line == 2
    assert finding.end_column is not None


def test_parse_error_returns_status_error() -> None:
    result = analyze_dataloader_source("loader = DataLoader(\n")

    assert result.status == "error"
    assert result.findings == []
    assert result.error is not None


def test_multiple_dataloader_calls_produce_multiple_findings() -> None:
    result = analyze_dataloader_source(
        "first = DataLoader(dataset)\n"
        "second = torch_data.DataLoader(dataset, num_workers=0, pin_memory=False)\n"
    )

    assert _finding_ids(result) == [
        "dataloader_missing_num_workers",
        "dataloader_missing_pin_memory",
        "dataloader_num_workers_zero",
        "dataloader_pin_memory_false",
    ]


def test_helper_functions_support_literal_and_dataloader_detection() -> None:
    call = ast.parse(
        "loader = torch_data.DataLoader(dataset, num_workers=4, pin_memory=False)\n"
    ).body[0].value

    assert isinstance(call, ast.Call)
    assert is_dataloader_call(call)
    assert literal_int_value(get_keyword(call, "num_workers").value) == 4
    assert literal_bool_value(get_keyword(call, "pin_memory").value) is False
    assert get_keyword(call, "batch_size") is None


def _finding_ids(result) -> list[str]:
    return [finding.id for finding in result.findings]
