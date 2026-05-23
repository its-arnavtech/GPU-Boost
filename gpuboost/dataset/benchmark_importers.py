"""Local-file benchmark context importers for GPUBoost datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from gpuboost.schemas.dataset import (
    BenchmarkContextRow,
    DatasetValue,
    create_timestamp,
)


_CSV_REQUIRED_COLUMNS = {"benchmark_name", "metric_name", "metric_value"}


def import_benchmark_context_json(
    filepath: str,
    source: str,
) -> list[BenchmarkContextRow]:
    """Import benchmark context rows from a local UTF-8 JSON file."""

    with Path(filepath).open(encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        records = payload["rows"]
    else:
        raise ValueError("Expected JSON list or object with a 'rows' list.")

    normalized_source = normalize_source_name(source)
    rows: list[BenchmarkContextRow] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Each JSON benchmark row must be an object.")
        rows.append(
            BenchmarkContextRow(
                row_id=f"{normalized_source}_{index}",
                created_at=create_timestamp(),
                source=normalized_source,
                benchmark_name=str(record.get("benchmark_name") or ""),
                workload_name=_optional_string(record.get("workload_name")),
                hardware_name=_optional_string(record.get("hardware_name")),
                software_stack=_dict_value(record.get("software_stack")),
                metrics=_dict_value(record.get("metrics")),
                units=_string_dict(record.get("units")),
                url=_optional_string(record.get("url")),
                notes=_optional_string(record.get("notes")),
                metadata=_dict_value(record.get("metadata")),
            )
        )
    return rows


def import_benchmark_context_csv(
    filepath: str,
    source: str,
) -> list[BenchmarkContextRow]:
    """Import benchmark context rows from a local UTF-8 CSV file."""

    normalized_source = normalize_source_name(source)
    with Path(filepath).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = _CSV_REQUIRED_COLUMNS - fieldnames
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Missing required CSV columns: {missing_columns}")

        rows: list[BenchmarkContextRow] = []
        for index, record in enumerate(reader):
            metric_name = (record.get("metric_name") or "").strip()
            metric_unit = (record.get("metric_unit") or "").strip()
            software_stack = {
                key: value
                for key in ("cuda", "framework")
                if (value := parse_scalar(record.get(key, ""))) is not None
            }

            rows.append(
                BenchmarkContextRow(
                    row_id=f"{normalized_source}_{index}",
                    created_at=create_timestamp(),
                    source=normalized_source,
                    benchmark_name=(record.get("benchmark_name") or "").strip(),
                    workload_name=_optional_string(record.get("workload_name")),
                    hardware_name=_optional_string(record.get("hardware_name")),
                    software_stack=software_stack,
                    metrics={
                        metric_name: parse_scalar(record.get("metric_value", ""))
                    },
                    units={metric_name: metric_unit} if metric_unit else {},
                    url=_optional_string(record.get("url")),
                    notes=_optional_string(record.get("notes")),
                )
            )
    return rows


def parse_scalar(value: str) -> str | int | float | bool | None:
    """Parse a scalar CSV value into a JSON-safe primitive."""

    stripped = value.strip()
    if stripped == "":
        return None

    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        return stripped


def normalize_source_name(source: str) -> str:
    """Normalize a benchmark source name for row identifiers and schema values."""

    normalized = source.lower().replace(" ", "_").replace("-", "_")
    return "".join(char for char in normalized if char.isalnum() or char == "_")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _dict_value(value: Any) -> dict[str, DatasetValue]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, str | int | float | bool) or item is None
    }


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if item is not None
    }
