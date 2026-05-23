"""Local MLCommons inference result intake helpers for GPUBoost Phase 11."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from gpuboost.dataset.export import export_benchmark_context_jsonl, export_validation_report
from gpuboost.dataset.validation import validate_benchmark_context_rows
from gpuboost.schemas.dataset import BenchmarkContextRow, create_timestamp


_DEFAULT_ROOT = "data/gpuboost/raw/mlcommons/inference_results_v6.0"
_DEFAULT_OUTPUT_DIR = "data/gpuboost/generated"
_DEFAULT_MANIFEST_DIR = "data/gpuboost/manifests"
_VENDORS = ("NVIDIA", "AMD", "Google", "CoreWeave")
_LIKELY_FILE_MARKERS = (
    "system_desc",
    "system",
    "results",
    "result",
    "performance",
    "accuracy",
    "measurements",
    "submission",
    "summary",
    "mlperf_log_summary",
    "out.csv",
)
_FORBIDDEN_KEY_PARTS = (
    "code",
    "source",
    "script",
    "log",
    "stdout",
    "stderr",
    "diff",
    "raw",
    "file_contents",
)
_STRING_METRIC_KEYS = {
    "result_valid",
}
_NUMERIC_METRIC_KEYS = {
    "result",
    "samples_per_sec",
    "samples_per_second",
    "queries_per_sec",
    "queries_per_second",
    "tokens_per_sec",
    "tokens_per_second",
    "latency_ms",
    "latency",
    "latency_99_percentile",
    "latency_90_percentile",
    "latency_50_percentile",
    "accuracy",
    "exact_match",
    "rougel",
    "rouge1",
    "rouge2",
    "f1_hierarchical",
    "vbench_score",
    "power_w",
    "energy",
}
_SYSTEM_METADATA_KEYS = (
    "accelerator_model_name",
    "accelerators_per_node",
    "system_type",
    "host_processor_model_name",
    "host_processors_per_node",
    "framework",
)


def inspect_mlcommons_source(
    mlcommons_root: str = _DEFAULT_ROOT,
) -> dict[str, Any]:
    """Inspect local MLCommons vendor folders and summarize available files."""

    root = Path(mlcommons_root)
    vendors: dict[str, Any] = {}

    for vendor in _VENDORS:
        vendor_path = root / "closed" / vendor
        warnings: list[str] = []
        exists = vendor_path.exists() and vendor_path.is_dir()
        file_paths = [path for path in vendor_path.rglob("*") if exists and path.is_file()]
        if not exists:
            warnings.append(f"Missing vendor folder: {vendor_path}")

        vendors[vendor] = {
            "exists": exists,
            "file_count": len(file_paths),
            "json_count": sum(1 for path in file_paths if path.suffix.lower() == ".json"),
            "csv_count": sum(1 for path in file_paths if path.suffix.lower() == ".csv"),
            "txt_count": sum(1 for path in file_paths if path.suffix.lower() == ".txt"),
            "likely_result_file_count": sum(1 for path in file_paths if _looks_like_result_file(path)),
            "likely_system_desc_count": sum(1 for path in file_paths if _looks_like_system_desc_file(path)),
            "warnings": warnings,
        }

    return {
        "mlcommons_root": str(root),
        "vendors": vendors,
    }


def extract_mlcommons_context_rows(
    mlcommons_root: str = _DEFAULT_ROOT,
) -> tuple[list[BenchmarkContextRow], list[str]]:
    """Extract conservative benchmark context rows from local MLCommons files."""

    root = Path(mlcommons_root)
    system_records: dict[str, dict[str, dict[str, Any]]] = {}
    rows: list[BenchmarkContextRow] = []
    warnings: list[str] = []
    invalid_json_count = 0
    unclear_file_count = 0

    for vendor in _VENDORS:
        vendor_path = root / "closed" / vendor
        if not vendor_path.exists():
            warnings.append(f"Missing vendor folder: {vendor_path}")
            continue

        vendor_systems, system_invalid_count = _load_vendor_system_records(vendor_path, vendor)
        invalid_json_count += system_invalid_count
        system_records[vendor] = vendor_systems
        rows.extend(
            row
            for row in (
                _build_system_context_row(record, vendor_path / "systems" / f"{system_name}.json", vendor)
                for system_name, record in vendor_systems.items()
            )
            if row is not None
        )

    for vendor in _VENDORS:
        vendor_path = root / "closed" / vendor
        if not vendor_path.exists():
            continue
        for path in sorted(vendor_path.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".json", ".csv"}:
                continue
            if _looks_like_system_desc_file(path):
                continue
            if _should_skip_file(path):
                unclear_file_count += 1
                continue

            try:
                if suffix == ".csv":
                    file_rows = _extract_rows_from_csv(path, vendor, vendor_path)
                else:
                    file_rows = _extract_rows_from_json(path, vendor, vendor_path, system_records.get(vendor, {}))
            except json.JSONDecodeError:
                invalid_json_count += 1
                continue
            except (csv.Error, UnicodeError, ValueError):
                unclear_file_count += 1
                continue

            if not file_rows:
                unclear_file_count += 1
                continue
            rows.extend(file_rows)

    deduped_rows, duplicate_count = _dedupe_rows(rows)
    if duplicate_count:
        warnings.append(f"Skipped duplicate MLCommons context rows: count={duplicate_count}.")
    if invalid_json_count:
        warnings.append(f"Skipped invalid MLCommons JSON files: count={invalid_json_count}.")
    if unclear_file_count:
        warnings.append(f"Skipped unclear MLCommons files: count={unclear_file_count}.")
    return deduped_rows, warnings


def parse_mlcommons_scalar(value: Any) -> str | int | float | bool | None:
    """Parse a local MLCommons scalar value into a safe primitive."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value

    text = str(value).strip()
    if not text or len(text) > 200:
        return None

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return float(text)
    return text


def infer_vendor_from_path(path: str) -> str | None:
    """Infer the MLCommons vendor folder from a path."""

    normalized = path.replace("\\", "/")
    for vendor in _VENDORS:
        if f"/closed/{vendor}/" in normalized:
            return vendor
    return None


def infer_workload_from_path_or_record(path: str, record: dict[str, Any]) -> str | None:
    """Infer MLPerf workload/model name from a record or path."""

    for key in ("MlperfModel", "Model", "model", "benchmark", "workload_name"):
        value = parse_mlcommons_scalar(record.get(key))
        if isinstance(value, str) and value:
            return value

    parts = path.replace("\\", "/").split("/")
    if "results" in parts:
        index = parts.index("results")
        if len(parts) >= index + 4:
            return parts[index + 2]
    return None


def infer_hardware_from_record(record: dict[str, Any]) -> str | None:
    """Infer a hardware/system name from a record."""

    for key in (
        "hardware_name",
        "accelerator_model_name",
        "accelerator_model",
        "gpu_name",
        "SystemName",
        "system_name",
        "Platform",
    ):
        value = parse_mlcommons_scalar(record.get(key))
        if isinstance(value, str) and value:
            return value
    return None


def flatten_safe_numeric_metrics(
    record: dict[str, Any],
) -> dict[str, str | int | float | bool | None]:
    """Extract a conservative set of safe scalar MLPerf metrics from a record."""

    metrics: dict[str, str | int | float | bool | None] = {}
    for key, value in record.items():
        normalized_key = _normalize_key(str(key))
        if any(part in normalized_key for part in _FORBIDDEN_KEY_PARTS):
            continue

        scalar = parse_mlcommons_scalar(value)
        if scalar is None:
            continue

        if normalized_key == "result":
            metric_name, metric_value = _metric_from_result_field(scalar)
            if metric_name and metric_value is not None:
                metrics[metric_name] = metric_value
            continue

        if normalized_key == "accuracy":
            accuracy_value = _extract_accuracy_value(scalar)
            if accuracy_value is not None:
                metrics["accuracy"] = accuracy_value
            continue

        if normalized_key in {"errors", "compliance", "result_valid"}:
            result_valid = _normalize_result_valid(scalar)
            if result_valid is not None:
                metrics["result_valid"] = result_valid
            continue

        if normalized_key in _STRING_METRIC_KEYS and isinstance(scalar, str):
            metrics[normalized_key] = scalar
            continue

        if normalized_key in _NUMERIC_METRIC_KEYS and _is_numeric_scalar(scalar):
            metric_name = _map_metric_name(normalized_key)
            if metric_name:
                metrics[metric_name] = scalar

    return metrics


def run_mlcommons_intake(
    mlcommons_root: str = _DEFAULT_ROOT,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    manifest_dir: str = _DEFAULT_MANIFEST_DIR,
) -> dict[str, Any]:
    """Run local MLCommons intake and write generated artifacts."""

    source_summary = inspect_mlcommons_source(mlcommons_root)
    rows, warnings = extract_mlcommons_context_rows(mlcommons_root)
    validation_report = validate_benchmark_context_rows(rows)

    output_path = Path(output_dir)
    manifest_path = Path(manifest_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path.mkdir(parents=True, exist_ok=True)

    rows_path = output_path / "mlcommons_inference_context.jsonl"
    validation_path = output_path / "mlcommons_inference_validation_report.json"
    report_json_path = manifest_path / "mlcommons_inference_intake_report.json"
    report_md_path = manifest_path / "mlcommons_inference_intake_report.md"

    export_benchmark_context_jsonl(rows, str(rows_path))
    export_validation_report(validation_report, str(validation_path))

    summary = {
        "generated_at": create_timestamp(),
        "row_count": len(rows),
        "validation_status": validation_report.status,
        "source_summary": source_summary,
        "output_files": {
            "jsonl": str(rows_path),
            "validation_report": str(validation_path),
            "intake_report_json": str(report_json_path),
            "intake_report_md": str(report_md_path),
        },
        "warnings": warnings,
        "skipped_file_count": _count_skipped_files(warnings),
    }

    report_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_md_path.write_text(_build_markdown_report(summary), encoding="utf-8")
    return summary


def _load_vendor_system_records(
    vendor_path: Path,
    vendor: str,
) -> tuple[dict[str, dict[str, Any]], int]:
    records: dict[str, dict[str, Any]] = {}
    invalid_count = 0
    for path in sorted(vendor_path.rglob("systems/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            invalid_count += 1
            continue
        if not isinstance(payload, dict):
            continue
        system_name = path.stem
        records[system_name] = {**payload, "vendor": vendor, "system_name": system_name}
    return records, invalid_count


def _extract_rows_from_csv(
    path: Path,
    vendor: str,
    vendor_root: Path,
) -> list[BenchmarkContextRow]:
    rows: list[BenchmarkContextRow] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for record in reader:
            if not isinstance(record, dict):
                continue
            row = _build_result_context_row(record, path, vendor, vendor_root, None)
            if row is not None:
                rows.append(row)
    return rows


def _extract_rows_from_json(
    path: Path,
    vendor: str,
    vendor_root: Path,
    system_records: dict[str, dict[str, Any]],
) -> list[BenchmarkContextRow]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        if not payload or not all(isinstance(item, dict) for item in payload[:10]):
            return []
        records = [item for item in payload if isinstance(item, dict)]
    else:
        return []

    rows: list[BenchmarkContextRow] = []
    for record in records:
        system_record = _match_system_record(path, record, system_records)
        row = _build_result_context_row(record, path, vendor, vendor_root, system_record)
        if row is not None:
            rows.append(row)
    return rows


def _build_result_context_row(
    record: dict[str, Any],
    path: Path,
    vendor: str,
    vendor_root: Path,
    system_record: dict[str, Any] | None,
) -> BenchmarkContextRow | None:
    merged_record = dict(system_record or {})
    merged_record.update(record)

    workload_name = infer_workload_from_path_or_record(str(path), merged_record)
    hardware_name = infer_hardware_from_record(merged_record)
    metrics = flatten_safe_numeric_metrics(merged_record)
    software_stack = _software_stack_from_record(merged_record)

    if not workload_name and not hardware_name and not metrics and not software_stack:
        return None
    if not metrics:
        return None

    units = _infer_units(metrics, merged_record)
    relative_path = _safe_relative_path(path, vendor_root)
    scenario = _infer_scenario(merged_record, path)
    metadata: dict[str, str | int | float | bool | None] = {
        "vendor": vendor,
        "relative_path": relative_path,
        "file_name": path.name,
        "source_repo": "mlcommons/inference_results_v6.0",
        "context_type": "benchmark_result",
    }
    if scenario:
        metadata["scenario"] = scenario

    row_id_parts = [
        "mlperf",
        vendor.lower(),
        normalize_identifier(hardware_name or path.parent.name),
        normalize_identifier(workload_name or path.stem),
        normalize_identifier(scenario or ""),
    ]

    return BenchmarkContextRow(
        row_id="_".join(part for part in row_id_parts if part),
        created_at=create_timestamp(),
        source="mlperf",
        benchmark_name="MLPerf Inference",
        workload_name=workload_name,
        hardware_name=hardware_name,
        software_stack=software_stack,
        metrics=metrics,
        units=units,
        url=None,
        notes="Local MLCommons inference result extraction",
        metadata=metadata,
    )


def _build_system_context_row(
    record: dict[str, Any],
    path: Path,
    vendor: str,
) -> BenchmarkContextRow | None:
    hardware_name = infer_hardware_from_record(record)
    software_stack = _software_stack_from_record(record)
    metadata = _system_metadata(record, vendor, path)
    if not hardware_name and not metadata:
        return None

    return BenchmarkContextRow(
        row_id=f"mlperf_{vendor.lower()}_system_{normalize_identifier(record.get('system_name') or hardware_name or path.stem)}",
        created_at=create_timestamp(),
        source="mlperf",
        benchmark_name="MLPerf Inference",
        workload_name=None,
        hardware_name=hardware_name,
        software_stack=software_stack,
        metrics={},
        units={},
        url=None,
        notes="Local MLCommons system description extraction",
        metadata=metadata,
    )


def _software_stack_from_record(record: dict[str, Any]) -> dict[str, str | int | float | bool | None]:
    software_stack: dict[str, str | int | float | bool | None] = {}
    framework_value = parse_mlcommons_scalar(record.get("framework"))
    if isinstance(framework_value, str):
        software_stack["framework"] = framework_value
        software_stack.update(_parse_framework_versions(framework_value))

    for key, aliases in {
        "backend": ("backend",),
        "driver": ("driver", "driver_version"),
        "system_type": ("system_type", "SystemType"),
        "cuda": ("cuda",),
        "tensorrt": ("tensorrt",),
        "pytorch": ("pytorch",),
    }.items():
        if key in software_stack:
            continue
        for alias in aliases:
            value = parse_mlcommons_scalar(record.get(alias))
            if value is not None:
                software_stack[key] = value
                break
    return software_stack


def _parse_framework_versions(framework_text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    patterns = {
        "cuda": r"CUDA\s+([0-9.]+)",
        "tensorrt": r"TensorRT(?:-LLM)?\s+([0-9.]+)",
        "pytorch": r"PyTorch\s+([0-9A-Za-z.+-]+)",
        "driver": r"Driver\s+([0-9.]+)",
        "backend": r"\b(vLLM|TensorRT-LLM|ROCm)\b",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, framework_text, re.I)
        if match:
            parsed[key] = match.group(1)
    return parsed


def _infer_units(
    metrics: dict[str, str | int | float | bool | None],
    record: dict[str, Any],
) -> dict[str, str]:
    units: dict[str, str] = {}
    raw_units = parse_mlcommons_scalar(record.get("Units"))
    unit_text = raw_units if isinstance(raw_units, str) else None
    for key in metrics:
        if key in {"qps", "samples_per_second"}:
            units[key] = unit_text or "samples/s"
        elif key == "tokens_per_second":
            units[key] = unit_text or "tokens/s"
        elif key == "latency_ms":
            units[key] = "ms"
        elif key == "power_w":
            units[key] = "W"
    return units


def _metric_from_result_field(value: str | int | float | bool) -> tuple[str | None, str | int | float | bool | None]:
    if isinstance(value, bool):
        return "result_valid", value
    if isinstance(value, int | float):
        return "qps", value
    numeric = parse_mlcommons_scalar(value)
    if isinstance(numeric, int | float):
        return "qps", numeric
    return None, None


def _extract_accuracy_value(value: str | int | float | bool) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
    if len(matches) == 1:
        number = float(matches[0])
        return int(number) if number.is_integer() else number
    return None


def _normalize_result_valid(value: str | int | float | bool) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value == 0
    lowered = value.strip().lower()
    if lowered in {"true", "pass", "passed", "valid", "closed"}:
        return True
    if lowered in {"false", "fail", "failed", "invalid", "open"}:
        return False
    return None


def _is_numeric_scalar(value: str | int | float | bool | None) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _map_metric_name(normalized_key: str) -> str | None:
    mapping = {
        "result": "qps",
        "samples_per_sec": "samples_per_second",
        "samples_per_second": "samples_per_second",
        "queries_per_sec": "qps",
        "queries_per_second": "qps",
        "tokens_per_sec": "tokens_per_second",
        "tokens_per_second": "tokens_per_second",
        "latency_ms": "latency_ms",
        "latency": "latency_ms",
        "latency_99_percentile": "latency_ms",
        "latency_90_percentile": "latency_ms",
        "latency_50_percentile": "latency_ms",
        "accuracy": "accuracy",
        "exact_match": "accuracy",
        "rougel": "accuracy",
        "rouge1": "accuracy",
        "rouge2": "accuracy",
        "vbench_score": "accuracy",
        "f1_hierarchical": "accuracy",
        "power_w": "power_w",
        "energy": "energy",
    }
    return mapping.get(normalized_key)


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _infer_scenario(record: dict[str, Any], path: Path) -> str | None:
    for key in ("Scenario", "scenario"):
        value = parse_mlcommons_scalar(record.get(key))
        if isinstance(value, str) and value:
            return value
    parts = path.as_posix().split("/")
    for part in ("Offline", "Server", "SingleStream", "Interactive", "MultiStream"):
        if part in parts:
            return part
    return None


def _match_system_record(
    path: Path,
    record: dict[str, Any],
    system_records: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in ("SystemName", "system_name", "Platform"):
        value = parse_mlcommons_scalar(record.get(key))
        if isinstance(value, str):
            for system_name, system_record in system_records.items():
                if system_name == value or system_name in value or value in system_name:
                    return system_record

    parts = path.as_posix().split("/")
    if "results" in parts:
        index = parts.index("results")
        if len(parts) > index + 1:
            return system_records.get(parts[index + 1])
    return None


def _dedupe_rows(rows: list[BenchmarkContextRow]) -> tuple[list[BenchmarkContextRow], int]:
    deduped: list[BenchmarkContextRow] = []
    seen: set[str] = set()
    duplicate_count = 0
    for row in rows:
        if row.row_id in seen:
            duplicate_count += 1
            continue
        seen.add(row.row_id)
        deduped.append(row)
    return deduped, duplicate_count


def _looks_like_result_file(path: Path) -> bool:
    normalized = path.as_posix().lower()
    return any(marker in normalized for marker in _LIKELY_FILE_MARKERS)


def _looks_like_system_desc_file(path: Path) -> bool:
    normalized = path.as_posix().lower()
    return "systems/" in normalized or "system_desc" in normalized


def _should_skip_file(path: Path) -> bool:
    normalized = path.as_posix().lower()
    if any(part in normalized for part in ("/code/", "/calibration/", "/documentation/")):
        return True
    if path.name.lower() == "mlperf_log_accuracy.json":
        return True
    if path.suffix.lower() == ".json" and "accuracy" in normalized and "summary" not in normalized:
        return True
    if path.suffix.lower() == ".txt" and "mlperf_log_summary" not in normalized:
        return True
    return not _looks_like_result_file(path)


def normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _safe_relative_path(path: Path, vendor_root: Path) -> str:
    try:
        return path.relative_to(vendor_root).as_posix()
    except ValueError:
        return path.as_posix()


def _system_metadata(
    record: dict[str, Any],
    vendor: str,
    path: Path,
) -> dict[str, str | int | float | bool | None]:
    metadata: dict[str, str | int | float | bool | None] = {
        "vendor": vendor,
        "relative_path": path.as_posix(),
        "file_name": path.name,
        "source_repo": "mlcommons/inference_results_v6.0",
        "context_type": "hardware_specs",
        "source_kind": "system_description",
    }
    for key in _SYSTEM_METADATA_KEYS:
        value = parse_mlcommons_scalar(record.get(key))
        if value is not None:
            metadata[key] = value
    return metadata


def _count_skipped_files(warnings: list[str]) -> int:
    total = 0
    for warning in warnings:
        match = re.search(r"count=(\d+)", warning)
        if match:
            total += int(match.group(1))
    return total


def _build_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# MLCommons Inference Intake Report",
        "",
        f"Generated at: {summary['generated_at']}",
        f"- Row count: {summary['row_count']}",
        f"- Validation status: {summary['validation_status']}",
        f"- Skipped file count: {summary['skipped_file_count']}",
        "",
        "## Source Summary",
    ]

    for vendor, vendor_summary in summary["source_summary"]["vendors"].items():
        lines.extend(
            [
                f"### {vendor}",
                f"- Exists: {vendor_summary['exists']}",
                f"- File count: {vendor_summary['file_count']}",
                f"- JSON count: {vendor_summary['json_count']}",
                f"- CSV count: {vendor_summary['csv_count']}",
                f"- TXT count: {vendor_summary['txt_count']}",
                f"- Likely result files: {vendor_summary['likely_result_file_count']}",
                f"- Likely system description files: {vendor_summary['likely_system_desc_count']}",
            ]
        )
        for warning in vendor_summary["warnings"]:
            lines.append(f"- Warning: {warning}")
        lines.append("")

    lines.append("## Warnings")
    if summary["warnings"]:
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.append("")

    if summary["row_count"] == 0:
        lines.extend(
            [
                "## Outcome",
                "- No clean MLCommons context rows could be extracted from the local files.",
                "",
            ]
        )

    return "\n".join(lines)
