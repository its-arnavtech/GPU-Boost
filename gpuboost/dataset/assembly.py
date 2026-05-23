"""Unified Phase 11 dataset assembly for future local model training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from gpuboost.dataset.export import (
    build_dataset_manifest,
    export_benchmark_context_jsonl,
    export_dataset_jsonl,
    export_manifest,
    export_validation_report,
)
from gpuboost.dataset.history_converter import history_records_to_dataset_rows
from gpuboost.dataset.readiness import (
    analyze_training_readiness,
    write_training_readiness_reports,
)
from gpuboost.dataset.splitting import assign_dataset_splits, split_counts as dataset_split_counts
from gpuboost.dataset.validation import (
    validate_benchmark_context_rows,
    validate_dataset_rows,
)
from gpuboost.history.store import default_history_db_path, list_history_runs
from gpuboost.schemas.dataset import (
    BenchmarkContextRow,
    DatasetLabel,
    DatasetPrivacyFlags,
    DatasetRow,
)


def assemble_training_dataset(
    history_db_path: str | None = None,
    external_context_paths: list[str] | None = None,
    output_dir: str = "data/gpuboost/generated",
    manifest_dir: str = "data/gpuboost/manifests",
    assign_splits: bool = True,
    seed: int = 42,
    outcome_dataset_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble history rows and safe context rows into Phase 11 outputs."""

    output_path = Path(output_dir)
    manifest_path = Path(manifest_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    history_rows = _load_history_dataset_rows(history_db_path, warnings)
    outcome_rows = _load_outcome_dataset_rows(
        outcome_dataset_paths,
        output_path,
    )
    dataset_rows, dataset_source_counts = _merge_dataset_rows(
        history_rows=history_rows,
        outcome_rows=outcome_rows,
        warnings=warnings,
    )
    context_rows = _load_external_context_rows(external_context_paths, warnings, output_path)

    if assign_splits and dataset_rows:
        dataset_rows, split_summary = assign_dataset_splits(dataset_rows, seed=seed)
        split_summary_dict = split_summary.to_dict()
    else:
        split_summary_dict = None

    dataset_validation = validate_dataset_rows(dataset_rows)
    context_validation = validate_benchmark_context_rows(context_rows)

    training_dataset_path = output_path / "training_dataset.jsonl"
    training_manifest_path = output_path / "training_dataset_manifest.json"
    training_validation_path = output_path / "training_dataset_validation_report.json"
    benchmark_context_path = output_path / "benchmark_context.jsonl"

    export_dataset_jsonl(dataset_rows, str(training_dataset_path))
    manifest = build_dataset_manifest(
        dataset_rows,
        dataset_name="training_dataset",
        dataset_version="phase11.7",
        warnings=warnings,
    )
    export_manifest(manifest, str(training_manifest_path))
    export_validation_report(dataset_validation, str(training_validation_path))

    if context_rows:
        export_benchmark_context_jsonl(context_rows, str(benchmark_context_path))

    label_counts = Counter(row.label.value for row in dataset_rows)
    labeled_count = sum(1 for row in dataset_rows if row.is_labeled())
    readiness = analyze_training_readiness(dataset_rows, context_rows=context_rows)
    if output_dir == "data/gpuboost/generated" and manifest_dir == "data/gpuboost/manifests":
        readiness = _attach_external_intake_status(readiness)
    readiness_json_path, readiness_md_path = write_training_readiness_reports(
        readiness,
        manifest_dir=str(manifest_path),
    )

    output_files = {
        "training_dataset_jsonl": str(training_dataset_path),
        "training_dataset_manifest": str(training_manifest_path),
        "training_dataset_validation_report": str(training_validation_path),
        "training_readiness_report_json": readiness_json_path,
        "training_readiness_report_md": readiness_md_path,
    }
    if context_rows:
        output_files["benchmark_context_jsonl"] = str(benchmark_context_path)

    return {
        "history_row_count": dataset_source_counts["history"],
        "outcome_row_count": dataset_source_counts["outcome"],
        "dataset_row_count": len(dataset_rows),
        "benchmark_context_row_count": len(context_rows),
        "labeled_count": labeled_count,
        "unlabeled_count": len(dataset_rows) - labeled_count,
        "label_counts": dict(label_counts),
        "split_counts": dataset_split_counts(dataset_rows),
        "validation_status": dataset_validation.status,
        "context_validation_status": context_validation.status,
        "readiness_status": readiness["status"],
        "output_files": output_files,
        "warnings": warnings,
        "split_assignment_summary": split_summary_dict,
    }


def load_dataset_rows_jsonl(path: str) -> list[DatasetRow]:
    """Load dataset rows from a UTF-8 JSONL file if it exists."""

    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[DatasetRow] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            if _looks_like_benchmark_context_payload(payload):
                continue
            rows.append(_dataset_row_from_dict(payload))
    return rows


def load_benchmark_context_rows_jsonl(path: str) -> list[BenchmarkContextRow]:
    """Load benchmark context rows from a UTF-8 JSONL file if it exists."""

    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[BenchmarkContextRow] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(_benchmark_context_row_from_dict(payload))
    return rows


def write_json(path: str, data: dict[str, Any]) -> None:
    """Write JSON to disk with stable formatting."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _load_history_dataset_rows(
    history_db_path: str | None,
    warnings: list[str],
) -> list[DatasetRow]:
    db_path = Path(history_db_path) if history_db_path is not None else default_history_db_path()
    if not db_path.exists():
        warnings.append(f"History DB not found: {db_path}")
        return []

    history_summary = list_history_runs(limit=1_000_000, db_path=db_path)
    if not history_summary.runs:
        warnings.append(f"No history runs found in: {db_path}")
        return []
    return history_records_to_dataset_rows(history_summary.runs)


def _load_outcome_dataset_rows(
    outcome_dataset_paths: list[str] | None,
    output_dir: Path,
) -> list[DatasetRow]:
    known_paths = _default_outcome_dataset_paths(output_dir)
    combined_paths = _unique_paths_preserving_order(
        [*known_paths, *(outcome_dataset_paths or [])]
    )

    rows: list[DatasetRow] = []
    for path in combined_paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        rows.extend(load_dataset_rows_jsonl(path))
    return rows


def _merge_dataset_rows(
    history_rows: list[DatasetRow],
    outcome_rows: list[DatasetRow],
    warnings: list[str],
) -> tuple[list[DatasetRow], dict[str, int]]:
    rows: list[DatasetRow] = []
    seen_row_ids: set[str] = set()
    source_counts = {"history": 0, "outcome": 0}
    duplicate_counts = {"history": 0, "outcome": 0}

    for source_name, source_rows in (
        ("history", history_rows),
        ("outcome", outcome_rows),
    ):
        for row in source_rows:
            if row.row_id and row.row_id in seen_row_ids:
                duplicate_counts[source_name] += 1
                continue
            if row.row_id:
                seen_row_ids.add(row.row_id)
            rows.append(row)
            source_counts[source_name] += 1

    if duplicate_counts["history"]:
        warnings.append(
            "Skipped "
            f"{duplicate_counts['history']} duplicate history-derived "
            "dataset rows by row_id."
        )
    if duplicate_counts["outcome"]:
        warnings.append(
            "Skipped "
            f"{duplicate_counts['outcome']} duplicate controlled outcome "
            "dataset rows by row_id."
        )

    return rows, source_counts


def _load_external_context_rows(
    external_context_paths: list[str] | None,
    warnings: list[str],
    output_dir: Path,
) -> list[BenchmarkContextRow]:
    known_paths = _default_context_paths(output_dir)
    combined_paths = _unique_preserving_order([*known_paths, *(external_context_paths or [])])

    rows: list[BenchmarkContextRow] = []
    seen_row_ids: set[str] = set()
    for path in combined_paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        duplicate_count = 0
        for row in load_benchmark_context_rows_jsonl(path):
            row_id = row.row_id
            if row_id in seen_row_ids:
                duplicate_count += 1
                continue
            seen_row_ids.add(row_id)
            rows.append(row)
        if duplicate_count:
            warnings.append(
                f"Skipped {duplicate_count} duplicate benchmark context rows from {path}."
            )
    return rows


def _default_context_paths(output_dir: Path) -> list[str]:
    consolidated_path = output_dir / "benchmark_context.jsonl"
    fallback_paths: list[Path] = [
        output_dir / "techpowerup_gpu_specs.jsonl",
        output_dir / "mlcommons_inference_context.jsonl",
    ]
    if consolidated_path.exists():
        consolidated_mtime = consolidated_path.stat().st_mtime
        paths = [str(consolidated_path)]
        for path in fallback_paths:
            if path.exists() and path.stat().st_mtime > consolidated_mtime:
                paths.append(str(path))
        return paths

    return [str(path) for path in fallback_paths if path.exists()]


def _default_outcome_dataset_paths(output_dir: Path) -> list[str]:
    outcome_path = output_dir / "outcomes" / "outcome_dataset.jsonl"
    return [str(outcome_path)] if outcome_path.exists() else []


def _dataset_row_from_dict(data: dict[str, Any]) -> DatasetRow:
    label_data = data.get("label") if isinstance(data.get("label"), dict) else {}
    privacy_data = data.get("privacy") if isinstance(data.get("privacy"), dict) else {}

    return DatasetRow(
        row_id=str(data.get("row_id") or ""),
        created_at=str(data.get("created_at") or ""),
        schema_version=str(data.get("schema_version") or "dataset.row.v1"),
        source=str(data.get("source") or ""),
        row_type=str(data.get("row_type") or ""),
        hardware=_safe_dataset_scalar_dict(data.get("hardware")),
        workload=_safe_dataset_scalar_dict(data.get("workload")),
        features=_safe_dataset_scalar_dict(data.get("features")),
        metrics=_safe_dataset_scalar_dict(data.get("metrics")),
        label=DatasetLabel(
            value=str(label_data.get("value") or "unknown"),
            source=str(label_data.get("source") or "unknown"),
            confidence=(
                label_data.get("confidence")
                if isinstance(label_data.get("confidence"), int | float)
                else None
            ),
            notes=_safe_text_or_none(label_data.get("notes")),
        ),
        privacy=DatasetPrivacyFlags(
            contains_raw_source=bool(privacy_data.get("contains_raw_source", False)),
            contains_raw_diff=bool(privacy_data.get("contains_raw_diff", False)),
            contains_stdout=bool(privacy_data.get("contains_stdout", False)),
            contains_stderr=bool(privacy_data.get("contains_stderr", False)),
            contains_sensitive_path=bool(privacy_data.get("contains_sensitive_path", False)),
            notes=[str(item) for item in privacy_data.get("notes", []) if isinstance(item, str)],
        ),
        split=str(data.get("split")) if data.get("split") is not None else None,
        quality_score=data.get("quality_score") if isinstance(data.get("quality_score"), int | float) else None,
        warnings=[
            item for item in data.get("warnings", []) if _is_safe_string_value(item)
        ],
        metadata=_safe_dataset_scalar_dict(data.get("metadata")),
    )


def _benchmark_context_row_from_dict(data: dict[str, Any]) -> BenchmarkContextRow:
    return BenchmarkContextRow(
        row_id=str(data.get("row_id") or ""),
        created_at=str(data.get("created_at") or ""),
        schema_version=str(data.get("schema_version") or "dataset.benchmark_context.v1"),
        source=str(data.get("source") or ""),
        benchmark_name=str(data.get("benchmark_name") or ""),
        workload_name=str(data.get("workload_name")) if data.get("workload_name") is not None else None,
        hardware_name=str(data.get("hardware_name")) if data.get("hardware_name") is not None else None,
        software_stack=_scalar_dict(data.get("software_stack")),
        metrics=_scalar_dict(data.get("metrics")),
        units=_string_dict(data.get("units")),
        url=str(data.get("url")) if data.get("url") is not None else None,
        notes=str(data.get("notes")) if data.get("notes") is not None else None,
        metadata=_scalar_dict(data.get("metadata")),
    )


def _scalar_dict(value: Any) -> dict[str, str | int | float | bool | None]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, str | int | float | bool) or item is None
    }


def _safe_dataset_scalar_dict(
    value: Any,
) -> dict[str, str | int | float | bool | None]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if _is_safe_dataset_key(key) and _is_safe_dataset_scalar_value(item)
    }


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if item is not None
    }


def _looks_like_benchmark_context_payload(data: dict[str, Any]) -> bool:
    schema_version = str(data.get("schema_version") or "")
    return schema_version == "dataset.benchmark_context.v1" or (
        "benchmark_name" in data and "label" not in data
    )


def _is_safe_dataset_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return normalized not in {
        "file_contents",
        "raw_diff",
        "raw_source",
        "source_code",
        "stderr",
        "stdout",
        "unified_diff",
    }


def _is_safe_dataset_scalar_value(value: Any) -> bool:
    if not isinstance(value, str | int | float | bool) and value is not None:
        return False
    if isinstance(value, str):
        return _is_safe_string_value(value)
    return True


def _is_safe_string_value(value: Any) -> bool:
    return isinstance(value, str) and not (
        _looks_like_unified_diff(value) or _looks_like_python_source(value)
    )


def _safe_text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if _is_safe_string_value(value):
        return value
    return None


def _looks_like_unified_diff(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith("--- ") and "\n+++ " in stripped


def _looks_like_python_source(value: str) -> bool:
    if len(value) < 120 or "\n" not in value:
        return False
    source_markers = ("def ", "class ", "import ", "from ")
    marker_count = sum(1 for marker in source_markers if marker in value)
    return marker_count >= 2 and ("    " in value or "\n\t" in value)


def _attach_external_intake_status(report: dict[str, Any]) -> dict[str, Any]:
    external_intake: dict[str, Any] = {}

    techpowerup_report_path = Path("data/gpuboost/manifests/techpowerup_gpu_specs_intake_report.json")
    if techpowerup_report_path.exists():
        try:
            techpowerup_report = json.loads(techpowerup_report_path.read_text(encoding="utf-8"))
            external_intake["techpowerup"] = {
                "row_count": techpowerup_report.get("row_count"),
                "validation_status": techpowerup_report.get("validation_status")
                or techpowerup_report.get("validation", {}).get("status"),
            }
        except json.JSONDecodeError:
            external_intake["techpowerup"] = {"warning": "Could not parse TechPowerUp intake report."}

    mlcommons_intake_report_path = Path("data/gpuboost/manifests/mlcommons_inference_intake_report.json")
    if mlcommons_intake_report_path.exists():
        try:
            mlcommons_intake_report = json.loads(mlcommons_intake_report_path.read_text(encoding="utf-8"))
            external_intake["mlcommons_inference"] = {
                "row_count": mlcommons_intake_report.get("row_count"),
                "validation_status": mlcommons_intake_report.get("validation_status"),
                "warnings": mlcommons_intake_report.get("warnings", []),
            }
        except json.JSONDecodeError:
            external_intake["mlcommons_inference"] = {"warning": "Could not parse MLCommons intake report."}

    mlcommons_report_path = Path("data/gpuboost/manifests/third_party_raw_inventory.json")
    if mlcommons_report_path.exists():
        try:
            mlcommons_report = json.loads(
                mlcommons_report_path.read_text(encoding="utf-8-sig")
            )
            external_intake["mlcommons_raw_inventory"] = {
                "generated_at": mlcommons_report.get("generated_at"),
                "repositories_collected": len(mlcommons_report.get("repositories_collected", [])),
            }
        except json.JSONDecodeError:
            external_intake["mlcommons_raw_inventory"] = {"warning": "Could not parse third-party raw inventory report."}

    if not any(source == "mlperf" for source in report["context"]["context_source_counts"]):
        report["recommendations"].append("Parse MLCommons local files into BenchmarkContextRow records.")

    pytorch_results_path = Path("data/gpuboost/raw/pytorch/benchmark/results")
    if not pytorch_results_path.exists():
        message = "PyTorch benchmark results folder was missing from collected repo; verify source path or skip for now."
        external_intake["pytorch_benchmark"] = {"warning": message}
        report["recommendations"].append(message)

    if external_intake:
        report["external_intake"] = external_intake
    report["recommendations"] = _unique_preserving_order(report["recommendations"])
    return report


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _unique_paths_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = _path_dedupe_key(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _path_dedupe_key(value: str) -> str:
    try:
        return str(Path(value).resolve()).casefold()
    except OSError:
        return str(Path(value)).casefold()
