"""Local TechPowerUp GPU HTML intake helpers for GPUBoost Phase 11."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from gpuboost.dataset.benchmark_importers import normalize_source_name
from gpuboost.dataset.export import export_validation_report
from gpuboost.dataset.validation import validate_benchmark_context_rows
from gpuboost.schemas.dataset import BenchmarkContextRow, create_timestamp

try:  # pragma: no cover - exercised indirectly when bs4 is installed.
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - stdlib fallback is covered in tests.
    BeautifulSoup = None


_SOURCE_NAME = "techpowerup"
_BENCHMARK_NAME = "TechPowerUp GPU Database"
_NOTES = "Local TechPowerUp GPU database HTML extraction"
_SCALAR_SPEC_FIELDS = (
    "gpu_name",
    "architecture",
    "generation_or_series",
    "manufacturer",
    "release_date",
    "launch_price",
    "gpu_chip",
    "process_size_nm",
    "transistors_million",
    "die_size_mm2",
    "memory_size_mb",
    "memory_type",
    "memory_bus_width_bit",
    "memory_bandwidth_gbps",
    "base_clock_mhz",
    "boost_clock_mhz",
    "memory_clock_mhz",
    "cuda_cores",
    "tensor_cores",
    "rt_cores",
    "tmus",
    "rops",
    "sm_count",
    "bus_interface",
    "tdp_w",
    "power_connectors",
    "directx",
    "opengl",
    "opencl",
    "vulkan",
    "shader_model",
    "cuda_capability",
    "is_laptop_gpu",
    "vram_gb",
    "series_family",
)


def parse_number(value: str) -> int | float | None:
    """Extract the first numeric value from text."""

    if not value:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", value.replace("_", ""))
    if match is None:
        return None
    number_text = match.group(0).replace(",", "")
    if "." in number_text:
        try:
            return float(number_text)
        except ValueError:
            return None
    try:
        return int(number_text)
    except ValueError:
        return None


def parse_memory_size_mb(value: str) -> int | None:
    """Normalize a memory size string into megabytes."""

    if not value:
        return None
    match = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(gb|gib|mb|mib)\b", value, re.I)
    if match is None:
        return None
    amount = float(match.group(1).replace(",", ""))
    unit = match.group(2).lower()
    if unit in {"gb", "gib"}:
        return int(round(amount * 1024))
    return int(round(amount))


def parse_bandwidth_gbps(value: str) -> float | None:
    """Normalize a bandwidth string into GB/s."""

    if not value:
        return None
    match = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(tb/s|gb/s)\b", value, re.I)
    if match is None:
        raw = parse_number(value)
        return float(raw) if raw is not None else None
    amount = float(match.group(1).replace(",", ""))
    unit = match.group(2).lower()
    if unit == "tb/s":
        return amount * 1024.0
    return amount


def parse_mhz(value: str) -> int | None:
    """Normalize a clock string into MHz."""

    raw = parse_number(value)
    if raw is None:
        return None
    return int(round(float(raw)))


def parse_watts(value: str) -> int | None:
    """Normalize a power string into watts."""

    raw = parse_number(value)
    if raw is None:
        return None
    return int(round(float(raw)))


def parse_bits(value: str) -> int | None:
    """Normalize a bit width string into bits."""

    raw = parse_number(value)
    if raw is None:
        return None
    return int(round(float(raw)))


def parse_process_nm(value: str) -> int | None:
    """Normalize a process-size string into nanometers."""

    raw = parse_number(value)
    if raw is None:
        return None
    return int(round(float(raw)))


def parse_die_size_mm2(value: str) -> float | None:
    """Normalize a die-size string into square millimeters."""

    raw = parse_number(value)
    if raw is None:
        return None
    return float(raw)


def derive_series_family(gpu_name: str) -> str:
    """Derive a normalized RTX family bucket from a GPU name."""

    text = gpu_name.lower()
    if re.search(r"\brtx\s*30\d{2}\b", text):
        return "rtx_30"
    if re.search(r"\brtx\s*40\d{2}\b", text):
        return "rtx_40"
    if re.search(r"\brtx\s*50\d{2}\b", text):
        return "rtx_50"
    return "unknown"


def derive_is_laptop_gpu(gpu_name: str) -> bool:
    """Return whether a GPU name appears to describe a mobile part."""

    text = gpu_name.lower()
    return any(
        token in text
        for token in (" laptop", " mobile", " max-q", " max q", " notebook")
    ) or text.startswith("laptop ")


def extract_techpowerup_gpu_specs(
    html_text: str,
) -> dict[str, str | int | float | bool | None]:
    """Extract a conservative scalar GPU spec dictionary from local HTML."""

    extracted = _extract_html_content(html_text)
    spec_map = extracted["spec_map"]
    title = extracted["title"]
    heading = extracted["h1"]
    og_title = extracted["og_title"]

    gpu_name = _clean_gpu_name(
        _first_non_empty(
            _lookup_value(spec_map, "card name", "graphics card", "product name"),
            og_title,
            heading,
            title,
        )
    )
    architecture = _lookup_value(spec_map, "architecture")
    generation_or_series = _normalize_generation(
        _first_non_empty(
            _lookup_value(spec_map, "generation", "series"),
            gpu_name,
        )
    )
    manufacturer = _derive_manufacturer(gpu_name)
    release_date = _lookup_value(spec_map, "release date", "launch date")
    launch_price = _parse_launch_price(_lookup_value(spec_map, "launch price", "msrp"))

    gpu_chip = _lookup_value(spec_map, "gpu name", "gpu chip", "graphics processor")
    process_size_nm = parse_process_nm(_lookup_value(spec_map, "process size"))
    transistors_million = _parse_transistors_million(
        _lookup_value(spec_map, "transistors")
    )
    die_size_mm2 = parse_die_size_mm2(_lookup_value(spec_map, "die size"))

    memory_size_mb = parse_memory_size_mb(_lookup_value(spec_map, "memory size"))
    memory_type = _lookup_value(spec_map, "memory type")
    memory_bus_width_bit = parse_bits(_lookup_value(spec_map, "memory bus"))
    memory_bandwidth_gbps = parse_bandwidth_gbps(_lookup_value(spec_map, "bandwidth"))

    base_clock_mhz = parse_mhz(_lookup_value(spec_map, "base clock"))
    boost_clock_mhz = parse_mhz(_lookup_value(spec_map, "boost clock"))
    memory_clock_mhz = parse_mhz(_lookup_value(spec_map, "memory clock"))

    cuda_cores = _parse_int_value(
        _first_non_empty(
            _lookup_value(spec_map, "cuda cores"),
            _lookup_value(spec_map, "shading units"),
        )
    )
    tensor_cores = _parse_int_value(_lookup_value(spec_map, "tensor cores"))
    rt_cores = _parse_int_value(_lookup_value(spec_map, "rt cores"))
    tmus = _parse_int_value(_lookup_value(spec_map, "tmus"))
    rops = _parse_int_value(_lookup_value(spec_map, "rops"))
    sm_count = _parse_int_value(_lookup_value(spec_map, "sm count", "sms"))

    bus_interface = _lookup_value(spec_map, "bus interface")
    tdp_w = parse_watts(_lookup_value(spec_map, "tdp", "board power"))
    power_connectors = _lookup_value(
        spec_map,
        "power connectors",
        "external power",
        "power connector",
    )

    directx = _lookup_value(spec_map, "directx")
    opengl = _lookup_value(spec_map, "opengl")
    opencl = _lookup_value(spec_map, "opencl")
    vulkan = _lookup_value(spec_map, "vulkan")
    shader_model = _lookup_value(spec_map, "shader model")
    cuda_capability = _parse_cuda_capability(_lookup_value(spec_map, "cuda"))

    vram_gb = round(memory_size_mb / 1024.0, 3) if memory_size_mb is not None else None
    series_family = derive_series_family(gpu_name or "")
    is_laptop_gpu = derive_is_laptop_gpu(gpu_name or "")

    specs: dict[str, str | int | float | bool | None] = {
        "gpu_name": gpu_name,
        "architecture": architecture,
        "generation_or_series": generation_or_series,
        "manufacturer": manufacturer,
        "release_date": release_date,
        "launch_price": launch_price,
        "gpu_chip": gpu_chip,
        "process_size_nm": process_size_nm,
        "transistors_million": transistors_million,
        "die_size_mm2": die_size_mm2,
        "memory_size_mb": memory_size_mb,
        "memory_type": memory_type,
        "memory_bus_width_bit": memory_bus_width_bit,
        "memory_bandwidth_gbps": memory_bandwidth_gbps,
        "base_clock_mhz": base_clock_mhz,
        "boost_clock_mhz": boost_clock_mhz,
        "memory_clock_mhz": memory_clock_mhz,
        "cuda_cores": cuda_cores,
        "tensor_cores": tensor_cores,
        "rt_cores": rt_cores,
        "tmus": tmus,
        "rops": rops,
        "sm_count": sm_count,
        "bus_interface": bus_interface,
        "tdp_w": tdp_w,
        "power_connectors": power_connectors,
        "directx": directx,
        "opengl": opengl,
        "opencl": opencl,
        "vulkan": vulkan,
        "shader_model": shader_model,
        "cuda_capability": cuda_capability,
        "is_laptop_gpu": is_laptop_gpu,
        "vram_gb": vram_gb,
        "series_family": series_family,
    }
    return {key: specs.get(key) for key in _SCALAR_SPEC_FIELDS}


def parse_techpowerup_gpu_html(
    html_text: str,
    source_path: str | None = None,
) -> BenchmarkContextRow | None:
    """Parse one local TechPowerUp GPU HTML page into a benchmark context row."""

    specs = extract_techpowerup_gpu_specs(html_text)
    gpu_name = specs.get("gpu_name")
    if not isinstance(gpu_name, str) or not gpu_name.strip():
        return None

    extracted = _extract_html_content(html_text)
    url = _safe_canonical_url(extracted["canonical_url"])
    metadata = {
        key: value
        for key, value in specs.items()
        if value is not None and not (isinstance(value, str) and not value.strip())
    }
    metadata["context_type"] = "hardware_specs"
    metadata["source_kind"] = "gpu_specs"
    if source_path:
        metadata["source_file"] = Path(source_path).name

    row_id_source = normalize_source_name(gpu_name)
    return BenchmarkContextRow(
        row_id=f"{_SOURCE_NAME}_{row_id_source}",
        created_at=create_timestamp(),
        source=_SOURCE_NAME,
        benchmark_name=_BENCHMARK_NAME,
        workload_name=None,
        hardware_name=gpu_name,
        software_stack={},
        metrics={},
        units={},
        url=url,
        notes=_NOTES,
        metadata=metadata,
    )


def import_techpowerup_html_folder(
    input_dir: str,
) -> tuple[list[BenchmarkContextRow], list[str]]:
    """Import local TechPowerUp HTML files from a directory tree."""

    directory = Path(input_dir)
    if not directory.exists():
        return [], [f"Missing input directory: {directory}"]
    if not directory.is_dir():
        return [], [f"Input path is not a directory: {directory}"]

    rows: list[BenchmarkContextRow] = []
    warnings: list[str] = []
    seen_gpu_names: set[str] = set()

    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in {".html", ".htm"} or not path.is_file():
            continue
        try:
            html_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            warnings.append(f"File could not be read: {path} ({exc})")
            continue

        try:
            row = parse_techpowerup_gpu_html(html_text, source_path=str(path))
        except Exception as exc:  # pragma: no cover - defensive safety path.
            warnings.append(f"File could not be parsed: {path} ({exc})")
            continue

        if row is None:
            warnings.append(f"No GPU specs found: {path}")
            continue

        gpu_name = (row.hardware_name or "").strip()
        dedupe_key = gpu_name.casefold()
        if dedupe_key in seen_gpu_names:
            warnings.append(f"Duplicate GPU name skipped: {gpu_name} ({path})")
            continue

        seen_gpu_names.add(dedupe_key)
        rows.append(row)

    return rows, warnings


def run_techpowerup_intake(
    input_dir: str = "data/gpuboost/raw/techpowerup/GPU_Info",
    output_dir: str = "data/gpuboost/generated",
    manifest_dir: str = "data/gpuboost/manifests",
) -> dict[str, Any]:
    """Run the local TechPowerUp intake and write dataset artifacts."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    manifest_path = Path(manifest_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path.mkdir(parents=True, exist_ok=True)

    file_count = 0
    if input_path.exists() and input_path.is_dir():
        file_count = sum(
            1
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".html", ".htm"}
        )

    rows, warnings = import_techpowerup_html_folder(str(input_path))
    validation_report = validate_benchmark_context_rows(rows)

    jsonl_path = output_path / "techpowerup_gpu_specs.jsonl"
    validation_path = output_path / "techpowerup_gpu_specs_validation_report.json"
    report_json_path = manifest_path / "techpowerup_gpu_specs_intake_report.json"
    report_md_path = manifest_path / "techpowerup_gpu_specs_intake_report.md"

    _write_jsonl([row.to_dict() for row in rows], jsonl_path)
    export_validation_report(validation_report, str(validation_path))

    series_counts = Counter(
        str(row.metadata.get("series_family", "unknown"))
        for row in rows
    )
    missing_fields_summary = {
        field: sum(1 for row in rows if row.metadata.get(field) is None)
        for field in _SCALAR_SPEC_FIELDS
        if any(row.metadata.get(field) is None for row in rows)
    }
    validation_status = {
        "status": validation_report.status,
        "row_count": validation_report.row_count,
        "valid_row_count": validation_report.valid_row_count,
        "invalid_row_count": validation_report.invalid_row_count,
        "issue_count": len(validation_report.issues),
    }

    intake_report: dict[str, Any] = {
        "generated_at": create_timestamp(),
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "file_count": file_count,
        "row_count": len(rows),
        "validation": validation_status,
        "warnings": warnings,
        "output_files": {
            "jsonl": str(jsonl_path),
            "validation_report": str(validation_path),
            "intake_report_json": str(report_json_path),
            "intake_report_md": str(report_md_path),
        },
        "extracted_series_counts": dict(series_counts),
        "missing_fields_summary": missing_fields_summary,
    }

    _write_json(intake_report, report_json_path)
    report_md_path.write_text(_build_markdown_report(intake_report), encoding="utf-8")
    return intake_report


def _extract_html_content(html_text: str) -> dict[str, Any]:
    if BeautifulSoup is not None:
        return _extract_with_bs4(html_text)
    return _extract_with_html_parser(html_text)


def _extract_with_bs4(html_text: str) -> dict[str, Any]:
    soup = BeautifulSoup(html_text, "html.parser")
    spec_map: dict[str, str] = {}

    for row in soup.find_all("tr"):
        cells = [_clean_whitespace(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        _add_cell_pairs(spec_map, cells)

    for definition_list in soup.find_all("dl"):
        terms = definition_list.find_all("dt")
        values = definition_list.find_all("dd")
        for term, value in zip(terms, values, strict=False):
            _add_label_value(
                spec_map,
                _clean_whitespace(term.get_text(" ", strip=True)),
                _clean_whitespace(value.get_text(" ", strip=True)),
            )

    canonical_url = None
    og_title = None
    canonical_link = soup.find("link", rel=lambda value: value and "canonical" in value)
    if canonical_link and canonical_link.get("href"):
        canonical_url = str(canonical_link["href"]).strip()
    if canonical_url is None:
        og_url = soup.find("meta", attrs={"property": "og:url"})
        if og_url and og_url.get("content"):
            canonical_url = str(og_url["content"]).strip()
    og_title_tag = soup.find("meta", attrs={"property": "og:title"})
    if og_title_tag and og_title_tag.get("content"):
        og_title = str(og_title_tag["content"]).strip()

    return {
        "title": _clean_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else None,
        "h1": _extract_bs4_heading(soup),
        "og_title": _clean_whitespace(og_title) if og_title else None,
        "canonical_url": canonical_url,
        "spec_map": spec_map,
    }


class _FallbackHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title: list[str] = []
        self.h1: list[str] = []
        self.current_tag: str | None = None
        self.link_attrs: list[dict[str, str]] = []
        self.meta_attrs: list[dict[str, str]] = []
        self.h1_attrs: list[dict[str, str]] = []
        self._current_h1_is_gpu = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.current_tag = tag.lower()
        if self.current_tag == "link":
            self.link_attrs.append({key.lower(): value or "" for key, value in attrs})
        elif self.current_tag == "meta":
            self.meta_attrs.append({key.lower(): value or "" for key, value in attrs})
        elif self.current_tag == "h1":
            attr_map = {key.lower(): value or "" for key, value in attrs}
            self.h1_attrs.append(attr_map)
            class_name = attr_map.get("class", "")
            self._current_h1_is_gpu = "gpudb-name" in class_name

    def handle_endtag(self, tag: str) -> None:
        if self.current_tag == tag.lower():
            self.current_tag = None
        if tag.lower() == "h1":
            self._current_h1_is_gpu = False

    def handle_data(self, data: str) -> None:
        if self.current_tag == "title":
            self.title.append(data)
        elif self.current_tag == "h1" and self._current_h1_is_gpu:
            self.h1.append(data)


def _extract_with_html_parser(html_text: str) -> dict[str, Any]:
    parser = _FallbackHTMLParser()
    parser.feed(html_text)

    spec_map: dict[str, str] = {}
    for match in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", html_text, re.I | re.S):
        row_html = match.group(1)
        cells = [
            _clean_html_text(cell.group(1))
            for cell in re.finditer(r"<(?:th|td)\b[^>]*>(.*?)</(?:th|td)>", row_html, re.I | re.S)
        ]
        _add_cell_pairs(spec_map, cells)

    dt_values = [
        _clean_html_text(match.group(1))
        for match in re.finditer(r"<dt\b[^>]*>(.*?)</dt>", html_text, re.I | re.S)
    ]
    dd_values = [
        _clean_html_text(match.group(1))
        for match in re.finditer(r"<dd\b[^>]*>(.*?)</dd>", html_text, re.I | re.S)
    ]
    for label, value in zip(dt_values, dd_values, strict=False):
        _add_label_value(spec_map, label, value)

    canonical_url = None
    og_title = None
    for attrs in parser.link_attrs:
        rel = attrs.get("rel", "").lower()
        href = attrs.get("href", "").strip()
        if "canonical" in rel and href:
            canonical_url = href
            break
    for attrs in parser.meta_attrs:
        if attrs.get("property", "").lower() == "og:title":
            content = attrs.get("content", "").strip()
            if content:
                og_title = content
                break

    return {
        "title": _clean_whitespace("".join(parser.title)) or None,
        "h1": _clean_whitespace(" ".join(parser.h1)) or None,
        "og_title": _clean_whitespace(og_title) if og_title else None,
        "canonical_url": canonical_url,
        "spec_map": spec_map,
    }


def _extract_bs4_heading(soup: BeautifulSoup) -> str | None:
    preferred_heading = soup.select_one("h1.gpudb-name")
    if preferred_heading is not None:
        text = _clean_whitespace(preferred_heading.get_text(" ", strip=True))
        return text or None
    if soup.h1 is not None:
        text = _clean_whitespace(soup.h1.get_text(" ", strip=True))
        return text or None
    return None


def _add_cell_pairs(spec_map: dict[str, str], cells: list[str]) -> None:
    if len(cells) < 2:
        return
    limit = len(cells) - 1
    for index in range(0, limit, 2):
        _add_label_value(spec_map, cells[index], cells[index + 1])


def _add_label_value(spec_map: dict[str, str], label: str, value: str) -> None:
    clean_label = _normalize_label(label)
    clean_value = _clean_whitespace(value)
    if not clean_label or not clean_value:
        return
    if len(clean_label) > 80 or len(clean_value) > 400:
        return
    spec_map.setdefault(clean_label, clean_value)


def _lookup_value(spec_map: dict[str, str], *labels: str) -> str | None:
    for label in labels:
        value = spec_map.get(_normalize_label(label))
        if value:
            return value
    return None


def _normalize_label(value: str | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(normalized.split())


def _clean_html_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return _clean_whitespace(unescape(without_tags))


def _clean_whitespace(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(unescape(value).replace("\xa0", " ").split())


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def _clean_gpu_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = _clean_whitespace(value)
    text = re.sub(r"\s*\|\s*TechPowerUp.*$", "", text, flags=re.I)
    text = re.sub(r"\s+Specs.*$", "", text, flags=re.I)
    text = re.sub(r"\s+GPU Database.*$", "", text, flags=re.I)
    return text.strip() or None


def _normalize_generation(value: str | None) -> str | None:
    if value is None:
        return None
    text = _clean_whitespace(value)
    match = re.search(r"\b(?:rtx|geforce)\s*(30|40|50)\b", text, re.I)
    if match:
        return f"RTX {match.group(1)}"
    match = re.search(r"\brtx\s*(30|40|50)\d{2}\b", text, re.I)
    if match:
        return f"RTX {match.group(1)}"
    return text or None


def _derive_manufacturer(gpu_name: str | None) -> str | None:
    if not gpu_name:
        return None
    text = gpu_name.lower()
    if "nvidia" in text or "geforce" in text or "rtx" in text:
        return "NVIDIA"
    if "radeon" in text or "amd" in text:
        return "AMD"
    if "intel" in text or "arc" in text:
        return "Intel"
    return None


def _parse_transistors_million(value: str | None) -> int | None:
    if not value:
        return None
    raw = parse_number(value)
    if raw is None:
        return None
    return int(round(float(raw)))


def _parse_launch_price(value: str | None) -> int | float | None:
    if not value:
        return None
    raw = parse_number(value)
    if raw is None:
        return None
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    return raw


def _parse_int_value(value: str | None) -> int | None:
    if not value:
        return None
    raw = parse_number(value)
    if raw is None:
        return None
    return int(round(float(raw)))


def _parse_cuda_capability(value: str | None) -> float | str | None:
    if not value:
        return None
    match = re.search(r"\d+(?:\.\d+)?", value)
    if match:
        number = float(match.group(0))
        if number.is_integer():
            return int(number)
        return number
    cleaned = _clean_whitespace(value)
    return cleaned or None


def _safe_canonical_url(url: str | None) -> str | None:
    if not url:
        return None
    text = url.strip()
    if re.match(r"^https?://", text, re.I):
        return text
    return None


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")


def _write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# TechPowerUp GPU Specs Intake Report",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## Summary",
        f"- Input directory: {report['input_dir']}",
        f"- Output directory: {report['output_dir']}",
        f"- HTML files seen: {report['file_count']}",
        f"- Rows written: {report['row_count']}",
        f"- Validation status: {report['validation']['status']}",
        "",
        "## Output Files",
        f"- JSONL: {report['output_files']['jsonl']}",
        f"- Validation report: {report['output_files']['validation_report']}",
        f"- Intake report JSON: {report['output_files']['intake_report_json']}",
        f"- Intake report Markdown: {report['output_files']['intake_report_md']}",
        "",
        "## Extracted Series Counts",
    ]

    series_counts = report.get("extracted_series_counts", {})
    if series_counts:
        for key, value in sorted(series_counts.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")

    lines.extend(["", "## Missing Fields Summary"])
    missing_fields = report.get("missing_fields_summary", {})
    if missing_fields:
        for key, value in sorted(missing_fields.items()):
            lines.append(f"- {key}: missing in {value} rows")
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings"])
    warnings = report.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Next Recommended Step",
            "- Parse/import these local files through the GPUBoost Phase 11 importers.",
            "",
        ]
    )
    return "\n".join(lines)


def benchmark_context_row_to_dict(row: BenchmarkContextRow) -> dict[str, Any]:
    """Return a benchmark context row as a plain dictionary."""

    return asdict(row)
