"""Tests for the TechPowerUp local HTML intake helpers."""

from __future__ import annotations

import json

from gpuboost.dataset.techpowerup_importer import (
    derive_is_laptop_gpu,
    derive_series_family,
    extract_techpowerup_gpu_specs,
    import_techpowerup_html_folder,
    parse_bandwidth_gbps,
    parse_bits,
    parse_memory_size_mb,
    parse_mhz,
    parse_techpowerup_gpu_html,
    parse_watts,
    run_techpowerup_intake,
)


def test_parse_gpu_name_from_title_and_h1() -> None:
    row = parse_techpowerup_gpu_html(_sample_html(), source_path="gpu.html")

    assert row is not None
    assert row.hardware_name == "NVIDIA GeForce RTX 4090"


def test_parse_memory_size_to_mb() -> None:
    assert parse_memory_size_mb("24 GB GDDR6X") == 24576
    assert parse_memory_size_mb("24576 MB") == 24576


def test_parse_bandwidth_to_gbps() -> None:
    assert parse_bandwidth_gbps("1008 GB/s") == 1008.0
    assert parse_bandwidth_gbps("1.5 TB/s") == 1536.0


def test_parse_clocks_to_mhz() -> None:
    assert parse_mhz("2235 MHz") == 2235
    assert parse_mhz("1313 MHz (21 Gbps effective)") == 1313


def test_parse_tdp_watts() -> None:
    assert parse_watts("450 W") == 450


def test_parse_bus_width_bits() -> None:
    assert parse_bits("384 bit") == 384


def test_derive_rtx_series_family() -> None:
    assert derive_series_family("NVIDIA GeForce RTX 3090") == "rtx_30"
    assert derive_series_family("NVIDIA GeForce RTX 4090") == "rtx_40"
    assert derive_series_family("NVIDIA GeForce RTX 5090") == "rtx_50"
    assert derive_series_family("NVIDIA A100") == "unknown"


def test_derive_laptop_gpu() -> None:
    assert derive_is_laptop_gpu("NVIDIA GeForce RTX 4090 Laptop GPU") is True
    assert derive_is_laptop_gpu("GeForce RTX 4080 Mobile") is True
    assert derive_is_laptop_gpu("GeForce RTX 3080 Max-Q") is True
    assert derive_is_laptop_gpu("NVIDIA GeForce RTX 4090") is False


def test_extract_specs_from_techpowerup_like_table() -> None:
    specs = extract_techpowerup_gpu_specs(_sample_html())

    assert specs["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert specs["architecture"] == "Ada Lovelace"
    assert specs["memory_size_mb"] == 24576
    assert specs["memory_bandwidth_gbps"] == 1008.0
    assert specs["base_clock_mhz"] == 2235
    assert specs["tdp_w"] == 450
    assert specs["cuda_cores"] == 16384
    assert specs["series_family"] == "rtx_40"
    assert specs["vram_gb"] == 24.0


def test_parse_html_returns_benchmark_context_row() -> None:
    row = parse_techpowerup_gpu_html(_sample_html(), source_path="gpu.html")

    assert row is not None
    assert row.source == "techpowerup"
    assert row.benchmark_name == "TechPowerUp GPU Database"
    assert row.metrics == {}
    assert row.software_stack == {}
    assert row.metadata["context_type"] == "hardware_specs"
    assert row.metadata["source_kind"] == "gpu_specs"
    assert row.metadata["gpu_name"] == "NVIDIA GeForce RTX 4090"


def test_no_raw_html_in_row_dict() -> None:
    row = parse_techpowerup_gpu_html(_sample_html(), source_path="gpu.html")

    assert row is not None
    payload = row.to_dict()
    serialized = json.dumps(payload)

    assert "<html" not in serialized.lower()
    assert "TechPowerUp GPU Database" in serialized
    assert "<table" not in serialized.lower()


def test_import_folder_parses_multiple_files(tmp_path) -> None:
    input_dir = tmp_path / "GPU_Info"
    input_dir.mkdir()
    (input_dir / "4090.html").write_text(_sample_html(), encoding="utf-8")
    (input_dir / "5090.html").write_text(
        _sample_html(
            gpu_name="NVIDIA GeForce RTX 5090",
            generation="RTX 50",
            memory_size="32768 MB",
            bandwidth="1792 GB/s",
            boost_clock="2650 MHz",
        ),
        encoding="utf-8",
    )

    rows, warnings = import_techpowerup_html_folder(str(input_dir))

    assert len(rows) == 2
    assert warnings == []
    assert {row.hardware_name for row in rows} == {
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 5090",
    }


def test_duplicate_gpu_names_warn_and_dedupe(tmp_path) -> None:
    input_dir = tmp_path / "GPU_Info"
    input_dir.mkdir()
    html = _sample_html()
    (input_dir / "first.html").write_text(html, encoding="utf-8")
    (input_dir / "second.html").write_text(html, encoding="utf-8")

    rows, warnings = import_techpowerup_html_folder(str(input_dir))

    assert len(rows) == 1
    assert any("Duplicate GPU name skipped" in warning for warning in warnings)


def test_missing_folder_returns_warning_and_empty_rows(tmp_path) -> None:
    rows, warnings = import_techpowerup_html_folder(str(tmp_path / "missing"))

    assert rows == []
    assert len(warnings) == 1
    assert "Missing input directory" in warnings[0]


def test_malformed_html_skipped_safely(tmp_path) -> None:
    input_dir = tmp_path / "GPU_Info"
    input_dir.mkdir()
    (input_dir / "broken.html").write_text("<html><body><div>not a gpu page", encoding="utf-8")

    rows, warnings = import_techpowerup_html_folder(str(input_dir))

    assert rows == []
    assert any("No GPU specs found" in warning for warning in warnings)


def test_run_techpowerup_intake_writes_outputs(tmp_path) -> None:
    input_dir = tmp_path / "GPU_Info"
    output_dir = tmp_path / "generated"
    manifest_dir = tmp_path / "manifests"
    input_dir.mkdir()
    (input_dir / "4090.html").write_text(_sample_html(), encoding="utf-8")

    report = run_techpowerup_intake(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        manifest_dir=str(manifest_dir),
    )

    jsonl_path = output_dir / "techpowerup_gpu_specs.jsonl"
    validation_path = output_dir / "techpowerup_gpu_specs_validation_report.json"
    report_json_path = manifest_dir / "techpowerup_gpu_specs_intake_report.json"
    report_md_path = manifest_dir / "techpowerup_gpu_specs_intake_report.md"

    assert report["row_count"] == 1
    assert jsonl_path.exists()
    assert validation_path.exists()
    assert report_json_path.exists()
    assert report_md_path.exists()

    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    intake_report = json.loads(report_json_path.read_text(encoding="utf-8"))

    assert records[0]["hardware_name"] == "NVIDIA GeForce RTX 4090"
    assert "raw_source" not in json.dumps(records[0])
    assert validation["status"] == "warning"
    assert intake_report["validation"]["status"] == "warning"
    assert records[0]["metadata"]["context_type"] == "hardware_specs"


def _sample_html(
    gpu_name: str = "NVIDIA GeForce RTX 4090",
    generation: str = "RTX 40",
    memory_size: str = "24576 MB",
    bandwidth: str = "1008 GB/s",
    boost_clock: str = "2520 MHz",
) -> str:
    return f"""
<!doctype html>
<html>
  <head>
    <title>{gpu_name} Specs | TechPowerUp GPU Database</title>
    <link rel="canonical" href="https://www.techpowerup.com/gpu-specs/example" />
  </head>
  <body>
    <h1>{gpu_name}</h1>
    <table>
      <tr><td>Architecture</td><td>Ada Lovelace</td><td>Generation</td><td>{generation}</td></tr>
      <tr><td>GPU Name</td><td>AD102</td><td>Process Size</td><td>5 nm</td></tr>
      <tr><td>Transistors</td><td>76,300 million</td><td>Die Size</td><td>608 mm²</td></tr>
      <tr><td>Release Date</td><td>Sep 20, 2022</td><td>Launch Price</td><td>$1599</td></tr>
      <tr><td>Memory Size</td><td>{memory_size}</td><td>Memory Type</td><td>GDDR6X</td></tr>
      <tr><td>Memory Bus</td><td>384 bit</td><td>Bandwidth</td><td>{bandwidth}</td></tr>
      <tr><td>Base Clock</td><td>2235 MHz</td><td>Boost Clock</td><td>{boost_clock}</td></tr>
      <tr><td>Memory Clock</td><td>1313 MHz</td><td>Bus Interface</td><td>PCIe 4.0 x16</td></tr>
      <tr><td>Shading Units</td><td>16384</td><td>Tensor Cores</td><td>512</td></tr>
      <tr><td>RT Cores</td><td>128</td><td>TMUs</td><td>512</td></tr>
      <tr><td>ROPs</td><td>176</td><td>SM Count</td><td>128</td></tr>
      <tr><td>TDP</td><td>450 W</td><td>Power Connectors</td><td>1x 16-pin</td></tr>
      <tr><td>DirectX</td><td>12 Ultimate</td><td>OpenGL</td><td>4.6</td></tr>
      <tr><td>OpenCL</td><td>3.0</td><td>Vulkan</td><td>1.4</td></tr>
      <tr><td>CUDA</td><td>8.9</td><td>Shader Model</td><td>6.8</td></tr>
    </table>
  </body>
</html>
"""
