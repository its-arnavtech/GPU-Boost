"""Tests for Phase 2 benchmark schemas."""

from gpuboost.schemas.benchmark_result import (
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkSuiteResult,
)


def test_benchmark_schemas_serialize_to_dict() -> None:
    result = BenchmarkResult(
        name="Example",
        status="ok",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        duration_sec=1.0,
        metrics=[BenchmarkMetric(name="throughput", value=123.4, unit="items/sec")],
        warnings=[],
        error=None,
    )
    suite = BenchmarkSuiteResult(
        generated_at="2026-01-01T00:00:01+00:00",
        gpu_name="NVIDIA Test GPU",
        cuda_available=True,
        device_index=0,
        results=[result],
        warnings=["test warning"],
    )

    data = suite.to_dict()

    assert data["gpu_name"] == "NVIDIA Test GPU"
    assert data["results"][0]["metrics"][0]["name"] == "throughput"
    assert data["warnings"] == ["test warning"]

