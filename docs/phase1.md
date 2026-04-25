# GPUBoost Phase 1

Phase 1 establishes GPUBoost as a clean Python CLI package for NVIDIA GPU,
PyTorch/CUDA, and host system inspection.

This phase is intentionally minimal. It provides the command entry point,
profile schemas, inspector modules, documentation, and basic tests. Full
benchmark logic, recommendations, reports, LLM optimization, dashboard code,
and daemon behavior are reserved for later phases.

## Commands

```bash
gpuboost info
gpuboost info --json
```

## Phase 1 Scope

- Package and CLI structure
- GPU inspection through NVML, PyTorch CUDA, and `nvidia-smi`
- System inspection through `platform` and `psutil`
- PyTorch/CUDA environment inspection
- Serializable GPUBoost profile schema
- Human-readable and JSON-oriented output paths
- Import and schema tests

## Out of Scope

- Benchmarks
- Optimization recommendations
- HTML or JSON report generation
- Dashboard
- Daemon or background monitoring
- Code patching
