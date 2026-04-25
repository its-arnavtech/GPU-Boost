# GPUBoost

GPUBoost is an open-source NVIDIA GPU optimization engine. Phase 1 provides a
focused Python CLI that inspects NVIDIA GPU, CUDA, PyTorch, and host system
information.

Phase 1 does not run benchmarks yet. It is the inspection foundation for later
benchmarking, recommendations, reports, LLM optimization, dashboard, and daemon
features.

## What Phase 1 Does

- Detects host OS, Python version, CPU details, and RAM
- Detects PyTorch installation and CUDA runtime availability
- Detects NVIDIA GPU details through NVML, PyTorch CUDA, and `nvidia-smi`
- Produces human-readable terminal output or valid JSON
- Handles CPU-only systems and missing NVIDIA tooling gracefully

## Install For Development

```bash
python -m venv .venv
pip install -e ".[dev]"
```

On Windows, activate the virtual environment first if desired:

```powershell
.\.venv\Scripts\activate
```

## Usage

```bash
gpuboost info
gpuboost info --json
python -m gpuboost info
```

Warnings are printed in human output and included in JSON output. They are not
fatal; GPUBoost should still report whatever information is available.

## Run Tests

```bash
pytest
```

The test suite does not require an NVIDIA GPU.

## Not Included Yet

- Benchmark code
- Dashboard code
- Daemon code
- Optimization recommendations
- Reports
- LLM optimization helpers
- Code patching

