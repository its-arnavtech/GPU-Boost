# GPUBoost

GPUBoost is an open-source NVIDIA(unaffliated) GPU optimization engine. Phase 1 provides a focused Python CLI that inspects NVIDIA GPU, CUDA, PyTorch, and host system
information. Phase 2 adds a synthetic benchmark suite for measuring common GPU
performance bottlenecks.

GPUBoost does not include optimization recommendations, code patching, reports,
LLM helpers, dashboard, or daemon features yet.

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

## Phase 2 Benchmarks

Phase 2 benchmarks use synthetic tensors and datasets only. They do not download
models or external data.

Available benchmarks:

- Matrix multiplication: compares FP32 and FP16 CUDA matmul throughput
- Mixed precision: compares FP32 and AMP synthetic training throughput
- Batch size sweep: measures synthetic image forward-pass throughput
- DataLoader: measures synthetic DataLoader throughput across workers and
  pinned memory settings

Run the quick benchmark subset:

```bash
gpuboost benchmark --quick
```

Run the full benchmark suite:

```bash
gpuboost benchmark
```

Select a CUDA device:

```bash
gpuboost benchmark --device 0
```

Emit JSON:

```bash
gpuboost benchmark --json
gpuboost benchmark --quick --json
```

Benchmark results vary with laptop power mode, thermals, background GPU load,
drivers, CUDA version, PyTorch build, and whether the system is plugged in.
CPU-only systems return skipped CUDA benchmark results instead of crashing.

## Run Tests

```bash
pytest
```

The test suite does not require an NVIDIA GPU.

## Not Included Yet

- Dashboard code
- Daemon code
- Optimization recommendations
- Reports
- LLM optimization helpers
- Code patching
