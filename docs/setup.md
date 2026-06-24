# GPUBoost Setup

This guide is for a fresh local checkout. The commands below install GPUBoost
for development, run lightweight validation, and avoid generated artifact churn.

## Prerequisites

- Python 3.9 or newer
- Git
- A shell:
  - Windows PowerShell on Windows
  - Bash, zsh, or another POSIX shell on macOS/Linux
- Optional: an NVIDIA GPU, CUDA-capable PyTorch build, and NVIDIA drivers for
  GPU benchmarks

CUDA is not required to install GPUBoost, run the setup checks, run static code
analysis, or run the test suite. CPU-only systems should report skipped CUDA
benchmark work instead of crashing.

## Clone

```bash
git clone <repo-url>
cd GPU-Boost
```

## Create a Virtual Environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If PowerShell blocks activation, either run commands through
`.\.venv\Scripts\python.exe -m ...` or allow activation for the current process:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

## Install

Install GPUBoost and the development tools:

```bash
python -m pip install -e ".[dev]"
```

This installs the package in editable mode plus `pytest` and `ruff`. Do not add
extra dependencies unless a specific local workflow requires them.

If you want the full benchmark and local model workflow in the same environment,
install the optional extra too:

```bash
python -m pip install -e ".[dev,all]"
```

## Verify Installation

Run the lightweight setup doctor:

```bash
python -m gpuboost doctor
python -m gpuboost doctor --json
```

The doctor checks the Python version, core imports, optional PyTorch
availability, dev tool imports, and generated-artifact ignore patterns when a
GPUBoost source repository is available. It does not require CUDA, run
benchmarks, train models, call external APIs, or write generated artifacts.

Run the standard local checks:

```bash
python -m ruff check .
python -m pytest
```

## Basic CLI Smoke Commands

These commands are safe first checks for a new install:

```bash
python -m gpuboost --help
python -m gpuboost info
python -m gpuboost info --json
python -m gpuboost analyze examples/bad_train_sample.txt
python -m gpuboost demo real-world-info
python -m gpuboost model safety-check --json
```

`python -m gpuboost benchmark --quick` is also available, but it may skip CUDA
work on CPU-only systems and should not be treated as a setup failure.

After editable installation, the console script should also be available:

```bash
gpuboost doctor
gpuboost info
```

## PyTorch and CUDA Notes

GPUBoost can inspect and validate setup without CUDA. PyTorch availability is
reported by `doctor` and `info`; CUDA availability is reported separately. If
your environment needs benchmark execution, advisory model artifacts, or a
specific CPU-only/CUDA-enabled PyTorch build, install the optional `all` extra
or follow the official PyTorch install selector for your platform, then rerun:

```bash
python -m gpuboost doctor
python -m gpuboost info
```

Do not treat `cuda_available=false` as a failed setup check unless your goal is
to run GPU benchmarks on a CUDA device.

## Generated Data and Artifacts

Generated outputs should stay local and ignored by Git. Important ignored paths
and patterns include:

```text
data/gpuboost/generated/
data/gpuboost/raw/
*.pt
*.pth
*.safetensors
*.onnx
*.pkl
*.joblib
*.db
*.sqlite
*.sqlite3
```

Do not commit benchmark outputs, model weights, generated model artifacts, local
SQLite databases, caches, logs, or environment files.
