# Dependency Review

GPUBoost `0.1.1` keeps a small dependency surface for local inspection,
analysis, advisory safety checks, and release validation, with PyTorch moved to
optional extras for benchmark and model workflows.

## Runtime Dependencies

Runtime dependencies are declared in `pyproject.toml`:

- `psutil`: used for local CPU, memory, and system inspection.
- `nvidia-ml-py`: used for NVIDIA GPU/NVML inspection when NVIDIA tooling is
  available.
- `rich`: used for human-readable CLI rendering when available.

## Optional And Development Dependencies

Optional dependency groups:

- `benchmark`: installs PyTorch for benchmark execution.
- `model`: installs PyTorch for local model artifact and training commands.
- `all`: installs the full optional PyTorch-backed feature set.

The `dev` optional dependency group contains:

- `pytest`: local test runner.
- `ruff`: local lint/static formatting check.

No additional release dependency is required for Phase 15E.

## PyTorch, NumPy, And CUDA Behavior

PyTorch is optional because GPUBoost's lightweight install paths can inspect the
environment, compare saved benchmark JSON, analyze source, inspect history, and
run setup or safety checks without importing it. Commands that execute
benchmarks or local model workflows still require a working PyTorch install.
CUDA hardware is optional for normal setup and test validation: CPU-only
systems should skip CUDA-specific benchmark work or report CUDA as unavailable
instead of failing the whole CLI.

GPUBoost does not directly import NumPy for supported runtime workflows, so
NumPy is not declared as a standalone runtime dependency. Avoiding eager torch
imports is enough to prevent PyTorch's optional NumPy warning from appearing in
lightweight CLI commands.

Local neural training commands require a working PyTorch install but are not
part of normal release validation. This review did not train models.

## External Services And Data

- No external API dependency is required for normal GPUBoost use, tests, or
  release checks.
- No external LLM API is required.
- No dataset download requirement exists for normal tests or docs validation.
- Existing manifests and examples are local fixtures; generated datasets remain
  local files.

## Generated Artifacts

Generated artifacts are ignored by default. The `.gitignore` excludes:

- `data/gpuboost/generated/`
- `data/gpuboost/raw/`
- model checkpoints and weights such as `*.pt`, `*.pth`, `*.ckpt`,
  `*.safetensors`, and `*.onnx`
- serialized model/data files such as `*.pkl` and `*.joblib`
- local databases such as `*.db`, `*.sqlite`, and `*.sqlite3`
- reports, caches, logs, temporary directories, virtual environments, and
  common secret file patterns

Release source control should contain source, docs, tests, examples, and
reviewed fixture/manifest files only. No generated artifacts should be included
in the release checkpoint.

## License Status

The repository includes `LICENSE` with the MIT License, and `pyproject.toml`
declares `license = { text = "MIT" }` with the matching MIT classifier.
