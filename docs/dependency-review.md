# Dependency Review

GPUBoost `0.1.0` keeps a small dependency surface for local inspection,
benchmarking, analysis, advisory model workflows, and release validation.

## Runtime Dependencies

Runtime dependencies are declared in `pyproject.toml`:

- `torch`: used for PyTorch environment inspection, CUDA checks, synthetic
  benchmark workloads, and local neural training commands.
- `psutil`: used for local CPU, memory, and system inspection.
- `nvidia-ml-py`: used for NVIDIA GPU/NVML inspection when NVIDIA tooling is
  available.
- `rich`: used for human-readable CLI rendering when available.

## Optional And Development Dependencies

The `dev` optional dependency group contains:

- `pytest`: local test runner.
- `ruff`: local lint/static formatting check.

No additional release dependency is required for Phase 15E.

## PyTorch And CUDA Behavior

PyTorch is part of the declared runtime environment because GPUBoost inspects
PyTorch, runs PyTorch-based benchmarks, and supports local model workflows.
CUDA hardware is optional for normal setup and test validation: CPU-only
systems should skip CUDA-specific benchmark work or report CUDA as unavailable
instead of failing the whole CLI.

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
