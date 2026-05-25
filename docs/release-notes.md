# GPUBoost Release Notes

## 0.1.0 Checkpoint

GPUBoost `0.1.0` is a clean local-first release checkpoint after Phases 11-15.
It packages the current deterministic inspection, benchmark, analysis, dataset,
local model, trial, demo, setup, documentation, and repository hygiene workflows
without adding new major features.

## Completed Capabilities

- NVIDIA GPU and system inspection with JSON and human-readable CLI output.
- Synthetic benchmark suites for matrix multiplication, mixed precision, batch
  size sweeps, and DataLoader behavior.
- Rule-based optimization advice and static PyTorch code analysis.
- Reviewable patch diff generation for selected low-risk suggestions.
- Deterministic agent workflows with safe trial workspaces.
- Local model interfaces, baseline evaluation, neural training, artifact
  validation, and advisory prediction commands.
- Local history, comparison, dataset export, and demo report workflows.

## Phase 11 Dataset Readiness

Phase 11 added privacy-safe dataset assembly, validation, split assignment, and
training readiness checks. The dataset workflow uses safe feature extraction,
measured benchmark JSON pairs, controlled outcome grids, and readiness reports
before any Phase 12 model training is considered.

## Phase 12 Local Model Workflow

Phase 12 introduced the local model lifecycle: baseline evaluation, small local
neural training, explicit artifact saving, artifact validation, artifact
inspection, and prediction from saved local artifacts. Model predictions are
advisory-only and deterministic GPUBoost checks remain authoritative.

## Phase 13 Testing And Release Hardening

Phase 13 added subprocess CLI smoke tests, cross-platform PowerShell/path
hardening, security and data leak checks, artifact ignore checks, CLI UX
polish, model safety checks, and release-readiness documentation.

## Phase 14 Real-World Validation Demos

Phase 14 added lightweight realistic PyTorch demo workloads for CNN,
transformer, and DataLoader scenarios. Demo commands are discoverable from the
CLI, use synthetic data, write generated outputs only under ignored paths, and
support before/after comparison plus collect-outcomes pair generation.

## Phase 15 Final Polish

Phase 15 completed release polish across setup/install docs, README and
quickstart structure, release notes and versioning, GitHub issue/PR templates,
contributing and security policies, dependency/license/security review docs,
and a final project summary. It did not add product features, train models,
download data, call external APIs, or commit generated artifacts.

## Safety Guarantees

- GPUBoost does not apply patches automatically.
- No automatic patching is part of this release.
- Patch diffs are review-only.
- Trial mode modifies only temporary workspace copies.
- Model predictions are advisory-only and cannot apply patches.
- Trained artifact metadata keeps `patch_application_allowed=false`.
- Deterministic checks and measured benchmark evidence remain authoritative.
- CLI JSON redacts raw diffs and trial stdout/stderr by default.
- Normal test and docs workflows do not call external APIs.
- Normal test workflows do not require CUDA.
- No LLM fine-tuning is included.

## Known Limitations

- No bundled/default trained GPUBoost model is included.
- Model quality depends on available safe measured outcome rows.
- The neural training target is aspirational, not a release guarantee.
- Demo workloads use synthetic data and should not be overclaimed.
- Results vary by GPU, driver, CUDA version, PyTorch build, thermals, power
  mode, and background load.
- CPU fallback is useful for smoke testing but does not represent CUDA
  performance.
- Normal release validation does not prove production speedup.
- Automatic benchmark-command comparison and automatic patch application remain
  future work.

## Notable Commands

```bash
python -m gpuboost --version
python -m gpuboost info --json
python -m gpuboost benchmark --quick --recommend
python -m gpuboost analyze examples/bad_train_sample.txt --patch
python -m gpuboost agent optimize examples/bad_train_sample.txt --trial
python -m gpuboost model safety-check --json
python -m gpuboost model evaluate-baselines --json
python -m gpuboost model train-neural --json
python -m gpuboost demo real-world-info --json
python -m gpuboost demo real-world-pairs --json
python -m ruff check .
python -m pytest
```

## Generated Artifacts

No generated artifacts included. Generated artifacts are ignored under
`data/gpuboost/generated/`, raw intake data is ignored under
`data/gpuboost/raw/`, and local model weights, databases, caches, reports, and
secrets are excluded from release source control.
