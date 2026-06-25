# GPUBoost 0.2.0

GPUBoost is a local-first Python CLI for inspecting NVIDIA GPU environments,
running bounded synthetic benchmarks, analyzing PyTorch scripts, and preparing
reviewable optimization advice.

> **Release status.** GPUBoost `0.2.0` includes human-approved agentic
> optimization: it proposes an exact deterministic diff, requires explicit
> approval, applies only approved deterministic edits, creates backups, validates
> the result, and can roll back automatically. Model output cannot directly
> modify files, unapproved patching is forbidden, Git commits/pushes are never
> automatic, and fully unattended autonomous patching is not supported.

- PyPI: <https://pypi.org/project/gpuboost/>
- Repository: <https://github.com/its-arnavtech/GPU-Boost>
- License: [MIT License](LICENSE); see [LICENSE](LICENSE)
- Maturity: pre-alpha local tooling. Useful today, but performance results are
  hardware and workload specific.

## What It Does

- Reports Python, OS, CPU, RAM, NVIDIA GPU, CUDA, cuDNN, and PyTorch details.
- Runs quick synthetic benchmark suites for matrix multiplication, mixed
  precision, batch-size sweeps, and DataLoader behavior.
- Produces deterministic rule-based recommendations from benchmark metrics.
- Statically analyzes PyTorch-like source files without importing or executing
  them.
- Builds reviewable patch plans and unified diffs for conservative suggestions.
- Validates generated patch plans in temporary trial workspaces when requested.
- Prepares approval-gated agentic optimization runs that can apply selected
  deterministic edits only after explicit human approval.
- Compares supported GPUBoost benchmark JSON files.
- Stores local history only when explicitly requested.
- Provides local dataset, model artifact, and advisory model workflows.
- Describes synthetic real-world demo workloads that use local data only.

## What It Does Not Do

- No automatic patch application to original source files; no unapproved patch
  application is allowed.
- Approved source edits require an immutable plan digest, the original file
  hash, and an explicit human confirmation phrase.
- No guaranteed speedups.
- Model output is advisory; it cannot apply patches or approve changes.
- Deterministic checks remain authoritative.
- `patch_application_allowed=false` is part of the model safety contract.
- GPU/CUDA is optional for core workflows such as help, doctor, static analysis,
  compare, history, demo discovery, and safety checks.
- It does not call external APIs, upload source, download datasets, train models,
  or run heavy benchmark grids during normal use.

## Installation

Base install:

```powershell
pip install gpuboost
```

Full ML/benchmark extras:

```powershell
pip install "gpuboost[all]"
```

Targeted extras:

```powershell
pip install "gpuboost[benchmark]"
pip install "gpuboost[model]"
```

The base package depends on `psutil`, `nvidia-ml-py`, and `rich`. The
`benchmark`, `model`, and `all` extras add PyTorch and NumPy.

## Quickstart

These commands were validated during the post-release audit:

```powershell
python -m gpuboost --version
python -m gpuboost doctor --json
python -m gpuboost info --json
python -m gpuboost analyze examples\bad_train_sample.txt --json
python -m gpuboost agent optimize examples\bad_train_sample.txt --json
python -m gpuboost agent optimize examples\bad_train_sample.txt --trial --json
python -m gpuboost benchmark --quick --json --recommend
python -m gpuboost demo real-world-info --json
python -m gpuboost demo real-world-pairs --json
```

`examples/bad_train_sample.txt` is intentionally a `.txt` file so formatters and
linters do not treat it as project Python code.

## Core Workflows

Inspect the environment:

```powershell
python -m gpuboost doctor
python -m gpuboost doctor --json
python -m gpuboost info
python -m gpuboost info --json
```

Analyze a script:

```powershell
python -m gpuboost analyze train.py
python -m gpuboost analyze train.py --json
python -m gpuboost analyze train.py --patch
```

Run the deterministic agent:

```powershell
python -m gpuboost agent optimize train.py --json
python -m gpuboost agent optimize train.py --trial --json
python -m gpuboost agent optimize train.py --save-history
```

Prepare a human-approved agentic optimization run:

```powershell
python -m gpuboost agent optimize train.py --prepare
python -m gpuboost agent show-plan <run_id>
python -m gpuboost agent approve <run_id> --confirm "APPLY <plan>"
python -m gpuboost agent apply <run_id>
python -m gpuboost agent rollback <run_id>
```

This lifecycle is included in GPUBoost `0.2.0`.

- `--prepare` writes only an ignored `.gpuboost/runs/<run_id>.json` audit record
  and never modifies source.
- `agent show-plan` displays the exact reviewable diff before any approval.
- `agent approve` is mandatory: it records a confirmation phrase bound to the
  immutable plan digest, the original file hash, and the selected action IDs.
- `agent apply` re-validates the original SHA-256 hash, creates a backup, applies
  only approved deterministic edits, validates the result, and rolls back
  automatically if validation fails.
- `agent rollback` restores the pre-application backup and verifies the restored
  hash matches the original.
- Model output is advisory only and can never modify files directly.
- No Git commit or push is ever performed automatically.
- Unattended or autonomous patching is not supported; a human must approve every
  applied change.

Compare supported GPUBoost benchmark JSON files:

```powershell
python -m gpuboost compare baseline.json optimized.json
python -m gpuboost compare baseline.json optimized.json --json
```

Use local history:

```powershell
python -m gpuboost history list
python -m gpuboost history show <run_id>
python -m gpuboost history compare <baseline_run_id> <optimized_run_id>
```

Use dataset and model commands:

```powershell
python -m gpuboost dataset collect-outcomes data\gpuboost\experiments\pairs.json
python -m gpuboost model evaluate-baselines --json
python -m gpuboost model train-neural --json
python -m gpuboost model list-artifacts
python -m gpuboost model safety-check --json
```

Model training is experimental and local. Saved model artifacts are generated
files and remain advisory only.

The model workflow uses safe training data loading, safe feature extraction,
baseline model comparison, MLP training, explicit artifact save steps, direct
artifact prediction, and advisory-only agent integration. It does not fine-tune
an LLM and does not call external services.

Use demo discovery:

```powershell
python -m gpuboost demo real-world-info --json
python -m gpuboost demo real-world-pairs --json
```

Demo workloads use synthetic local data and are not proof that a change will
speed up a production workload.

## Does It Actually Help? Measured Results

GPUBoost is advisory: it does not make code faster by itself. It measures your
GPU and recommends changes (mixed precision, batch sizing, Tensor-Core-friendly
shapes, DataLoader settings); the speedup comes from applying the **right**
recommendation to the **right** workload. The numbers below were measured on
2026-06-25 on one machine (NVIDIA GeForce RTX 4060 Laptop GPU, CUDA 12.8,
PyTorch 2.11.0+cu128, Windows 11). They are evidence from this machine, not a
universal promise — re-measure on your own hardware with your own workload.

### `pip install gpuboost` works

Verified in a clean, isolated virtual environment installing from PyPI:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install gpuboost     # installs gpuboost 0.2.0
.\.venv\Scripts\python -m gpuboost --version        # gpuboost 0.2.0
.\.venv\Scripts\python -m gpuboost doctor --json    # all required checks pass
.\.venv\Scripts\python -m gpuboost analyze train.py --json
```

`pip install gpuboost` installed `gpuboost 0.2.0` (a ~215 KB pure-Python wheel
plus `rich`, `psutil`, `nvidia-ml-py`). Every **required** `doctor` check passed;
`analyze` correctly flagged all five issues in the sample script (`num_workers=0`,
missing `pin_memory`, missing AMP autocast, a `.item()` sync inside the loop, and
missing `cudnn.benchmark`); `model safety-check` reported `ok`. PyTorch is an
optional extra (`pip install "gpuboost[all]"`), needed only for the GPU benchmark
and model commands.

### Where it helps (measured)

| Recommendation | Workload where it helps | Measured speedup |
|---|---|---:|
| Tensor Cores / FP16 | 4096×4096 matmul, FP16 vs FP32 | **4.40×** (31.5 vs 7.2 TFLOPS) |
| Larger batch size | CNN throughput sweep | **4.64×** at batch 32 vs batch 1 |
| Mixed precision (AMP) | compute-heavy CNN (64–256 channels), batch ≥ 64 | **1.47–1.52×** |
| Mixed precision (AMP) | 4096-dim MLP, batch 256 | **1.17×** |
| Mixed precision (AMP) | small CNN at large batch (256) | **1.34×** |

Domain summary: the advice pays off on **compute-bound, matrix-heavy training** —
large `Linear`/`Conv`/attention layers with dimensions divisible by 8/16, run at
a batch size large enough to saturate the GPU, on a Tensor-Core GPU.

### Where it does not help — or hurts (measured)

| Setting | Workload where it backfires | Measured result |
|---|---|---:|
| AMP | tiny transformer (d_model=32, 1 layer), any batch | **0.70–0.77×** (slower) |
| AMP | small CNN / MLP at small batch (16–64) | **0.54–0.81×** (slower) |
| `num_workers > 0` | synthetic in-memory dataset on Windows | **~100× slower** (37–54 vs 5194 samples/s) |
| `pin_memory=True` | data already cheap to transfer / in-memory | **0.45×** (≈2.2× slower) |
| Batch size past the sweet spot | batch 64/128 vs 32 | slower (1531/1802 vs 2170 img/s) |

Domain summary: the advice does little or backfires on **tiny / launch-bound
models**, **very small batches**, and — especially on Windows — **DataLoader
`num_workers`/`pin_memory` tweaks for in-memory or trivial datasets**, where
process-spawn and host-pinning overhead dominate. GPUBoost's own
`benchmark --recommend` measured this on this machine and downgraded the
DataLoader-workers recommendation with an explicit warning, so trust the measured
numbers over the generic rule.

### Reproduce it

```powershell
python -m gpuboost benchmark --quick --json --recommend
python examples\real_world\pytorch_cnn_baseline.py --benchmark-json --device cuda --steps 50
python examples\real_world\pytorch_cnn_optimized.py --benchmark-json --device cuda --steps 50
```

The bundled demo workloads are intentionally tiny, so their per-step times are
noisy run-to-run; use `--steps 50` or more (not `--quick`'s 2 steps) and compare
medians of several runs. The `benchmark --recommend` suite uses larger, steadier
workloads and is the more reliable signal.

## Published 0.1.2 Audit Evidence

Validated on one local machine.

Fresh audit evidence is in [docs/post-release-audit.md](docs/post-release-audit.md).
The table below is the published `0.1.2` post-release audit evidence. It is
hardware/workload specific and should not be generalized.

| Item | Result |
|---|---:|
| GPUBoost version | 0.1.2 |
| Python | 3.12.10 |
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MB VRAM |
| CUDA available | true |
| PyTorch | 2.11.0+cu128 |
| Ruff | all checks passed |
| Pytest | 1044 passed |
| Static analysis sample | 5 findings |
| Agent optimize sample | 5 completed actions |
| Agent trial sample | passed; original file unchanged |
| Trial syntax-check duration | 0.024519 seconds |
| Quick benchmark sections | 4 successful sections |
| Best FP32 matmul | 7.167453 TFLOPS |
| Best FP16 matmul | 31.337318 TFLOPS |
| FP16/FP32 matmul ratio | 4.372169x |
| AMP synthetic throughput | 24402.147 samples/sec |
| FP32 synthetic throughput | 15938.796 samples/sec |
| AMP synthetic ratio | 1.530991x |
| Best batch sweep result | batch size 8, 2607.100 images/sec |
| CNN demo baseline | 353.727 samples/sec |
| CNN demo optimized | 862.134 samples/sec |

The benchmark and demo results used synthetic or controlled data on one laptop.
They are evidence that the commands ran successfully on that machine, not a
promise that other workloads will see the same ratios.

## Current 0.2.0 Validation

Current release validation after the human-approved agentic apply changes:

| Item | Result |
|---|---:|
| Package version | 0.2.0 |
| Ruff | all checks passed |
| Pytest | 1061 passed, 1 skipped |
| Expected skip | Windows symlink privilege-dependent test |

The published `0.1.2` audit remains above as historical evidence for the prior
review-first release. The `0.2.0` validation count is separate and includes the
human-approved agentic apply workflow.

## Safety Model

- Patch suggestions are reviewable diffs.
- Trial mode applies patch plans only to temporary copies.
- The original source file is not modified by trial mode.
- Human-approved agentic optimization never mutates source during `--prepare`.
- `agent apply` requires a persisted approval tied to the plan digest and
  original file hash, creates a pre-application backup, validates the result,
  and rolls back automatically on validation or acceptance-policy failure.
- No unapproved patch application is allowed.
- Model output cannot directly modify files; only approved deterministic plan
  edits are ever written.
- No Git commit or push is performed automatically by any agent command.
- Unattended autonomous patching is not supported; every applied change requires
  an explicit human confirmation phrase.
- Static analysis parses code without importing or executing the target script.
- Raw diffs and trial stdout/stderr are redacted from agent JSON by default.
- Model output is advisory and cannot override tests, trials, or benchmark
  evidence.
- Advisory-only model output is a triage aid only.
- Deterministic checks authoritative: deterministic GPUBoost checks remain authoritative.
- Generated raw data, generated datasets, local DBs, model weights, build
  artifacts, temp validation environments, and local reports are ignored.
- Generated artifacts ignored: generated artifacts are ignored by default.

## Documentation

- [Setup](docs/setup.md)
- [Quickstart](docs/quickstart.md)
- [Agent CLI](docs/agent-cli.md)
- [Trial Workspace](docs/trial-workspace.md)
- [Comparison](docs/comparison.md)
- [History](docs/history.md)
- [Model Interface](docs/model-interface.md)
- [Model Training](docs/model-training.md)
- [Demo Workflow](docs/demo-workflow.md)
- [Real-World Validation](docs/real-world-validation.md)
- [Dependency Review](docs/dependency-review.md)
- [Security Review](docs/security-review.md)
- [Final Project Summary](docs/final-project-summary.md)
- [Release Notes](docs/release-notes.md)
- [Release Checklist](docs/release-checklist.md)
- [Phase 13 Testing](docs/phase-13-testing.md)
- [Phase 13 Release Readiness](docs/phase-13-release-readiness.md)
- [Phase 14 Validation Summary](docs/phase-14-validation-summary.md)
- [Post-Release Audit](docs/post-release-audit.md)

## Limitations

- Synthetic benchmarks are signals, not production performance proof.
- GPU results vary with GPU model, driver, CUDA/PyTorch versions, thermals,
  power mode, background load, and workload shape.
- The local model workflow is experimental and has no bundled production model.
- Real-world demo JSON currently records useful workload metrics, but
  `gpuboost compare` did not compare those demo JSON files during the audit.
- Raw third-party intake data is intentionally ignored and should be reviewed
  before being kept in or copied from a local checkout.
- PowerShell `Tee-Object` can write UTF-16 JSON on some systems; use UTF-8 when
  saving JSON intended for `gpuboost compare`.
- Agent benchmark acceptance policies require an explicit benchmark command
  that emits JSON metrics such as `speedup_percent` or `regression_percent`.

## Development

```powershell
python -m ruff check .
python -m pytest -q
python -m build
python -m twine check --strict dist/*
```

Do not commit generated artifacts, raw intake data, model weights, local
databases, `.env` files, temp validation environments, or build outputs.
