# GPUBoost Release Notes

## 0.2.0 - 2026-06-25

GPUBoost `0.2.0` adds human-approved agentic optimization for deterministic
source edits. GPUBoost proposes an exact diff, requires explicit approval, and
can apply only the approved deterministic edits with backups, validation, and
rollback.

- Added an explicit approval-gated lifecycle:
  `agent optimize --prepare`, `agent show-plan`, `agent approve`,
  `agent reject`, `agent apply`, `agent rollback`, and `agent status`.
- `agent show-plan` is read-only and displays the exact reviewable diff before
  any approval; model output can never modify files directly.
- Operations are constrained to the active repository root: path traversal,
  absolute paths outside the root, symlink escapes, escaping backup directories,
  and cross-repository run records belonging to a different repository are all
  rejected.
- No Git commit or push is ever performed automatically, and unattended
  autonomous patching is not supported.
- `--prepare` is non-mutating and writes an ignored local run record with the
  deterministic plan, risk labels, immutable plan digest, original file hash,
  and reviewable diff.
- `agent approve` records human approval tied to the run ID, plan ID, plan
  digest, approved action IDs, approver, timestamp, and original target file
  hash.
- Partial approvals are supported: users can approve only selected deterministic
  actions from the proposed plan.
- `agent apply` mutates source only after approval, creates a backup, validates
  Python syntax/bytecode, supports explicit validation/test commands, and rolls
  back automatically on validation or acceptance-policy failure.
- Optional benchmark acceptance policies can require JSON metrics such as
  `speedup_percent` or `regression_percent` from a user-provided benchmark
  command.
- Safety policy wording now distinguishes no automatic patching and no
  unapproved patch application from human-approved deterministic apply.
- Persisted lifecycle records capture the run ID, plan ID, plan digest, original
  source hash, approval details, backup path, validation result, benchmark
  result, rollback result, and final status.
- Model-originated patching remains forbidden: model output is advisory and
  cannot approve, apply, or widen a deterministic plan.

## 0.1.2 Packaging Fix

GPUBoost `0.1.2` completes the follow-up packaging fix after local validation
of `0.1.1` showed that Torch-backed optional installs could still surface
PyTorch's missing-NumPy warning.

- Added NumPy to the Torch-backed optional extras: `benchmark`, `model`, and
  `all`.
- Prevented the PyTorch missing-NumPy warning in `gpuboost[benchmark]`,
  `gpuboost[model]`, and `gpuboost[all]` installs.
- Preserved the lightweight base install for inspection, analysis, compare,
  history, demo discovery, and safety-check workflows.
- Kept PyTorch optional instead of moving Torch or NumPy into base runtime
  dependencies.

## 0.1.1 Maintenance Release

GPUBoost `0.1.1` is the follow-up release prepared after TestPyPI validation of
`0.1.0` uncovered installed-package issues. This update keeps the same product
scope while fixing lightweight CLI import behavior, installed-package doctor
and model safety-check behavior outside a source checkout, and the package's
dependency story around optional PyTorch-backed workflows.

- Lightweight commands such as `python -m gpuboost --help`,
  `python -m gpuboost --version`, `python -m gpuboost compare --help`,
  `python -m gpuboost agent --help`, and `python -m gpuboost demo --help` no
  longer eagerly import PyTorch.
- `doctor` now separates installed-runtime checks from repository-only
  `.gitignore` policy checks and supports `--repo-root` for explicit source
  validation.
- `model safety-check` now reports package/runtime guardrails outside the
  repository and marks repository-only checks as skipped or warning instead of
  failing falsely.
- PyTorch moved behind optional extras for benchmark and model workflows; the
  default install remains useful for inspection, analysis, compare, history,
  demo discovery, and safety validation.

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

- GPUBoost does not apply patches without approval.
- Patch diffs are review-only unless the user enters the explicit
  `--prepare` -> `approve` -> `apply` lifecycle.
- Trial mode modifies only temporary workspace copies.
- Model predictions are advisory-only and cannot apply patches or approve
  changes.
- Trained artifact metadata keeps `patch_application_allowed=false`.
- Deterministic checks and measured benchmark evidence remain authoritative.
- Approved deterministic edits may be applied automatically only after explicit
  human approval tied to the plan digest and original source hash.
- Backups, validation, benchmark acceptance policies, automatic rollback, and
  explicit rollback are built into the apply lifecycle.
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
- Fully unattended autonomous patching remains unsupported.
- Benchmark threshold policies require a user-provided command that emits
  simple JSON metrics.

## Notable Commands

```bash
python -m gpuboost --version
python -m gpuboost info --json
python -m gpuboost benchmark --quick --recommend
python -m gpuboost analyze examples/bad_train_sample.txt --patch
python -m gpuboost agent optimize examples/bad_train_sample.txt --trial
python -m gpuboost agent optimize examples/agentic_apply_demo.txt --prepare
python -m gpuboost agent show-plan <run_id>
python -m gpuboost agent approve <run_id> --confirm "APPLY <plan>"
python -m gpuboost agent apply <run_id>
python -m gpuboost agent rollback <run_id>
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
