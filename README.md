# GPUBoost

GPUBoost is a local-first NVIDIA GPU performance assistant for CUDA and
PyTorch projects. It inspects hardware, runs lightweight benchmarks, performs
static analysis, suggests reviewable patches, tests patches in temporary trial
workspaces, prepares dataset/model artifacts, and demonstrates the workflow on
small real-world-style examples.

GPUBoost is not affiliated with NVIDIA. It is designed for evidence-based local
optimization work, not autonomous production patching.

Current checkpoint version: `0.1.0`. The package version source is
`gpuboost/__init__.py`; `pyproject.toml` reads it through Hatch dynamic
versioning.

## What GPUBoost Is

- A Python CLI for GPU/system inspection, synthetic benchmarks, static PyTorch
  analysis, optimization advice, and review-only patch suggestions.
- A deterministic agent workflow that can inspect, benchmark, analyze code,
  plan changes, generate diffs, and optionally test those diffs in a copied
  trial workspace.
- A local dataset and model workflow for safe structured features, readiness
  checks, local training experiments, and advisory-only model artifacts.
- A demo suite for validating the workflow against lightweight CNN,
  transformer, and DataLoader examples using synthetic data.

## What It Does

- Detects NVIDIA GPU, CUDA, PyTorch, cuDNN, Tensor Core, CPU, RAM, OS, and
  Python environment details.
- Runs CPU-safe benchmark commands, including matrix multiplication, AMP,
  batch-size sweeps, and DataLoader behavior.
- Produces rule-based recommendations from benchmark output.
- Statically analyzes PyTorch scripts without importing or executing user code.
- Generates unified diffs for selected low-risk changes, such as AMP,
  DataLoader, inference-mode, and cuDNN benchmark suggestions.
- Creates trial workspaces where generated diffs can be applied to a temporary
  copy of a script and syntax-checked.
- Stores local run history when explicitly requested.
- Provides local model artifact commands for evaluation, packaging, checking,
  and advisory prediction.

## What It Does Not Do

- No guaranteed speedup. Results depend on hardware, drivers, thermals, power
  mode, workload shape, PyTorch/CUDA versions, and background load.
- No production autonomous patching and no `--apply` command for original
  source files.
- No automatic patch application. Patch suggestions are review-only unless
  trial mode applies them to a temporary copy.
- No LLM fine-tuning and no external LLM API calls.
- No external API dependency for normal local use, tests, or release checks.
- No model authority over deterministic checks. Model predictions are
  advisory-only model signals.
- No automatic benchmark-command before/after execution in the agent.
- No commitment of raw/generated data or model weights.
- No CUDA requirement for normal tests.

## Quickstart

See [Setup](docs/setup.md) for full install notes and [Quickstart](docs/quickstart.md)
for the shortest path.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m gpuboost doctor
python -m gpuboost --version
python -m gpuboost --help
python -m gpuboost agent optimize examples/bad_train_sample.txt --json
python -m gpuboost agent optimize examples/bad_train_sample.txt --trial --json
python -m gpuboost model safety-check --json
```

On Windows PowerShell, activate the environment first if desired:

```powershell
.\.venv\Scripts\activate
```

The sample file is intentionally named `examples/bad_train_sample.txt` so
formatters and linters do not treat it as project Python code.

## Core Workflows

Inspect the local system:

```bash
python -m gpuboost doctor
python -m gpuboost doctor --json
python -m gpuboost info
python -m gpuboost info --json
```

Run benchmarks and recommendations:

```bash
python -m gpuboost benchmark --quick
python -m gpuboost benchmark --quick --recommend
python -m gpuboost benchmark --quick --json --recommend
```

Analyze a PyTorch script:

```bash
python -m gpuboost analyze train.py
python -m gpuboost analyze train.py --json
python -m gpuboost analyze train.py --patch
```

Compare saved benchmark JSON files:

```bash
python -m gpuboost compare baseline.json optimized.json
python -m gpuboost compare baseline.json optimized.json --json
```

Comparison reads existing files only. It does not run workloads, apply patches,
or decide that a change is production-ready.

## Agent Optimize

`agent optimize` is the one-shot deterministic workflow:

```bash
python -m gpuboost agent optimize
python -m gpuboost agent optimize --json
python -m gpuboost agent optimize train.py --json
python -m gpuboost agent optimize train.py --save-history
```

Without a script path, the agent inspects the system, runs the quick benchmark,
builds advisor recommendations, and reports a summary. With a script path, it
also analyzes the file and includes a reviewable diff when safe patch
suggestions exist.

JSON output uses schema version `agent.optimize.v1`. `partial` statuses are
non-fatal and exit with code `0`; `error` exits with code `1`.

See [Agent CLI](docs/agent-cli.md) for JSON shape, model-artifact behavior,
history fields, and CLI examples.

## Trial Workspace

Trial mode applies generated patch suggestions only to a copied file in a
temporary workspace:

```bash
python -m gpuboost agent optimize train.py --trial --json
python -m gpuboost agent optimize train.py --trial --test "pytest"
```

The original source file is never modified. Syntax checks validate the copied
file without importing or running the script. A user-provided test command runs
only when `--trial --test "<command>"` is explicitly passed.

See [Trial Workspace](docs/trial-workspace.md).

## Dataset And Model Workflow

GPUBoost includes a local structured workflow for dataset readiness, baseline
evaluation, small PyTorch MLP training, explicit artifact packaging, and
advisory agent integration:

```bash
python -m gpuboost model evaluate-baselines --json
python -m gpuboost model train-neural --json
python -m gpuboost model train-neural --save-artifact --json
python -m gpuboost model list-artifacts
python -m gpuboost model check-artifact <manifest_path> --min-test-macro-f1 0.75 --require-beats-baseline
python -m gpuboost agent optimize train.py --model-artifact <manifest_path> --json
```

The trained model remains advisory only. It cannot apply patches, edit files,
override deterministic checks, override trials, replace tests, or replace
benchmark evidence. Deterministic GPUBoost checks remain authoritative.

Model training uses safe feature extraction and must not train on raw source,
raw diffs, stdout, stderr, target-derived verdicts, or comparison labels.
Artifacts are saved only when explicitly requested.

See [Model Training](docs/model-training.md), [Model Interface](docs/model-interface.md),
and [Phase 12 Release Readiness](docs/phase-12-release-readiness.md).

## Real-World Demo Workflow

GPUBoost includes lightweight realistic demos for validating the surrounding
workflow:

```bash
python -m gpuboost demo real-world-info --json
python -m gpuboost demo real-world-pairs --json
```

The demos use synthetic data and are validation/demo coverage, not universal
proof of optimization impact. Hardware variability can change results, and demo
outcomes should not be generalized to every model or GPU.

Generated demo output is written under ignored generated paths such as
`data/gpuboost/generated/demo_real_world/`.

See [Demo Workflow](docs/demo-workflow.md), [Real-World Validation](docs/real-world-validation.md),
and [Phase 14 Validation Summary](docs/phase-14-validation-summary.md).

## Safety Guarantees

- GPUBoost does not apply patches automatically.
- There is no automatic patch application.
- Reviewable diffs are generated before any trial modification.
- Trial mode modifies only temporary copies, never the original source file.
- Static analysis parses user code without importing or executing it.
- Test commands are opt-in and may execute arbitrary user-provided code.
- Deterministic checks authoritative: deterministic GPUBoost checks remain
  authoritative alongside measured benchmark evidence.
- Model predictions are advisory only.
- Advisory-only model predictions cannot apply patches, edit files, or approve
  changes.
- No external LLM APIs are used.
- User code is not uploaded anywhere.
- Local run history stays local unless explicitly exported or contributed.
- Local run history does not store raw source, raw diffs, trial stdout, or
  trial stderr by default.
- Model features do not store raw source, raw diffs, stdout, or stderr.
- Generated artifacts are ignored by default; generated artifacts ignored by
  `.gitignore` include `data/gpuboost/generated/`, raw intake data, model
  weights, databases, caches, and local reports.
- Raw/generated data is not committed.

## Limitations

- Benchmarks are small local signals, not production performance proof.
- Synthetic demos are validation/demo, not universal proof.
- CPU-only machines skip CUDA benchmark work instead of failing hard.
- Local model artifacts are experimental advisory aids, not a production
  optimizer.
- GPUBoost does not include a bundled/default trained production model.
- Before/after validation currently compares saved benchmark JSON files; the
  agent does not run arbitrary before/after benchmark commands automatically.
- Power mode, thermal throttling, driver versions, CUDA versions, and workload
  noise can dominate results.

## Docs Index

- [Setup](docs/setup.md)
- [Quickstart](docs/quickstart.md)
- [Agent CLI](docs/agent-cli.md)
- [Agent Core](docs/agent-core.md)
- [Trial Workspace](docs/trial-workspace.md)
- [Comparison](docs/comparison.md)
- [Local History](docs/history.md)
- [Model Interface](docs/model-interface.md)
- [Model Training](docs/model-training.md)
- [Dependency Review](docs/dependency-review.md)
- [Security Review](docs/security-review.md)
- [Final Project Summary](docs/final-project-summary.md)
- [Demo Workflow](docs/demo-workflow.md)
- [Real-World Validation](docs/real-world-validation.md)
- [Demo Report Template](docs/demo-report-template.md)
- [Phase 12 Release Readiness](docs/phase-12-release-readiness.md)
- [Phase 13 Testing](docs/phase-13-testing.md)
- [Phase 13 Release Readiness](docs/phase-13-release-readiness.md)
- [Phase 14 Validation Summary](docs/phase-14-validation-summary.md)
- [Release Notes](docs/release-notes.md)
- [Release Checklist](docs/release-checklist.md)
- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)

## License And Release Audit

GPUBoost is released under the MIT License; see [LICENSE](LICENSE). The release
audit docs cover [Dependency Review](docs/dependency-review.md),
[Security Review](docs/security-review.md), and the
[Final Project Summary](docs/final-project-summary.md), including generated
artifacts, external API requirements, advisory-only model behavior, and ignore
rules.

## Test And Validation Status

The intended validation commands for this phase are:

```bash
python -m ruff check .
python -m pytest
```

The test suite is designed to run without an NVIDIA GPU. Some runtime benchmark
results will be skipped or CPU-safe on systems without CUDA.
