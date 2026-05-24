# GPUBoost

GPUBoost is an open-source, agentic NVIDIA GPU performance engineer that
inspects hardware, benchmarks workloads, analyzes PyTorch code, generates
optimization recommendations, and produces safe reviewable patch diffs.

The current implementation supports this workflow:

```text
inspect -> benchmark -> recommend -> analyze code -> generate reviewable patch diffs
```

The planned agentic workflow is:

```text
goal -> plan -> execute tools -> validate changes -> compare results -> remember run history -> local model interface
```

GPUBoost is not affiliated with NVIDIA. It is designed as a local-first,
evidence-based optimization tool for CUDA and PyTorch development.

## Current Capabilities

### Phase 1: GPU/System Inspector

- Detects NVIDIA GPU details, VRAM, CUDA availability, PyTorch environment,
  cuDNN, compute capability, and Tensor Core support
- Detects host CPU, RAM, operating system, and Python version
- Produces human-readable terminal output or JSON
- Handles CPU-only systems and missing NVIDIA tooling gracefully

### Phase 2: Benchmark Suite

- Matrix multiplication FP32/FP16 benchmark
- Mixed precision AMP synthetic training benchmark
- Batch size sweep benchmark
- DataLoader benchmark for worker and pinned-memory behavior
- JSON output and CPU-safe benchmark behavior

### Phase 3: Optimization Advisor

- Rule-based recommendations from benchmark results
- Mixed precision, Tensor Core, batch size, DataLoader, and warning-aware
  recommendations
- Recommendations prioritized by impact, confidence, and effort
- Optional advisor output from benchmark commands

### Phase 4: Code Analyzer + Reviewable Patch Diffs

- Static analysis for PyTorch scripts without executing user code
- Detects DataLoader issues, sync calls in loops, missing AMP/autocast,
  missing `torch.no_grad()` or `torch.inference_mode()`, and missing
  `torch.backends.cudnn.benchmark = True`
- Generates safe review-only unified diffs for selected low-risk changes
- Does not apply patches automatically

### Phase 5: Agent Core

- Deterministic, non-LLM agent core is implemented
- Supports goal schemas, run state, action registry, deterministic planning,
  executor, real handlers, and report builder
- Agent core does not apply patches automatically
- Current behavior remains safe and review-only

### Phase 6: Agent CLI

- Exposes `gpuboost agent optimize` for a one-shot deterministic workflow
- Supports human-readable reports and stable JSON output
- Uses schema version `agent.optimize.v1` for JSON automation
- Includes review-only patch diffs in `artifacts.diff` when safe suggestions
  exist
- Defaults to the quick benchmark path, matching the current implemented agent
  action set, with `quick=True`

### Phase 7: Safe Trial Workspace

- Adds `gpuboost agent optimize train.py --trial`
- Creates a temporary workspace and copies the target file into it
- Applies generated patch suggestions only to the copied trial file
- Runs a Python syntax check on the copied file without executing user code
- Optionally runs an explicit user-provided test command with `--test`
- Never modifies the original source file and does not provide `--apply`

### Phase 10: Local Model Interface

- Adds `gpuboost agent optimize --model`
- Routes safe summary features through a local model provider interface
- Includes `NullModelProvider` fallback when no provider is configured
- Does not include a trained GPUBoost model yet
- Does not call external LLM APIs, train a model, load a real model, collect
  datasets, or export data
- Keeps deterministic GPUBoost logic and measured benchmark data as the source
  of truth

### Phase 11: Data Collection and Readiness

- Adds privacy-safe dataset assembly, validation reports, split assignment,
  and training readiness analysis
- Collects controlled before/after outcome rows from measured benchmark JSON
  without executing arbitrary collector commands
- Includes local controlled workload grids for generating measured outcome
  pairs across dataloader, AMP, batch-size, and neutral-control cases
- Phase 11 readiness now gates Phase 12: model training should begin only when
  the readiness report has no hard blockers
- Phase 12 training must use the safe training feature extraction layer, not
  raw `DatasetRow.to_dict()` output

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

Show GPU and system information:

```bash
gpuboost info
gpuboost info --json
python -m gpuboost info
```

Warnings are printed in human output and included in JSON output. They are not
fatal; GPUBoost should still report whatever information is available.

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

Generate optimization advice from benchmark results:

```bash
gpuboost benchmark --quick --recommend
gpuboost benchmark --quick --json --recommend
```

Benchmark results vary with laptop power mode, thermals, background GPU load,
drivers, CUDA version, PyTorch build, and whether the system is plugged in.
CPU-only systems return skipped CUDA benchmark results instead of crashing.

Run static code analysis on a training or inference script:

```bash
gpuboost analyze train.py
gpuboost analyze train.py --json
```

Generate review-only unified diffs for safe patch suggestions:

```bash
gpuboost analyze train.py --patch
gpuboost analyze train.py --json --patch
```

Patch suggestions are unified diffs for review. GPUBoost never applies changes
automatically.

Run the deterministic agent optimize workflow:

```bash
gpuboost agent optimize
gpuboost agent optimize --json
gpuboost agent optimize train.py
gpuboost agent optimize train.py --json
gpuboost agent optimize train.py --quick
gpuboost agent optimize train.py --trial
gpuboost agent optimize train.py --trial --json
gpuboost agent optimize train.py --trial --test "pytest"
gpuboost agent optimize train.py --save-history
python -m gpuboost agent optimize --model
python -m gpuboost agent optimize --model --json
python -m gpuboost agent optimize .\examples\bad_train_sample.txt --model --trial --json
```

Without a script path, the agent performs system-level optimization analysis:

- inspect system
- quick benchmark
- advisor recommendations
- summary

With a script path, the agent also analyzes PyTorch code, creates a safe patch
plan, and generates a reviewable unified diff when patchable findings exist.
GPUBoost never applies patches automatically; diffs are review-only.

With `--trial`, GPUBoost creates a temporary copy of the script, applies the
generated patch plan only to that copy, runs syntax validation, and reports the
trial result. The original source file is never modified.

With `--trial --test "<command>"`, GPUBoost also runs the explicit command in
the temporary trial workspace. Test commands may execute arbitrary user-provided
code and never run unless `--test` is passed with `--trial`.

With `--model`, GPUBoost runs the Phase 10 local model interface over safe
feature summaries. Without a saved local artifact, it falls back to
`NullModelProvider` and reports `model_available: false`, `fallback_used: true`,
and `status: "fallback"` in `artifacts.model`.

With `--model-artifact <manifest>`, GPUBoost loads a saved local artifact for
advisory prediction:

```bash
python -m gpuboost agent optimize train.py --model-artifact data/gpuboost/generated/model_training/artifacts/<id>/manifest.json
```

The model prediction is advisory only. It cannot apply patches, edit files, or
override deterministic checks, trials, tests, or benchmark evidence.

JSON output uses schema version `agent.optimize.v1` and includes
`schema_version`, `command`, `result`, `report`, `artifacts.diff`, and
`artifacts.trial`. It also includes `artifacts.comparison`, currently `null`
unless comparison data is attached, `artifacts.history_run_id`, which is set
only when `--save-history` succeeds, and `artifacts.model`, which is `null`
unless `--model` is used.
`quick=True` is the default. A `partial` status can occur when optional steps
fail, such as missing script files.

Agent exit-code policy:

- `ok` -> `0`
- `partial` -> `0`
- `error` -> `1`

See [Agent CLI](docs/agent-cli.md) for examples and the JSON shape.

Save and inspect local run history:

```bash
gpuboost agent optimize train.py --save-history
gpuboost history list
gpuboost history list --json
gpuboost history show <run_id>
gpuboost history show <run_id> --json
gpuboost history compare <left_run_id> <right_run_id>
gpuboost history compare <left_run_id> <right_run_id> --json
```

History is local-only and defaults to `~/.gpuboost/gpuboost.db`. It stores
script path, script SHA256, statuses, counts, warnings, and safe summaries. It
does not store raw source code, raw diffs, trial stdout, or trial stderr by
default. Use `--db-path` on history commands or `--history-db-path` on
`agent optimize --save-history` for temporary development databases. See
[Local History](docs/history.md).

Try the static analysis demo sample:

```bash
gpuboost agent optimize examples/bad_train_sample.txt
gpuboost agent optimize examples/bad_train_sample.txt --json
gpuboost agent optimize examples/bad_train_sample.txt --trial
gpuboost agent optimize examples/bad_train_sample.txt --trial --json
```

The sample is intentionally kept as `.txt` so formatters and linters do not
treat it as project Python code.

Compare saved benchmark JSON files:

```bash
gpuboost benchmark --quick --json > baseline.json
gpuboost benchmark --quick --json > optimized.json
gpuboost compare baseline.json optimized.json
gpuboost compare baseline.json optimized.json --json
```

Comparison JSON uses schema version `comparison.v1`. The command compares
files only; it does not run benchmark commands, apply patches, or execute
before/after workloads automatically. See [Comparison](docs/comparison.md) for
the JSON shape, verdict meanings, limitations, and future benchmark-command
design.

## Near-Term Commands

Current commands:

```bash
gpuboost info
gpuboost info --json
gpuboost benchmark --quick
gpuboost benchmark --quick --recommend
gpuboost benchmark --json --recommend
gpuboost analyze train.py
gpuboost analyze train.py --json
gpuboost analyze train.py --patch
gpuboost analyze train.py --json --patch
gpuboost compare baseline.json optimized.json
gpuboost compare baseline.json optimized.json --json
gpuboost agent optimize
gpuboost agent optimize --json
gpuboost agent optimize train.py
gpuboost agent optimize train.py --json
gpuboost agent optimize train.py --quick
gpuboost agent optimize train.py --trial
gpuboost agent optimize train.py --trial --test "pytest"
gpuboost agent optimize train.py --save-history
python -m gpuboost agent optimize --model
python -m gpuboost agent optimize --model --json
python -m gpuboost agent optimize .\examples\bad_train_sample.txt --model --trial --json
gpuboost history list
gpuboost history show <run_id>
gpuboost history compare <left_run_id> <right_run_id>
python -m gpuboost model train-neural --json
python -m gpuboost model train-neural --save-artifact --json
python -m gpuboost model list-artifacts
python -m gpuboost model show-artifact <manifest_path>
python -m gpuboost model validate-artifact <manifest_path>
python -m gpuboost model check-artifact <manifest_path> --min-test-macro-f1 0.75 --require-beats-baseline
python -m gpuboost model predict-artifact <manifest_path> --features-json '{...}' --json
```

Planned commands, not yet implemented:

```bash
gpuboost agent optimize train.py --trial --benchmark-command "..."
gpuboost agent ask "Why is AMP slower on my machine?"
```

## Agentic AI Roadmap

Phases 5-10 pivot GPUBoost from a benchmark and analysis CLI into an agentic AI
production system. Phases 5-7 are implemented as deterministic local tooling.
Later phases add before/after validation, history, the local model interface,
data validation, and GPUBoost's own model.

### Phase 5: Agent Core - State, Actions, Planner, Executor

- Built a deterministic agent backbone
- Added `AgentState`, `AgentGoal`, `AgentAction`, `AgentPlan`,
  `AgentRunResult`, and `AgentEvent`
- Converts optimization goals into execution plans
- Executes plans through deterministic handlers and builds agent reports
- No LLM and no file modification

### Phase 6: Agent CLI - One-Shot Optimize Workflow

- Implemented `gpuboost agent optimize [script_path]`
- Agent runs inspector, quick benchmark, advisor, code analyzer, patch planner,
  diff generator, and final report when a script path is provided
- JSON output is stable and versioned with `agent.optimize.v1`
- Reviewable patch diffs are included in human output and `artifacts.diff`
- Still review-only and safe by default

### Phase 7: Safe Trial Workspace

- Implemented `--trial`
- Copies the target script into a temporary workspace
- Applies patches only to the copy
- Runs syntax checks without executing user code
- Optionally runs a user-provided test command
- Never modifies the original file

### Phase 8: Before/After Validation and Comparison

- Implemented `gpuboost compare baseline.json optimized.json`
- Supports stable `comparison.v1` JSON output
- Compares saved benchmark JSON files only
- Agent optimize JSON reserves `artifacts.comparison` for future integration
- Future benchmark-command support remains explicit opt-in design only

### Phase 9: Local Agent Memory and Run History

- Stores local SQLite data under `~/.gpuboost/`
- Stores run history, script hash, safe summaries, statuses, counts,
  warnings, and trial results
- Does not store raw source code, raw diffs, stdout, or stderr by default
- Adds `gpuboost history list`, `gpuboost history show <run_id>`, and
  `gpuboost history compare`
- Keeps everything local and private by default

### Phase 10: Local Model Interface / Model-Ready Agent Layer

- Adds local model schemas, safe feature extraction, provider interfaces, and
  inference artifacts
- `--model` routes through the local interface and falls back to
  `NullModelProvider` unless a provider is configured later
- No trained GPUBoost model is included yet
- No external LLM APIs are used
- No model training, real model loading, dataset export, or data collection is
  implemented
- Deterministic GPUBoost logic remains the source of truth
- The model layer may later rank, score, or predict confidence, but cannot
  apply patches or override measured benchmark data
- Features are safe summaries only: no raw source code, raw diffs, stdout, or
  stderr

### Phase 11: Data Collection and Validation

- Implements privacy-safe data collection, validation, controlled outcome
  grids, manifests, split assignment, and readiness reports
- Readiness reports are the gate into Phase 12 training: `ready` means no hard
  blockers remain; `warning` requires review before proceeding; `not_ready`
  blocks training

### Phase 12: GPUBoost Model Training and Integration

- Future phase to train and integrate GPUBoost's own model, starting with a
  baseline structured model rather than fine-tuning
- Phase 12.1 adds the safe training dataset loader, feature/label encoding,
  evaluation utilities, and a majority-class sanity baseline. See
  [Model Training](docs/model-training.md).
- Phase 12.2 adds dependency-free baseline comparison for
  majority-class, seeded random, nearest-centroid, and simple KNN models:
  `python -m gpuboost model evaluate-baselines --json`
- Baseline reports are written under
  `data/gpuboost/generated/model_training/` by default; no production model
  artifact is saved and no predictions are integrated into the agent yet
- Phase 12.3 adds a small PyTorch MLP trained from scratch on safe encoded
  structured features: `python -m gpuboost model train-neural --json`
- Neural training runs a modest validation-selected hyperparameter search,
  compares against the best baseline, treats `0.85` macro F1 as aspirational,
  and reports honestly when the target is missed
- Neural reports are evaluation artifacts only; no production model checkpoint
  is saved, no LLM is fine-tuned, no external API is called, and no agent
  integration is changed
- Phase 12.4 adds explicit local artifact packaging only when requested:
  `python -m gpuboost model train-neural --save-artifact --json`
- Artifacts can be checked with
  `python -m gpuboost model validate-artifact <manifest_path> --json` and used
  for standalone local predictions with
  `python -m gpuboost model predict-artifact <manifest_path> --features-json '{...}' --json`
- Phase 12.5 allows advisory agent predictions with
  `python -m gpuboost agent optimize train.py --model-artifact <manifest_path>`;
  the flag automatically enables model inference
- Phase 12.6 adds artifact lifecycle polish:
  `python -m gpuboost model list-artifacts`,
  `python -m gpuboost model show-artifact <manifest_path>`, and
  `python -m gpuboost model check-artifact <manifest_path> --min-test-macro-f1 0.75 --require-beats-baseline`
- Saved artifacts live under ignored generated paths by default; model
  predictions must never apply patches, edit files, or override deterministic
  GPUBoost checks
- A future phase should package/integrate a local model only if validation/test
  evaluation is strong and it meaningfully beats structured baselines
- Training must use safe feature extraction and must not train on
  target-derived comparison fields such as verdicts, before/after metrics,
  deltas, labels, raw diffs, stdout, or stderr
- Controlled outcome data is useful for baseline learning, but it is limited:
  controlled rows are synthetic workload measurements, not real user-script
  outcomes
- Third-party benchmark data is context and provenance, not direct GPUBoost
  labels
- The model may rank or score recommendations, but must not apply patches
  directly; deterministic GPUBoost logic remains authoritative

### Phase 13: Production System Testing

- Agent unit tests
- Agent integration tests
- Trial workspace safety tests
- Before/after comparison tests
- History database tests
- Model interface safety tests with stub providers
- End-to-end CLI smoke tests
- CPU-only CI compatibility
- Guarantee original files are not modified by default

## Architecture Direction

Current architecture:

```text
Inspector -> Benchmarks -> Advisor -> Code Analyzer -> Patch Planner -> Unified Diff
```

Agentic architecture direction:

```text
Goal -> Planner -> Actions -> Executor -> State -> Validation -> Report -> Memory -> Local Model Interface
```

## Safety Principles

- GPUBoost never applies patches automatically by default.
- Reviewable diffs are generated before changes.
- Trial mode applies patches only in temporary workspaces.
- Syntax checks validate Python syntax without importing or running scripts.
- Test commands are opt-in and may execute arbitrary user-provided code.
- Measured benchmark data takes priority over model-generated signals.
- The model layer is optional and cannot override deterministic metrics.
- No external LLM APIs are used.
- User code is not uploaded anywhere.
- Local run history stays local unless the user explicitly exports or
  contributes data.
- Local run history does not store raw source code, raw diffs, trial stdout, or
  trial stderr by default.
- Model features do not store raw source code, raw diffs, stdout, or stderr.

## Run Tests

```bash
pytest
```

The test suite does not require an NVIDIA GPU.

## Not Included Yet

- `--apply` or original source editing
- A bundled/default trained GPUBoost model
- External LLM provider integrations
- Phase 12 model training
- Phase 13 production-system testing
- Dashboard code
- Daemon code
