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
goal -> plan -> execute tools -> validate changes -> compare results -> remember run history -> explain results
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
- Not yet exposed as a CLI command
- Phase 6 will add `gpuboost agent optimize train.py`
- Agent core does not apply patches automatically
- Current behavior remains safe and review-only

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
```

Planned commands, not yet implemented:

```bash
gpuboost agent optimize train.py
gpuboost agent optimize train.py --json
gpuboost agent optimize train.py --trial
gpuboost agent optimize train.py --trial --test "pytest"
gpuboost agent compare baseline.json optimized.json
gpuboost history list
gpuboost history show <run_id>
gpuboost agent ask "Why is AMP slower on my machine?"
```

## Agentic AI Roadmap

Phases 5-10 pivot GPUBoost from a benchmark and analysis CLI into an agentic AI
production system. Phase 5 is implemented as a deterministic core; later phases
will add CLI access, trial workspaces, validation, history, and optional LLM
explanations.

### Phase 5: Agent Core - State, Actions, Planner, Executor

- Built a deterministic agent backbone
- Added `AgentState`, `AgentGoal`, `AgentAction`, `AgentPlan`,
  `AgentRunResult`, and `AgentEvent`
- Converts optimization goals into execution plans
- Executes plans through deterministic handlers and builds agent reports
- No LLM and no file modification

### Phase 6: Agent CLI - One-Shot Optimize Workflow

- Add `gpuboost agent optimize train.py`
- Agent runs inspector, quick benchmark, advisor, code analyzer, patch planner,
  diff generator, and final report
- Add `--json`
- Still review-only and safe by default

### Phase 7: Safe Trial Workspace

- Add `--trial`
- Copy the target script into a temporary workspace
- Apply patches only to the copy
- Run syntax checks
- Optionally run a user-provided test command
- Never modify the original file

### Phase 8: Before/After Validation and Comparison

- Compare baseline and optimized benchmark JSON
- Report speedups and regressions
- Keep comparisons evidence-based
- Add optional future benchmark command support

### Phase 9: Local Agent Memory and Run History

- Store local SQLite data under `~/.gpuboost/`
- Store run history, script hash, findings, recommendations, patches,
  warnings, and trial results
- Add `gpuboost history list`, `gpuboost history show <run_id>`, and
  `gpuboost history compare`
- Keep everything local and private by default

### Phase 10: Optional LLM Explanation / Natural-Language Agent Layer

- Add an optional AI explanation layer
- LLM can summarize, explain, answer questions, and generate human-readable
  reports
- LLM cannot invent metrics, override benchmark results, apply patches, or
  modify files
- Deterministic engine remains the source of truth
- Tests use a stub provider, with no API key required

## Production Agent Testing Phase

### Phase 11: Production Testing

- Agent unit tests
- Agent integration tests
- Trial workspace safety tests
- Before/after comparison tests
- History database tests
- LLM safety tests with a stub provider
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
Goal -> Planner -> Actions -> Executor -> State -> Validation -> Report -> Memory -> Optional LLM Explanation
```

## Safety Principles

- GPUBoost never applies patches automatically by default.
- Reviewable diffs are generated before changes.
- Trial mode will apply patches only in temporary workspaces.
- Measured benchmark data takes priority over AI-generated explanations.
- The LLM layer is optional and cannot override deterministic metrics.
- User code is not uploaded anywhere by default.
- Local run history stays local unless the user explicitly exports or
  contributes data.

## Run Tests

```bash
pytest
```

The test suite does not require an NVIDIA GPU.

## Not Included Yet

- Agent CLI
- Trial workspace patch application
- Before/after benchmark comparison
- Local run history database
- Optional LLM explanation layer
- Dashboard code
- Daemon code
