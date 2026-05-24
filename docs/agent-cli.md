# Agent CLI

The Phase 6 agent CLI exposes GPUBoost's deterministic optimize workflow:

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
python -m gpuboost agent optimize train.py --model-artifact data/gpuboost/generated/model_training/artifacts/<id>/manifest.json
python -m gpuboost agent optimize .\examples\bad_train_sample.txt --model --trial --json
```

The current agent workflow defaults to `quick=True` because the implemented
benchmark action is the quick benchmark path. Full benchmark agent mode is a
future feature.

## Purpose

`gpuboost agent optimize` runs a one-shot, non-LLM optimization workflow.

Without a script path, it performs system-level analysis:

- inspect system
- run the quick benchmark
- generate advisor recommendations
- summarize the run

With a script path, it also:

- analyzes PyTorch code statically
- creates a safe patch plan
- generates a reviewable unified diff when safe suggestions exist

With `--trial`, it also:

- creates a temporary trial workspace
- copies the target file into that workspace
- applies generated patch suggestions only to the copy
- runs Python syntax validation without executing user code
- optionally runs an explicit `--test` command in the trial workspace

The original source file is never modified. GPUBoost does not implement
`--apply`.

With `--model`, it also routes safe summary features through the local model
interface. Without an artifact, the workflow falls back to `NullModelProvider`.

With `--model-artifact <manifest_path>`, model inference is enabled
automatically and GPUBoost loads the local trained artifact through
`TrainedLocalModelProvider`. The prediction is advisory only. It cannot apply
patches, edit files, override deterministic checks, or call external APIs.
Missing or invalid artifacts produce a clean model fallback/error result instead
of a traceback.

The demo file `examples/bad_train_sample.txt` is intentionally kept as `.txt`
so project linters do not treat it as Python:

```bash
gpuboost agent optimize examples/bad_train_sample.txt
gpuboost agent optimize examples/bad_train_sample.txt --trial
```

## Human Output

Human output is concise and review-oriented. A successful script run looks like:

```text
GPUBoost Agent
Command: optimize
Status: ok
Script: examples/bad_train_sample.txt

Summary:
The agent workflow completed successfully.

Plan:
- inspect_system: completed
- run_quick_benchmark: completed
- generate_recommendations: completed
- analyze_code: completed
- create_patch_plan: completed
- generate_diff: completed
- summarize_results: completed

Reviewable Patch Diff:
GPUBoost does not apply patches automatically. Review the diff before applying changes.

--- examples/bad_train_sample.txt
+++ examples/bad_train_sample.txt (GPUBoost suggested)
...

Safety:
GPUBoost does not apply patches automatically. Review generated diffs before applying changes.
```

If no diff exists, the `Reviewable Patch Diff` section is omitted.

When `--save-history` is passed, the human output includes:

```text
History:
- Saved run: run_...
```

## JSON Output

JSON output is valid JSON only and uses schema version `agent.optimize.v1`:

```json
{
  "schema_version": "agent.optimize.v1",
  "command": "agent optimize",
  "result": {},
  "report": {},
  "artifacts": {
    "diff": null,
    "trial": null,
    "comparison": null,
    "history_run_id": null,
    "model": null
  }
}
```

When a reviewable diff exists, `artifacts.diff` contains the unified diff text.
The same diff is also exposed in `result.artifacts.diff`.

When trial mode is enabled, `artifacts.trial` contains a `TrialResult` with the
workspace record, step list, patch status, syntax status, optional test command
status, warnings, and `original_file_unchanged`.

When `--save-history` succeeds, `artifacts.history_run_id` contains the saved
history run ID. Without `--save-history`, it is `null`.

Without `--model` or `--model-artifact`, `artifacts.model` is `null`. With
model inference enabled, `artifacts.model` contains a stable advisory summary:

```json
{
  "status": "ok",
  "provider": "trained_local_model",
  "prediction": {"label": "improved", "confidence": 0.91},
  "probabilities": {"improved": 0.91, "regressed": 0.09},
  "patch_application_allowed": false,
  "warnings": []
}
```

Fallback results keep `model_available: false`, `fallback_used: true`, and a
warning explaining why a trained model was not used.

Unexpected workflow exceptions return valid JSON with `result` and `report`
set to `null`:

```json
{
  "schema_version": "agent.optimize.v1",
  "command": "agent optimize",
  "result": null,
  "report": null,
  "artifacts": {
    "diff": null,
    "trial": null,
    "comparison": null,
    "history_run_id": null,
    "model": null
  },
  "error": "error message"
}
```

## Safety Model

GPUBoost never applies patches to original files. Patch diffs are review-only.
Trial mode applies patches only to a temporary copy. Syntax checks use Python
compilation only and do not import or run the target script. A `--test` command
is explicit opt-in and may execute arbitrary user-provided code in the trial
workspace.

Phase 7 implements the safe trial workspace. Phase 9 adds optional local
history. Phase 10 adds the local model interface and fallback provider. Phase
12.5 allows a saved local model artifact to provide advisory predictions in the
agent report path.

`--save-history` stores a local SQLite history record under
`~/.gpuboost/gpuboost.db` by default. It stores script path, script SHA256,
statuses, counts, warnings, and safe summaries. It does not store raw source
code, raw diffs, trial stdout, or trial stderr by default. Use
`--history-db-path` to point at a temporary database for development or tests.
See [Local History](history.md).

The model feature layer follows the same safety boundary: it uses safe summaries
only and excludes raw source code, raw diffs, stdout, stderr, final comparison
labels, and target-derived before/after/delta fields. The deterministic
GPUBoost advisor, trial workspace, syntax checks, explicit tests, and measured
benchmark data remain authoritative. The model may rank, score, or predict
confidence, but it cannot apply patches or override measured results. See
[Local Model Interface](model-interface.md).

## Exit Codes

- `ok` -> `0`
- `partial` -> `0`
- `error` -> `1`
- unexpected CLI exception -> `1`

`partial` can occur when optional steps fail, such as code analysis for a
missing script path. In that case, human output shows the failed actions and
JSON preserves action statuses and errors in `result.plan.actions`.

## Current Limitations

- No auto-apply or `--apply`
- No external LLM APIs
- Trained model artifacts are advisory only and not auto-applied
- Quick benchmark only for now
- Full benchmark agent mode is not implemented yet

## Phase 6 Completion Checklist

- Agent CLI command available
- Human output available
- JSON output available
- Reviewable diffs displayed
- No auto-apply
- Error/partial handling implemented
- CPU-safe test coverage
- Phase 7: trial workspace
- Phase 8: before/after comparison
- Phase 9: optional local run history
- Phase 10: local model interface with safe fallback
