# Agent CLI

The Phase 6 agent CLI exposes GPUBoost's deterministic optimize workflow:

```bash
gpuboost agent optimize
gpuboost agent optimize --json
gpuboost agent optimize train.py
gpuboost agent optimize train.py --json
gpuboost agent optimize train.py --quick
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

The demo file `examples/bad_train_sample.txt` is intentionally kept as `.txt`
so project linters do not treat it as Python:

```bash
gpuboost agent optimize examples/bad_train_sample.txt
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

## JSON Output

JSON output is valid JSON only and uses schema version `agent.optimize.v1`:

```json
{
  "schema_version": "agent.optimize.v1",
  "command": "agent optimize",
  "result": {},
  "report": {},
  "artifacts": {
    "diff": null
  }
}
```

When a reviewable diff exists, `artifacts.diff` contains the unified diff text.
The same diff is also exposed in `result.artifacts.diff`.

Unexpected workflow exceptions return valid JSON with `result` and `report`
set to `null`:

```json
{
  "schema_version": "agent.optimize.v1",
  "command": "agent optimize",
  "result": null,
  "report": null,
  "artifacts": {
    "diff": null
  },
  "error": "error message"
}
```

## Safety Model

GPUBoost never applies patches automatically. Patch diffs are review-only.
The agent does not edit source files, create trial workspaces, store history, or
call an LLM in Phase 6.

## Exit Codes

- `ok` -> `0`
- `partial` -> `0`
- `error` -> `1`
- unexpected CLI exception -> `1`

`partial` can occur when optional steps fail, such as code analysis for a
missing script path. In that case, human output shows the failed actions and
JSON preserves action statuses and errors in `result.plan.actions`.

## Current Limitations

- No trial workspace yet
- No auto-apply or `--apply`
- No LLM layer yet
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
- Future: trial workspace, history, LLM explanations
