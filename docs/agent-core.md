# Agent Core

GPUBoost Phase 5 implements a deterministic, non-LLM agent core. Phase 6 wraps
this core in the `gpuboost agent optimize` CLI while keeping the workflow
review-only and deterministic.

## Architecture

```text
AgentGoal -> AgentPlan -> AgentExecutor -> AgentRunResult -> AgentReport
```

Main modules:

- `gpuboost/schemas/agent.py`: goal, action, plan, event, and run-result schemas
- `gpuboost/agent/state.py`: single-run working memory
- `gpuboost/agent/actions.py`: deterministic action registry
- `gpuboost/agent/planner.py`: goal-to-plan conversion
- `gpuboost/agent/executor.py`: injected-handler plan execution
- `gpuboost/agent/handlers.py`: handlers that call existing GPUBoost modules
- `gpuboost/agent/report.py`: stable report builder
- `gpuboost/agent/workflow.py`: internal optimize-script workflow helper

## Internal Workflow Helper

`create_optimize_script_goal()` creates a safe `optimize_script` goal with
review-only constraints. `run_optimize_script_workflow()` runs the deterministic
Phase 5 flow and returns an `AgentRunResult` plus `AgentReport`.

This helper is exposed by the Phase 6 `gpuboost agent optimize` CLI. Tests still
exercise the helper directly with fake handlers so benchmark and analyzer logic
do not run in unit coverage.

## Safety Principles

- No source files are edited.
- No patches are applied automatically.
- Patch output is review-only.
- No LLM is used in Phase 5 or the Phase 6 CLI wrapper.
- No network access is required for agent tests.
- Tests are CPU-safe and use fake handlers where integration coverage is needed.

## Known Follow-Ups

- Safe trial workspace support is not implemented yet.
- Full benchmark agent mode is not implemented yet; the current agent workflow
  uses the quick benchmark action.
