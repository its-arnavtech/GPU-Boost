# Agent Core

GPUBoost Phase 5 implements a deterministic, non-LLM agent core. It is a
foundation for the future Phase 6 CLI command, not a user-facing command yet.

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

This helper is not exposed as a CLI command yet. Phase 6 will wrap it in
`gpuboost agent optimize`.

## Safety Principles

- No source files are edited.
- No patches are applied automatically.
- Patch output is review-only.
- No LLM is used in Phase 5.
- No network access is required for agent tests.
- Tests are CPU-safe and use fake handlers where integration coverage is needed.

## Known Follow-Ups

- Review dependency semantics for skipped actions before Phase 6 CLI integration.
- Decide whether `AgentRunResult` should expose selected summary metadata, or
  whether richer Phase 6 reporting should consume `AgentState` directly.
