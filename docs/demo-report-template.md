# GPUBoost Demo Validation Report Template

Generated demo validation reports summarize Phase 14 real-world demo runs without
including generated benchmark payloads in the repository. Reports are written
under `data/gpuboost/generated/`, which is ignored by git.

## What the Report Contains

- Generation timestamp and report schema version.
- Workload summaries for realistic before/after validation pairs.
- Before/after verdicts from GPUBoost comparison results.
- Metric deltas when comparable metrics are available.
- Optional model advisory predictions, such as suggested optimization impact or
  confidence.
- Safety notes that make clear the report is advisory and does not authorize
  automatic patch application.
- Limitations for interpreting demo results.

## How to Interpret It

An `improved` verdict means the optimized benchmark improved on the comparable
metrics that GPUBoost could evaluate. A `regressed` verdict means at least one
comparable metric moved in the wrong direction. A `mixed` verdict means the
before/after comparison had both improvements and regressions. An `unknown` or
`error` result usually means comparable metrics were missing or an input was
incomplete.

Model advisory predictions are suggestions only. Deterministic GPUBoost checks
remain authoritative, including actual benchmark comparisons. The demo report
does not apply patches, approve patches, or include model weights.

## Safety Guarantees

- Model advisory only.
- `patch_application_allowed=false`.
- Deterministic checks are authoritative.
- No automatic patch application.
- No raw source, raw diffs, stdout, stderr, tracebacks, or model weights are
  included in the report.

## Limitations

- Demo workloads use synthetic data.
- Demo workloads are lightweight approximations of real user workloads.
- Results vary by hardware, driver, PyTorch version, and system load.
