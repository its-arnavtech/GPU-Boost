# Training Readiness Report

Status: ready

## Row Counts
- Total rows: 104
- Labeled rows: 103
- Unlabeled rows: 1

## Labels
- improved: 42
- neutral: 26
- regressed: 35
- unknown: 1

## Feature Coverage
- action_count: 1
- completed_action_count: 1
- controlled_grid: 100
- failed_action_count: 1
- has_comparison: 1
- has_diff: 1
- has_trial: 1
- improved_metric_count: 103
- overall_verdict: 103
- phase: 100
- regressed_metric_count: 103
- section_count: 103
- trial_original_file_unchanged: 1
- trial_patch_applied: 1
- trial_status: 1
- trial_syntax_check_status: 1
- trial_test_status: 1
- unchanged_metric_count: 103
- workload_family: 100

## Context
- Context rows: 177
- Hardware specs rows: 116
- Benchmark result context rows: 61
- MLPerf context rows: 78

## External Intake
- mlcommons_inference: {"row_count": 78, "validation_status": "warning", "warnings": ["Skipped duplicate MLCommons context rows: count=2.", "Skipped invalid MLCommons JSON files: count=2.", "Skipped unclear MLCommons files: count=432."]}
- mlcommons_raw_inventory: {"warning": "Could not parse third-party raw inventory report."}
- pytorch_benchmark: {"warning": "PyTorch benchmark results folder was missing from collected repo; verify source path or skip for now."}
- techpowerup: {"row_count": 99, "validation_status": "warning"}

## Blockers
- none

## Recommendations
- Keep TechPowerUp rows as hardware context, not labels.
- PyTorch benchmark results folder was missing from collected repo; verify source path or skip for now.

Phase 12 training may begin because no hard blockers remain.
