# Training Readiness Report

Status: not_ready

## Row Counts
- Total rows: 1
- Labeled rows: 0
- Unlabeled rows: 1

## Labels
- unknown: 1

## Feature Coverage
- action_count: 1
- completed_action_count: 1
- failed_action_count: 1
- has_comparison: 1
- has_diff: 1
- has_trial: 1
- trial_original_file_unchanged: 1
- trial_patch_applied: 1
- trial_status: 1
- trial_syntax_check_status: 1
- trial_test_status: 1

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
- Not enough total rows for training.
- No labeled optimization outcome rows found.
- No comparison-derived labels found.

## Recommendations
- Run GPUBoost with --save-history on real scripts.
- Run before/after comparisons to create improved/regressed/neutral labels.
- Keep TechPowerUp rows as hardware context, not labels.
- Capture GPU/hardware details consistently in history rows.
- PyTorch benchmark results folder was missing from collected repo; verify source path or skip for now.

Phase 12 training should not begin until blockers are resolved.
