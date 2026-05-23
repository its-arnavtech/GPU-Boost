# Third-Party Raw Inventory

Generated at: 2026-05-22T14:51:34.6358371-05:00

## What was collected

### MLCommons inference_results_v6.0
- Repo URL: https://github.com/mlcommons/inference_results_v6.0
- Saved under: data/gpuboost/raw/mlcommons/inference_results_v6.0
- Branch: main
- Commit: 4d3916ac9cf474b679cdfcf492d43a0559418ad1
- Requested folders: closed/NVIDIA, closed/AMD, closed/Google, closed/CoreWeave
- Approximate collected files: 4332
- Approximate collected size: 346045654 bytes
- Key extensions found: .py, .txt, .json, .sh, .md, .conf, .yaml, .hpp, .cpp, .h, .adoc, .cu, .patch, .cfg, .cache, .cc, .cuh, [no extension], .rankfile, .html
- Warnings/issues:
  - Sparse checkout on Windows hit invalid-path errors elsewhere in the MLCommons repository under open/AMD during branch checkout.
  - Requested folders were materialized with per-subtree git archive fallback rather than a clean sparse checkout worktree.
  - Archive extraction reported Invalid argument for parts of Google/CoreWeave (src/code or src), so those folders may be partial.

### PyTorch benchmark
- Repo URL: https://github.com/pytorch/benchmark
- Saved under: data/gpuboost/raw/pytorch/benchmark
- Branch: main
- Commit: 3568dd6898f28439eb76e692efb0362fd76b6259
- Requested folders: results
- Approximate collected files: 0
- Approximate collected size: 0 bytes
- Key extensions found: none
- Warnings/issues:
  - Default branch main at the collected commit does not contain the requested results/ path.
  - The sparse-checkout pattern is set to results, but the folder is absent on disk and absent from the current HEAD tree.
  - Missing requested folder: data\gpuboost\raw\pytorch\benchmark\results

## Next recommended step
Parse/import these local files through the GPUBoost Phase 11 importers.
