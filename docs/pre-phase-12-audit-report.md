# Pre-Phase-12 Repository Audit Report

Date/time: 2026-05-23T17:18:59.7491694-05:00  
Branch: `dataset-pipelines`  
Repository: GitHub CLI repo metadata unavailable because `gh auth status` reported an invalid token.

## Validation Results

- `python -m ruff check .`: passed.
- `python -m pytest`: passed, 798 tests.
- Pytest warning: local cache write failed under `.pytest_cache` with access denied. This appears local/environmental and did not fail tests.

## Git And GitHub State

- Current branch: `dataset-pipelines`.
- Recent commits:
  - `46205d4 Complete Phase 11 dataset readiness pipeline`
  - `e3afbd3 phase 11 data validated and tested`
  - `cfb61f6 dataset labelling`
  - `59b601c data_validated`
  - `c049a66 Merge pull request #13 from its-arnavtech/phase-10`
- Initial worktree state: clean, aside from Git warnings about inaccessible global ignore config.
- `main...HEAD` diff: 118 files changed, 25,706 insertions and 83 deletions.
- GitHub CLI:
  - `gh auth status`: failed, active token is invalid.
  - `gh repo view --json nameWithOwner,url`: failed due unavailable GitHub access in this environment.
  - `gh issue list --state open --limit 100 --json number,title,state,labels,url`: failed due unavailable GitHub access in this environment.
- GitHub issue creation was skipped per instructions. Local issue drafts were written to `docs/pre-phase-12-issue-drafts.md`.

## Security Scan Summary

- No tracked files found under `data/gpuboost/raw` or `data/gpuboost/generated`.
- No tracked database/model artifact files found for `*.db`, `*.sqlite`, `*.sqlite3`, `*.pt`, `*.pth`, `*.onnx`, `*.safetensors`, `*.pkl`, or `*.joblib`.
- No tracked `.env`, `.env.*`, key, cert, token, or credential filename matches found.
- PR-branch diff filename scan found no raw/generated/env/model/credential path matches.
- Secret-pattern scan found no confirmed real secrets. Matches were test fixtures, token parsing names, ignore rules, and local safety examples.
- Security risks worth tracking:
  - `.gitignore` unignores `.env.staging`.
  - Tracked raw-inventory manifests contain local absolute path fields.
  - CLI JSON can expose raw diff and trial stdout/stderr artifacts.

## Dataset And Model Readiness Findings

The current readiness report says `ready`, with 104 total rows, 103 labeled rows, three known label classes, and no listed blockers. However, the audit found one Phase 12 blocker:

- Target-derived comparison fields are present in dataset features/metrics while labels are derived from those same comparison verdicts. This creates label leakage unless Phase 12 excludes them before training.

Additional readiness risks:

- Current split assignment is row-level random, not grouped by controlled workload family or template. This can leak correlated controlled-grid variants across train/validation/test splits.
- The raw-inventory report is valid-looking JSON with a UTF-8 BOM, but readiness reports it as unparseable because the parser reads it as plain UTF-8 before `json.loads`.
- The legacy controlled outcome PowerShell runner writes JSON through shell redirection, which can produce collector-incompatible encoding in Windows PowerShell.
- Docs should be clearer that Phase 12 should start with a baseline model and that controlled synthetic workload rows are limited training data, not real user-script outcomes.

## GitHub Issues Created

None. GitHub CLI authentication failed, so issue creation was intentionally skipped.

## Local Issue Drafts

The following GitHub-ready issue drafts were written to `docs/pre-phase-12-issue-drafts.md`:

1. `[high][dataset][phase-12] Remove target-derived comparison fields from Phase 12 training features`
2. `[medium][dataset][phase-12] Use grouped and stratified splits for controlled outcome rows`
3. `[medium][security] Stop unignoring .env.staging`
4. `[medium][dataset][windows] Normalize PowerShell-generated JSON encoding in Phase 11 scripts and manifests`
5. `[medium][security][dataset] Remove local absolute paths from tracked raw-inventory manifests`
6. `[medium][cli][security] Redact or opt in to raw diff and trial stdout/stderr in JSON artifacts`
7. `[low][docs][phase-12] Document the Phase 12 baseline-first training plan and controlled-data limitations`

## Existing Issues Skipped As Duplicates

None identified locally. Open GitHub issues could not be fetched because GitHub CLI authentication is invalid, so these drafts still need deduplication after re-authentication.

## No-Issue Findings

- CI workflow exists and covers Ubuntu and Windows across Python 3.10, 3.11, and 3.12.
- Ruff and the full test suite pass.
- Raw/generated data directories are ignored and not tracked.
- Manifest files are tracked, which matches the stated repository policy.
- Tests did not rely on real local data and did not require GPU availability in this environment.
- Broad exception handling observed around optional Rich imports, GPU probes, cache cleanup, and model-provider fallback looked intentional or non-fatal.
- Some docs contain stale or low-priority wording, but only the Phase 12 baseline/limitations doc gap was elevated to a draft issue.

## Final Recommendation

Do not start Phase 12 training on the current assembled dataset as-is. Fix or explicitly filter the target-derived comparison fields first. Grouped/stratified validation splits should also be addressed before trusting Phase 12 validation or test metrics.

Proceed to Phase 12 implementation only after the leakage blocker is fixed or the Phase 12 training code has a tested feature allowlist that excludes the leaked fields.
