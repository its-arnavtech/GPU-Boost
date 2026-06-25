# GPUBoost Post-Release Audit

Generated: 2026-06-24

## 1. Release And Version Audited

- Repository: GPU-Boost
- Package: `gpuboost`
- Public release audited: `0.1.2`
- Branch: `main`
- Commit: `3e263aa5ac2755745b64072c501277a0a7ad7386`
- `origin/main`: `3e263aa5ac2755745b64072c501277a0a7ad7386`
- Local `main` matched `origin/main` at audit start.
- Initial tracked working tree: clean.

## 2. Environment Summary

Fresh evidence command: `python -m gpuboost info --json`

| Field | Value |
|---|---:|
| OS | Windows-11-10.0.26200-SP0 |
| Python | 3.12.10 |
| Logical CPU cores | 22 |
| Physical CPU cores | 16 |
| RAM | 23.37 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Total VRAM | 8188 MB |
| CUDA available | true |
| PyTorch | 2.11.0+cu128 |
| Torch CUDA | 12.8 |
| cuDNN | 91900 |

The GPU UUID reported by the local tool output is intentionally omitted.

## 3. Sensitive-File Audit

Tracked-file searches found no committed `.env` files, private keys, local
databases, model binaries, raw/generated data, build artifacts, or caches.

Content searches found only these tracked categories:

- Virtualenv activation examples in documentation.
- Synthetic private-path fixtures in tests that verify redaction.
- A synthetic private-key marker in `tests/test_phase_13_security_audit.py`.

Ignored raw third-party intake data under `data/gpuboost/raw/` contains many
variable-name markers such as `HF_TOKEN`, `API_KEY`, and `TOKEN` in public
MLCommons scripts and examples. No values were printed. This directory is
ignored and untracked; it was retained for manual review rather than deleted.

## 4. Tracked-Artifact Audit

The following targeted checks returned no tracked files:

```text
git ls-files ".env" ".env.*"
git ls-files "*.pem" "*.key" "*.p12" "*.pfx" "*.jks"
git ls-files "*.db" "*.sqlite" "*.sqlite3"
git ls-files "*.pt" "*.pth" "*.onnx" "*.safetensors" "*.pkl" "*.pickle" "*.joblib"
git ls-files "dist/**" "build/**" "run-output/**"
git ls-files "data/gpuboost/raw/**"
git ls-files "data/gpuboost/generated/**"
git ls-files "**/__pycache__/**" "**/.pytest_cache/**" "**/.ruff_cache/**"
git ls-files "*.egg-info/**"
```

## 5. `.gitignore` Changes

The root `.gitignore` was reorganized into explicit sections for Python caches,
virtual environments, tests, coverage, lint/type-check caches, packaging
outputs, IDE files, OS files, logs/temp files, databases, secrets, GPUBoost
datasets, model artifacts, benchmark/demo outputs, publishing credentials,
package-manager caches, and temporary validation environments.

Representative checks:

| Path | Result |
|---|---|
| `.env` | ignored |
| `.env.local` | ignored |
| `.env.example` | trackable |
| `dist/gpuboost.whl` | ignored |
| `run-output/demo.json` | ignored |
| `data/gpuboost/raw/private.jsonl` | ignored |
| `data/gpuboost/generated/dataset.jsonl` | ignored |
| `models/model.pt` | ignored |
| `artifacts/model.safetensors` | ignored |
| `local.sqlite3` | ignored |
| `.pytest_cache/` | ignored |
| `.venv/` | ignored |
| `examples/outcome_collection/pairs.example.json` | trackable |
| `data/gpuboost/manifests/training_readiness_report.json` | trackable |

Git emitted a local user-level ignore permission warning during these checks;
repo-local ignore rules still evaluated correctly.

## 6. Cleanup Actions

Deletion candidates were classified before removal.

Safe generated/local-only targets removed:

| Path | Files | Bytes | Reason |
|---|---:|---:|---|
| `.venv/` | 24688 | 4692500501 | Reproducible local virtual environment; active Python was outside the repo |
| `dist/` | 2 | 540318 | Reproducible package build output |
| `tmp/` | 25924 | 702269583 | Temporary validation and wheel-smoke environments |
| `pytest_tmp/` | 17 | 25775 | Test output |
| `pytest-cache-files-*` | 0 | 0 | Empty pytest cache directories |
| `.ruff_cache/` | 11 | 11577 | Reproducible lint cache |
| `data/gpuboost/generated/` | 44 | 1450065 | Ignored generated datasets/model artifacts |
| `data/gpuboost/experiments/amp_001/` | 2 | 1402 | Ignored local experiment output |
| `data/gpuboost/experiments/batch_001/` | 2 | 1390 | Ignored local experiment output |
| `data/gpuboost/experiments/dataloader_001/` | 2 | 1345 | Ignored local experiment output |
| `data/gpuboost/experiments/grid/` | 200 | 178047 | Ignored local experiment output |
| `data/gpuboost/experiments/grid_pairs.json` | 1 | 72682 | Ignored generated grid manifest |
| `data/gpuboost/experiments/grid_runner_manifest.json` | 1 | 171356 | Ignored generated grid manifest |
| `**/__pycache__/` | many | not separately totaled | Reproducible Python bytecode |

Retained intentionally:

- `data/gpuboost/raw/`: 9742 files, 560231747 bytes. Ignored and untracked,
  but retained for manual review because it is raw third-party intake data.
- `.pytest_cache/`: ignored cache directory, but unreadable in this sandboxed
  workspace during cleanup.
- New small ignored audit output under `run-output/post-release-audit/`.

## 7. Demo Commands

Fresh representative commands:

```powershell
python -m gpuboost --version
python -m gpuboost doctor --json
python -m gpuboost info --json
python -m gpuboost analyze examples\bad_train_sample.txt --json
python -m gpuboost agent optimize examples\bad_train_sample.txt --json
python -m gpuboost agent optimize examples\bad_train_sample.txt --trial --json
python -m gpuboost benchmark --quick --json --recommend
python -m gpuboost demo real-world-info --json
python -m gpuboost demo real-world-pairs --json
python examples\real_world\pytorch_cnn_baseline.py --quick --benchmark-json
python examples\real_world\pytorch_cnn_optimized.py --quick --benchmark-json
python -m gpuboost compare run-output\post-release-audit\cnn_baseline_utf8.json run-output\post-release-audit\cnn_optimized_utf8.json --json
```

No external APIs, dataset downloads, model training, long benchmark grids, or
destructive GPU stress tests were run.

## 8. Raw Measured Metrics

Version and setup:

| Metric | Value |
|---|---:|
| `python -m gpuboost --version` | `gpuboost 0.1.2` |
| Doctor status | `ok` |
| Doctor required checks | passed |
| CUDA required | false |

Static and agent workflow:

| Metric | Value |
|---|---:|
| Static analysis findings | 5 |
| Agent optimize status | `ok` |
| Agent optimize actions | 5 completed |
| Agent trial status | `passed` |
| Agent trial actions | 6 completed |
| Trial patch applied to temp copy | true |
| Original file unchanged | true |
| Trial syntax-check status | `passed` |
| Trial syntax-check duration | 0.024519 seconds |
| Trial test command | skipped, none supplied |
| Trial workspace cleanup | passed |

Quick synthetic benchmark:

| Metric | Value |
|---|---:|
| Benchmark sections | 4 successful |
| Matrix duration | 0.9341 seconds |
| Best FP32 matmul | 7.167453 TFLOPS |
| Best FP16 matmul | 31.337318 TFLOPS |
| FP16/FP32 matmul ratio | 4.372169x |
| Mixed precision duration | 5.716 seconds |
| FP32 synthetic throughput | 15938.796 samples/sec |
| AMP synthetic throughput | 24402.147 samples/sec |
| AMP synthetic ratio | 1.530991x |
| Batch sweep duration | 33.2951 seconds |
| Best batch size | 8 |
| Best batch throughput | 2607.100 images/sec |
| Batch-size speedup vs batch 1 | 3.115839x |
| DataLoader duration | 10.8698 seconds |
| Best DataLoader workers | 0 |
| Best DataLoader pin_memory | false |
| Best DataLoader throughput | 6804.981 samples/sec |

Quick CNN synthetic demo:

| Metric | Baseline | Optimized |
|---|---:|---:|
| Device | cuda | cuda |
| Batch size | 4 | 8 |
| AMP used | false | true |
| Non-blocking transfer | false | true |
| Throughput | 353.727 samples/sec | 862.134 samples/sec |
| Median step | 11.308 ms | 9.279 ms |

Manual ratio from the two demo runs: optimized throughput was 2.437285x the
baseline and median step time was 17.94% lower. This is one synthetic local run,
not a generalized claim.

## 9. Comparison Results

The quick CNN baseline and optimized scripts produced JSON metrics successfully.
Two comparison attempts were made:

- JSON captured through PowerShell `Tee-Object` failed because the file was
  UTF-16LE in this shell.
- After rewriting the files as UTF-8, `gpuboost compare` still returned
  `No comparable metrics were found.`

The README therefore does not claim that `gpuboost compare` validates the
real-world demo script outputs.

## 10. Test Results

Final validation expected after the README/test updates:

```text
python -m ruff check .
python -m pytest -q
```

Results:

- Ruff: all checks passed.
- Pytest: 1044 passed.

## 11. Build And Archive Results

Final validation after the README/test updates:

```text
python -m build
python -m build --no-isolation
python -m twine check --strict dist/*
```

Results:

- `python -m build` failed because isolated build attempted to install
  `hatchling` from PyPI and network access was blocked in the sandbox.
- `python -m build --no-isolation` succeeded using the locally installed
  backend and produced `gpuboost-0.1.2-py3-none-any.whl` and
  `gpuboost-0.1.2.tar.gz`.
- Twine strict check passed.
- Archive scan found no forbidden paths in the wheel or sdist.
- Wheel scan: 111 files, 0 sensitive/local marker hits.
- Sdist scan: 274 files, 0 forbidden paths, 4 marker-name hits. The hits were
  in this audit document and synthetic security tests; no secret values or
  private key material were found.

## 12. README Claim-To-Evidence Mapping

| README claim | Evidence |
|---|---|
| Current release is `0.1.2` | `gpuboost/__init__.py`, `python -m gpuboost --version` |
| Base install is lightweight | `pyproject.toml` dependencies |
| Benchmark/model extras include Torch and NumPy | `pyproject.toml` optional dependencies |
| GPU/CUDA optional for core workflows | doctor reports `cuda_required=false`; tests run without requiring CUDA |
| No automatic patch application | CLI has no original-file apply command; trial output shows temp copy and original unchanged |
| Model output advisory | model CLI help and safety-check contract include advisory-only and `patch_application_allowed=false` |
| Deterministic checks authoritative | docs/tests and model safety contract |
| Generated/raw/model artifacts ignored | `.gitignore`, `git ls-files` artifact checks |
| Fresh validation metrics | commands and tables in this audit |

## 13. Limitations

- Results are from one Windows laptop with one RTX 4060 Laptop GPU.
- Benchmarks and demo workloads use synthetic or controlled data.
- Benchmark ratios are not universal speedup claims.
- Raw third-party intake data remains locally present and ignored, but should be
  manually reviewed before reuse or copying.
- `gpuboost compare` did not compare quick CNN demo JSON metrics.
- PowerShell UTF-16 JSON capture can surprise `compare`; save JSON as UTF-8.
- Git emitted a user-home ignore permission warning unrelated to repo contents.

## 14. Remaining Risks

- Real-world demo comparison tooling needs either broader metric support or
  documentation that it is not the comparison path for those JSON files.
- `.pytest_cache/` remained inaccessible in this sandboxed checkout.
- Historical phase docs still exist and should be treated as historical unless
  refreshed against this audit.
- Raw third-party intake data is ignored and untracked but still present.

## 15. Final Recommendation

Recommendation: **CLEAN AFTER SMALL FIXES**.

The tracked repository state is release-maintainable after the README and
ignore-policy updates. No tracked sensitive/generated files were found. The
remaining small fixes are the real-world demo comparison gap, the local
PowerShell JSON encoding caveat, and manual review or removal of ignored raw
third-party intake data outside release artifacts.
