# Publishing GPUBoost

This page documents the manual package publishing flow. Do not commit
generated `dist/`, `build/`, `*.egg-info/`, or `wheelhouse/` artifacts.

## Local Build

Install packaging tools if needed:

```bash
python -m pip install --upgrade build twine
```

Clean old local artifacts:

```bash
python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('dist'), pathlib.Path('build')]]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]"
```

Build the source distribution and wheel:

```bash
python -m build
```

Check package metadata:

```bash
python -m twine check --strict dist/*
```

Before any upload, inspect the wheel and sdist contents. Package artifacts must
not contain raw data, generated datasets, model artifacts, checkpoints, local
databases, virtual environments, caches, `.env` files, or secrets.

## TestPyPI

Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI in a clean environment:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ gpuboost
```

If you want benchmark or local model commands in that validation environment,
install the optional extra:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "gpuboost[all]"
```

`gpuboost[benchmark]`, `gpuboost[model]`, and `gpuboost[all]` should all pull
in both PyTorch and NumPy, while the base `gpuboost` install should keep both
dependencies optional.

For `0.1.2`, validate at least these installed-package commands from outside
the repository:

```bash
python -m gpuboost --version
python -m gpuboost --help
python -m gpuboost doctor --json
python -m gpuboost model safety-check --json
python -m gpuboost compare --help
python -m gpuboost agent --help
python -m gpuboost demo --help
```

TestPyPI is temporary and may be incomplete or ephemeral. Treat it as a
validation environment, not a durable release archive.

## PyPI

After TestPyPI validation, upload the same checked artifacts to PyPI:

```bash
python -m twine upload dist/*
```

Trusted Publishing is recommended for future GitHub Actions publishing so
uploads can use PyPI's OIDC workflow instead of long-lived API tokens.
