"""Lightweight safety checks for the local Phase 12 model workflow."""

from __future__ import annotations

from pathlib import Path


MODEL_WORKFLOW_SAFETY_SCHEMA_VERSION = "training.model_workflow_safety.v1"
_ARTIFACT_EXTENSIONS = ("*.pt", "*.pth", "*.onnx", "*.safetensors", "*.pkl", "*.joblib")


def verify_model_workflow_safety() -> dict[str, object]:
    """Return JSON-safe release-readiness safety checks for Phase 12."""

    gitignore = _read_text(Path(".gitignore"))
    provider_source = _read_text(Path("gpuboost/model/provider.py"))
    docs_text = "\n".join(
        _read_text(path)
        for path in [
            Path("README.md"),
            Path("docs/model-training.md"),
            Path("docs/agent-cli.md"),
        ]
    )
    cli_source = _read_text(Path("gpuboost/cli/main.py"))

    checks = {
        "generated_dir_ignored": "data/gpuboost/generated/" in gitignore,
        "artifact_extensions_ignored": all(
            extension in gitignore for extension in _ARTIFACT_EXTENSIONS
        ),
        "raw_data_ignored": "data/gpuboost/raw/" in gitignore,
        "model_patch_application_allowed_false_documented": (
            "patch_application_allowed=false" in docs_text
            or "patch_application_allowed: false" in docs_text
        ),
        "provider_patch_application_allowed_false": (
            '"patch_application_allowed": False' in provider_source
        ),
        "no_default_artifact_path_required": (
            "--save-artifact" in cli_source
            and "--model-artifact" in cli_source
            and "artifact_manifest_path" not in _read_text(Path("pyproject.toml"))
        ),
    }
    warnings = [
        f"Safety check failed: {name}"
        for name, passed in checks.items()
        if not passed
    ]
    hard_failures = [
        name
        for name in (
            "generated_dir_ignored",
            "artifact_extensions_ignored",
            "raw_data_ignored",
            "provider_patch_application_allowed_false",
        )
        if not checks[name]
    ]
    if hard_failures:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "ok"
    return {
        "schema_version": MODEL_WORKFLOW_SAFETY_SCHEMA_VERSION,
        "status": status,
        **checks,
        "warnings": warnings,
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
