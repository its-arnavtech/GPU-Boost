"""Lightweight safety checks for the local Phase 12 model workflow."""

from __future__ import annotations

import importlib
from pathlib import Path

from gpuboost.repository import resolve_repository_context


MODEL_WORKFLOW_SAFETY_SCHEMA_VERSION = "training.model_workflow_safety.v1"
_ARTIFACT_EXTENSIONS = ("*.pt", "*.pth", "*.onnx", "*.safetensors", "*.pkl", "*.joblib")
_LOCAL_DB_PATTERNS = ("*.db", "*.sqlite", "*.sqlite3")
_CACHE_PATTERNS = ("__pycache__/", ".pytest_cache/", ".ruff_cache/", ".cache/")
_ENV_SECRET_PATTERNS = (
    ".env",
    ".env.*",
    "secrets/",
    "credentials/",
    "*.pem",
    "*.key",
    "*.token",
    "*.secret",
)
_REPOSITORY_ONLY_CHECKS = (
    "generated_dir_ignored",
    "artifact_extensions_ignored",
    "local_db_artifacts_ignored",
    "cache_dirs_ignored",
    "env_secret_patterns_ignored",
    "raw_data_ignored",
    "model_patch_application_allowed_false_documented",
)


def verify_model_workflow_safety(
    repo_root: str | None = None,
) -> dict[str, object]:
    """Return JSON-safe release-readiness safety checks for Phase 12."""

    repository = resolve_repository_context(repo_root)
    gitignore = _read_repo_text(repository.root, ".gitignore")
    docs_text = "\n".join(
        _read_repo_text(repository.root, relative_path)
        for relative_path in (
            "README.md",
            "docs/model-training.md",
            "docs/agent-cli.md",
        )
    )
    provider_source = _read_module_text("gpuboost.model.provider")
    cli_source = _read_module_text("gpuboost.cli.main")

    checks: dict[str, bool | None] = {
        "generated_dir_ignored": _repo_check(
            repository.root,
            "data/gpuboost/generated/" in gitignore,
        ),
        "artifact_extensions_ignored": _repo_check(
            repository.root,
            all(extension in gitignore for extension in _ARTIFACT_EXTENSIONS),
        ),
        "local_db_artifacts_ignored": _repo_check(
            repository.root,
            all(pattern in gitignore for pattern in _LOCAL_DB_PATTERNS),
        ),
        "cache_dirs_ignored": _repo_check(
            repository.root,
            all(pattern in gitignore for pattern in _CACHE_PATTERNS),
        ),
        "env_secret_patterns_ignored": _repo_check(
            repository.root,
            all(pattern in gitignore for pattern in _ENV_SECRET_PATTERNS),
        ),
        "raw_data_ignored": _repo_check(
            repository.root,
            "data/gpuboost/raw/" in gitignore,
        ),
        "patch_application_allowed": False,
        "model_patch_application_allowed_false_documented": _repo_check(
            repository.root,
            (
                "patch_application_allowed=false" in docs_text
                or "patch_application_allowed: false" in docs_text
            ),
        ),
        "provider_patch_application_allowed_false": (
            '"patch_application_allowed": False' in provider_source
        ),
        "no_default_artifact_path_required": (
            "--save-artifact" in cli_source
            and "--model-artifact" in cli_source
        ),
    }

    warnings: list[str] = []
    skipped_checks: list[str] = []

    if repository.root is None:
        warnings.append(f"{repository.message} Repository-only checks skipped.")
        skipped_checks.extend(_REPOSITORY_ONLY_CHECKS)
    else:
        for name in _REPOSITORY_ONLY_CHECKS:
            if checks[name] is False:
                warnings.append(f"Safety check failed: {name}")

    for name in (
        "provider_patch_application_allowed_false",
        "no_default_artifact_path_required",
    ):
        if checks[name] is False:
            warnings.append(f"Safety check failed: {name}")

    hard_failures = [
        name
        for name in (
            "generated_dir_ignored",
            "artifact_extensions_ignored",
            "local_db_artifacts_ignored",
            "cache_dirs_ignored",
            "env_secret_patterns_ignored",
            "raw_data_ignored",
            "provider_patch_application_allowed_false",
        )
        if checks[name] is False
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
        "repo_root": str(repository.root) if repository.root is not None else None,
        "repo_root_status": repository.status,
        "repo_root_message": repository.message,
        "skipped_checks": skipped_checks,
        **checks,
        "warnings": warnings,
    }


def _repo_check(root: Path | None, passed: bool) -> bool | None:
    if root is None:
        return None
    return passed


def _read_repo_text(root: Path | None, relative_path: str) -> str:
    if root is None:
        return ""
    return _read_text(root / relative_path)


def _read_module_text(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return ""
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        return ""
    return _read_text(Path(module_file))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
