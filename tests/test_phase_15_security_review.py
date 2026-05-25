"""Phase 15E dependency, license, and security review docs."""

from __future__ import annotations

from pathlib import Path


DEPENDENCY_REVIEW = Path("docs/dependency-review.md")
SECURITY_REVIEW = Path("docs/security-review.md")
README = Path("README.md")
LICENSE = Path("LICENSE")


def test_dependency_review_exists_and_covers_release_requirements() -> None:
    text = DEPENDENCY_REVIEW.read_text(encoding="utf-8").lower()

    for phrase in [
        "runtime dependencies",
        "torch",
        "psutil",
        "nvidia-ml-py",
        "rich",
        "optional and development dependencies",
        "pytest",
        "ruff",
        "pyTorch".lower(),
        "cuda hardware is optional",
        "no external api dependency",
        "no dataset download requirement",
        "generated artifacts are ignored",
    ]:
        assert phrase in text


def test_security_review_exists_and_covers_release_requirements() -> None:
    text = SECURITY_REVIEW.read_text(encoding="utf-8").lower()

    for phrase in [
        "ignored sensitive files",
        "generated artifacts are ignored",
        "raw intake data is ignored",
        "model artifacts",
        "no secrets are required",
        "external api",
        "advisory-only",
        "patch_application_allowed=false",
        "raw diffs and trial stdout/stderr",
        "known remaining risks",
    ]:
        assert phrase in text


def test_readme_links_dependency_security_and_license_docs() -> None:
    text = README.read_text(encoding="utf-8")

    for phrase in [
        "docs/dependency-review.md",
        "docs/security-review.md",
        "[LICENSE](LICENSE)",
        "MIT License",
    ]:
        assert phrase in text


def test_license_status_is_documented() -> None:
    assert LICENSE.exists()

    license_text = LICENSE.read_text(encoding="utf-8")
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    dependency_text = DEPENDENCY_REVIEW.read_text(encoding="utf-8")

    assert "MIT License" in license_text
    assert 'license = { text = "MIT" }' in pyproject_text
    assert "MIT License" in readme_text
    assert "MIT License" in dependency_text
