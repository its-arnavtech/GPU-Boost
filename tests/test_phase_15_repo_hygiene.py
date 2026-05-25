"""Repository hygiene checks for Phase 15D."""

from __future__ import annotations

from pathlib import Path


ISSUE_TEMPLATES = [
    Path(".github/ISSUE_TEMPLATE/bug_report.md"),
    Path(".github/ISSUE_TEMPLATE/feature_request.md"),
    Path(".github/ISSUE_TEMPLATE/security_or_data_leak.md"),
]
PR_TEMPLATE = Path(".github/pull_request_template.md")
CONTRIBUTING = Path("CONTRIBUTING.md")
SECURITY = Path("SECURITY.md")


def test_issue_templates_exist() -> None:
    for path in ISSUE_TEMPLATES:
        assert path.exists()


def test_security_issue_template_warns_against_public_secret_pastes() -> None:
    text = Path(".github/ISSUE_TEMPLATE/security_or_data_leak.md").read_text(
        encoding="utf-8"
    )
    lowered = text.lower()

    for phrase in [
        "do not paste secrets",
        "redact",
        "tokens",
        "raw private data",
        "public issue",
    ]:
        assert phrase in lowered


def test_pr_template_includes_artifact_and_safety_checks() -> None:
    text = PR_TEMPLATE.read_text(encoding="utf-8").lower()

    for phrase in [
        "python -m ruff check .",
        "python -m pytest",
        "no generated or raw data",
        "no model artifacts",
        "no secrets",
        "advisory-only",
        "deterministic checks remain authoritative",
        "docs are updated",
    ]:
        assert phrase in text


def test_contributing_mentions_tests_ruff_and_project_policies() -> None:
    text = CONTRIBUTING.read_text(encoding="utf-8").lower()

    for phrase in [
        "python -m ruff check .",
        "python -m pytest",
        "coding style",
        "generated data",
        "artifact policy",
        "safety model",
        "pull request expectations",
    ]:
        assert phrase in text


def test_security_doc_warns_not_to_post_secrets_publicly() -> None:
    text = SECURITY.read_text(encoding="utf-8").lower()

    for phrase in [
        "do not paste secrets",
        "public issues",
        "private vulnerability reporting",
        "generated artifacts",
        "model artifacts",
        "advisory-only",
        "does not guarantee speedups",
    ]:
        assert phrase in text

