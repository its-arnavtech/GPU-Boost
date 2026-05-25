"""Documentation checks for Phase 15 README and quickstart polish."""

from __future__ import annotations

from pathlib import Path


README = Path("README.md")
QUICKSTART = Path("docs/quickstart.md")
FINAL_SUMMARY = Path("docs/final-project-summary.md")


def test_readme_links_quickstart() -> None:
    text = README.read_text(encoding="utf-8")

    assert "docs/quickstart.md" in text


def test_quickstart_exists() -> None:
    assert QUICKSTART.exists()


def test_final_project_summary_exists_and_covers_release_boundaries() -> None:
    text = FINAL_SUMMARY.read_text(encoding="utf-8").lower()

    for phrase in [
        "completed capabilities",
        "architecture overview",
        "key cli workflows",
        "dataset, model, and demo workflow",
        "safety model",
        "validation status",
        "known limitations",
        "future work",
        "release readiness recommendation",
        "model predictions are advisory-only",
        "deterministic gpuboost checks remain authoritative",
        "does not guarantee speedups",
        "no external api dependency",
    ]:
        assert phrase in text


def test_readme_mentions_phase_15_safety_boundaries() -> None:
    text = README.read_text(encoding="utf-8").lower()

    for phrase in [
        "advisory-only model",
        "no automatic patch application",
        "deterministic checks authoritative",
        "deterministic gpuboost checks remain authoritative",
        "generated artifacts ignored",
        "generated artifacts are ignored",
    ]:
        assert phrase in text

    assert "docs/final-project-summary.md" in text
