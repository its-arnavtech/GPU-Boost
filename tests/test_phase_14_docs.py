"""Documentation checks for Phase 14 real-world validation docs."""

from __future__ import annotations

from pathlib import Path


PHASE_14_DOCS = [
    Path("docs/real-world-validation.md"),
    Path("docs/demo-workflow.md"),
    Path("docs/model-training.md"),
    Path("docs/agent-cli.md"),
    Path("README.md"),
]


def test_phase_14_required_docs_exist() -> None:
    for doc_path in PHASE_14_DOCS:
        assert doc_path.exists()


def test_phase_14_docs_describe_validation_boundaries() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PHASE_14_DOCS)
    lowered = combined.lower()

    for phrase in [
        "advisory-only",
        "no automatic patch application",
        "synthetic data",
        "hardware variability",
        "generated artifacts are ignored",
        "deterministic gpuboost checks remain authoritative",
    ]:
        assert phrase in lowered


def test_real_world_validation_doc_lists_required_workflow_steps() -> None:
    text = Path("docs/real-world-validation.md").read_text(encoding="utf-8")

    for phrase in [
        "phase 14",
        "examples/real_world/",
        "run_real_world_demo_benchmarks.ps1",
        "python -m gpuboost compare",
        "python -m gpuboost dataset collect-outcomes",
        "--model-artifact",
        "improved",
        "regressed",
        "neutral",
        "cpu fallback",
    ]:
        assert phrase in text.lower()


def test_readme_links_to_phase_14_docs() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    for link in [
        "docs/real-world-validation.md",
        "docs/demo-workflow.md",
        "docs/release-checklist.md",
    ]:
        assert link in text
