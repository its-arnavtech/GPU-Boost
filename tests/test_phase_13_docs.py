"""Documentation checks for Phase 13E release-readiness docs."""

from __future__ import annotations

from pathlib import Path


PHASE_13_DOCS = [
    Path("docs/phase-13-testing.md"),
    Path("docs/demo-workflow.md"),
    Path("docs/release-checklist.md"),
    Path("docs/phase-13-release-readiness.md"),
]


def test_phase_13_docs_exist() -> None:
    for doc_path in PHASE_13_DOCS:
        assert doc_path.exists()


def test_phase_13_docs_describe_safety_boundaries() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PHASE_13_DOCS)
    lowered = combined.lower()

    for phrase in [
        "advisory-only",
        "generated artifacts remain ignored",
        "no llm fine-tuning",
        "does not apply patches automatically",
        "deterministic gpuboost checks",
    ]:
        assert phrase in lowered


def test_phase_13_testing_doc_lists_required_commands() -> None:
    text = Path("docs/phase-13-testing.md").read_text(encoding="utf-8")

    for command in [
        "python -m ruff check .",
        "python -m pytest",
        "python -m gpuboost model safety-check --json",
        "powershell -ExecutionPolicy Bypass -File .\\scripts\\smoke_phase_12_model_workflow.ps1",
    ]:
        assert command in text


def test_demo_workflow_lists_end_to_end_commands() -> None:
    text = Path("docs/demo-workflow.md").read_text(encoding="utf-8")

    for phrase in [
        "agent optimize examples/bad_train_sample.txt",
        "--trial",
        "model evaluate-baselines",
        "model train-neural",
        "--save-artifact",
        "model validate-artifact",
        "model check-artifact",
        "model list-artifacts",
        "model show-artifact",
        "model predict-artifact",
        "--model-artifact",
        "patch_application_allowed=false",
    ]:
        assert phrase in text


def test_readme_links_to_phase_13_docs() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    for link in [
        "docs/model-training.md",
        "docs/agent-cli.md",
        "docs/demo-workflow.md",
        "docs/release-checklist.md",
        "docs/phase-13-testing.md",
        "docs/phase-13-release-readiness.md",
    ]:
        assert link in text
