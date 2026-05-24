"""Release-readiness checks for Phase 12 documentation."""

from __future__ import annotations

from pathlib import Path


def test_phase_12_docs_describe_completed_workflow_and_safety() -> None:
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in [
            "README.md",
            "docs/model-training.md",
            "docs/agent-cli.md",
            "docs/phase-12-release-readiness.md",
        ]
    ).lower()

    for phrase in [
        "safe training data loading",
        "safe feature extraction",
        "baseline model comparison",
        "mlp training",
        "artifact save",
        "direct artifact prediction",
        "advisory-only agent integration",
        "does not fine-tune an llm",
        "does not call external",
        "model predictions are advisory only",
        "patch_application_allowed=false",
        "data/gpuboost/generated/",
        "deterministic",
    ]:
        assert phrase in combined


def test_phase_12_docs_list_full_lifecycle_commands() -> None:
    text = Path("docs/model-training.md").read_text(encoding="utf-8")

    for command in [
        "python -m gpuboost model evaluate-baselines --json",
        "python -m gpuboost model train-neural --max-epochs 50 --max-candidates 12 --target-macro-f1 0.85 --json",
        "python -m gpuboost model train-neural --max-epochs 50 --max-candidates 12 --target-macro-f1 0.85 --save-artifact --json",
        "python -m gpuboost model list-artifacts",
        "python -m gpuboost model show-artifact <manifest>",
        "python -m gpuboost model check-artifact <manifest> --min-test-macro-f1 0.75 --require-beats-baseline",
        "python -m gpuboost model validate-artifact <manifest>",
        "python -m gpuboost model predict-artifact <manifest> --features-json",
        "python -m gpuboost agent optimize <script> --model-artifact <manifest> --json",
    ]:
        assert command in text
