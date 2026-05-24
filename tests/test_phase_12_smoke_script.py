"""Static checks for the manual Phase 12 model workflow smoke script."""

from __future__ import annotations

from pathlib import Path


def test_phase_12_smoke_script_exists_and_mentions_required_commands() -> None:
    script = Path("scripts/smoke_phase_12_model_workflow.ps1")
    text = script.read_text(encoding="utf-8")

    assert script.exists()
    for command in [
        "evaluate-baselines",
        "train-neural",
        "validate-artifact",
        "check-artifact",
        "predict-artifact",
        "agent\", \"optimize",
    ]:
        assert command in text
    assert "--save-artifact" in text
    assert "--model-artifact" in text
    assert "patch_application_allowed" in text
    assert "advisory only" in text
    assert "--max-epochs\", \"20" in text
    assert "--max-candidates\", \"4" in text


def test_phase_12_smoke_script_avoids_plain_json_redirection() -> None:
    text = Path("scripts/smoke_phase_12_model_workflow.ps1").read_text(
        encoding="utf-8"
    )

    assert ">" not in text.replace("-ne", "").replace("->", "")
    assert "Set-Content" not in text or "-Encoding utf8" in text
