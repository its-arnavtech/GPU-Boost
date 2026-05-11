"""Tests for Phase 9.6 history comparisons."""

from __future__ import annotations

import json

from gpuboost.cli import main as cli_main
from gpuboost.history.compare import compare_history_runs
from gpuboost.history.store import insert_history_run
from gpuboost.schemas.history import HistoryRunRecord


def test_compare_history_runs_no_changes() -> None:
    left = _make_record("left")
    right = _make_record("right")

    result = compare_history_runs(left, right)

    assert result.status == "ok"
    assert result.summary == "No tracked fields changed."
    assert result.changed_fields == {}


def test_compare_history_runs_changed_status() -> None:
    result = compare_history_runs(
        _make_record("left", status="ok"),
        _make_record("right", status="partial"),
    )

    assert result.changed_fields["status"] == "ok -> partial"
    assert result.summary == "Changed fields: status"


def test_compare_history_runs_changed_trial_status() -> None:
    result = compare_history_runs(
        _make_record("left", trial_status="passed"),
        _make_record("right", trial_status="failed"),
    )

    assert result.changed_fields["trial_status"] == "passed -> failed"


def test_compare_history_runs_changed_script_hash() -> None:
    result = compare_history_runs(
        _make_record("left", script_sha256="aaa"),
        _make_record("right", script_sha256="bbb"),
    )

    assert result.changed_fields["script_sha256"] == "aaa -> bbb"


def test_compare_history_runs_omits_raw_payloads() -> None:
    result = compare_history_runs(
        _make_record("left"),
        _make_record("right", status="partial"),
    )
    data = result.to_dict()

    assert "source code" not in str(data)
    assert "--- train.py" not in str(data)
    assert "stdout" not in str(data)
    assert "stderr" not in str(data)


def test_history_compare_human_output(tmp_path, capsys) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_make_record("left", status="ok"), db_path=db_path)
    insert_history_run(_make_record("right", status="partial"), db_path=db_path)

    exit_code = cli_main.main(
        ["history", "compare", "left", "right", "--db-path", str(db_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GPUBoost History Compare" in captured.out
    assert "Left: left" in captured.out
    assert "Right: right" in captured.out
    assert "- status: ok -> partial" in captured.out


def test_history_compare_json_valid_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_make_record("left"), db_path=db_path)
    insert_history_run(_make_record("right"), db_path=db_path)

    exit_code = cli_main.main(
        ["history", "compare", "left", "right", "--db-path", str(db_path), "--json"]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 0
    assert data["schema_version"] == "history.compare.v1"
    assert data["comparison"]["summary"] == "No tracked fields changed."


def test_history_compare_missing_left_run_clean_error(tmp_path, capsys) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_make_record("right"), db_path=db_path)

    exit_code = cli_main.main(
        ["history", "compare", "missing", "right", "--db-path", str(db_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "History run not found: missing" in captured.out
    assert "Traceback" not in captured.out


def test_history_compare_missing_right_run_clean_error(tmp_path, capsys) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_make_record("left"), db_path=db_path)

    exit_code = cli_main.main(
        ["history", "compare", "left", "missing", "--db-path", str(db_path), "--json"]
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert exit_code == 1
    assert data["comparison"] is None
    assert data["error"] == "History run not found: missing"


def test_history_compare_db_path_respected(tmp_path, capsys) -> None:
    first_db = tmp_path / "first.db"
    second_db = tmp_path / "second.db"
    insert_history_run(_make_record("left"), db_path=first_db)
    insert_history_run(_make_record("right"), db_path=second_db)

    exit_code = cli_main.main(
        ["history", "compare", "left", "right", "--db-path", str(first_db)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "History run not found: right" in captured.out


def _make_record(
    run_id: str,
    *,
    status: str = "ok",
    script_sha256: str = "abc",
    trial_status: str = "passed",
) -> HistoryRunRecord:
    return HistoryRunRecord(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        status=status,
        command="agent optimize",
        schema_version="history.run.v1",
        goal_kind="optimize_script",
        goal_description="Optimize train.py",
        script_path="train.py",
        script_sha256=script_sha256,
        gpu_name="NVIDIA Test GPU",
        cuda_available=True,
        trial_summary={"status": trial_status},
        comparison_summary={"overall_verdict": "improved"},
        metadata={
            "has_diff": False,
            "has_trial": True,
            "has_comparison": True,
        },
    )
