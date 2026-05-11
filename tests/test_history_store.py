"""Tests for Phase 9.2 SQLite history store."""

from __future__ import annotations

import sqlite3

from gpuboost.history.store import (
    default_history_db_path,
    default_history_dir,
    delete_history_run,
    initialize_history_store,
    insert_history_run,
    list_history_runs,
    load_history_run,
)
from gpuboost.schemas.history import HistoryRunRecord, HistorySummary


def test_initialize_history_store_creates_db_and_parent_dir(tmp_path) -> None:
    db_path = tmp_path / "nested" / "history.db"

    resolved = initialize_history_store(db_path)

    assert resolved == db_path
    assert db_path.exists()
    assert db_path.parent.exists()


def test_insert_and_load_record_round_trip(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    record = _make_record("run-001")

    insert_history_run(record, db_path=db_path)
    loaded = load_history_run("run-001", db_path=db_path)

    assert loaded == record


def test_insert_replaces_same_run_id(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_make_record("run-001", status="partial"), db_path=db_path)
    insert_history_run(_make_record("run-001", status="ok"), db_path=db_path)

    loaded = load_history_run("run-001", db_path=db_path)

    assert loaded is not None
    assert loaded.status == "ok"


def test_load_missing_returns_none(tmp_path) -> None:
    db_path = tmp_path / "history.db"

    assert load_history_run("missing", db_path=db_path) is None


def test_list_history_runs_returns_newest_first(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(
        _make_record("older", created_at="2026-01-01T00:00:00+00:00"),
        db_path=db_path,
    )
    insert_history_run(
        _make_record("newer", created_at="2026-01-02T00:00:00+00:00"),
        db_path=db_path,
    )

    summary = list_history_runs(db_path=db_path)

    assert [record.run_id for record in summary.runs] == ["newer", "older"]


def test_list_history_runs_respects_limit(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(
        _make_record("run-001", created_at="2026-01-01T00:00:00+00:00"),
        db_path=db_path,
    )
    insert_history_run(
        _make_record("run-002", created_at="2026-01-02T00:00:00+00:00"),
        db_path=db_path,
    )

    summary = list_history_runs(limit=1, db_path=db_path)

    assert summary.total_runs == 1
    assert [record.run_id for record in summary.runs] == ["run-002"]


def test_empty_store_returns_history_summary_with_total_runs_zero(tmp_path) -> None:
    summary = list_history_runs(db_path=tmp_path / "history.db")

    assert isinstance(summary, HistorySummary)
    assert summary.total_runs == 0
    assert summary.runs == []


def test_delete_history_run_returns_true_and_false(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_make_record("run-001"), db_path=db_path)

    assert delete_history_run("run-001", db_path=db_path) is True
    assert delete_history_run("run-001", db_path=db_path) is False


def test_corrupt_record_json_raises_value_error(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    initialize_history_store(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO history_runs (
                run_id,
                created_at,
                status,
                command,
                schema_version,
                goal_kind,
                goal_description,
                record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "bad",
                "2026-01-01T00:00:00+00:00",
                "ok",
                "agent optimize",
                "history.run.v1",
                "optimize",
                "Optimize.",
                "{bad json",
            ),
        )

    try:
        load_history_run("bad", db_path=db_path)
    except ValueError as error:
        assert "run_id 'bad' is corrupt" in str(error)
    else:
        raise AssertionError("Expected ValueError for corrupt history JSON.")


def test_no_raw_source_code_field_exists_in_table_or_record(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    record = _make_record("run-001")
    insert_history_run(record, db_path=db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(history_runs)").fetchall()
        }
        record_json = connection.execute(
            "SELECT record_json FROM history_runs WHERE run_id = ?",
            ("run-001",),
        ).fetchone()[0]

    assert "source_code" not in columns
    assert "raw_source" not in columns
    assert "code" not in columns
    assert "source_code" not in record.to_dict()
    assert "raw_source" not in record.to_dict()
    assert "source_code" not in record_json
    assert "raw_source" not in record_json


def test_cuda_available_stored_and_reconstructed_correctly(tmp_path) -> None:
    db_path = tmp_path / "history.db"
    insert_history_run(_make_record("true", cuda_available=True), db_path=db_path)
    insert_history_run(_make_record("false", cuda_available=False), db_path=db_path)
    insert_history_run(_make_record("none", cuda_available=None), db_path=db_path)

    with sqlite3.connect(db_path) as connection:
        rows = dict(
            connection.execute(
                "SELECT run_id, cuda_available FROM history_runs"
            ).fetchall()
        )

    assert rows == {"true": 1, "false": 0, "none": None}
    assert load_history_run("true", db_path=db_path).cuda_available is True
    assert load_history_run("false", db_path=db_path).cuda_available is False
    assert load_history_run("none", db_path=db_path).cuda_available is None


def test_default_path_helpers_return_expected_suffixes() -> None:
    history_dir = default_history_dir()
    db_path = default_history_db_path()

    assert history_dir.name == ".gpuboost"
    assert db_path.name == "gpuboost.db"
    assert db_path.parent == history_dir


def _make_record(
    run_id: str,
    *,
    created_at: str = "2026-01-01T00:00:00+00:00",
    status: str = "ok",
    cuda_available: bool | None = True,
) -> HistoryRunRecord:
    return HistoryRunRecord(
        run_id=run_id,
        created_at=created_at,
        status=status,
        command="agent optimize",
        schema_version="history.run.v1",
        goal_kind="optimize_script",
        goal_description="Optimize train.py for NVIDIA GPU performance",
        script_path="train.py",
        script_sha256="abc123",
        gpu_name="NVIDIA Test GPU",
        cuda_available=cuda_available,
        benchmark_summary={"metric_count": 1},
        action_statuses={"inspect_system": "completed"},
        metadata={"event_count": 1},
    )
