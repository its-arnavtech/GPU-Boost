"""SQLite persistence for local GPUBoost run history."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from gpuboost.schemas.history import (
    HistoryRunRecord,
    HistorySummary,
    create_timestamp,
)


def default_history_dir() -> Path:
    """Return the default local GPUBoost history directory."""

    return Path.home() / ".gpuboost"


def default_history_db_path() -> Path:
    """Return the default local GPUBoost history database path."""

    return default_history_dir() / "gpuboost.db"


def initialize_history_store(db_path: str | Path | None = None) -> Path:
    """Create the local history database and table if needed."""

    resolved_path = _resolve_db_path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolved_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS history_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                command TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                goal_kind TEXT NOT NULL,
                goal_description TEXT NOT NULL,
                script_path TEXT,
                script_sha256 TEXT,
                gpu_name TEXT,
                cuda_available INTEGER,
                record_json TEXT NOT NULL
            )
            """
        )

    return resolved_path


def insert_history_run(
    record: HistoryRunRecord,
    db_path: str | Path | None = None,
) -> None:
    """Insert or replace a history run record."""

    resolved_path = initialize_history_store(db_path)
    record_json = json.dumps(record.to_dict(), sort_keys=True)
    cuda_available = _bool_to_sqlite(record.cuda_available)

    with sqlite3.connect(resolved_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO history_runs (
                run_id,
                created_at,
                status,
                command,
                schema_version,
                goal_kind,
                goal_description,
                script_path,
                script_sha256,
                gpu_name,
                cuda_available,
                record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.created_at,
                record.status,
                record.command,
                record.schema_version,
                record.goal_kind,
                record.goal_description,
                record.script_path,
                record.script_sha256,
                record.gpu_name,
                cuda_available,
                record_json,
            ),
        )


def load_history_run(
    run_id: str,
    db_path: str | Path | None = None,
) -> HistoryRunRecord | None:
    """Load one history run record by run ID."""

    resolved_path = initialize_history_store(db_path)

    with sqlite3.connect(resolved_path) as connection:
        row = connection.execute(
            "SELECT record_json FROM history_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    return _record_from_json(row[0], run_id=run_id)


def list_history_runs(
    limit: int = 20,
    db_path: str | Path | None = None,
) -> HistorySummary:
    """List newest history records first."""

    resolved_path = initialize_history_store(db_path)
    safe_limit = max(0, int(limit))

    with sqlite3.connect(resolved_path) as connection:
        rows = connection.execute(
            """
            SELECT run_id, record_json
            FROM history_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    runs = [_record_from_json(row[1], run_id=row[0]) for row in rows]
    return HistorySummary(
        generated_at=create_timestamp(),
        total_runs=len(runs),
        runs=runs,
    )


def delete_history_run(
    run_id: str,
    db_path: str | Path | None = None,
) -> bool:
    """Delete a history run by run ID."""

    resolved_path = initialize_history_store(db_path)

    with sqlite3.connect(resolved_path) as connection:
        cursor = connection.execute(
            "DELETE FROM history_runs WHERE run_id = ?",
            (run_id,),
        )
        deleted_count = cursor.rowcount

    return deleted_count > 0


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is None:
        return default_history_db_path()
    return Path(db_path)


def _record_from_json(record_json: str, run_id: str) -> HistoryRunRecord:
    try:
        data = json.loads(record_json)
        if not isinstance(data, dict):
            raise TypeError("record_json did not decode to an object")
        return HistoryRunRecord(**data)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"History record for run_id {run_id!r} is corrupt.") from error


def _bool_to_sqlite(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0
