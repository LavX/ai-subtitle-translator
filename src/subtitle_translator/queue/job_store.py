"""SQLite persistence layer for translation jobs."""

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from subtitle_translator.crypto import decrypt, encrypt
from subtitle_translator.queue.job_manager import Job, JobStatus, JobType

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
)
"""

_CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    request_data TEXT NOT NULL,
    api_key_override TEXT,
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    job_name TEXT,
    file_name TEXT,
    source_language TEXT,
    target_language TEXT,
    title TEXT,
    media_type TEXT,
    model TEXT,
    total_lines INTEGER,
    total_batches INTEGER,
    completed_batches INTEGER NOT NULL DEFAULT 0,
    completed_lines INTEGER NOT NULL DEFAULT 0,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL DEFAULT 0.0
)
"""


def _dt_to_str(dt: datetime | None) -> str | None:
    """Serialize a datetime to an ISO8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(value: str | None) -> datetime | None:
    """Deserialize an ISO8601 string to a UTC-aware datetime, or None."""
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _row_to_job(row: sqlite3.Row, crypto_key: bytes | None) -> Job:
    """Convert a DB row to a Job instance."""
    api_key: str | None = row["api_key_override"]
    if api_key is not None and crypto_key is not None and api_key.startswith("enc:"):
        api_key = decrypt(api_key, crypto_key)

    result_raw: str | None = row["result"]
    result: Any = json.loads(result_raw) if result_raw is not None else None

    return Job(
        id=row["id"],
        job_type=JobType(row["job_type"]),
        status=JobStatus(row["status"]),
        progress=row["progress"],
        message=row["message"],
        request_data=json.loads(row["request_data"]),
        api_key_override=api_key,
        result=result,
        error=row["error"],
        created_at=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
        started_at=_str_to_dt(row["started_at"]),
        completed_at=_str_to_dt(row["completed_at"]),
        job_name=row["job_name"],
        file_name=row["file_name"],
        source_language=row["source_language"],
        target_language=row["target_language"],
        title=row["title"],
        media_type=row["media_type"],
        model=row["model"],
        total_lines=row["total_lines"],
        total_batches=row["total_batches"],
        completed_batches=row["completed_batches"],
        completed_lines=row["completed_lines"],
        tokens_used=row["tokens_used"],
        total_cost=row["total_cost"],
    )


class JobStore:
    """SQLite-backed persistent store for translation jobs."""

    def __init__(self, db_path: str, crypto_key: bytes | None = None) -> None:
        """Open (or create) the database and ensure the schema exists.

        Args:
            db_path: Path to the SQLite database file.
            crypto_key: 32-byte AES-256 key used to encrypt api_key_override at
                rest. If None, the field is stored as plaintext.
        """
        self._crypto_key = crypto_key
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        # Restrict DB file permissions (contains encrypted API keys)
        try:
            os.chmod(db_path, 0o600)
        except OSError:
            pass
        logger.debug("JobStore opened: %s", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create tables and insert the schema_version row if needed."""
        with self._conn:
            self._conn.execute(_CREATE_SCHEMA_VERSION)
            self._conn.execute(_CREATE_JOBS)

            row = self._conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,)
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_job(self, job: Job) -> None:
        """INSERT OR REPLACE the job into the database.

        The api_key_override field is encrypted when a crypto_key is set.
        """
        api_key = job.api_key_override
        if api_key is not None and self._crypto_key is not None:
            api_key = encrypt(api_key, self._crypto_key)

        result_json = json.dumps(job.result) if job.result is not None else None

        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    id, job_type, status, progress, message,
                    request_data, api_key_override, result, error,
                    created_at, started_at, completed_at,
                    job_name, file_name, source_language, target_language,
                    title, media_type, model,
                    total_lines, total_batches, completed_batches,
                    completed_lines, tokens_used, total_cost
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
                    job.id,
                    job.job_type.value,
                    job.status.value,
                    job.progress,
                    job.message,
                    json.dumps(job.request_data),
                    api_key,
                    result_json,
                    job.error,
                    _dt_to_str(job.created_at),
                    _dt_to_str(job.started_at),
                    _dt_to_str(job.completed_at),
                    job.job_name,
                    job.file_name,
                    job.source_language,
                    job.target_language,
                    job.title,
                    job.media_type,
                    job.model,
                    job.total_lines,
                    job.total_batches,
                    job.completed_batches,
                    job.completed_lines,
                    job.tokens_used,
                    job.total_cost,
                ),
            )
        logger.debug("Saved job %s (status=%s)", job.id, job.status.value)

    def load_active_jobs(self) -> list[Job]:
        """Return all jobs with status QUEUED or PROCESSING."""
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE status IN (?, ?) ORDER BY created_at ASC",
            (JobStatus.QUEUED.value, JobStatus.PROCESSING.value),
        ).fetchall()
        return [_row_to_job(row, self._crypto_key) for row in rows]

    def load_all_jobs(self, limit: int = 100) -> list[Job]:
        """Return all jobs ordered by created_at descending."""
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_job(row, self._crypto_key) for row in rows]

    def delete_job(self, job_id: str) -> None:
        """Delete a job by ID."""
        with self._conn:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        logger.debug("Deleted job %s", job_id)

    def cleanup_expired(self, retention_hours: int) -> int:
        """Delete completed/partial/failed/cancelled jobs older than retention_hours.

        Returns:
            Number of rows deleted.
        """
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        cutoff_str = _dt_to_str(cutoff)
        terminal_statuses = (
            JobStatus.COMPLETED.value,
            JobStatus.PARTIAL.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        )
        placeholders = ",".join("?" * len(terminal_statuses))
        with self._conn:
            cursor = self._conn.execute(
                f"DELETE FROM jobs WHERE status IN ({placeholders}) AND completed_at < ?",
                (*terminal_statuses, cutoff_str),
            )
        count = cursor.rowcount
        if count:
            logger.info("cleanup_expired: removed %d job(s) older than %dh", count, retention_hours)
        return count

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
        logger.debug("JobStore closed")
