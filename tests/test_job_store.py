"""Tests for the SQLite job persistence layer."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from subtitle_translator.crypto import generate_key
from subtitle_translator.queue.job_manager import Job, JobStatus, JobType
from subtitle_translator.queue.job_store import JobStore


def make_job(
    job_id: str = "test-job-1",
    status: JobStatus = JobStatus.QUEUED,
    api_key_override: str | None = None,
    completed_at: datetime | None = None,
) -> Job:
    """Build a minimal Job for testing."""
    return Job(
        id=job_id,
        job_type=JobType.TRANSLATE_CONTENT,
        status=status,
        progress=0,
        message="test message",
        request_data={"sourceLanguage": "en", "targetLanguage": "hu"},
        api_key_override=api_key_override,
        result=None,
        error=None,
        created_at=datetime.now(UTC),
        started_at=None,
        completed_at=completed_at,
        job_name="my-job",
        file_name="movie.srt",
        source_language="en",
        target_language="hu",
        title="Breaking Bad",
        media_type="Episode",
        model="google/gemini-2.5-flash",
        total_lines=42,
        total_batches=5,
        completed_batches=2,
        completed_lines=20,
        tokens_used=1500,
        total_cost=0.003,
    )


class TestSchemaCreation:
    def test_schema_creation(self, tmp_path: Path) -> None:
        """Fresh DB gets both jobs and schema_version tables."""
        db_path = str(tmp_path / "test.db")
        store = JobStore(db_path)
        store.close()

        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()

        assert "jobs" in tables
        assert "schema_version" in tables


class TestSaveAndLoad:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Save a job then load_all returns it with matching fields."""
        db_path = str(tmp_path / "test.db")
        store = JobStore(db_path)

        job = make_job()
        job.result = {"lines": ["Hello", "World"]}
        job.started_at = datetime.now(UTC)
        store.save_job(job)

        jobs = store.load_all_jobs()
        store.close()

        assert len(jobs) == 1
        loaded = jobs[0]

        assert loaded.id == job.id
        assert loaded.job_type == job.job_type
        assert loaded.status == job.status
        assert loaded.progress == job.progress
        assert loaded.message == job.message
        assert loaded.request_data == job.request_data
        assert loaded.result == job.result
        assert loaded.error == job.error
        assert loaded.job_name == job.job_name
        assert loaded.file_name == job.file_name
        assert loaded.source_language == job.source_language
        assert loaded.target_language == job.target_language
        assert loaded.title == job.title
        assert loaded.media_type == job.media_type
        assert loaded.model == job.model
        assert loaded.total_lines == job.total_lines
        assert loaded.total_batches == job.total_batches
        assert loaded.completed_batches == job.completed_batches
        assert loaded.completed_lines == job.completed_lines
        assert loaded.tokens_used == job.tokens_used
        assert abs(loaded.total_cost - job.total_cost) < 1e-9
        # Datetimes round-trip via ISO8601 (microsecond precision, tz-aware)
        assert loaded.created_at.replace(tzinfo=UTC) == job.created_at.replace(tzinfo=UTC)

    def test_save_overwrite(self, tmp_path: Path) -> None:
        """Saving the same job_id a second time updates the existing row."""
        db_path = str(tmp_path / "test.db")
        store = JobStore(db_path)

        job = make_job()
        store.save_job(job)

        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.message = "done"
        store.save_job(job)

        jobs = store.load_all_jobs()
        store.close()

        assert len(jobs) == 1
        assert jobs[0].status == JobStatus.COMPLETED
        assert jobs[0].progress == 100
        assert jobs[0].message == "done"


class TestLoadActiveJobs:
    def test_load_active_jobs_filters(self, tmp_path: Path) -> None:
        """Only queued and processing jobs are returned by load_active_jobs."""
        db_path = str(tmp_path / "test.db")
        store = JobStore(db_path)

        store.save_job(make_job("job-queued", JobStatus.QUEUED))
        store.save_job(make_job("job-processing", JobStatus.PROCESSING))
        store.save_job(make_job("job-completed", JobStatus.COMPLETED))
        store.save_job(make_job("job-failed", JobStatus.FAILED))
        store.save_job(make_job("job-cancelled", JobStatus.CANCELLED))
        store.save_job(make_job("job-partial", JobStatus.PARTIAL))

        active = store.load_active_jobs()
        store.close()

        active_ids = {j.id for j in active}
        assert active_ids == {"job-queued", "job-processing"}


class TestDeleteJob:
    def test_delete_job(self, tmp_path: Path) -> None:
        """Deleted job is no longer returned by load_all_jobs."""
        db_path = str(tmp_path / "test.db")
        store = JobStore(db_path)

        store.save_job(make_job("job-a"))
        store.save_job(make_job("job-b"))

        store.delete_job("job-a")

        jobs = store.load_all_jobs()
        store.close()

        ids = {j.id for j in jobs}
        assert "job-a" not in ids
        assert "job-b" in ids


class TestCleanupExpired:
    def test_cleanup_expired(self, tmp_path: Path) -> None:
        """Old completed/failed jobs are removed; fresh ones and active ones are kept."""
        db_path = str(tmp_path / "test.db")
        store = JobStore(db_path)

        old_completed_at = datetime.now(UTC) - timedelta(hours=25)
        fresh_completed_at = datetime.now(UTC) - timedelta(minutes=30)

        # Old jobs that should be deleted
        old_completed = make_job(
            "old-completed", JobStatus.COMPLETED, completed_at=old_completed_at
        )
        old_completed.completed_at = old_completed_at
        old_failed = make_job("old-failed", JobStatus.FAILED, completed_at=old_completed_at)
        old_failed.completed_at = old_completed_at

        # Fresh completed job that should be kept
        fresh_completed = make_job(
            "fresh-completed", JobStatus.COMPLETED, completed_at=fresh_completed_at
        )
        fresh_completed.completed_at = fresh_completed_at

        # Active job that should never be deleted by cleanup
        active_job = make_job("active-queued", JobStatus.QUEUED)

        store.save_job(old_completed)
        store.save_job(old_failed)
        store.save_job(fresh_completed)
        store.save_job(active_job)

        deleted = store.cleanup_expired(retention_hours=24)
        jobs = store.load_all_jobs()
        store.close()

        assert deleted == 2
        ids = {j.id for j in jobs}
        assert "old-completed" not in ids
        assert "old-failed" not in ids
        assert "fresh-completed" in ids
        assert "active-queued" in ids


class TestEncryption:
    def test_api_key_encrypted_at_rest(self, tmp_path: Path) -> None:
        """With a crypto_key, the raw DB column stores 'enc:...' not plaintext."""
        db_path = str(tmp_path / "test.db")
        key = generate_key()
        store = JobStore(db_path, crypto_key=key)

        job = make_job(api_key_override="sk-supersecret-key")
        store.save_job(job)
        store.close()

        # Inspect the raw DB value directly
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT api_key_override FROM jobs WHERE id = ?", (job.id,)).fetchone()
        conn.close()

        raw_value = row[0]
        assert raw_value is not None
        assert raw_value.startswith("enc:")
        assert "sk-supersecret-key" not in raw_value

    def test_api_key_roundtrip_with_encryption(self, tmp_path: Path) -> None:
        """Save with crypto_key, load decrypts back to original plaintext."""
        db_path = str(tmp_path / "test.db")
        key = generate_key()
        store = JobStore(db_path, crypto_key=key)

        job = make_job(api_key_override="sk-supersecret-key")
        store.save_job(job)

        jobs = store.load_all_jobs()
        store.close()

        assert len(jobs) == 1
        assert jobs[0].api_key_override == "sk-supersecret-key"

    def test_works_without_encryption(self, tmp_path: Path) -> None:
        """No crypto_key: api_key is NOT persisted (stored as NULL to avoid plaintext on disk)."""
        db_path = str(tmp_path / "test.db")
        store = JobStore(db_path)  # no crypto_key

        job = make_job(api_key_override="sk-plaintext-key")
        store.save_job(job)

        # Raw DB value should be NULL (not stored in plaintext)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT api_key_override FROM jobs WHERE id = ?", (job.id,)).fetchone()
        conn.close()

        assert row[0] is None

        # And loading should return None
        jobs = store.load_all_jobs()
        store.close()
        assert jobs[0].api_key_override is None
