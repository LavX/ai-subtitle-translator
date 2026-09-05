"""End-to-end persistence tests: save jobs, destroy manager, reopen DB, verify recovery."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from subtitle_translator.crypto import generate_key
from subtitle_translator.queue.job_manager import Job, JobManager, JobStatus, JobType
from subtitle_translator.queue.job_store import JobStore


def _make_job(
    job_id: str,
    status: JobStatus = JobStatus.QUEUED,
    api_key_override: str | None = None,
    result=None,
    error: str | None = None,
) -> Job:
    return Job(
        id=job_id,
        job_type=JobType.TRANSLATE_CONTENT,
        status=status,
        progress=100 if status == JobStatus.COMPLETED else 0,
        message="",
        request_data={"sourceLanguage": "en", "targetLanguage": "hu", "lines": []},
        api_key_override=api_key_override,
        result=result,
        error=error,
        created_at=datetime.now(UTC),
        started_at=datetime.now(UTC) if status != JobStatus.QUEUED else None,
        completed_at=datetime.now(UTC) if status in (JobStatus.COMPLETED, JobStatus.FAILED) else None,
        job_name="test",
        source_language="en",
        target_language="hu",
        model="test-model",
        total_lines=10,
    )


class TestRestartSimulation:
    """Simulate a full container restart: save, close, reopen, recover."""

    @pytest.mark.asyncio
    async def test_completed_jobs_survive_restart(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "jobs.db")

        # Session 1: create manager, submit and complete jobs
        store1 = JobStore(db_path)
        manager1 = JobManager()
        manager1.set_store(store1)

        completed_job = _make_job("job-completed", JobStatus.COMPLETED, result={"lines": [{"position": 0, "line": "Hola"}]})
        manager1.jobs[completed_job.id] = completed_job
        store1.save_job(completed_job)

        queued_job = _make_job("job-queued", JobStatus.QUEUED)
        manager1.jobs[queued_job.id] = queued_job
        store1.save_job(queued_job)

        failed_job = _make_job("job-failed", JobStatus.FAILED, error="timeout")
        manager1.jobs[failed_job.id] = failed_job
        store1.save_job(failed_job)

        # Destroy session 1
        store1.close()
        del manager1

        # Session 2: fresh manager, same DB file
        store2 = JobStore(db_path)
        manager2 = JobManager()
        manager2.set_store(store2)
        requeued = await manager2.recover_jobs()

        # Queued job should be re-queued
        assert requeued == 1

        # All three jobs should be in memory
        assert manager2.get_job("job-completed") is not None
        assert manager2.get_job("job-completed").status == JobStatus.COMPLETED
        assert manager2.get_job("job-completed").result == {"lines": [{"position": 0, "line": "Hola"}]}

        assert manager2.get_job("job-queued") is not None
        assert manager2.get_job("job-queued").status == JobStatus.QUEUED
        assert manager2.get_job("job-queued").message == "Recovered after restart"

        assert manager2.get_job("job-failed") is not None
        assert manager2.get_job("job-failed").status == JobStatus.FAILED
        assert manager2.get_job("job-failed").error == "timeout"

        store2.close()

    @pytest.mark.asyncio
    async def test_encrypted_api_key_survives_restart(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "jobs.db")
        crypto_key = generate_key()

        # Session 1: save job with encrypted API key
        store1 = JobStore(db_path, crypto_key=crypto_key)
        job = _make_job("job-enc", JobStatus.QUEUED, api_key_override="sk-or-secret-key-123")
        store1.save_job(job)
        store1.close()

        # Session 2: reopen with same key, verify decryption
        store2 = JobStore(db_path, crypto_key=crypto_key)
        manager2 = JobManager()
        manager2.set_store(store2)
        requeued = await manager2.recover_jobs()

        assert requeued == 1
        recovered = manager2.get_job("job-enc")
        assert recovered is not None
        assert recovered.api_key_override == "sk-or-secret-key-123"

        store2.close()

    @pytest.mark.asyncio
    async def test_wrong_key_skips_corrupt_job(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "jobs.db")
        key1 = generate_key()
        key2 = generate_key()

        # Session 1: save with key1
        store1 = JobStore(db_path, crypto_key=key1)
        store1.save_job(_make_job("job-k1", JobStatus.QUEUED, api_key_override="secret"))
        store1.save_job(_make_job("job-nokey", JobStatus.COMPLETED))  # no API key, should survive
        store1.close()

        # Session 2: open with key2 (wrong key)
        store2 = JobStore(db_path, crypto_key=key2)
        manager2 = JobManager()
        manager2.set_store(store2)
        await manager2.recover_jobs()

        # job-k1 should be skipped (decrypt fails), job-nokey should load fine
        assert manager2.get_job("job-nokey") is not None
        assert manager2.get_job("job-nokey").status == JobStatus.COMPLETED

        store2.close()

    @pytest.mark.asyncio
    async def test_plaintext_api_key_not_persisted_without_encryption(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "jobs.db")

        # No crypto key = encryption disabled
        store = JobStore(db_path, crypto_key=None)
        job = _make_job("job-plain", JobStatus.QUEUED, api_key_override="sk-or-my-secret")
        store.save_job(job)

        # Verify the raw DB does NOT contain the plaintext key
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT api_key_override FROM jobs WHERE id = ?", ("job-plain",)).fetchone()
        conn.close()

        assert row["api_key_override"] is None

        store.close()

    @pytest.mark.asyncio
    async def test_processing_job_requeued_as_queued_on_restart(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "jobs.db")

        store1 = JobStore(db_path)
        job = _make_job("job-proc", JobStatus.PROCESSING)
        store1.save_job(job)
        store1.close()

        store2 = JobStore(db_path)
        manager2 = JobManager()
        manager2.set_store(store2)
        requeued = await manager2.recover_jobs()

        assert requeued == 1
        recovered = manager2.get_job("job-proc")
        assert recovered.status == JobStatus.QUEUED
        assert recovered.message == "Recovered after restart"

        store2.close()
