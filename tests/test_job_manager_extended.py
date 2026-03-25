"""Extended tests for job_manager.py covering missing lines."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from subtitle_translator.queue.job_manager import Job, JobManager, JobStatus, JobType


@pytest.fixture
def manager():
    """Create a fresh JobManager for each test."""
    return JobManager(max_concurrent=2, max_jobs=10, job_ttl_hours=1)


def _make_job(job_id="test-1", status=JobStatus.QUEUED, **kwargs):
    defaults = {
        "id": job_id,
        "job_type": JobType.TRANSLATE_CONTENT,
        "status": status,
        "request_data": {"sourceLanguage": "en", "targetLanguage": "hu", "lines": []},
        "created_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return Job(**defaults)


# --- recover_jobs with no store (line 135) ---


class TestRecoverJobs:
    @pytest.mark.asyncio
    async def test_recover_jobs_no_store_returns_zero(self, manager):
        assert manager._store is None
        result = await manager.recover_jobs()
        assert result == 0


# --- update_progress for non-existent job (line 306) ---


class TestUpdateProgress:
    def test_update_progress_nonexistent_job_does_nothing(self, manager):
        # Should not raise, just silently return
        manager.update_progress("nonexistent-id", 50, message="halfway")
        assert "nonexistent-id" not in manager.jobs

    def test_update_progress_with_store_persists(self, manager):
        store = MagicMock()
        manager.set_store(store)
        job = _make_job(status=JobStatus.PROCESSING)
        manager.jobs["test-1"] = job

        manager.update_progress("test-1", 50, message="halfway")
        store.save_job.assert_called_once_with(job)
        assert job.progress == 50


# --- set_job_partial (lines 355-366) ---


class TestSetJobPartial:
    def test_set_job_partial_sets_status_and_result(self, manager):
        job = _make_job(status=JobStatus.PROCESSING)
        manager.jobs["test-1"] = job

        manager.set_job_partial("test-1", result={"lines": [1, 2]}, error="batch 3 failed")

        assert job.status == JobStatus.PARTIAL
        assert job.result == {"lines": [1, 2]}
        assert job.error == "batch 3 failed"
        assert job.completed_at is not None
        assert "batch 3 failed" in job.message

    def test_set_job_partial_calculates_progress_from_batches(self, manager):
        job = _make_job(status=JobStatus.PROCESSING)
        job.total_batches = 10
        job.completed_batches = 7
        manager.jobs["test-1"] = job

        manager.set_job_partial("test-1", result={"partial": True}, error="timeout")

        assert job.progress == 70

    def test_set_job_partial_zero_total_batches_skips_progress(self, manager):
        job = _make_job(status=JobStatus.PROCESSING)
        job.total_batches = 0
        job.completed_batches = 0
        manager.jobs["test-1"] = job

        manager.set_job_partial("test-1", result={}, error="oops")

        # progress should remain at default (0), not crash on division
        assert job.progress == 0

    def test_set_job_partial_nonexistent_does_nothing(self, manager):
        manager.set_job_partial("no-such-id", result={}, error="err")
        assert "no-such-id" not in manager.jobs

    def test_set_job_partial_with_store(self, manager):
        store = MagicMock()
        manager.set_store(store)
        job = _make_job(status=JobStatus.PROCESSING)
        manager.jobs["test-1"] = job

        manager.set_job_partial("test-1", result={}, error="partial fail")
        store.save_job.assert_called_once_with(job)


# --- cancel_job edge cases (lines 388, 410) ---


class TestCancelJob:
    def test_cancel_nonexistent_job_returns_false(self, manager):
        assert manager.cancel_job("does-not-exist") is False

    def test_cancel_processing_job_returns_false(self, manager):
        job = _make_job(status=JobStatus.PROCESSING)
        manager.jobs["test-1"] = job
        assert manager.cancel_job("test-1") is False
        assert job.status == JobStatus.PROCESSING

    def test_cancel_queued_job_with_store(self, manager):
        store = MagicMock()
        manager.set_store(store)
        job = _make_job(status=JobStatus.QUEUED)
        manager.jobs["test-1"] = job

        result = manager.cancel_job("test-1")
        assert result is True
        store.save_job.assert_called_once_with(job)


# --- delete_job edge cases (lines 410, 436) ---


class TestDeleteJob:
    def test_delete_nonexistent_job_returns_false(self, manager):
        assert manager.delete_job("does-not-exist") is False

    def test_delete_active_job_returns_false(self, manager):
        job = _make_job(status=JobStatus.PROCESSING)
        manager.jobs["test-1"] = job
        assert manager.delete_job("test-1") is False
        assert "test-1" in manager.jobs

    def test_delete_queued_job_returns_false(self, manager):
        job = _make_job(status=JobStatus.QUEUED)
        manager.jobs["test-1"] = job
        assert manager.delete_job("test-1") is False

    def test_delete_completed_job_with_store(self, manager):
        store = MagicMock()
        manager.set_store(store)
        job = _make_job(status=JobStatus.COMPLETED)
        manager.jobs["test-1"] = job

        result = manager.delete_job("test-1")
        assert result is True
        assert "test-1" not in manager.jobs
        store.delete_job.assert_called_once_with("test-1")


# --- get_queue_position edge cases (lines 436, 487) ---


class TestGetQueuePosition:
    def test_position_nonexistent_job_returns_none(self, manager):
        assert manager.get_queue_position("no-such-id") is None

    def test_position_non_queued_job_returns_none(self, manager):
        job = _make_job(status=JobStatus.PROCESSING)
        manager.jobs["test-1"] = job
        assert manager.get_queue_position("test-1") is None

    def test_position_completed_job_returns_none(self, manager):
        job = _make_job(status=JobStatus.COMPLETED)
        manager.jobs["test-1"] = job
        assert manager.get_queue_position("test-1") is None


# --- _worker edge cases (lines 594-596, 603, 606-607) ---


class TestWorker:
    @pytest.mark.asyncio
    async def test_worker_skips_missing_job(self, manager):
        """Job removed between queue put and worker pickup (line 594-596)."""
        handler = AsyncMock()
        manager.set_worker_handler(handler)
        # Put a job ID in the queue but don't add it to manager.jobs
        await manager.queue.put(("ghost-job", JobType.TRANSLATE_CONTENT))

        task = asyncio.create_task(manager._worker(0))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_skips_cancelled_job(self, manager):
        """Job cancelled before worker picks it up (line 603, 606-607)."""
        handler = AsyncMock()
        manager.set_worker_handler(handler)
        job = _make_job(status=JobStatus.CANCELLED)
        manager.jobs["test-1"] = job
        await manager.queue.put(("test-1", JobType.TRANSLATE_CONTENT))

        task = asyncio.create_task(manager._worker(0))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        handler.assert_not_called()
        # Job status should remain cancelled, not changed to processing
        assert manager.jobs["test-1"].status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_worker_skips_completed_job(self, manager):
        """Job already completed somehow."""
        handler = AsyncMock()
        manager.set_worker_handler(handler)
        job = _make_job(status=JobStatus.COMPLETED)
        manager.jobs["test-1"] = job
        await manager.queue.put(("test-1", JobType.TRANSLATE_CONTENT))

        task = asyncio.create_task(manager._worker(0))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_unexpected_exception_in_queue_get(self, manager):
        """Unexpected error during queue.get triggers except block (lines 594-596)."""
        handler = AsyncMock()
        manager.set_worker_handler(handler)

        # Monkey-patch queue.get to raise a non-CancelledError exception once,
        # then raise CancelledError to stop the loop.
        call_count = 0

        async def exploding_get():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("unexpected queue error")
            # On second call, block until cancelled
            raise asyncio.CancelledError()

        manager.queue.get = exploding_get

        task = asyncio.create_task(manager._worker(0))
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        handler.assert_not_called()
        assert call_count >= 1


# --- _cleanup_loop exception handling (lines 606-607) ---


class TestCleanupLoopError:
    @pytest.mark.asyncio
    async def test_cleanup_loop_survives_unexpected_error(self, manager):
        """Non-CancelledError in cleanup loop is caught (lines 606-607)."""
        call_count = 0

        async def bad_cleanup():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("cleanup boom")

        manager._cleanup_expired_jobs = bad_cleanup

        # Patch sleep to be very short so we actually hit the cleanup call
        original_sleep = asyncio.sleep

        async def fast_sleep(seconds):
            await original_sleep(0.01)

        old_sleep = asyncio.sleep
        asyncio.sleep = fast_sleep
        try:
            task = asyncio.create_task(manager._cleanup_loop())
            await original_sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            asyncio.sleep = old_sleep

        assert call_count >= 1


# --- get_queue_position fallback return None (line 487) ---


class TestGetQueuePositionFallback:
    def test_position_returns_none_when_job_not_in_queued_list(self, manager):
        """
        Cover the fallback return None at line 487.
        This happens if a job has status QUEUED but its id is not found
        when iterating queued_jobs, which is theoretically unreachable
        but we can trigger it by mutating the job mid-iteration.
        """
        job = _make_job(job_id="test-1", status=JobStatus.QUEUED)
        manager.jobs["test-1"] = job

        # Patch the job's id to something different so the loop won't match
        original_id = job.id
        job.id = "tampered-id"
        result = manager.get_queue_position("test-1")
        job.id = original_id  # restore

        assert result is None


# --- set_job_failed with store (line 388) ---


class TestSetJobFailed:
    def test_set_job_failed_with_store(self, manager):
        store = MagicMock()
        manager.set_store(store)
        job = _make_job(status=JobStatus.PROCESSING)
        manager.jobs["test-1"] = job

        manager.set_job_failed("test-1", "some error")
        store.save_job.assert_called_once_with(job)
        assert job.status == JobStatus.FAILED


# --- _cleanup_expired_jobs with store (lines 631-632) ---


class TestCleanupExpiredJobs:
    @pytest.mark.asyncio
    async def test_cleanup_calls_store_cleanup(self, manager):
        store = MagicMock()
        manager.set_store(store)

        await manager._cleanup_expired_jobs()
        store.cleanup_expired.assert_called_once_with(1)  # job_ttl_hours=1

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired_jobs(self, manager):
        old_time = datetime.now(UTC) - timedelta(hours=2)
        job = _make_job(
            status=JobStatus.COMPLETED,
            completed_at=old_time,
        )
        manager.jobs["test-1"] = job

        await manager._cleanup_expired_jobs()
        assert "test-1" not in manager.jobs

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_jobs(self, manager):
        job = _make_job(
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(UTC),
        )
        manager.jobs["test-1"] = job

        await manager._cleanup_expired_jobs()
        assert "test-1" in manager.jobs


# --- _cleanup_loop error handling (lines 603, 606-607) ---


class TestCleanupLoop:
    @pytest.mark.asyncio
    async def test_cleanup_loop_handles_cancellation(self, manager):
        """The loop breaks on CancelledError (line 604-605)."""
        task = asyncio.create_task(manager._cleanup_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.done()
