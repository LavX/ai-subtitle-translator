"""Tests for the job manager."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from subtitle_translator.queue.job_manager import Job, JobManager, JobStatus, JobType


@pytest.fixture
def manager():
    """Create a fresh JobManager for each test."""
    return JobManager(max_concurrent=2, max_jobs=5, job_ttl_hours=1)


@pytest.fixture
def metadata():
    """Sample metadata dict."""
    return {
        "job_name": "my-job",
        "file_name": "movie.srt",
        "source_language": "en",
        "target_language": "hu",
        "title": "Breaking Bad",
        "media_type": "Episode",
        "model": "google/gemini-2.5-flash-preview-09-2025",
        "total_lines": 42,
    }


class TestJobSubmission:
    """Tests for submitting jobs."""

    @pytest.mark.asyncio
    async def test_submit_job_returns_id(self, manager):
        job_id = await manager.submit_job(
            request_data={"sourceLanguage": "en"},
            job_type=JobType.TRANSLATE_CONTENT,
        )
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    @pytest.mark.asyncio
    async def test_submit_job_creates_queued_job(self, manager):
        job_id = await manager.submit_job(
            request_data={"sourceLanguage": "en"},
            job_type=JobType.TRANSLATE_CONTENT,
        )
        job = manager.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.QUEUED
        assert job.job_type == JobType.TRANSLATE_CONTENT

    @pytest.mark.asyncio
    async def test_submit_job_with_metadata(self, manager, metadata):
        job_id = await manager.submit_job(
            request_data={"sourceLanguage": "en"},
            job_type=JobType.TRANSLATE_FILE,
            metadata=metadata,
        )
        job = manager.get_job(job_id)
        assert job.job_name == "my-job"
        assert job.file_name == "movie.srt"
        assert job.source_language == "en"
        assert job.target_language == "hu"
        assert job.title == "Breaking Bad"
        assert job.media_type == "Episode"
        assert job.model == "google/gemini-2.5-flash-preview-09-2025"
        assert job.total_lines == 42

    @pytest.mark.asyncio
    async def test_submit_job_without_metadata(self, manager):
        job_id = await manager.submit_job(
            request_data={"sourceLanguage": "en"},
            job_type=JobType.TRANSLATE_CONTENT,
        )
        job = manager.get_job(job_id)
        assert job.job_name is None
        assert job.file_name is None
        assert job.source_language is None
        assert job.total_lines is None

    @pytest.mark.asyncio
    async def test_submit_job_with_api_key_override(self, manager):
        job_id = await manager.submit_job(
            request_data={},
            job_type=JobType.TRANSLATE_CONTENT,
            api_key_override="sk-test-123",
        )
        job = manager.get_job(job_id)
        assert job.api_key_override == "sk-test-123"

    @pytest.mark.asyncio
    async def test_submit_job_max_limit(self, manager):
        for _ in range(5):
            await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        with pytest.raises(RuntimeError, match="Maximum job limit"):
            await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)

    @pytest.mark.asyncio
    async def test_submit_job_adds_to_queue(self, manager):
        await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        assert not manager.queue.empty()


class TestJobRetrieval:
    """Tests for getting jobs."""

    @pytest.mark.asyncio
    async def test_get_existing_job(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        assert manager.get_job(job_id) is not None

    def test_get_nonexistent_job(self, manager):
        assert manager.get_job("nonexistent-id") is None


class TestJobStatusTransitions:
    """Tests for status transitions."""

    @pytest.mark.asyncio
    async def test_set_job_processing(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        job = manager.get_job(job_id)
        assert job.status == JobStatus.PROCESSING
        assert job.started_at is not None
        assert job.message == "Processing translation..."

    @pytest.mark.asyncio
    async def test_set_job_completed(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        manager.set_job_completed(job_id, {"lines": [], "model_used": "test"})
        job = manager.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100
        assert job.completed_at is not None
        assert job.result == {"lines": [], "model_used": "test"}

    @pytest.mark.asyncio
    async def test_set_job_failed(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        manager.set_job_failed(job_id, "Something went wrong")
        job = manager.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert job.error == "Something went wrong"
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        result = manager.cancel_job(job_id)
        assert result is True
        job = manager.get_job(job_id)
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_processing_job_fails(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        result = manager.cancel_job(job_id)
        assert result is False
        assert manager.get_job(job_id).status == JobStatus.PROCESSING

    def test_cancel_nonexistent_job(self, manager):
        assert manager.cancel_job("nonexistent") is False

    @pytest.mark.asyncio
    async def test_set_processing_nonexistent_job(self, manager):
        manager.set_job_processing("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_set_completed_nonexistent_job(self, manager):
        manager.set_job_completed("nonexistent", {})  # should not raise

    @pytest.mark.asyncio
    async def test_set_failed_nonexistent_job(self, manager):
        manager.set_job_failed("nonexistent", "err")  # should not raise


class TestUpdateProgress:
    """Tests for update_progress with metrics."""

    @pytest.mark.asyncio
    async def test_update_progress_basic(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.update_progress(job_id, 50, "Halfway")
        job = manager.get_job(job_id)
        assert job.progress == 50
        assert job.message == "Halfway"

    @pytest.mark.asyncio
    async def test_update_progress_clamps(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.update_progress(job_id, 150)
        assert manager.get_job(job_id).progress == 100
        manager.update_progress(job_id, -10)
        assert manager.get_job(job_id).progress == 0

    @pytest.mark.asyncio
    async def test_update_progress_with_metrics(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.update_progress(
            job_id,
            50,
            "Processing",
            total_batches=10,
            completed_batches=5,
            completed_lines=100,
            tokens_used=5000,
            total_cost=0.03,
        )
        job = manager.get_job(job_id)
        assert job.total_batches == 10
        assert job.completed_batches == 5
        assert job.completed_lines == 100
        assert job.tokens_used == 5000
        assert job.total_cost == pytest.approx(0.03)

    @pytest.mark.asyncio
    async def test_update_progress_partial_metrics(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.update_progress(job_id, 25, tokens_used=1000)
        job = manager.get_job(job_id)
        assert job.tokens_used == 1000
        assert job.total_batches is None  # not set
        assert job.completed_batches == 0  # default

    @pytest.mark.asyncio
    async def test_update_progress_nonexistent_job(self, manager):
        manager.update_progress("nonexistent", 50)  # should not raise

    @pytest.mark.asyncio
    async def test_update_progress_preserves_message(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.update_progress(job_id, 25, "First")
        manager.update_progress(job_id, 50)  # empty message
        job = manager.get_job(job_id)
        assert job.message == "First"  # preserved


class TestDeleteJob:
    """Tests for job deletion."""

    @pytest.mark.asyncio
    async def test_delete_completed_job(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        manager.set_job_completed(job_id, {})
        assert manager.delete_job(job_id) is True
        assert manager.get_job(job_id) is None

    @pytest.mark.asyncio
    async def test_delete_failed_job(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        manager.set_job_failed(job_id, "err")
        assert manager.delete_job(job_id) is True

    @pytest.mark.asyncio
    async def test_delete_cancelled_job(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.cancel_job(job_id)
        assert manager.delete_job(job_id) is True

    @pytest.mark.asyncio
    async def test_cannot_delete_queued_job(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        assert manager.delete_job(job_id) is False

    @pytest.mark.asyncio
    async def test_cannot_delete_processing_job(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        assert manager.delete_job(job_id) is False

    def test_delete_nonexistent_job(self, manager):
        assert manager.delete_job("nonexistent") is False


class TestListJobs:
    """Tests for listing and filtering jobs."""

    @pytest.mark.asyncio
    async def test_list_all_jobs(self, manager):
        await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_FILE)
        jobs = manager.list_jobs()
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_list_jobs_with_status_filter(self, manager):
        id1 = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(id1)
        queued = manager.list_jobs(status_filter=JobStatus.QUEUED)
        assert len(queued) == 1
        processing = manager.list_jobs(status_filter=JobStatus.PROCESSING)
        assert len(processing) == 1

    @pytest.mark.asyncio
    async def test_list_jobs_sorted_newest_first(self, manager):
        id1 = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        id2 = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        jobs = manager.list_jobs()
        assert jobs[0].id == id2
        assert jobs[1].id == id1

    @pytest.mark.asyncio
    async def test_list_jobs_with_limit(self, manager):
        for _ in range(3):
            await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        jobs = manager.list_jobs(limit=2)
        assert len(jobs) == 2


class TestQueuePosition:
    """Tests for queue position tracking."""

    @pytest.mark.asyncio
    async def test_queue_position(self, manager):
        id1 = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        id2 = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        assert manager.get_queue_position(id1) == 1
        assert manager.get_queue_position(id2) == 2

    @pytest.mark.asyncio
    async def test_queue_position_processing_returns_none(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        assert manager.get_queue_position(job_id) is None

    def test_queue_position_nonexistent(self, manager):
        assert manager.get_queue_position("nonexistent") is None


class TestStats:
    """Tests for statistics."""

    @pytest.mark.asyncio
    async def test_stats_empty(self, manager):
        stats = manager.get_stats()
        assert stats["total"] == 0
        assert stats["queued"] == 0

    @pytest.mark.asyncio
    async def test_stats_mixed(self, manager):
        id1 = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(id1)
        manager.set_job_failed(id1, "err")
        stats = manager.get_stats()
        assert stats["total"] == 2
        assert stats["queued"] == 1
        assert stats["failed"] == 1


class TestSetMaxConcurrent:
    """Tests for dynamic worker scaling."""

    @pytest.mark.asyncio
    async def test_set_max_concurrent(self, manager):
        await manager.set_max_concurrent(5)
        assert manager.max_concurrent == 5

    @pytest.mark.asyncio
    async def test_set_max_concurrent_invalid(self, manager):
        with pytest.raises(ValueError):
            await manager.set_max_concurrent(0)
        with pytest.raises(ValueError):
            await manager.set_max_concurrent(11)


class TestJobToDict:
    """Tests for Job.to_dict serialization."""

    @pytest.mark.asyncio
    async def test_to_dict(self, manager, metadata):
        job_id = await manager.submit_job(
            request_data={"sourceLanguage": "en"},
            job_type=JobType.TRANSLATE_CONTENT,
            metadata=metadata,
        )
        job = manager.get_job(job_id)
        d = job.to_dict()
        assert d["jobId"] == job_id
        assert d["jobType"] == "translate_content"
        assert d["status"] == "queued"
        assert d["result"] is None
        assert d["error"] is None


class TestCleanup:
    """Tests for expired job cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_jobs(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        manager.set_job_completed(job_id, {})
        # Manually set completed_at to be past TTL
        job = manager.get_job(job_id)
        job.completed_at = datetime.now(UTC) - timedelta(hours=2)
        await manager._cleanup_expired_jobs()
        assert manager.get_job(job_id) is None

    @pytest.mark.asyncio
    async def test_cleanup_keeps_fresh_jobs(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        manager.set_job_completed(job_id, {})
        await manager._cleanup_expired_jobs()
        assert manager.get_job(job_id) is not None

    @pytest.mark.asyncio
    async def test_cleanup_keeps_active_jobs(self, manager):
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        await manager._cleanup_expired_jobs()
        assert manager.get_job(job_id) is not None


class TestWorkerLifecycle:
    """Tests for worker start/stop."""

    @pytest.mark.asyncio
    async def test_start_without_handler_does_nothing(self, manager):
        await manager.start_workers()
        assert not manager._workers_started

    @pytest.mark.asyncio
    async def test_start_with_handler(self, manager):
        async def dummy_handler(jm, jid, jt):
            pass

        manager.set_worker_handler(dummy_handler)
        await manager.start_workers()
        assert manager._workers_started
        assert len(manager._workers) == 2
        await manager.stop_workers()
        assert not manager._workers_started

    @pytest.mark.asyncio
    async def test_start_workers_twice_warns(self, manager):
        async def dummy_handler(jm, jid, jt):
            pass

        manager.set_worker_handler(dummy_handler)
        await manager.start_workers()
        await manager.start_workers()  # should warn but not crash
        assert manager._workers_started
        await manager.stop_workers()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, manager):
        await manager.stop_workers()  # should not raise


class TestWorkerProcessing:
    """Tests for _worker job processing."""

    @pytest.mark.asyncio
    async def test_worker_processes_job(self, manager):
        processed = []

        async def handler(jm, jid, jt):
            processed.append(jid)
            jm.set_job_completed(jid, {"done": True})

        manager.set_worker_handler(handler)
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        await manager.start_workers()
        # Wait for the job to be processed
        await asyncio.sleep(0.1)
        await manager.stop_workers()

        assert job_id in processed
        job = manager.get_job(job_id)
        assert job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_worker_handles_handler_exception(self, manager):
        async def failing_handler(jm, jid, jt):
            raise ValueError("handler boom")

        manager.set_worker_handler(failing_handler)
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        await manager.start_workers()
        await asyncio.sleep(0.1)
        await manager.stop_workers()

        job = manager.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert "handler boom" in job.error

    @pytest.mark.asyncio
    async def test_worker_skips_missing_job(self, manager):
        async def handler(jm, jid, jt):
            pass

        manager.set_worker_handler(handler)
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        # Remove job before worker picks it up
        del manager.jobs[job_id]
        await manager.start_workers()
        await asyncio.sleep(0.1)
        await manager.stop_workers()

    @pytest.mark.asyncio
    async def test_worker_skips_cancelled_job(self, manager):
        processed = []

        async def handler(jm, jid, jt):
            processed.append(jid)

        manager.set_worker_handler(handler)
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.cancel_job(job_id)
        await manager.start_workers()
        await asyncio.sleep(0.1)
        await manager.stop_workers()

        assert job_id not in processed

    @pytest.mark.asyncio
    async def test_worker_exits_when_exceeds_max_concurrent(self, manager):
        async def handler(jm, jid, jt):
            pass

        manager.set_worker_handler(handler)
        await manager.start_workers()
        # Reduce max_concurrent so excess workers exit
        manager.max_concurrent = 1
        # Submit a job to wake the queue
        await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        await asyncio.sleep(0.2)
        await manager.stop_workers()

    @pytest.mark.asyncio
    async def test_set_max_concurrent_adds_workers(self, manager):
        async def handler(jm, jid, jt):
            pass

        manager.set_worker_handler(handler)
        await manager.start_workers()
        assert len(manager._workers) == 2
        await manager.set_max_concurrent(4)
        assert len(manager._workers) >= 4
        await manager.stop_workers()


class TestStorePersistence:
    """Tests for write-through store integration."""

    @pytest.mark.asyncio
    async def test_write_through_on_submit(self, manager):
        store = MagicMock()
        manager.set_store(store)
        await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        store.save_job.assert_called_once()
        saved_job = store.save_job.call_args[0][0]
        assert saved_job.status == JobStatus.QUEUED

    @pytest.mark.asyncio
    async def test_write_through_on_complete(self, manager):
        store = MagicMock()
        manager.set_store(store)
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        store.save_job.reset_mock()
        manager.set_job_processing(job_id)
        manager.set_job_completed(job_id, {"lines": []})
        # save_job called twice: once for processing, once for completed
        assert store.save_job.call_count == 2
        last_saved = store.save_job.call_args_list[-1][0][0]
        assert last_saved.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_recover_jobs(self):
        manager = JobManager(max_concurrent=2, max_jobs=10)
        store = MagicMock()

        queued_job = Job(
            id="job-queued",
            job_type=JobType.TRANSLATE_CONTENT,
            status=JobStatus.QUEUED,
            request_data={},
            created_at=datetime.now(UTC),
        )
        processing_job = Job(
            id="job-processing",
            job_type=JobType.TRANSLATE_FILE,
            status=JobStatus.PROCESSING,
            request_data={},
            created_at=datetime.now(UTC),
        )
        completed_job = Job(
            id="job-completed",
            job_type=JobType.TRANSLATE_CONTENT,
            status=JobStatus.COMPLETED,
            request_data={},
            created_at=datetime.now(UTC),
        )
        store.load_all_jobs.return_value = [queued_job, processing_job, completed_job]

        manager.set_store(store)
        count = await manager.recover_jobs()

        # Return value is the count of re-queued (active) jobs only
        assert count == 2
        # All 3 jobs loaded into memory
        assert "job-queued" in manager.jobs
        assert "job-processing" in manager.jobs
        assert "job-completed" in manager.jobs
        assert manager.jobs["job-queued"].status == JobStatus.QUEUED
        assert manager.jobs["job-processing"].status == JobStatus.QUEUED
        assert manager.jobs["job-completed"].status == JobStatus.COMPLETED
        assert manager.jobs["job-queued"].message == "Recovered after restart"
        assert manager.jobs["job-processing"].message == "Recovered after restart"
        assert not manager.queue.empty()
        # save_job called once per re-queued job (not for completed)
        assert store.save_job.call_count == 2

    @pytest.mark.asyncio
    async def test_existing_tests_pass_without_store(self, manager):
        # store is None by default; all operations should work without error
        assert manager._store is None
        job_id = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id)
        manager.update_progress(job_id, 50, "halfway")
        manager.set_job_completed(job_id, {"result": "ok"})
        assert manager.get_job(job_id).status == JobStatus.COMPLETED
        assert manager.delete_job(job_id) is True

        job_id2 = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        manager.set_job_processing(job_id2)
        manager.set_job_failed(job_id2, "boom")
        assert manager.get_job(job_id2).status == JobStatus.FAILED

        job_id3 = await manager.submit_job(request_data={}, job_type=JobType.TRANSLATE_CONTENT)
        assert manager.cancel_job(job_id3) is True
