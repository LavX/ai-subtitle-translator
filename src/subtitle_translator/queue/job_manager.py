"""In-memory job queue manager for async translation processing."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from subtitle_translator.queue.job_store import JobStore

from subtitle_translator.queue.job_events import JobEvents

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    """Status of a translation job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(StrEnum):
    """Type of translation job."""

    TRANSLATE_CONTENT = "translate_content"
    TRANSLATE_FILE = "translate_file"


class Job(BaseModel):
    """Represents a translation job in the queue."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    job_type: JobType
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    message: str = ""
    request_data: dict[str, Any]
    api_key_override: str | None = None
    result: Any | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    # Input metadata
    job_name: str | None = None
    file_name: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    title: str | None = None
    media_type: str | None = None
    model: str | None = None
    total_lines: int | None = None
    # Processing metrics
    total_batches: int | None = None
    completed_batches: int = 0
    completed_lines: int = 0
    tokens_used: int = 0
    total_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API response."""
        return {
            "jobId": self.id,
            "jobType": self.job_type.value,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "createdAt": self.created_at.isoformat() + "Z",
            "startedAt": self.started_at.isoformat() + "Z" if self.started_at else None,
            "completedAt": self.completed_at.isoformat() + "Z" if self.completed_at else None,
            "result": self.result if self.status == JobStatus.COMPLETED else None,
            "error": self.error if self.status == JobStatus.FAILED else None,
        }


class JobManager:
    """
    In-memory job queue manager.

    Manages translation jobs using an asyncio-based queue system.
    Jobs are processed by background workers.
    """

    def __init__(
        self,
        max_concurrent: int = 2,
        max_jobs: int = 100,
        job_ttl_hours: int = 1,
    ):
        """
        Initialize the job manager.

        Args:
            max_concurrent: Maximum number of concurrent translation jobs
            max_jobs: Maximum number of jobs to keep in memory
            job_ttl_hours: Time-to-live for completed/failed jobs in hours
        """
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.max_jobs = max_jobs
        self.job_ttl = timedelta(hours=job_ttl_hours)
        self._workers_started = False
        self._workers: list[asyncio.Task] = []
        self._admission = asyncio.Condition()
        self._running_jobs = 0
        self._scheduled_jobs: set[str] = set()
        # Handler task per running job, so a user can interrupt work already started.
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._cancel_requested: set[str] = set()
        self._cleanup_task: asyncio.Task | None = None
        self._worker_handler: Any | None = None
        self._store: JobStore | None = None
        self.events = JobEvents()

    def _publish(self, job: Job) -> None:
        owner = job.request_data.get("_ui_owner")
        if isinstance(owner, str):
            self.events.notify(owner)

    def set_store(self, store: JobStore) -> None:
        """Set the persistent job store for write-through caching."""
        self._store = store

    def set_worker_handler(self, handler: Any) -> None:
        """
        Set the worker handler function.

        The handler should be an async function that takes (job_manager, job_id, job_type).
        """
        self._worker_handler = handler

    async def recover_jobs(self) -> int:
        """Load jobs from persistent store into memory.

        Active jobs (queued/processing) are re-queued for processing.
        Terminal jobs (completed/partial/failed/cancelled) are loaded into
        the in-memory dict so they remain visible via the API after restarts.
        """
        if not self._store:
            return 0

        # Accepted work survives even if the admission limit has since decreased.
        all_jobs = self._store.load_active_jobs() + self._store.load_all_jobs(
            limit=self.max_jobs, terminal_only=True
        )
        requeued = 0
        restored = 0
        for job in all_jobs:
            if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                if job.id in self._scheduled_jobs:
                    continue
                self._scheduled_jobs.add(job.id)
                job.status = JobStatus.QUEUED
                job.message = "Recovered after restart"
                self.jobs[job.id] = job
                await self.queue.put((job.id, job.job_type))
                self._store.save_job(job)
                self._publish(job)
                requeued += 1
            else:
                self.jobs[job.id] = job
                restored += 1

        if requeued:
            logger.info(f"Re-queued {requeued} active jobs from previous session")
        if restored:
            logger.info(f"Restored {restored} completed jobs from previous session")
        return requeued

    async def start_workers(self) -> None:
        """Start background workers to process jobs."""
        if self._workers_started:
            logger.warning("Workers already started")
            return

        if self._worker_handler is None:
            logger.error("Worker handler not set, cannot start workers")
            return

        logger.info(f"Starting {self.max_concurrent} job queue workers")

        for i in range(self.max_concurrent):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._workers_started = True
        logger.info("Job queue workers started successfully")

    async def stop_workers(self) -> None:
        """Stop all workers gracefully."""
        if not self._workers_started:
            return

        logger.info("Stopping job queue workers")

        # Cancel all worker tasks
        for task in self._workers:
            task.cancel()

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Wait for all tasks to complete
        await asyncio.gather(*self._workers, self._cleanup_task, return_exceptions=True)

        self._workers = []
        self._cleanup_task = None
        self._workers_started = False

        logger.info("Job queue workers stopped")

    async def submit_job(
        self,
        request_data: dict[str, Any],
        job_type: JobType,
        api_key_override: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Submit a new job to the queue.

        Args:
            request_data: The request data for the translation job
            job_type: Type of translation job
            api_key_override: API key to use for this job (kept separate from request_data)
            metadata: Optional input metadata (job_name, file_name, languages, etc.)

        Returns:
            The job ID

        Raises:
            RuntimeError: If max jobs limit is reached
        """
        # Check job limit
        active_jobs = sum(
            1 for j in self.jobs.values() if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
        )

        if active_jobs >= self.max_jobs:
            raise RuntimeError(f"Maximum job limit ({self.max_jobs}) reached")

        # Create job
        job_id = str(uuid.uuid4())
        meta = metadata or {}
        job = Job(
            id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            request_data=request_data,
            api_key_override=api_key_override,
            created_at=datetime.now(UTC),
            job_name=meta.get("job_name"),
            file_name=meta.get("file_name"),
            source_language=meta.get("source_language"),
            target_language=meta.get("target_language"),
            title=meta.get("title"),
            media_type=meta.get("media_type"),
            model=meta.get("model"),
            total_lines=meta.get("total_lines"),
        )

        self.jobs[job_id] = job
        self._scheduled_jobs.add(job_id)
        if self._store:
            self._store.save_job(job)
        self._publish(job)
        await self.queue.put((job_id, job_type))

        logger.info(f"Job {job_id} submitted (type: {job_type.value})")

        return job_id

    def get_job(self, job_id: str) -> Job | None:
        """
        Get a job by ID.

        Checks the in-memory dict first, then falls back to the persistent
        store for jobs that were cleaned from memory but still on disk.

        Args:
            job_id: The job ID

        Returns:
            The job if found, None otherwise
        """
        job = self.jobs.get(job_id)
        if job is not None:
            return job
        # Fallback to persistent store (job may have been cleaned from memory)
        if self._store:
            job = self._store.load_job(job_id)
            if job is not None:
                self.jobs[job_id] = job
                return job
        return None

    def update_progress(
        self,
        job_id: str,
        progress: int,
        message: str = "",
        total_batches: int | None = None,
        completed_batches: int | None = None,
        completed_lines: int | None = None,
        tokens_used: int | None = None,
        total_cost: float | None = None,
    ) -> None:
        """
        Update job progress and metrics.

        Args:
            job_id: The job ID
            progress: Progress percentage (0-100)
            message: Optional status message
            total_batches: Total number of batches
            completed_batches: Number of completed batches
            completed_lines: Number of lines translated so far
            tokens_used: Total tokens consumed so far
            total_cost: Total cost in USD so far
        """
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.progress = min(max(progress, 0), 100)
            if message:
                job.message = message
            if total_batches is not None:
                job.total_batches = total_batches
            if completed_batches is not None:
                job.completed_batches = completed_batches
            if completed_lines is not None:
                job.completed_lines = completed_lines
            if tokens_used is not None:
                job.tokens_used = tokens_used
            if total_cost is not None:
                job.total_cost = total_cost
            if self._store:
                self._store.save_job(job)
            self._publish(job)

    def set_job_model(self, job_id: str, model: str) -> None:
        """Persist the effective model selected when a job starts."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.model = model
            if self._store:
                self._store.save_job(job)
            self._publish(job)

    def set_job_processing(self, job_id: str) -> None:
        """Mark a job as processing."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(UTC)
            job.message = "Processing translation..."
            if self._store:
                self._store.save_job(job)
            self._publish(job)

    def set_job_completed(
        self,
        job_id: str,
        result: Any,
    ) -> None:
        """
        Mark a job as completed.

        Args:
            job_id: The job ID
            result: The translation result
        """
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.result = result
            job.completed_at = datetime.now(UTC)
            job.message = "Translation completed"
            logger.info(f"Job {job_id} completed successfully")
            if self._store:
                self._store.save_job(job)
            self._publish(job)

    def set_job_partial(
        self,
        job_id: str,
        result: Any,
        error: str,
    ) -> None:
        """
        Mark a job as partially completed.

        Args:
            job_id: The job ID
            result: The partial translation result
            error: Description of what failed
        """
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = JobStatus.PARTIAL
            job.result = result
            job.error = error
            job.completed_at = datetime.now(UTC)
            if job.total_batches and job.total_batches > 0:
                job.progress = int((job.completed_batches / job.total_batches) * 100)
            job.message = f"Partial translation: {error}"
            logger.warning(f"Job {job_id} partially completed: {error}")
            if self._store:
                self._store.save_job(job)
            self._publish(job)

    def set_job_failed(
        self,
        job_id: str,
        error: str,
    ) -> None:
        """
        Mark a job as failed.

        Args:
            job_id: The job ID
            error: Error message
        """
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = JobStatus.FAILED
            job.error = error
            job.completed_at = datetime.now(UTC)
            job.message = f"Translation failed: {error}"
            logger.error(f"Job {job_id} failed: {error}")
            if self._store:
                self._store.save_job(job)
            self._publish(job)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job if it's still queued.

        Args:
            job_id: The job ID

        Returns:
            True if job was cancelled, False otherwise
        """
        job = self.jobs.get(job_id)
        if job is None:
            return False

        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(UTC)
            job.message = "Job cancelled by user"
            logger.info(f"Job {job_id} cancelled")
            if self._store:
                self._store.save_job(job)
            self._publish(job)
            return True

        # A running job is interrupted through its handler task. The worker records
        # the cancelled status once the handler has actually stopped, so partial
        # progress it wrote stays consistent with what was really done.
        task = self._active_tasks.get(job_id)
        if job.status == JobStatus.PROCESSING and task is not None and not task.done():
            self._cancel_requested.add(job_id)
            task.cancel()
            logger.info(f"Job {job_id}: cancellation requested while processing")
            return True

        return False

    def is_cancelling(self, job_id: str) -> bool:
        """Whether a running job has been asked to stop and has not recorded it yet."""
        return job_id in self._cancel_requested

    def _record_cancelled(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(UTC)
        done = (
            f" after {job.completed_batches}/{job.total_batches} batches"
            if job.total_batches
            else ""
        )
        job.message = f"Job cancelled by user{done}"
        logger.info(f"Job {job_id} cancelled while processing{done}")
        if self._store:
            self._store.save_job(job)
        self._publish(job)

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from memory.

        Args:
            job_id: The job ID

        Returns:
            True if job was deleted, False if not found
        """
        if job_id in self.jobs:
            job = self.jobs[job_id]
            # Only delete completed, partial, failed, or cancelled jobs
            if job.status in (
                JobStatus.COMPLETED,
                JobStatus.PARTIAL,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                del self.jobs[job_id]
                if self._store:
                    self._store.delete_job(job_id)
                self._publish(job)
                logger.info(f"Job {job_id} deleted")
                return True
        return False

    def list_jobs(
        self,
        status_filter: JobStatus | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """
        List jobs with optional filtering.

        Args:
            status_filter: Optional status to filter by
            limit: Maximum number of jobs to return

        Returns:
            List of jobs
        """
        jobs = list(self.jobs.values())

        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]

        # Sort by created_at descending (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def get_queue_position(self, job_id: str) -> int | None:
        """
        Get the position of a job in the queue.

        Args:
            job_id: The job ID

        Returns:
            Position (1-based) if queued, None otherwise
        """
        job = self.jobs.get(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            return None

        queued_jobs = [j for j in self.jobs.values() if j.status == JobStatus.QUEUED]
        queued_jobs.sort(key=lambda j: j.created_at)

        for i, j in enumerate(queued_jobs):
            if j.id == job_id:
                return i + 1

        return None

    def get_stats(self) -> dict[str, int]:
        """
        Get job queue statistics.

        Returns:
            Dictionary with job counts by status
        """
        stats = {
            "total": len(self.jobs),
            "queued": 0,
            "processing": 0,
            "completed": 0,
            "partial": 0,
            "failed": 0,
            "cancelled": 0,
        }

        for job in self.jobs.values():
            stats[job.status.value] += 1

        return stats

    async def set_max_concurrent(self, new_max: int) -> None:
        """
        Update max concurrent workers at runtime.

        This allows dynamically scaling the number of worker tasks.
        If increasing, new workers are started immediately.
        If decreasing, workers are allowed to finish naturally.

        Args:
            new_max: New maximum number of concurrent workers (1-10)

        Raises:
            ValueError: If new_max is not between 1 and 10
        """
        if new_max < 1 or new_max > 10:
            raise ValueError("max_concurrent must be between 1 and 10")

        old_max = self.max_concurrent
        self.max_concurrent = new_max

        logger.info(f"Updating max concurrent workers: {old_max} -> {new_max}")

        # Keep ownership of idle workers across decreases. They park at admission,
        # so a later increase reuses them instead of duplicating worker IDs.
        if self._workers_started:
            for i in range(len(self._workers), new_max):
                self._workers.append(asyncio.create_task(self._worker(i)))
        async with self._admission:
            self._admission.notify_all()

    async def _worker(self, worker_id: int) -> None:
        """Process queued work subject to the current global admission limit."""
        logger.info(f"Worker {worker_id} started")
        while True:
            try:
                job_id, job_type = await self.queue.get()
                admitted = False
                try:
                    async with self._admission:
                        await self._admission.wait_for(
                            lambda: self._running_jobs < self.max_concurrent
                        )
                        job = self.jobs.get(job_id)
                        if job is None or job.status != JobStatus.QUEUED:
                            continue
                        self.set_job_processing(job_id)
                        self._running_jobs += 1
                        admitted = True
                    handler = asyncio.create_task(self._worker_handler(self, job_id, job_type))
                    self._active_tasks[job_id] = handler
                    try:
                        await handler
                    except asyncio.CancelledError:
                        if job_id not in self._cancel_requested:
                            # The worker itself is being stopped; take the handler down too.
                            handler.cancel()
                            raise
                        self._record_cancelled(job_id)
                    except Exception as e:
                        logger.exception(f"Worker {worker_id}: Job {job_id} failed: {e}")
                        self.set_job_failed(job_id, str(e))
                    finally:
                        self._active_tasks.pop(job_id, None)
                        self._cancel_requested.discard(job_id)
                finally:
                    if admitted:
                        async with self._admission:
                            self._running_jobs -= 1
                            self._admission.notify_all()
                    self._scheduled_jobs.discard(job_id)
                    self.queue.task_done()
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id}: Unexpected error: {e}")
                await asyncio.sleep(1)

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired jobs."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_expired_jobs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Cleanup loop error: {e}")

    async def _cleanup_expired_jobs(self) -> None:
        """Remove expired completed/failed jobs."""
        now = datetime.now(UTC)
        expired_ids = []

        for job_id, job in self.jobs.items():
            if job.status in (
                JobStatus.COMPLETED,
                JobStatus.PARTIAL,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                if job.completed_at and (now - job.completed_at) > self.job_ttl:
                    expired_ids.append(job_id)

        expired_jobs = [self.jobs[job_id] for job_id in expired_ids]
        for job_id in expired_ids:
            del self.jobs[job_id]

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired jobs")

        if self._store:
            hours = int(self.job_ttl.total_seconds() / 3600)
            self._store.cleanup_expired(hours)
        for job in expired_jobs:
            self._publish(job)


# Global job manager instance
job_manager = JobManager()
