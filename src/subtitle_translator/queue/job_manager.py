"""In-memory job queue manager for async translation processing."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a translation job."""
    
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
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
    request_data: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
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
        self.jobs: Dict[str, Job] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.max_jobs = max_jobs
        self.job_ttl = timedelta(hours=job_ttl_hours)
        self._workers_started = False
        self._workers: List[asyncio.Task] = []
        self._cleanup_task: Optional[asyncio.Task] = None
        self._worker_handler: Optional[Any] = None
    
    def set_worker_handler(self, handler: Any) -> None:
        """
        Set the worker handler function.
        
        The handler should be an async function that takes (job_manager, job_id, job_type).
        """
        self._worker_handler = handler
    
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
        request_data: Dict[str, Any],
        job_type: JobType,
    ) -> str:
        """
        Submit a new job to the queue.
        
        Args:
            request_data: The request data for the translation job
            job_type: Type of translation job
            
        Returns:
            The job ID
            
        Raises:
            RuntimeError: If max jobs limit is reached
        """
        # Check job limit
        active_jobs = sum(
            1 for j in self.jobs.values()
            if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
        )
        
        if active_jobs >= self.max_jobs:
            raise RuntimeError(f"Maximum job limit ({self.max_jobs}) reached")
        
        # Create job
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            request_data=request_data,
            created_at=datetime.utcnow(),
        )
        
        self.jobs[job_id] = job
        await self.queue.put((job_id, job_type))
        
        logger.info(f"Job {job_id} submitted (type: {job_type.value})")
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job by ID.
        
        Args:
            job_id: The job ID
            
        Returns:
            The job if found, None otherwise
        """
        return self.jobs.get(job_id)
    
    def update_progress(
        self,
        job_id: str,
        progress: int,
        message: str = "",
    ) -> None:
        """
        Update job progress.
        
        Args:
            job_id: The job ID
            progress: Progress percentage (0-100)
            message: Optional status message
        """
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.progress = min(max(progress, 0), 100)
            if message:
                job.message = message
    
    def set_job_processing(self, job_id: str) -> None:
        """Mark a job as processing."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()
            job.message = "Processing translation..."
    
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
            job.completed_at = datetime.utcnow()
            job.message = "Translation completed"
            logger.info(f"Job {job_id} completed successfully")
    
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
            job.completed_at = datetime.utcnow()
            job.message = f"Translation failed: {error}"
            logger.error(f"Job {job_id} failed: {error}")
    
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
            job.completed_at = datetime.utcnow()
            job.message = "Job cancelled by user"
            logger.info(f"Job {job_id} cancelled")
            return True
        
        return False
    
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
            # Only delete completed, failed, or cancelled jobs
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                del self.jobs[job_id]
                logger.info(f"Job {job_id} deleted")
                return True
        return False
    
    def list_jobs(
        self,
        status_filter: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> List[Job]:
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
    
    def get_queue_position(self, job_id: str) -> Optional[int]:
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
        
        queued_jobs = [
            j for j in self.jobs.values()
            if j.status == JobStatus.QUEUED
        ]
        queued_jobs.sort(key=lambda j: j.created_at)
        
        for i, j in enumerate(queued_jobs):
            if j.id == job_id:
                return i + 1
        
        return None
    
    def get_stats(self) -> Dict[str, int]:
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
        
        # Start additional workers if needed
        if new_max > old_max and self._workers_started:
            for i in range(old_max, new_max):
                logger.info(f"Starting additional worker {i}")
                task = asyncio.create_task(self._worker(i))
                self._workers.append(task)
        
        # Note: We don't stop existing workers when decreasing.
        # They will naturally stop when the queue is empty or
        # will be cancelled when stop_workers() is called.
    
    async def _worker(self, worker_id: int) -> None:
        """
        Background worker that processes jobs from the queue.
        
        Args:
            worker_id: Worker identifier for logging
        """
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get next job from queue
                job_id, job_type = await self.queue.get()
                
                # Check if job still exists and is queued
                job = self.jobs.get(job_id)
                if job is None:
                    logger.warning(f"Worker {worker_id}: Job {job_id} not found")
                    self.queue.task_done()
                    continue
                
                if job.status != JobStatus.QUEUED:
                    logger.warning(
                        f"Worker {worker_id}: Job {job_id} status is {job.status}, skipping"
                    )
                    self.queue.task_done()
                    continue
                
                logger.info(f"Worker {worker_id}: Processing job {job_id}")
                
                # Mark as processing
                self.set_job_processing(job_id)
                
                # Process the job using the handler
                try:
                    await self._worker_handler(self, job_id, job_type)
                except Exception as e:
                    logger.exception(f"Worker {worker_id}: Job {job_id} failed with error: {e}")
                    self.set_job_failed(job_id, str(e))
                
                self.queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id}: Unexpected error: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
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
        now = datetime.utcnow()
        expired_ids = []
        
        for job_id, job in self.jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at and (now - job.completed_at) > self.job_ttl:
                    expired_ids.append(job_id)
        
        for job_id in expired_ids:
            del self.jobs[job_id]
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired jobs")


# Global job manager instance
job_manager = JobManager()