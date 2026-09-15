"""Cancellation through the public job API."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from subtitle_translator.api import routes
from subtitle_translator.queue.job_manager import JobManager, JobStatus, JobType
from subtitle_translator.queue.job_store import JobStore


@pytest.fixture
def persisted_manager(tmp_path, monkeypatch):
    manager = JobManager()
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    manager.set_store(store)
    monkeypatch.setattr(routes, "job_manager", manager)
    monkeypatch.setattr(routes, "_auth_token", None)
    yield manager, store
    store.close()


@pytest.fixture
async def client(persisted_manager):
    app = FastAPI()
    app.include_router(routes.jobs_router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


@pytest.fixture
async def queued_job(persisted_manager):
    manager, _ = persisted_manager
    return await manager.submit_job(
        job_type=JobType.TRANSLATE_FILE,
        request_data={"content": "1\n00:00:00,000 --> 00:00:01,000\nHello\n"},
    )


async def _until(predicate, timeout=5.0):
    """Wait for a condition the worker settles asynchronously."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached in time")


def advance_job(manager, job_id, status):
    if status == JobStatus.CANCELLED:
        manager.cancel_job(job_id)
        return
    manager.set_job_processing(job_id)
    if status == JobStatus.COMPLETED:
        manager.set_job_completed(job_id, {"content": "Translated subtitle"})
    elif status == JobStatus.PARTIAL:
        manager.set_job_partial(job_id, {"content": "Partial subtitle"}, "Provider interrupted")
    elif status == JobStatus.FAILED:
        manager.set_job_failed(job_id, "Provider unavailable")


async def test_only_queued_cancels_and_retains_queued_job(client, persisted_manager, queued_job):
    manager, store = persisted_manager
    response = await client.delete(f"/api/v1/jobs/{queued_job}?onlyQueued=true")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert manager.get_job(queued_job).status == JobStatus.CANCELLED
    assert store.load_job(queued_job).status == JobStatus.CANCELLED


@pytest.mark.parametrize(
    "status",
    [
        JobStatus.PROCESSING,
        JobStatus.COMPLETED,
        JobStatus.PARTIAL,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    ],
)
async def test_only_queued_preserves_job_that_advanced_before_request(
    client, persisted_manager, queued_job, status
):
    manager, store = persisted_manager
    advance_job(manager, queued_job, status)
    before = manager.get_job(queued_job).model_dump()
    persisted_before = store.load_job(queued_job).model_dump()

    response = await client.delete(f"/api/v1/jobs/{queued_job}?onlyQueued=true")

    assert response.status_code == 200
    assert response.json()["status"] == status.value
    assert manager.get_job(queued_job).model_dump() == before
    assert store.load_job(queued_job).model_dump() == persisted_before
    if status == JobStatus.PROCESSING:
        assert response.json()["message"] == "Cannot cancel job that is currently processing"


@pytest.mark.parametrize("query", ["", "?onlyQueued=false"])
@pytest.mark.parametrize(
    "status", [JobStatus.COMPLETED, JobStatus.PARTIAL, JobStatus.FAILED, JobStatus.CANCELLED]
)
async def test_default_and_false_keep_existing_terminal_deletion(
    client, persisted_manager, queued_job, status, query
):
    manager, store = persisted_manager
    advance_job(manager, queued_job, status)

    response = await client.delete(f"/api/v1/jobs/{queued_job}{query}")

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    assert manager.get_job(queued_job) is None
    assert store.load_job(queued_job) is None


class TestCancellingARunningJob:
    """A job that has started must be stoppable through the public API.

    The GUI could already do this: its route asks the manager to interrupt the
    handler task. The public route, which is the one Bazarr+ calls, refused and
    still answered 200, so a caller was told the job had been dealt with while it
    kept running, kept billing and kept holding its worker.
    """

    @pytest.fixture
    async def running(self, persisted_manager):
        """A job the worker has actually started, blocked inside its handler."""
        manager, _ = persisted_manager
        entered = asyncio.Event()

        async def handler(job_manager, job_id, job_type):
            entered.set()
            await asyncio.Event().wait()

        manager.set_worker_handler(handler)
        await manager.start_workers()
        job_id = await manager.submit_job(
            job_type=JobType.TRANSLATE_FILE,
            request_data={"content": "1\n00:00:00,000 --> 00:00:01,000\nHello\n"},
        )
        await asyncio.wait_for(entered.wait(), 5)
        assert manager.get_job(job_id).status == JobStatus.PROCESSING
        try:
            yield job_id
        finally:
            await manager.stop_workers()

    async def test_delete_cancels_a_running_job(self, client, persisted_manager, running):
        manager, store = persisted_manager

        response = await client.delete(f"/api/v1/jobs/{running}")

        assert response.status_code == 200
        assert response.json()["status"] in ("cancelling", "cancelled")
        await _until(lambda: manager.get_job(running).status == JobStatus.CANCELLED)
        assert store.load_job(running).status == JobStatus.CANCELLED

    async def test_only_queued_still_leaves_a_running_job_alone(
        self, client, persisted_manager, running
    ):
        """onlyQueued means 'cancel only if it has not started', and still does."""
        manager, _ = persisted_manager

        response = await client.delete(f"/api/v1/jobs/{running}?onlyQueued=true")

        assert response.status_code == 200
        assert response.json()["status"] == JobStatus.PROCESSING.value
        assert response.json()["message"] == "Cannot cancel job that is currently processing"
        assert manager.get_job(running).status == JobStatus.PROCESSING

    async def test_delete_says_so_when_a_running_job_has_no_handler_to_stop(
        self, client, persisted_manager, queued_job
    ):
        """A job restored as processing after a restart has no task to interrupt,
        and the caller must be told that rather than shown a false success."""
        manager, _ = persisted_manager
        manager.set_job_processing(queued_job)

        response = await client.delete(f"/api/v1/jobs/{queued_job}")

        assert response.status_code == 200
        assert response.json()["status"] == JobStatus.PROCESSING.value
        assert response.json()["message"] == "Cannot cancel job that is currently processing"
        assert manager.get_job(queued_job).status == JobStatus.PROCESSING
