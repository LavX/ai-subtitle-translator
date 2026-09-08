"""Queued-only cancellation must preserve jobs that have already advanced."""

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
