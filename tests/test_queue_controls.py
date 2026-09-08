"""Queue controls the browser relies on: cancel a running job, forget a finished one.

Cancelling used to stop at the queue. A job that had already started could only
be waited out, which with a stalled provider meant minutes of paid attempts the
user had already given up on. Forget only hid a row in the browser; the job and
its result stayed on the server.
"""

import asyncio

import pytest
from fastapi import HTTPException

from subtitle_translator import gui, ui_api
from subtitle_translator.queue.job_manager import JobManager, JobStatus, JobType
from subtitle_translator.queue.job_store import JobStore

REQUEST = {"content": "1\n00:00:01,000 --> 00:00:02,000\nsource\n", "targetLanguage": "hu"}


@pytest.fixture
def manager(tmp_path):
    manager = JobManager(max_concurrent=1)
    manager.set_store(JobStore(db_path=str(tmp_path / "jobs.db")))
    return manager


async def _until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


class TestCancelRunningJob:
    @pytest.mark.asyncio
    async def test_cancel_interrupts_the_running_handler_and_frees_the_worker(self, manager):
        """The handler is interrupted, the job ends cancelled, the next job still runs."""
        entered = asyncio.Event()
        interrupted = asyncio.Event()
        finished = []

        async def handler(job_manager, job_id, job_type):
            if job_manager.get_job(job_id).request_data.get("hang"):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    interrupted.set()
                    raise
            finished.append(job_id)
            job_manager.set_job_completed(job_id, {"content": "done"})

        manager.set_worker_handler(handler)
        await manager.start_workers()
        try:
            hanging = await manager.submit_job({**REQUEST, "hang": True}, JobType.TRANSLATE_FILE)
            await asyncio.wait_for(entered.wait(), 5)
            assert manager.get_job(hanging).status == JobStatus.PROCESSING

            assert manager.cancel_job(hanging) is True
            await asyncio.wait_for(interrupted.wait(), 5)
            await _until(lambda: manager.get_job(hanging).status == JobStatus.CANCELLED)
            persisted = manager._store.load_job(hanging)
            assert persisted.status == JobStatus.CANCELLED
            assert persisted.completed_at is not None

            follow_up = await manager.submit_job(REQUEST, JobType.TRANSLATE_FILE)
            await _until(lambda: manager.get_job(follow_up).status == JobStatus.COMPLETED)
            assert finished == [follow_up]
        finally:
            await manager.stop_workers()

    @pytest.mark.asyncio
    async def test_cancel_without_a_running_task_is_still_refused(self, manager):
        """A job marked processing by hand has no task to interrupt; the old answer stands."""
        job_id = await manager.submit_job(REQUEST, JobType.TRANSLATE_FILE)
        manager.set_job_processing(job_id)
        assert manager.cancel_job(job_id) is False
        assert manager.get_job(job_id).status == JobStatus.PROCESSING


class TestUiControls:
    @pytest.fixture(autouse=True)
    def wire(self, manager, monkeypatch):
        monkeypatch.setattr(ui_api, "job_manager", manager)
        self.manager = manager
        self.identity = ui_api.UiIdentity(api_key="sk-or-test", owner="owner-a")
        self.stranger = ui_api.UiIdentity(api_key="sk-or-other", owner="owner-b")

    async def _own(self, **extra):
        return await self.manager.submit_job(
            {**REQUEST, "_ui_owner": self.identity.owner, **extra}, JobType.TRANSLATE_FILE
        )

    @pytest.mark.asyncio
    async def test_cancel_reports_cancelling_for_a_running_job(self):
        entered = asyncio.Event()

        async def handler(job_manager, job_id, job_type):
            entered.set()
            await asyncio.Event().wait()

        self.manager.set_worker_handler(handler)
        await self.manager.start_workers()
        try:
            job_id = await self._own()
            await asyncio.wait_for(entered.wait(), 5)
            response = await ui_api.cancel_job(job_id, self.identity)
            assert response.status in ("cancelling", "cancelled")
            await _until(lambda: self.manager.get_job(job_id).status == JobStatus.CANCELLED)
        finally:
            await self.manager.stop_workers()

    @pytest.mark.asyncio
    async def test_forget_deletes_a_finished_job_from_memory_and_store(self):
        job_id = await self._own()
        self.manager.set_job_completed(job_id, {"content": "done"})

        result = await ui_api.forget_job(job_id, self.identity)

        assert result == {"jobId": job_id, "deleted": True}
        assert self.manager.get_job(job_id) is None
        assert self.manager._store.load_job(job_id) is None

    @pytest.mark.asyncio
    async def test_forget_refuses_an_active_job(self):
        job_id = await self._own()
        with pytest.raises(HTTPException) as caught:
            await ui_api.forget_job(job_id, self.identity)
        assert caught.value.status_code == 409
        assert self.manager.get_job(job_id) is not None

    @pytest.mark.asyncio
    async def test_forget_respects_ownership(self):
        job_id = await self._own()
        self.manager.set_job_completed(job_id, {"content": "done"})
        with pytest.raises(HTTPException) as caught:
            await ui_api.forget_job(job_id, self.stranger)
        assert caught.value.status_code == 404
        assert self.manager.get_job(job_id) is not None

    @pytest.mark.asyncio
    async def test_snapshot_metadata_carries_creation_time(self):
        """The browser orders the queue newest first and needs a stable timestamp."""
        job_id = await self._own()
        value = gui.metadata(self.manager.get_job(job_id))
        assert value["createdAt"] == self.manager.get_job(job_id).created_at.isoformat()
        assert value["completedAt"] is None
        self.manager.set_job_completed(job_id, {"content": "done"})
        finished = gui.metadata(self.manager.get_job(job_id))
        assert finished["completedAt"] == self.manager.get_job(job_id).completed_at.isoformat()


class TestCancelThenShutdown:
    @pytest.mark.asyncio
    async def test_stopping_workers_right_after_a_cancel_still_stops(self, manager):
        """A worker whose handler was cancelled for the user must still obey stop_workers.

        Task.cancel() on the worker only cancels the handler it is awaiting, so the
        worker saw the user's cancel, recorded it and went back to the queue while
        stop_workers() waited on it forever.
        """
        entered = asyncio.Event()

        async def handler(job_manager, job_id, job_type):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)
                raise

        manager.set_worker_handler(handler)
        await manager.start_workers()
        job_id = await manager.submit_job(REQUEST, JobType.TRANSLATE_FILE)
        await asyncio.wait_for(entered.wait(), 5)

        assert manager.cancel_job(job_id) is True
        await asyncio.sleep(0)
        await asyncio.wait_for(manager.stop_workers(), 2)

        assert manager.get_job(job_id).status == JobStatus.CANCELLED
        assert manager._workers == []
