"""Restart and admission regressions using real queues and disposable storage."""

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from subtitle_translator.api import routes
from subtitle_translator.config import Settings
from subtitle_translator.main import app
from subtitle_translator.queue.job_manager import Job, JobManager, JobStatus, JobType
from subtitle_translator.queue.job_store import JobStore


async def settle():
    for _ in range(30):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_restart_recovers_active_jobs_beyond_history_and_lowered_limit(tmp_path):
    path = str(tmp_path / "jobs.db")
    store = JobStore(path)
    statuses = [JobStatus.QUEUED, JobStatus.PROCESSING] + [
        JobStatus.COMPLETED,
        JobStatus.PARTIAL,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    ] * 2
    for i, status in enumerate(statuses):
        store.save_job(
            Job(
                id=str(i),
                job_type=JobType.TRANSLATE_CONTENT,
                status=status,
                request_data={},
                created_at=datetime.now(UTC) + timedelta(seconds=i),
            )
        )
    store.close()
    for _ in range(2):
        store = JobStore(path)
        manager = JobManager(max_jobs=1)
        manager.set_store(store)
        try:
            assert await manager.recover_jobs() == 2
            assert await manager.recover_jobs() == 0
            assert manager.queue.qsize() == 2
            assert len(manager.jobs) == 3
            assert [manager.queue.get_nowait()[0] for _ in range(2)] == ["0", "1"]
            with pytest.raises(RuntimeError, match="Maximum job limit"):
                await manager.submit_job({}, JobType.TRANSLATE_CONTENT)
            assert store.load_job("0").status == JobStatus.QUEUED
            assert store.load_job("1").status == JobStatus.QUEUED
        finally:
            store.close()


@pytest.mark.asyncio
async def test_http_resize_cycles_preserve_admission_and_queue_accounting(monkeypatch):
    manager = JobManager(max_concurrent=2)
    monkeypatch.setattr(routes, "job_manager", manager)
    monkeypatch.setattr(routes, "get_settings", lambda: Settings(_env_file=None, admin_api_key=""))
    running = set()
    seen = []
    release = asyncio.Queue()

    async def handler(manager, job_id, job_type):
        running.add(job_id)
        seen.append(job_id)
        try:
            await release.get()
            manager.set_job_completed(job_id, {})
        finally:
            running.remove(job_id)

    manager.set_worker_handler(handler)
    await manager.start_workers()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:

        async def resize(count):
            response = await client.put("/api/v1/config", json={"maxConcurrentJobs": count})
            assert response.status_code == 200
            await settle()

        try:
            await settle()
            for _ in range(3):
                await resize(1)
                await resize(2)
            ids = [await manager.submit_job({}, JobType.TRANSLATE_CONTENT) for _ in range(8)]
            await settle()
            assert len(running) == 2
            await resize(1)
            assert len(running) == 2  # Existing jobs finish without cancellation.
            await release.put(None)
            await settle()
            assert len(running) == 1
            assert len(seen) == 2
            for _ in range(3):
                await resize(2)
                assert len(running) == 2
                await resize(1)
                await release.put(None)
                await settle()
                assert len(running) == 1
            await resize(2)
            for _ in ids:
                await release.put(None)
            await asyncio.wait_for(manager.queue.join(), 2)
            assert set(seen) == set(ids)
            assert len(seen) == len(ids)
        finally:
            tasks = list(manager._workers)
            await manager.stop_workers()
            assert all(t.done() for t in tasks)
            assert not running


@pytest.mark.asyncio
async def test_lifespan_applies_history_limit_and_awaits_batches_before_close(
    tmp_path, monkeypatch
):
    from subtitle_translator import main
    from subtitle_translator.core.translator import SubtitleTranslator
    from subtitle_translator.providers.openrouter import OpenRouterProvider
    from subtitle_translator.queue.worker import process_content_translation_job

    path = str(tmp_path / "lifespan.db")
    store = JobStore(path)
    for i in range(3):
        store.save_job(
            Job(
                id=str(i),
                job_type=JobType.TRANSLATE_CONTENT,
                status=JobStatus.COMPLETED,
                request_data={},
                created_at=datetime.now(UTC),
            )
        )
    store.close()
    runtime_settings = Settings(
        _env_file=None,
        encryption_enabled=False,
        db_path=path,
        job_queue_max_jobs=1,
        job_queue_max_concurrent=1,
        parallel_batches_per_job=3,
    )
    manager = JobManager()
    entered = asyncio.Event()
    active = set()
    order = []

    async def send(request):
        active.add(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            active.remove(asyncio.current_task())
            order.append("request cancelled")

    provider = OpenRouterProvider(runtime_settings)
    runtime_settings.openrouter_api_key = "synthetic-lifecycle"
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    provider._model_params_fetched = True
    translator = SubtitleTranslator(provider, runtime_settings)

    async def handler(manager, job_id, job_type):
        await process_content_translation_job(manager, job_id, translator)

    async def close_translator():
        assert not active
        order.append("provider closed")
        await translator.close()

    class OrderedStore(JobStore):
        def close(self):
            assert not active
            order.append("store closed")
            super().close()

    monkeypatch.setattr(main, "get_settings", lambda: runtime_settings)
    monkeypatch.setattr(main, "job_manager", manager)
    monkeypatch.setattr(main, "job_worker_handler", handler)
    monkeypatch.setattr(main, "close_translator", close_translator)
    monkeypatch.setattr(main, "JobStore", OrderedStore)
    async with main.lifespan(app):
        assert len(manager.jobs) == 1
        job_id = await manager.submit_job(
            {
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [{"position": i, "line": "source"} for i in range(250)],
            },
            JobType.TRANSLATE_CONTENT,
        )
        await asyncio.wait_for(entered.wait(), 2)
    assert order[0] == "request cancelled"
    assert set(order[1:]) == {"provider closed", "store closed"}
    assert not any("_run_batch" in t.get_coro().__qualname__ for t in asyncio.all_tasks())
    assert manager.queue._unfinished_tasks == 0
    store = JobStore(path)
    try:
        restored = JobManager(max_jobs=1)
        restored.set_store(store)
        assert await restored.recover_jobs() == 1
        assert restored.get_job(job_id).status == JobStatus.QUEUED
    finally:
        store.close()
