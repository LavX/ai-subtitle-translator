"""Batch output, cancellation and timeout regressions with fake HTTP."""

import asyncio
import json

import httpx
import pytest

from subtitle_translator.config import Settings
from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.batch_sizing import get_batch_size_resolver
from subtitle_translator.core.translator import SubtitleTranslator
from subtitle_translator.providers.openrouter import OpenRouterProvider
from subtitle_translator.queue.job_manager import JobManager, JobStatus, JobType
from subtitle_translator.queue.worker import (
    process_content_translation_job,
    process_file_translation_job,
)


def response(lines, tokens=10, cost=0.01):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps({"translations": lines})}}],
            "usage": {"total_tokens": tokens, "cost": cost},
        },
    )


def settings(**kwargs):
    return Settings(
        _env_file=None,
        openrouter_api_key="synthetic-test-only",
        max_retries=1,
        retry_delay=0,
        parallel_batches_per_job=1,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def reset_sizing():
    get_batch_size_resolver().reset()
    yield
    get_batch_size_resolver().reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("file_job", [False, True])
@pytest.mark.parametrize(
    "outcome,count,status",
    [
        ("success", 10, JobStatus.COMPLETED),
        ("partial", 5, JobStatus.PARTIAL),
        ("failure", 0, JobStatus.FAILED),
    ],
)
async def test_adaptive_jobs_retain_unique_output_and_all_billed_usage(
    file_job, outcome, count, status
):
    calls = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append(len(lines))
        if len(lines) == 10:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "invalid"}}],
                    "usage": {"total_tokens": 7, "cost": 0.007},
                },
            )
        if outcome == "failure" or (outcome == "partial" and lines[0]["index"] == "6"):
            return httpx.Response(401)
        translated = [{"index": x["index"], "content": "translated " + x["index"]} for x in lines]
        return response(translated + translated[:1] + [{"index": "999", "content": "unrequested"}])

    provider = OpenRouterProvider(settings())
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    translator = SubtitleTranslator(provider, provider.settings)
    manager = JobManager()
    data = {"sourceLanguage": "en", "targetLanguage": "hu"}
    if file_job:
        data["content"] = "\n\n".join(
            f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},500\nsource {i}" for i in range(1, 11)
        )
        handler = process_file_translation_job
        job_type = JobType.TRANSLATE_FILE
    else:
        data["lines"] = [{"position": i, "line": f"source {i}"} for i in range(1, 11)]
        handler = process_content_translation_job
        job_type = JobType.TRANSLATE_CONTENT
    job_id = await manager.submit_job(data, job_type)
    try:
        await handler(manager, job_id, translator)
        job = manager.get_job(job_id)
        assert job.status == status
        assert job.completed_lines == count
        expected_tokens = {"success": 27, "partial": 17, "failure": 7}[outcome]
        assert job.tokens_used == expected_tokens
        assert job.total_cost == pytest.approx(expected_tokens / 1000)
        if count:
            text = job.result["content"] if file_job else str(job.result["lines"])
            assert "translated 1" in text
            assert "unrequested" not in text
            if status == JobStatus.PARTIAL:
                assert "5/10 lines translated" in job.error
                assert "source 6" in text
    finally:
        await translator.close()


@pytest.mark.asyncio
async def test_actual_provider_timeout_splits_and_floor_is_bounded():
    calls = []
    activity = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append(len(lines))
        if len(lines) > 5:
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        return response([{**x, "content": "translated"} for x in lines])

    provider = OpenRouterProvider(settings())
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(10)],
            "en",
            "hu",
            progress_callback=lambda progress: activity.append(progress.message),
        )
        assert result.success
        assert calls == [10, 5, 5]
        assert any(
            "recovering after timeout" in message and "5 lines" in message for message in activity
        )
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["cancel", "callback_error"])
async def test_batch_parent_awaits_active_and_delayed_children(exit_kind):
    entered = asyncio.Event()
    release = asyncio.Event()
    active = set()
    finished = []
    progress = []

    async def send(request):
        task = asyncio.current_task()
        active.add(task)
        entered.set()
        try:
            await release.wait()
            finished.append(task)
            return response([{"index": "0", "content": "translated"}])
        finally:
            active.remove(task)

    provider = OpenRouterProvider(settings())
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    provider.settings.parallel_batches_per_job = 3

    def callback(value):
        progress.append(value.completed_batches)
        if exit_kind == "callback_error" and value.completed_batches:
            raise RuntimeError("progress sink unavailable")

    task = asyncio.create_task(
        BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(3)],
            "en",
            "hu",
            batch_size=1,
            progress_callback=callback,
        )
    )
    await entered.wait()
    try:
        if exit_kind == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            with pytest.raises(RuntimeError, match="progress sink"):
                await task
        assert not active
        children = [t for t in asyncio.all_tasks() if "_run_batch" in t.get_coro().__qualname__]
        assert not children
        await provider.close()
        writes = len(progress)
        release.set()
        await asyncio.sleep(0)
        assert len(progress) == writes
        if exit_kind == "cancel":
            assert not finished
    finally:
        release.set()
        for child in asyncio.all_tasks():
            if "_run_batch" in child.get_coro().__qualname__:
                child.cancel()
                await asyncio.gather(child, return_exceptions=True)
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_success", [False, True])
async def test_floor_timeout_is_not_repeated_and_stops_later_groups(prior_success):
    calls = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append(lines[0]["index"])
        if prior_success and lines[0]["index"] == "0":
            return response([{**x, "content": "translated"} for x in lines])
        raise httpx.ReadTimeout("private provider payload must not enter activity", request=request)

    provider = OpenRouterProvider(settings())
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(15)], "en", "hu", batch_size=5
        )
        assert calls == (["0", "5"] if prior_success else ["0"])
        assert result.progress.completed_batches == (2 if prior_success else 1)
        assert result.progress.total_batches == 3
        assert result.progress.completed_lines == (5 if prior_success else 0)
        assert not result.success
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_adaptive_timeout_budget_preserves_finished_subbatch_and_cleans_request():
    calls = []
    active = set()

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append(len(lines))
        active.add(asyncio.current_task())
        try:
            if len(lines) == 10 or lines[0]["index"] == "5":
                await asyncio.Event().wait()
            await asyncio.sleep(0.02)
            return response([{**x, "content": "translated"} for x in lines])
        finally:
            active.remove(asyncio.current_task())

    provider = OpenRouterProvider(settings(request_timeout=0.05))
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    start = asyncio.get_running_loop().time()
    try:
        result = await asyncio.wait_for(
            BatchProcessor(provider, provider.settings).process_all_batches(
                [{"index": str(i), "content": "source"} for i in range(10)], "en", "hu"
            ),
            0.3,
        )
        elapsed = asyncio.get_running_loop().time() - start
        assert elapsed < 0.2
        assert calls == [10, 5, 5]
        assert not result.success
        assert len(result.all_translations) == 5
        assert result.total_tokens == 10
        assert not active
        assert "budget" in result.batch_results[0].error.lower()
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("file_job", [False, True])
async def test_timeout_recovery_activity_reaches_job_api_without_false_progress(
    tmp_path, monkeypatch, file_job
):
    from subtitle_translator.api import routes
    from subtitle_translator.main import app
    from subtitle_translator.queue.job_store import JobStore

    started = asyncio.Event()
    release = asyncio.Event()
    history = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        if len(lines) == 10:
            raise httpx.ReadTimeout("sensitive upstream text", request=request)
        started.set()
        await release.wait()
        return response([{**x, "content": "translated"} for x in lines])

    provider = OpenRouterProvider(settings())
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    translator = SubtitleTranslator(provider, provider.settings)
    manager = JobManager(max_concurrent=1)
    store = JobStore(str(tmp_path / "activity.db"))
    manager.set_store(store)
    monkeypatch.setattr(routes, "job_manager", manager)
    monkeypatch.setattr(routes, "_auth_token", None)
    original_update = manager.update_progress

    def update(*args, **kwargs):
        original_update(*args, **kwargs)
        job = manager.get_job(args[0])
        history.append((job.message, job.completed_lines, job.completed_batches, job.tokens_used))

    monkeypatch.setattr(manager, "update_progress", update)

    async def handler(manager, job_id, job_type):
        if file_job:
            await process_file_translation_job(manager, job_id, translator)
        else:
            await process_content_translation_job(manager, job_id, translator)

    manager.set_worker_handler(handler)
    data = {"sourceLanguage": "en", "targetLanguage": "hu"}
    if file_job:
        data["content"] = "\n\n".join(
            f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},500\nsource" for i in range(1, 11)
        )
    else:
        data["lines"] = [{"position": i, "line": "source"} for i in range(10)]
    job_id = await manager.submit_job(
        data, JobType.TRANSLATE_FILE if file_job else JobType.TRANSLATE_CONTENT
    )
    await manager.start_workers()
    try:
        await asyncio.wait_for(started.wait(), 2)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as api:
            state = (await api.get(f"/api/v1/jobs/{job_id}")).json()
        assert "request" in state["message"].lower()
        assert any("timed out" in value[0].lower() for value in history)
        assert any("smaller" in value[0].lower() for value in history)
        assert all(value[1:] == (0, 0, 0) for value in history)
        assert all("sensitive upstream" not in value[0] for value in history)
        assert store.load_job(job_id).message == state["message"]
        release.set()
        await asyncio.wait_for(manager.queue.join(), 2)
        assert manager.get_job(job_id).status == JobStatus.COMPLETED
    finally:
        await manager.stop_workers()
        await translator.close()
        writes = len(history)
        store.close()
        release.set()
        await asyncio.sleep(0)
        assert len(history) == writes


@pytest.mark.asyncio
async def test_cancel_awaits_multiple_requests_and_network_retry_backoff():
    entered = asyncio.Event()
    requests = []
    active = set()

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        requests.append(lines[0]["index"])
        if len(requests) == 3:
            entered.set()
        if lines[0]["index"] == "0":
            raise httpx.ConnectError("synthetic network failure", request=request)
        active.add(asyncio.current_task())
        try:
            await asyncio.Event().wait()
        finally:
            active.remove(asyncio.current_task())

    provider = OpenRouterProvider(settings())
    provider.settings.parallel_batches_per_job = 3
    provider.settings.retry_delay = 60
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    task = asyncio.create_task(
        BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(3)], "en", "hu", batch_size=1
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert len(active) == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not active
        assert requests == ["0", "1", "2"]
        assert not any("_run_batch" in t.get_coro().__qualname__ for t in asyncio.all_tasks())
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,attempts", [("network", 2), ("auth", 1), ("rate_limit", 4)])
async def test_non_timeout_provider_errors_keep_distinct_attempts_without_splitting(
    failure, attempts
):
    calls = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append(len(lines))
        if failure == "network":
            raise httpx.ConnectError(
                "network timeout wording is not a typed timeout", request=request
            )
        if failure == "auth":
            return httpx.Response(401)
        return httpx.Response(429, headers={"retry-after": "0.001"})

    provider = OpenRouterProvider(settings())
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(10)], "en", "hu"
        )
        assert calls == [10] * attempts
        assert not result.success
        assert not result.batch_results[0].timed_out
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("file_job", [False, True])
@pytest.mark.parametrize("prior_success", [False, True])
async def test_terminal_timeout_summary_preserves_unattempted_work_and_stop_reason(
    tmp_path, monkeypatch, file_job, prior_success
):
    from subtitle_translator.api import routes
    from subtitle_translator.main import app
    from subtitle_translator.queue.job_store import JobStore

    calls = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append((lines[0]["index"], len(lines)))
        if prior_success and lines[0]["index"] == "1":
            return response([{**x, "content": "translated"} for x in lines])
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    provider = OpenRouterProvider(settings())
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    translator = SubtitleTranslator(provider, provider.settings)
    manager = JobManager(max_concurrent=1)
    store = JobStore(str(tmp_path / "terminal.db"))
    manager.set_store(store)
    monkeypatch.setattr(routes, "job_manager", manager)
    monkeypatch.setattr(routes, "_auth_token", None)

    async def handler(manager, job_id, job_type):
        if file_job:
            await process_file_translation_job(manager, job_id, translator)
        else:
            await process_content_translation_job(manager, job_id, translator)

    manager.set_worker_handler(handler)
    data = {"sourceLanguage": "en", "targetLanguage": "hu"}
    if file_job:
        data["content"] = "\n\n".join(
            f"{i}\n00:00:01,000 --> 00:00:02,000\nsource {i}" for i in range(1, 301)
        )
    else:
        data["lines"] = [{"position": i, "line": f"source {i}"} for i in range(1, 301)]
    job_id = await manager.submit_job(
        data, JobType.TRANSLATE_FILE if file_job else JobType.TRANSLATE_CONTENT
    )
    await manager.start_workers()
    try:
        await asyncio.wait_for(manager.queue.join(), 2)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as api:
            state = (await api.get(f"/api/v1/jobs/{job_id}")).json()
        persisted = store.load_job(job_id)
        attempted, unattempted = (2, 1) if prior_success else (1, 2)
        expected = (
            f"{attempted} of 3 batches attempted; 1 failed; {unattempted} not attempted. "
            "Provider requests timed out without usable output; remaining batches stopped. "
        )
        assert expected in state["error"]
        assert expected in state["message"]
        assert persisted.error == state["error"]
        assert persisted.message == state["message"]
        assert (state["totalBatches"], state["completedBatches"]) == (3, attempted)
        assert state["completedLines"] == (100 if prior_success else 0)
        assert persisted.tokens_used == (10 if prior_success else 0)
        assert persisted.total_cost == pytest.approx(0.01 if prior_success else 0)
        if prior_success:
            assert state["status"] == "partial"
            output = persisted.result["content"] if file_job else str(persisted.result["lines"])
            assert "translated" in output
            assert "source 101" in output
            assert calls == [("1", 100), ("101", 100), ("101", 50)]
        else:
            assert state["status"] == "failed"
            assert calls == [("1", 100), ("1", 50)]
    finally:
        await manager.stop_workers()
        await translator.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("file_translation", [False, True])
@pytest.mark.parametrize("prior_success", [False, True])
async def test_synchronous_timeout_error_preserves_stop_summary(
    monkeypatch, file_translation, prior_success
):
    from subtitle_translator.api import routes
    from subtitle_translator.api.models import TranslateContentRequest
    from subtitle_translator.main import app

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        if prior_success and lines[0]["index"] == "1":
            return response([{**x, "content": "translated"} for x in lines])
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    provider = OpenRouterProvider(settings())
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    translator = SubtitleTranslator(provider, provider.settings)
    data = {"sourceLanguage": "en", "targetLanguage": "hu"}
    if file_translation:
        data["content"] = "\n\n".join(
            f"{i}\n00:00:01,000 --> 00:00:02,000\nsource {i}" for i in range(1, 301)
        )
    else:
        data["lines"] = [{"position": i, "line": f"source {i}"} for i in range(1, 301)]
    try:
        if file_translation:
            result = await translator.translate_file(data["content"], "en", "hu")
        else:
            result = await translator.translate_content(TranslateContentRequest(**data))
        attempted, not_attempted = (2, 1) if prior_success else (1, 2)
        expected = (
            f"{attempted} of 3 batches attempted; 1 failed; {not_attempted} not attempted. "
            "Provider requests timed out without usable output; remaining batches stopped. "
        )
        assert not result.success
        assert expected in result.error
        assert result.tokens_used == (10 if prior_success else 0)
        if prior_success:
            output = result.content if file_translation else str(result.lines)
            assert "translated" in output
            assert "source 101" in output
        else:
            get_batch_size_resolver().reset()

            async def get_translator():
                return translator

            monkeypatch.setattr(routes, "get_translator", get_translator)
            monkeypatch.setattr(routes, "get_settings", lambda: provider.settings)
            monkeypatch.setattr(routes, "_auth_token", None)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as api:
                endpoint = (
                    "/api/v1/translate/file" if file_translation else "/api/v1/translate/content"
                )
                failed = await api.post(endpoint, json=data)
            assert failed.status_code == 500
            assert expected in failed.json()["detail"]["message"]
    finally:
        await translator.close()
