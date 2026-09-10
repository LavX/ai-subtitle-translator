"""Dispatch-time adaptive sizing with real batches and synthetic HTTP responses."""

import asyncio
import json

import httpx
import pytest

from subtitle_translator.api.models import TranslationConfig
from subtitle_translator.config import Settings
from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.batch_sizing import get_batch_size_resolver
from subtitle_translator.providers.base import TranslationBatch
from subtitle_translator.providers.openrouter import OpenRouterProvider


@pytest.fixture(autouse=True)
def reset_sizing():
    get_batch_size_resolver().reset()
    yield
    get_batch_size_resolver().reset()


def response(lines, tokens=10):
    translations = [{**line, "content": "translated"} for line in lines]
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps({"translations": translations})}}],
            "usage": {"total_tokens": tokens, "cost": tokens / 1000},
        },
    )


def provider_with_transport(send, **kwargs):
    provider = OpenRouterProvider(
        Settings(
            _env_file=None,
            openrouter_api_key="synthetic-test-only",
            max_retries=1,
            retry_delay=0,
            parallel_batches_per_job=1,
            **kwargs,
        )
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    provider._model_params_fetched = True
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("resized_request_fails", [False, True])
async def test_retry_resize_preserves_partial_usage_and_consumed_retry_budget(
    resized_request_fails,
):
    model = "test/retry-resize-accounting"
    resolver = get_batch_size_resolver()
    resolver.record_failure(model, 100)
    waiting = asyncio.Event()
    calls = []
    billed_tokens = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        first, size = int(lines[0]["index"]), len(lines)
        calls.append((first, size, resolver.limit_planned_size(model, 100)))
        if first == 0 and size == 25:
            billed_tokens.append(3)
            return response(lines[:1], tokens=3)
        if first == 0 and (size == 50 or (size == 5 and resized_request_fails)):
            tokens = 2 if size == 50 else 4
            billed_tokens.append(tokens)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "invalid synthetic response"}}],
                    "usage": {"total_tokens": tokens, "cost": tokens / 1000},
                },
            )
        billed_tokens.append(5)
        return response(lines, tokens=5)

    provider = provider_with_transport(send)
    provider.settings.retry_delay = 0.05
    processor = BatchProcessor(provider, provider.settings)

    def activity(message):
        if "retry 1 after incomplete or invalid response" in message:
            waiting.set()

    task = asyncio.create_task(
        processor.process_batch(
            TranslationBatch(
                [{"index": str(i), "content": "synthetic"} for i in range(100)], "en", "hu"
            ),
            0,
            model=model,
            _activity_callback=activity,
        )
    )
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        # Another batch lowers the shared cap through size failures while this one
        # waits out its retry delay. Recorded directly so the lowering is complete
        # before this batch re-plans; a stalled request no longer lowers the cap.
        resolver.record_failure(model, 20)
        resolver.record_failure(model, 10)
        assert resolver.limit_planned_size(model, 100) == 5
        result = await asyncio.wait_for(task, 2)
        assert result.success is not resized_request_fails
        assert all(size <= cap for _, size, cap in calls)
        assert result.retries == 1
        assert result.tokens_used == sum(billed_tokens)
        assert result.cost == pytest.approx(sum(billed_tokens) / 1000)
        if resized_request_fails:
            # The failed first child does not cost its siblings: every line after
            # it is still requested, at whatever size the cap allows by then.
            requested_after_failure = set()
            for first, size, _ in calls[calls.index((0, 5, 5)) + 1 :]:
                requested_after_failure.update(range(first, first + size))
            assert requested_after_failure == set(range(5, 100))
            assert {line["index"] for line in result.translations} == {"0"} | {
                str(i) for i in range(5, 100)
            }
            assert "cues 0-4 at size 5" in result.error
        else:
            assert {line["index"] for line in result.translations} == {str(i) for i in range(100)}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel_batches", [1, 2])
async def test_undispatched_roots_do_not_inherit_a_limit_from_a_timeout(parallel_batches):
    """A stall is not size evidence: measured live, a 5-line request timed out
    after 600s in the same window a 100-line request finished in 28s. One
    stalled root used to cap every undispatched root at half size for nothing."""
    calls = []
    completed = []
    resolver = get_batch_size_resolver()
    model = "test/dispatch-adaptation"

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append((lines[0]["index"], len(lines), resolver.resolve(model)))
        if len(lines) > 50:
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        return response([{**line, "content": "translated"} for line in lines])

    provider = OpenRouterProvider(
        Settings(
            _env_file=None,
            openrouter_api_key="synthetic-test-only",
            max_retries=1,
            retry_delay=0,
            parallel_batches_per_job=parallel_batches,
        )
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(200)],
            "en",
            "hu",
            model=model,
            batch_size=100,
            progress_callback=lambda progress: completed.append(
                (progress.completed_batches, progress.completed_lines, progress.total_batches)
            ),
        )
        second_root = next(call for call in calls if call[0] == "100")
        assert second_root[1] <= second_root[2], calls
        assert [size for _, size, _ in calls] == [100, 100, 50, 50, 100, 100, 50, 50]
        assert result.success
        assert len(result.all_translations) == 200
        assert {line["index"] for line in result.all_translations} == {str(i) for i in range(200)}
        assert result.total_tokens == 40
        assert result.progress.total_cost == pytest.approx(0.04)
        assert [item.batch_index for item in result.batch_results] == [0, 1]
        assert set(completed) == {(0, 0, 2), (1, 100, 2), (2, 200, 2)}
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_successful_smaller_requests_allow_later_roots_to_grow():
    calls = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append(len(lines))
        if len(calls) == 1:
            # A size failure teaches the resolver; a stall deliberately does not.
            return response(lines[:1], tokens=0)
        return response([{**line, "content": "translated"} for line in lines])

    provider = provider_with_transport(send)
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(300)],
            "en",
            "hu",
            model="test/dispatch-growth",
            batch_size=100,
        )
        assert calls == [100, 50, 49, 50, 50, 100]
        assert result.success
        assert len(result.all_translations) == 300
        assert result.progress.completed_batches == result.progress.total_batches == 3
        assert result.total_tokens == 50
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_explicit_initial_batch_size_is_not_replaced_by_metadata_or_default():
    calls = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append(len(lines))
        return response(lines)

    provider = provider_with_transport(send, batch_size=10)
    provider._model_metadata_cache = {"test/explicit-batch": {"max_batch_size": 5}}
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(20)],
            "en",
            "hu",
            model="test/explicit-batch",
            batch_size=20,
        )
        assert result.success
        assert calls == [20]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_pending_children_observe_a_limit_lowered_by_another_batch():
    model = "test/shared-cap"
    resolver = get_batch_size_resolver()
    resolver.record_failure(model, 100)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    calls = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        first = int(lines[0]["index"])
        calls.append((first, len(lines), resolver.resolve(model)))
        if first >= 1000:
            # Lowers the shared cap through a size failure; a stall would not.
            return response(lines[:1], tokens=0)
        if first == 0:
            first_entered.set()
            await release_first.wait()
        return response(lines)

    provider = provider_with_transport(send)
    processor = BatchProcessor(provider, provider.settings)
    task = asyncio.create_task(
        processor.process_batch(
            TranslationBatch(
                lines=[{"index": str(i), "content": "source"} for i in range(100)],
                source_language="en",
                target_language="hu",
            ),
            batch_index=0,
            model=model,
        )
    )
    try:
        await asyncio.wait_for(first_entered.wait(), 1)
        other = await processor.process_batch(
            TranslationBatch(
                lines=[{"index": str(i), "content": "source"} for i in range(1000, 1020)],
                source_language="en",
                target_language="hu",
            ),
            batch_index=1,
            model=model,
        )
        release_first.set()
        result = await task
        assert not other.success
        # 20 splits to two children of 10; the first child's short count and retry
        # record the floor, so its sibling is re-planned as two children of 5.
        # The line the short reply did answer is not sent again, so the sibling
        # holds nine lines and splits into 5 + 4 at the learned floor.
        assert [size for first, size, _ in calls if first >= 1000] == [20, 10, 10, 5, 5, 4, 4]
        main_calls = [call for call in calls if call[0] < 1000]
        assert main_calls[0] == (0, 50, 50)
        assert main_calls[1] == (50, 5, 5)
        assert all(size <= cap for _, size, cap in main_calls)
        assert result.success
        assert len(result.translations) == 100
        assert len({line["index"] for line in result.translations}) == 100
    finally:
        release_first.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await provider.close()


@pytest.mark.asyncio
async def test_preemptive_children_recover_once_and_retain_partial_output_and_usage():
    model = "test/preemptive-partial"
    get_batch_size_resolver().record_failure(model, 100)
    calls = []

    async def send(request):
        payload = json.loads(request.content)
        assert payload["model"] == model
        lines = json.loads(payload["messages"][-1]["content"])
        first = lines[0]["index"]
        calls.append((first, len(lines)))
        if first == "50" and len(lines) == 50:
            return response([lines[0]], tokens=7)
        if first == "76":
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "invalid synthetic response"}}],
                    "usage": {"total_tokens": 3, "cost": 0.003},
                },
            )
        return response([{**line, "content": "translated"} for line in lines])

    provider = provider_with_transport(send)
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(100)],
            "en",
            "hu",
            model="test/ignored-by-override",
            batch_size=100,
            config_override=TranslationConfig(model=model),
        )
        # Line 50 came back with the short reply, so recovery requests 51-99 only.
        assert calls == [("0", 50), ("50", 50), ("51", 25), ("76", 24), ("76", 24)]
        assert not result.success
        assert {line["index"] for line in result.all_translations} == {str(i) for i in range(76)}
        assert result.total_tokens == 33
        assert result.progress.total_cost == pytest.approx(0.033)
        assert result.progress.completed_batches == result.progress.total_batches == 1
        assert result.progress.completed_lines == 76
        assert result.progress.failed_batches == 1
        assert result.batch_results[0].retries == 1
        assert not result.batch_results[0].timed_out
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_preemptive_children_and_recovery_share_the_root_timeout_budget():
    model = "test/preemptive-budget"
    get_batch_size_resolver().record_failure(model, 100)
    calls = []
    active = set()

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append(len(lines))
        task = asyncio.current_task()
        active.add(task)
        try:
            if lines[0]["index"] == "0":
                await asyncio.sleep(0.06)
                return response(lines)
            await asyncio.Event().wait()
        finally:
            active.remove(task)

    provider = provider_with_transport(send, request_timeout=5)
    try:
        start = asyncio.get_running_loop().time()
        result = await asyncio.wait_for(
            BatchProcessor(provider, provider.settings).process_all_batches(
                [{"index": str(i), "content": "source"} for i in range(100)],
                "en",
                "hu",
                model=model,
                batch_size=100,
                # Compress the request override below its public API minimum for this test.
                config_override=TranslationConfig.model_construct(request_timeout=0.1),
            ),
            0.5,
        )
        # Three request windows: attempt, one same-size retry, then the split.
        assert asyncio.get_running_loop().time() - start < 0.38
        assert calls == [50, 50, 50, 25]
        assert not result.success
        assert result.progress.completed_lines == 50
        assert result.total_tokens == 10
        assert result.batch_results[0].timed_out
        assert "budget" in result.batch_results[0].error.lower()
        assert not active
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_cancelling_a_preemptive_child_finishes_provider_cleanup():
    model = "test/preemptive-cancellation"
    get_batch_size_resolver().record_failure(model, 100)
    entered = asyncio.Event()
    active = set()
    calls = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append(len(lines))
        if lines[0]["index"] == "0":
            return response(lines)
        task = asyncio.current_task()
        active.add(task)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            active.remove(task)

    provider = provider_with_transport(send)
    task = asyncio.create_task(
        BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(100)],
            "en",
            "hu",
            model=model,
            batch_size=100,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == [50, 50]
        assert not active
        assert not [
            child for child in asyncio.all_tasks() if "_run_batch" in child.get_coro().__qualname__
        ]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await provider.close()


@pytest.mark.asyncio
async def test_completed_parallel_root_preserves_the_other_roots_current_activity():
    second_entered = asyncio.Event()
    release_second = asyncio.Event()
    first_completed = asyncio.Event()
    progress_events = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        if lines[0]["index"] == "0":
            await second_entered.wait()
        else:
            second_entered.set()
            await release_second.wait()
        return response(lines)

    def progress(value):
        progress_events.append(
            (value.completed_batches, value.completed_lines, value.total_tokens, value.message)
        )
        if value.completed_batches == 1:
            first_completed.set()

    provider = provider_with_transport(send)
    provider.settings.parallel_batches_per_job = 2
    task = asyncio.create_task(
        BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(200)],
            "en",
            "hu",
            model="test/parallel-activity",
            batch_size=100,
            progress_callback=progress,
        )
    )
    try:
        await asyncio.wait_for(first_completed.wait(), 2)
        first_update = progress_events[-1]
        assert first_update[:3] == (1, 100, 10)
        assert "Batch 2:" in first_update[3]
        assert "request in progress for 100 lines" in first_update[3]
        assert all(event[:3] == (0, 0, 0) for event in progress_events[:-1])
        release_second.set()
        result = await task
        assert result.success
        assert progress_events[-1] == (2, 200, 20, "")
    finally:
        release_second.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await provider.close()
