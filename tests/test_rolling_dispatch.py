"""Rolling root admission: a stalled request must not idle the other slots.

Roots used to run in fixed groups of parallel_count, and the next group was
only created once every root of the current group had returned. With two
slots, one stalled request left the other slot idle for the stall's whole
budget. Roots are now admitted whenever a slot frees, in index order, while
the fixed cohorts stay the unit of the all-timeout early stop.
"""

import asyncio
import json
import time

import httpx
import pytest

from subtitle_translator.config import Settings
from subtitle_translator.core.batch_processor import BatchProcessor, summarize_batch_failure
from subtitle_translator.core.batch_sizing import get_batch_size_resolver
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


def provider_with_transport(send, parallel, **kwargs):
    provider = OpenRouterProvider(
        Settings(
            _env_file=None,
            openrouter_api_key="synthetic-test-only",
            max_retries=1,
            retry_delay=0,
            parallel_batches_per_job=parallel,
            **kwargs,
        )
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    provider._model_params_fetched = True
    return provider


def root_of(request):
    lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
    return int(lines[0]["index"]) // 5, lines


class Tracker:
    """Records which roots are in flight and when each one started."""

    def __init__(self):
        self.active: set[int] = set()
        self.peak = 0
        self.started: list[tuple[int, float]] = []
        self.origin = time.monotonic()

    def enter(self, root):
        self.active.add(root)
        self.peak = max(self.peak, len(self.active))
        self.started.append((root, time.monotonic() - self.origin))

    def leave(self, root):
        self.active.discard(root)


async def run(provider, count, progress=None):
    return await BatchProcessor(provider, provider.settings).process_all_batches(
        [{"index": str(i), "content": "source"} for i in range(count)],
        "en",
        "hu",
        batch_size=5,
        model="test/rolling",
        progress_callback=progress,
    )


@pytest.mark.asyncio
async def test_a_freed_slot_is_refilled_while_a_peer_is_still_stalled():
    tracker = Tracker()
    peer_started = asyncio.Event()
    release_peer = asyncio.Event()
    later_started = asyncio.Event()
    first_finished = asyncio.Event()

    async def send(request):
        root, lines = root_of(request)
        tracker.enter(root)
        try:
            if root == 0:
                await peer_started.wait()
            elif root == 1:
                peer_started.set()
                await release_peer.wait()
            else:
                later_started.set()
                await release_peer.wait()
            return response(lines)
        finally:
            tracker.leave(root)

    def progress(value):
        if value.completed_batches == 1:
            first_finished.set()

    provider = provider_with_transport(send, parallel=2)
    task = asyncio.create_task(run(provider, 20, progress))
    try:
        await asyncio.wait_for(first_finished.wait(), 2)
        # The freed slot is refilled at once, not after the stall or a pacing delay.
        await asyncio.wait_for(later_started.wait(), 0.3)
        assert tracker.active == {1, 2}
        assert tracker.peak == 2
        # Roots admitted together are still staggered, so they do not burst.
        assert tracker.started[1][1] - tracker.started[0][1] >= 0.4
        release_peer.set()
        result = await asyncio.wait_for(task, 3)
        assert result.success
        assert [r.batch_index for r in result.batch_results] == [0, 1, 2, 3]
        assert result.progress.completed_lines == 20
        assert result.total_tokens == 40
        assert tracker.peak == 2 and not tracker.active
    finally:
        release_peer.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_success", [False, True])
async def test_an_all_timeout_cohort_stops_admission_of_later_cohorts(prior_success):
    """The cohort after a wholly timed-out one is never sent, even with a free slot."""
    calls = []

    async def send(request):
        root, lines = root_of(request)
        calls.append(root)
        if prior_success and root < 2:
            return response(lines)
        raise httpx.ReadTimeout("stalled", request=request)

    provider = provider_with_transport(send, parallel=2)
    try:
        result = await run(provider, 25)
        assert sorted(calls) == ([0, 1, 2, 3] if prior_success else [0, 1])
        assert result.progress.total_batches == 5
        assert result.progress.completed_batches == (4 if prior_success else 2)
        assert result.progress.completed_lines == (10 if prior_success else 0)
        assert not result.success
        summary = summarize_batch_failure(result, 5)
        attempted = 4 if prior_success else 2
        assert summary.startswith(
            f"{attempted} of 5 batches attempted; 2 failed; {5 - attempted} not attempted. "
            "Provider requests timed out without usable output; remaining batches stopped."
        )
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_a_stop_waits_for_an_earlier_root_and_keeps_its_output():
    """Cohort 1 times out entirely while cohort 0's slow root is still running.

    Nothing after cohort 1 is admitted, the slow root is neither cancelled nor
    forgotten, and its activity stays visible until it finishes.
    """
    tracker = Tracker()
    release_slow = asyncio.Event()
    stopped = asyncio.Event()
    events = []

    async def send(request):
        root, lines = root_of(request)
        tracker.enter(root)
        try:
            if root == 0:
                await release_slow.wait()
                return response(lines)
            if root == 1:
                return response(lines)
            raise httpx.ReadTimeout("stalled", request=request)
        finally:
            tracker.leave(root)

    def progress(value):
        events.append((value.completed_batches, value.completed_lines, value.message))
        if value.completed_batches == 3:
            stopped.set()

    provider = provider_with_transport(send, parallel=2)
    task = asyncio.create_task(run(provider, 30, progress))
    try:
        await asyncio.wait_for(stopped.wait(), 3)
        await asyncio.sleep(0.1)
        assert tracker.active == {0}
        assert sorted(r for r, _ in tracker.started) == [0, 1, 2, 3]
        assert events[-1][0] == 3 and events[-1][2].startswith("Batch 1:")
        release_slow.set()
        result = await asyncio.wait_for(task, 3)
        assert sorted(r for r, _ in tracker.started) == [0, 1, 2, 3]
        assert result.progress.completed_batches == 4
        assert result.progress.completed_lines == 10
        assert [r.batch_index for r in result.batch_results] == [0, 1, 2, 3]
        assert result.total_tokens == 20
        assert summarize_batch_failure(result, 6).startswith(
            "4 of 6 batches attempted; 2 failed; 2 not attempted. "
            "Provider requests timed out without usable output; remaining batches stopped."
        )
    finally:
        release_slow.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await provider.close()


@pytest.mark.asyncio
async def test_a_timed_out_root_beside_a_successful_peer_does_not_stop_admission():
    calls = []

    async def send(request):
        root, lines = root_of(request)
        calls.append(root)
        if root == 0:
            raise httpx.ReadTimeout("stalled", request=request)
        return response(lines)

    provider = provider_with_transport(send, parallel=2)
    try:
        result = await run(provider, 20)
        assert sorted(calls) == [0, 1, 2, 3]
        assert result.progress.completed_batches == 4
        assert result.progress.completed_lines == 15
        assert not result.success
        assert summarize_batch_failure(result, 4).startswith("1 of 4 batches failed")
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_a_root_gets_its_full_budget_when_it_finally_launches():
    """Time spent waiting for a slot must not eat into a root's request budget."""

    async def send(request):
        _, lines = root_of(request)
        await asyncio.sleep(0.15)
        return response(lines)

    provider = provider_with_transport(send, parallel=1, request_timeout=0.2)
    try:
        result = await run(provider, 30)
        assert result.success
        assert result.progress.completed_lines == 30
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_simultaneous_completions_are_aggregated_once_and_in_order():
    tracker = Tracker()

    async def send(request):
        root, lines = root_of(request)
        tracker.enter(root)
        try:
            return response(lines)
        finally:
            tracker.leave(root)

    provider = provider_with_transport(send, parallel=3)
    try:
        result = await run(provider, 25)
        assert result.success
        assert [r.batch_index for r in result.batch_results] == [0, 1, 2, 3, 4]
        assert result.progress.completed_lines == 25
        assert len(result.all_translations) == 25
        assert result.total_tokens == 50
        assert tracker.peak <= 3
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_nothing_is_admitted_after_cancellation():
    tracker = Tracker()
    both_started = asyncio.Event()

    async def send(request):
        root, lines = root_of(request)
        tracker.enter(root)
        try:
            if len(tracker.active) == 2:
                both_started.set()
            await asyncio.Event().wait()
        finally:
            tracker.leave(root)
        return response(lines)

    provider = provider_with_transport(send, parallel=2)
    task = asyncio.create_task(run(provider, 20))
    try:
        await asyncio.wait_for(both_started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not tracker.active
        assert sorted(r for r, _ in tracker.started) == [0, 1]
    finally:
        await provider.close()
