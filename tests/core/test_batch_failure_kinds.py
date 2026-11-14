"""A failed batch says why, a connection-level failure stops new batches, and usage is reported."""

import asyncio

import pytest

from subtitle_translator.core.batch_processor import RATE_LIMIT_INLINE_WAIT_MAX, BatchProcessor
from subtitle_translator.core.errors import (
    AuthenticationError,
    DependencyUnavailableError,
    ErrorKind,
    InvalidResponseError,
    ProviderTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
    TranslationProviderError,
)
from subtitle_translator.core.provider import TranslationProvider, TranslationResult
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.usage import UsageMeter, reserve_request


class Scripted(TranslationProvider):
    """Answers each call from a script: an exception to raise, or 'ok' to translate. With a price,
    it holds that worst case for each request itself, the way OpenRouter does. With a delay,
    each call takes that long, so parallel batches are in flight together."""

    def __init__(self, script, price=None, delay=0.0):
        self.script = list(script)
        self.calls = 0
        self.price = price
        self.delay = delay
        self.prices_requests = price is not None
        self.held_when_sent: list[float] = []

    @property
    def provider_name(self):
        return "scripted"

    async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
        if self.price is not None:
            await reserve_request(config_override, self.price, "scripted")
        meter = getattr(config_override, "_usage_meter", None)
        self.held_when_sent.append(meter.reserved if meter is not None else 0.0)
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(step, BaseException):
            raise step
        return TranslationResult(
            translations=[{"index": line["index"], "content": f"T{line['content']}"} for line in batch.lines],
            model_used="m",
            total_tokens=10,
            cost=0.01,
        )

    async def health_check(self):
        return True

    async def close(self):
        pass

    async def get_available_models(self):
        return []


def lines(n):
    return [{"index": str(i), "content": f"line {i}"} for i in range(n)]


def settings(parallel=1):
    return CoreSettings(max_retries=1, retry_delay=0.0, request_timeout=5.0, parallel_batches_per_job=parallel)


@pytest.mark.asyncio
async def test_quota_stops_new_batches_and_is_not_retried():
    provider = Scripted([QuotaExhaustedError("daily limit", retry_after=3600)])
    result = await BatchProcessor(provider, settings()).process_all_batches(
        lines(20), "en", "hu", batch_size=5, model="m"
    )
    assert provider.calls == 1
    assert result.stopped_by is ErrorKind.QUOTA_EXHAUSTED
    failed = [r for r in result.batch_results if not r.success]
    assert failed[0].error_kind is ErrorKind.QUOTA_EXHAUSTED and failed[0].retry_after == 3600
    assert len(result.batch_results) == 1  # the other three batches were never sent


@pytest.mark.asyncio
async def test_auth_failure_stops_new_batches():
    provider = Scripted([AuthenticationError("bad key")])
    result = await BatchProcessor(provider, settings()).process_all_batches(lines(10), "en", "hu", batch_size=5, model="m")
    assert result.stopped_by is ErrorKind.AUTH_REQUIRED and provider.calls == 1


@pytest.mark.asyncio
async def test_a_long_rate_limit_wait_is_escalated_not_slept():
    provider = Scripted([RateLimitError("slow", retry_after=RATE_LIMIT_INLINE_WAIT_MAX + 1)])
    started = asyncio.get_running_loop().time()
    result = await BatchProcessor(provider, settings()).process_all_batches(lines(5), "en", "hu", batch_size=5, model="m")
    assert asyncio.get_running_loop().time() - started < 2
    assert result.batch_results[0].error_kind is ErrorKind.RATE_LIMITED
    assert result.batch_results[0].retry_after == RATE_LIMIT_INLINE_WAIT_MAX + 1


@pytest.mark.asyncio
async def test_a_short_rate_limit_is_still_retried_in_place():
    provider = Scripted([RateLimitError("blip", retry_after=0.01), "ok"])
    result = await BatchProcessor(provider, settings()).process_all_batches(lines(5), "en", "hu", batch_size=5, model="m")
    assert result.success and provider.calls == 2


@pytest.mark.asyncio
async def test_a_server_error_after_retries_is_dependency_unavailable():
    boom = TranslationProviderError("OpenRouter server error: 503", retryable=True, status_code=503)
    provider = Scripted([boom, boom, boom, boom])
    result = await BatchProcessor(provider, settings()).process_all_batches(lines(5), "en", "hu", batch_size=5, model="m")
    assert result.batch_results[0].error_kind is ErrorKind.DEPENDENCY_UNAVAILABLE


@pytest.mark.asyncio
async def test_an_unreachable_back_end_stops_new_batches_once_its_retries_are_used():
    down = DependencyUnavailableError("Server could not be reached (ConnectError)")
    provider = Scripted([down] * 10)
    result = await BatchProcessor(provider, settings()).process_all_batches(lines(20), "en", "hu", batch_size=5, model="m")
    assert provider.calls == 2  # the first batch and its one retry
    assert result.stopped_by is ErrorKind.DEPENDENCY_UNAVAILABLE
    assert len(result.batch_results) == 1  # three batches were never sent
    assert result.batch_results[0].unreachable is True


@pytest.mark.asyncio
async def test_an_unreachable_back_end_met_by_a_smaller_request_stops_new_batches():
    # A garbled reply splits the 10-line batch; its first 5-line request then finds the back
    # end down twice. The split batch carries that up, and no other batch is sent.
    down = DependencyUnavailableError("Server could not be reached (ConnectError)")
    provider = Scripted([InvalidResponseError("garbled"), down, down])
    result = await BatchProcessor(provider, settings()).process_all_batches(
        lines(30), "en", "hu", batch_size=10, model="m"
    )
    assert provider.calls == 3
    assert result.stopped_by is ErrorKind.DEPENDENCY_UNAVAILABLE and len(result.batch_results) == 1
    assert result.batch_results[0].unreachable is True


@pytest.mark.asyncio
async def test_one_stalled_batch_does_not_stop_the_others():
    provider = Scripted([ProviderTimeoutError("late"), "ok", "ok", "ok"])
    result = await BatchProcessor(provider, settings(parallel=2)).process_all_batches(
        lines(20), "en", "hu", batch_size=5, model="m"
    )
    assert result.stopped_by is None and len(result.batch_results) == 4
    assert result.batch_results[0].error_kind is ErrorKind.DEPENDENCY_UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first", [RateLimitError("slow down", retry_after=0.0), InvalidResponseError("garbled")]
)
async def test_a_stall_after_an_earlier_failure_does_not_stop_the_others(first):
    # The batch's last failure is a stall; the rate limit or bad reply before it does not make
    # the back end unreachable, although 2.1.1 clears timed_out after it.
    provider = Scripted([first, ProviderTimeoutError("late")])
    result = await BatchProcessor(provider, settings()).process_all_batches(
        lines(20), "en", "hu", batch_size=5, model="m"
    )
    assert result.stopped_by is None and len(result.batch_results) == 4
    stalled = result.batch_results[0]
    assert stalled.error_kind is ErrorKind.DEPENDENCY_UNAVAILABLE and stalled.unreachable is False


@pytest.mark.asyncio
async def test_a_content_failure_is_partial_and_does_not_stop_the_job():
    bad = InvalidResponseError("garbled")
    provider = Scripted([bad, bad, "ok"])
    result = await BatchProcessor(provider, settings()).process_all_batches(lines(10), "en", "hu", batch_size=5, model="m")
    kinds = [r.error_kind for r in result.batch_results]
    assert ErrorKind.PARTIAL in kinds and result.stopped_by is None
    assert len(result.batch_results) == 2  # the second batch still ran


@pytest.mark.asyncio
async def test_usage_is_reported_after_every_call_including_failed_ones():
    reports = []

    async def sink(usage):
        reports.append(usage.tokens)

    failed = InvalidResponseError("garbled", tokens_used=7, cost=0.001)
    provider = Scripted([failed, "ok"])
    meter = UsageMeter(sink=sink)
    await BatchProcessor(provider, settings(), meter=meter).process_all_batches(
        lines(5), "en", "hu", batch_size=5, model="m"
    )
    assert reports == [7, 10]
    assert meter.tokens == 17


@pytest.mark.asyncio
async def test_a_request_whose_worst_case_does_not_fit_is_not_sent():
    provider = Scripted([], price=0.05)  # worst case 0.05, actual charge 0.01
    meter = UsageMeter(budget=0.055)
    result = await BatchProcessor(provider, settings(), meter=meter).process_all_batches(
        lines(10), "en", "hu", batch_size=5, model="m"
    )
    assert provider.calls == 1  # 0.05 fits 0.055; after 0.01 is spent, 0.045 is left
    assert result.stopped_by is ErrorKind.CANCELLED
    assert "budget" in result.batch_results[-1].error.lower()
    assert meter.reserved == 0.0  # the call that was sent released its hold


@pytest.mark.asyncio
async def test_batches_in_flight_hold_their_worst_case_from_each_other():
    # Two batches run in parallel; the second starts while the first is still waiting for its
    # answer. 0.05 is held for the first, so 0.03 is left and the second's 0.05 does not fit.
    provider = Scripted([], price=0.05, delay=0.8)
    meter = UsageMeter(budget=0.08)
    result = await BatchProcessor(provider, settings(parallel=2), meter=meter).process_all_batches(
        lines(10), "en", "hu", batch_size=5, model="m"
    )
    assert provider.calls == 1 and provider.held_when_sent == [pytest.approx(0.05)]
    assert result.stopped_by is ErrorKind.CANCELLED
    assert meter.cost == pytest.approx(0.01) and meter.reserved == 0.0


@pytest.mark.asyncio
async def test_a_back_end_without_prices_stops_only_when_nothing_is_left():
    provider = Scripted([])  # reports 0.01 per call, prices nothing
    meter = UsageMeter(budget=0.015)
    await BatchProcessor(provider, settings(), meter=meter).process_all_batches(
        lines(15), "en", "hu", batch_size=5, model="m"
    )
    assert provider.calls == 2  # 0.00 < 0.015 and 0.01 < 0.015 go out; 0.02 does not


@pytest.mark.asyncio
async def test_a_call_the_provider_timed_out_keeps_its_hold():
    # The provider stopped waiting, but the server may have answered and billed the request, so
    # its worst case stays held: of 0.08, 0.03 is left, and the same-size retry's 0.05 is not sent.
    provider = Scripted([ProviderTimeoutError("late")], price=0.05)
    meter = UsageMeter(budget=0.08)
    result = await BatchProcessor(provider, settings(), meter=meter).process_all_batches(
        lines(10), "en", "hu", batch_size=10, model="m"
    )
    assert provider.calls == 1 and meter.reserved == pytest.approx(0.05)
    assert result.stopped_by is ErrorKind.CANCELLED
    assert result.batch_results[0].reason == "budget_exhausted"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,held",
    [
        (ProviderTimeoutError("late"), 0.05),
        (DependencyUnavailableError("connection reset (ReadError)", billing_unknown=True), 0.05),
        (DependencyUnavailableError("could not connect (ConnectError)"), 0.0),
        (TranslationProviderError("server error: 503", retryable=True, status_code=503), 0.0),
    ],
)
async def test_only_a_call_that_may_have_been_billed_keeps_its_hold(failure, held):
    provider = Scripted([failure, "ok"], price=0.05)
    meter = UsageMeter(budget=1.0)
    result = await BatchProcessor(provider, settings(), meter=meter).process_all_batches(
        lines(10), "en", "hu", batch_size=10, model="m"
    )
    assert result.success and provider.calls == 2
    assert meter.reserved == pytest.approx(held) and meter.cost == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_each_finished_root_batch_is_announced():
    seen = []

    async def on_root(result, requested):
        seen.append((result.batch_index, [line["index"] for line in requested]))

    provider = Scripted([])
    await BatchProcessor(provider, settings()).process_all_batches(
        lines(10), "en", "hu", batch_size=5, model="m", on_root_result=on_root
    )
    assert sorted(seen) == [(0, ["0", "1", "2", "3", "4"]), (1, ["5", "6", "7", "8", "9"])]
