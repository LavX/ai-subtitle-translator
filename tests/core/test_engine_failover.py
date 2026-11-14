"""The standalone's ordered list: fail over on the four classes only, resume from checkpoints."""

import asyncio

import pytest

from subtitle_translator.core.backends import Connection
from subtitle_translator.core.context import Language
from subtitle_translator.core.engine import (
    EngineError,
    EngineEvents,
    Line,
    TranslationEngine,
    TranslationJob,
)
from subtitle_translator.core.errors import (
    AuthenticationError,
    ErrorKind,
    InvalidConfigError,
    InvalidResponseError,
    ProviderTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
)
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.usage import reserve_request
from tests.core.engine_fakes import ScriptedBackend, ScriptedProvider, registry_with

BASE = CoreSettings(max_retries=1, retry_delay=0.0, request_timeout=5.0)
A, B = Connection("a", "scripted"), Connection("b", "scripted")


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def job(n=10):
    return TranslationJob(Language("en", "English"), Language("hu", "Hungarian"), [Line(i, f"l{i}") for i in range(n)])


def engine(providers, clock=None):
    return TranslationEngine(registry_with(ScriptedBackend(providers)), BASE, clock=clock or Clock())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        QuotaExhaustedError("daily", retry_after=600),
        RateLimitError("slow", retry_after=120),
        AuthenticationError("bad key"),
    ],
)
async def test_failover_continues_from_checkpoints(failure):
    a = ScriptedProvider(["ok", failure])
    b = ScriptedProvider()
    failovers = []
    outcome = await engine({"a": a, "b": b}).translate(job(10), [A, B], events=EngineEvents(on_failover=failovers.append))
    assert outcome.status == "complete" and outcome.connection_id == "b"
    assert [line["index"] for batch in b.batches for line in batch.lines] == ["5", "6", "7", "8", "9"]
    assert [f.connection_id for f in failovers] == ["a"]
    assert outcome.tokens == 20  # a's first batch plus b's batch, counted once


@pytest.mark.asyncio
async def test_content_and_config_errors_never_fail_over():
    bad = InvalidResponseError("garbled")
    b = ScriptedProvider()
    partial = await engine({"a": ScriptedProvider(["ok", bad, bad]), "b": b}).translate(job(10), [A, B])
    assert partial.status == "partial" and b.batches == []
    with pytest.raises(EngineError) as caught:
        await engine({"a": ScriptedProvider([InvalidConfigError("no model")]), "b": b}).translate(job(10), [A, B])
    assert caught.value.kind is ErrorKind.INVALID_CONFIG and b.batches == []


@pytest.mark.asyncio
async def test_all_connections_failing_reports_the_last_error_and_all_spend():
    a = ScriptedProvider(["ok", QuotaExhaustedError("a spent", retry_after=600)])
    b = ScriptedProvider([RateLimitError("b slow", retry_after=90)])
    with pytest.raises(EngineError) as caught:
        await engine({"a": a, "b": b}).translate(job(10), [A, B])
    error = caught.value
    assert error.kind is ErrorKind.RATE_LIMITED and error.connection_id == "b"
    assert error.tokens == 10 and len(error.checkpoints) == 1
    assert [f.connection_id for f in error.failovers] == ["a", "b"]


@pytest.mark.asyncio
async def test_a_pinned_connection_never_fails_over_even_when_cooling():
    clock = Clock()
    a = ScriptedProvider([QuotaExhaustedError("spent", retry_after=600), QuotaExhaustedError("spent", retry_after=600)])
    b = ScriptedProvider()
    eng = engine({"a": a, "b": b}, clock)
    with pytest.raises(EngineError):
        await eng.translate(job(5), [A, B], pinned="a")
    with pytest.raises(EngineError):
        await eng.translate(job(5), [A, B], pinned="a")  # tried again: the user asked for it
    assert len(a.batches) == 2 and b.batches == []


@pytest.mark.asyncio
async def test_a_cooling_connection_is_skipped_until_its_wait_ends():
    clock = Clock()
    a = ScriptedProvider([RateLimitError("slow", retry_after=100)])
    b = ScriptedProvider()
    eng = engine({"a": a, "b": b}, clock)
    await eng.translate(job(5), [A, B])
    calls = len(a.batches)
    await eng.translate(job(5), [A, B])
    assert len(a.batches) == calls  # a was skipped
    clock.now += 101
    await eng.translate(job(5), [A, B])
    assert len(a.batches) == calls + 1


@pytest.mark.asyncio
async def test_disabled_and_missing_connections():
    eng = engine({"a": ScriptedProvider(), "b": ScriptedProvider()})
    with pytest.raises(EngineError) as none_on:
        await eng.translate(job(3), [Connection("a", "scripted", enabled=False)])
    assert none_on.value.kind is ErrorKind.INVALID_CONFIG
    with pytest.raises(EngineError) as unknown:
        await eng.translate(job(3), [A], pinned="zzz")
    assert unknown.value.kind is ErrorKind.INVALID_CONFIG
    with pytest.raises(EngineError) as off:
        await eng.translate(job(3), [Connection("a", "scripted", enabled=False)], pinned="a")
    assert off.value.kind is ErrorKind.INVALID_CONFIG


@pytest.mark.asyncio
async def test_every_connection_cooling_on_a_provider_wait_is_rate_limited_with_the_soonest_wait():
    # Both waits are the provider's own (a Retry-After), so they are honoured: nothing is sent.
    clock = Clock()
    a = ScriptedProvider([RateLimitError("x", retry_after=50)])
    b = ScriptedProvider([RateLimitError("y", retry_after=80)])
    eng = engine({"a": a, "b": b}, clock)
    with pytest.raises(EngineError):
        await eng.translate(job(5), [A, B])
    sent = len(a.batches), len(b.batches)
    with pytest.raises(EngineError) as caught:
        await eng.translate(job(5), [A, B])
    assert caught.value.kind is ErrorKind.RATE_LIMITED and caught.value.retry_after == pytest.approx(50)
    assert (len(a.batches), len(b.batches)) == sent


@pytest.mark.asyncio
async def test_the_only_connection_is_tried_again_after_a_timeout_with_no_provider_wait():
    # A timeout names no wait, so the default cooldown is a guess. With nothing else to try,
    # the next request is sent instead of being refused.
    clock = Clock()
    only = ScriptedProvider([ProviderTimeoutError("read timed out")])
    eng = engine({"a": only}, clock)
    with pytest.raises(EngineError) as first:
        await eng.translate(job(5), [A])
    assert first.value.kind is ErrorKind.DEPENDENCY_UNAVAILABLE
    sent = len(only.batches)
    again = await eng.translate(job(5), [A])
    assert again.status == "complete" and len(only.batches) > sent


@pytest.mark.asyncio
async def test_the_only_connection_is_tried_again_after_a_402_once_the_operator_tops_up():
    clock = Clock()
    only = ScriptedProvider([QuotaExhaustedError("payment required")])
    eng = engine({"a": only}, clock)
    with pytest.raises(EngineError):
        await eng.translate(job(5), [A])
    sent = len(only.batches)
    # Same connection, same key, no time passed: the credits may have been topped up.
    again = await eng.translate(job(5), [A])
    assert again.status == "complete" and len(only.batches) > sent


@pytest.mark.asyncio
async def test_the_only_connection_is_tried_again_after_a_401_with_no_provider_wait():
    only = ScriptedProvider([AuthenticationError("bad key")])
    eng = engine({"a": only})
    with pytest.raises(EngineError):
        await eng.translate(job(5), [A])
    again = await eng.translate(job(5), [A])
    assert again.status == "complete"


@pytest.mark.asyncio
async def test_with_every_connection_cooling_the_one_whose_wait_ends_soonest_is_tried():
    clock = Clock()
    # Both fail with no provider wait: a sits out the 60 s default, b (a 402) the 3600 s one.
    a = ScriptedProvider([ProviderTimeoutError("late")])
    b = ScriptedProvider([QuotaExhaustedError("spent")])
    eng = engine({"a": a, "b": b}, clock)
    with pytest.raises(EngineError):
        await eng.translate(job(5), [A, B])
    calls = len(a.batches), len(b.batches)
    again = await eng.translate(job(5), [A, B])
    assert again.connection_id == "a"
    assert len(a.batches) == calls[0] + 1 and len(b.batches) == calls[1]


@pytest.mark.asyncio
async def test_a_cooling_connection_is_still_skipped_while_another_one_is_healthy():
    clock = Clock()
    a = ScriptedProvider([ProviderTimeoutError("late")])
    b = ScriptedProvider()
    eng = engine({"a": a, "b": b}, clock)
    await eng.translate(job(5), [A, B])
    calls = len(a.batches)
    for _ in range(3):
        outcome = await eng.translate(job(5), [A, B])
        assert outcome.connection_id == "b"
    assert len(a.batches) == calls


@pytest.mark.asyncio
async def test_a_success_clears_the_connections_cooldown():
    clock = Clock()
    a = ScriptedProvider([RateLimitError("slow", retry_after=100)])
    eng = engine({"a": a, "b": ScriptedProvider()}, clock)
    await eng.translate(job(5), [A, B])
    clock.now += 101
    assert (await eng.translate(job(5), [A, B])).connection_id == "a"
    assert (await eng.translate(job(5), [A, B])).connection_id == "a"


@pytest.mark.asyncio
async def test_budget_is_shared_across_the_list():
    # a spends 0.04 on its first batch, then runs out of quota; b gets what is left (0.02),
    # which cannot cover its 0.04 estimate, so b sends nothing.
    a = ScriptedProvider(["ok", QuotaExhaustedError("spent")], cost=0.04, price=0.01)
    b = ScriptedProvider(cost=0.04, price=0.04)
    big = job(15)
    big.budget = 0.06
    with pytest.raises(EngineError) as caught:
        await engine({"a": a, "b": b}).translate(big, [A, B])
    assert caught.value.kind is ErrorKind.CANCELLED and b.batches == []
    assert caught.value.cost == pytest.approx(0.04)


@pytest.mark.asyncio
async def test_a_call_cut_off_on_one_connection_stays_held_on_the_next():
    # a's first request outlives the batch deadline and never reports, so it may have been
    # billed: its worst case (0.05) stays held. Of the 0.10 budget, b may promise 0.05, which
    # covers one request, not two.
    class Stalls(ScriptedProvider):
        async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
            await reserve_request(config_override, self.price, self.name)
            self.batches.append(batch)
            await asyncio.sleep(30)

    a, b = Stalls(price=0.05), ScriptedProvider(price=0.05)
    quick = CoreSettings(max_retries=1, retry_delay=0.0, request_timeout=0.2)
    big = job(10)
    big.budget = 0.10
    with pytest.raises(EngineError) as caught:
        await TranslationEngine(registry_with(ScriptedBackend({"a": a, "b": b})), quick).translate(big, [A, B])
    assert caught.value.kind is ErrorKind.CANCELLED and len(b.batches) == 1
    assert caught.value.reserved == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_a_call_the_provider_timed_out_stays_held_on_the_next():
    # a's first request ends with the provider's own timeout, which the server may have billed,
    # so its worst case (0.05) stays held when the stall moves the job on. Of the 0.10 budget, b
    # may promise 0.05, which covers one request, not two.
    a = ScriptedProvider([ProviderTimeoutError("late")], price=0.05)
    b = ScriptedProvider(price=0.05)
    big = job(10)
    big.budget = 0.10
    with pytest.raises(EngineError) as caught:
        await engine({"a": a, "b": b}).translate(big, [A, B])
    assert len(a.batches) == 1 and len(b.batches) == 1
    assert caught.value.kind is ErrorKind.CANCELLED and caught.value.reserved == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_a_connection_saved_with_a_new_key_is_not_held_by_the_old_cooldown():
    clock = Clock()
    a = ScriptedProvider([AuthenticationError("bad key")])
    b = ScriptedProvider()
    eng = engine({"a": a, "b": b}, clock)
    await eng.translate(job(5), [Connection("a", "scripted", secrets={"api_key": "old"}), B])
    calls = len(a.batches)
    await eng.translate(job(5), [Connection("a", "scripted", secrets={"api_key": "old"}), B])
    assert len(a.batches) == calls  # the rejected key sits out its hour
    fixed = await eng.translate(job(5), [Connection("a", "scripted", secrets={"api_key": "new"}), B])
    assert len(a.batches) == calls + 1 and fixed.connection_id == "a"


@pytest.mark.asyncio
async def test_each_failover_logs_one_info_line_with_ids_and_kind_only(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="subtitle_translator.core.engine")
    a = ScriptedProvider([RateLimitError("https://secret.example/x?key=SECRET-KEY", retry_after=120)])
    b = ScriptedProvider([AuthenticationError("bad SECRET-KEY")])
    with pytest.raises(EngineError):
        await engine({"a": a, "b": b}).translate(
            job(5), [Connection("a", "scripted", secrets={"api_key": "SECRET-KEY"}), B]
        )
    lines = [r.getMessage() for r in caplog.records if r.name == "subtitle_translator.core.engine"]
    assert len(lines) == 2 and all(r.levelno == logging.INFO for r in caplog.records if r.name.endswith("engine"))
    assert lines[0] == "Failover: connection a ended with rate_limited, next is b"
    assert lines[1].startswith("Failover: connection b ended with auth_required") and "no connection left" in lines[1]
    assert "SECRET" not in " ".join(lines) and "secret.example" not in " ".join(lines)


@pytest.mark.asyncio
async def test_progress_after_a_failover_carries_what_earlier_attempts_spent():
    a = ScriptedProvider(["ok", QuotaExhaustedError("daily", retry_after=600)], tokens=10, cost=0.01)
    b = ScriptedProvider(tokens=20, cost=0.02)
    seen = []
    outcome = await engine({"a": a, "b": b}).translate(
        job(10), [A, B], events=EngineEvents(on_progress=seen.append)
    )
    assert outcome.status == "complete" and outcome.cost == pytest.approx(0.03)
    from_b = [p for p in seen if p.connection_id == "b"]
    assert from_b, "the second connection reported no progress"
    assert from_b[-1].cost == pytest.approx(0.03) and from_b[-1].tokens == 30
    assert all(p.cost >= 0.01 and p.tokens >= 10 for p in from_b)
    costs = [p.cost for p in seen]
    assert costs == sorted(costs)


@pytest.mark.asyncio
async def test_usage_and_checkpoint_totals_after_a_failover_run_on_from_earlier_attempts():
    a = ScriptedProvider(["ok", QuotaExhaustedError("daily", retry_after=600)], tokens=10, cost=0.01)
    b = ScriptedProvider(tokens=20, cost=0.02)
    usage, checkpoints = [], []

    async def on_usage(record):
        usage.append(record)

    async def on_checkpoint(checkpoint):
        checkpoints.append(checkpoint)

    outcome = await engine({"a": a, "b": b}).translate(
        job(10), [A, B], events=EngineEvents(on_usage=on_usage, on_checkpoint=on_checkpoint)
    )
    assert [(u.tokens, u.total_tokens) for u in usage] == [(10, 10), (20, 30)]
    assert usage[-1].total_cost == pytest.approx(0.03)
    assert checkpoints[-1].total_tokens == 30 == outcome.tokens
    assert checkpoints[-1].total_cost == pytest.approx(outcome.cost)


@pytest.mark.asyncio
async def test_a_job_that_finishes_late_does_not_clear_a_newer_cooldown():
    clock = Clock()
    release = asyncio.Event()
    started = asyncio.Event()

    class Slow(ScriptedProvider):
        async def translate_batch(self, batch, **kwargs):
            started.set()
            await release.wait()
            return await super().translate_batch(batch, **kwargs)

    a = Slow()
    eng = engine({"a": a, "b": ScriptedProvider()}, clock)
    running = asyncio.ensure_future(eng.translate(job(5), [A]))
    await started.wait()
    # Meanwhile another job's request on the same connection is told to wait.
    eng._start_cooling(A, clock.now + 600, True)
    release.set()
    assert (await running).connection_id == "a"
    assert eng._cooling_until(A) == clock.now + 600


@pytest.mark.asyncio
async def test_a_short_failover_wait_does_not_shorten_a_longer_running_one():
    clock = Clock()
    eng = engine({"a": ScriptedProvider(), "b": ScriptedProvider()}, clock)
    eng._start_cooling(A, clock.now + 600, True)
    eng._start_cooling(A, clock.now + 30, False)
    assert eng._cooling_until(A) == clock.now + 600 and eng._cooling_on_provider_wait(A)
