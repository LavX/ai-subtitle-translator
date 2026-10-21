"""One attempt against one connection: checkpoints in and out, usage, budget, kinds."""

import asyncio

import pytest

from subtitle_translator.core.backends import Connection
from subtitle_translator.core.checkpoints import Checkpoint, CheckpointLine, source_digest
from subtitle_translator.core.context import Language, MediaContext
from subtitle_translator.core.engine import (
    EngineError,
    EngineEvents,
    Line,
    TranslationEngine,
    TranslationJob,
)
from subtitle_translator.core.errors import (
    DeclinedError,
    ErrorKind,
    InvalidResponseError,
    QuotaExhaustedError,
)
from subtitle_translator.core.models import MAX_LINE_CHARS
from subtitle_translator.core.settings import CoreSettings
from tests.core.engine_fakes import ScriptedBackend, ScriptedProvider, registry_with

HEBREW = Language("he", "Hebrew", "Hebr", rtl=True)
ENGLISH = Language("en", "English")
BASE = CoreSettings(max_retries=1, retry_delay=0.0, request_timeout=5.0)


def job(n=10, **kw):
    return TranslationJob(
        source=ENGLISH, target=Language("hu", "Hungarian"), lines=[Line(i, f"l{i}") for i in range(n)], **kw
    )


def engine(providers, **kw):
    return TranslationEngine(registry_with(ScriptedBackend(providers, **kw)), BASE)


CONN = Connection("c1", "scripted")


@pytest.mark.asyncio
async def test_a_clean_attempt_is_complete_and_reports_every_call():
    usage, checkpoints = [], []

    async def on_usage(u):
        usage.append(u)

    async def on_checkpoint(c):
        checkpoints.append(c)

    outcome = await engine({"c1": ScriptedProvider()}).attempt(
        job(10), CONN, EngineEvents(on_usage=on_usage, on_checkpoint=on_checkpoint)
    )
    assert outcome.status == "complete" and outcome.missing == []
    assert [line.text for line in outcome.lines] == [f"Tl{i}" for i in range(10)]
    assert len(usage) == 2 and len(checkpoints) == 2
    assert outcome.tokens == sum(u.tokens for u in usage) == checkpoints[-1].total_tokens
    assert outcome.cost == pytest.approx(sum(u.cost for u in usage))


@pytest.mark.asyncio
async def test_prior_checkpoints_are_not_translated_again():
    provider = ScriptedProvider()
    done = Checkpoint(
        {i: CheckpointLine(source_digest(f"l{i}"), f"old{i}") for i in range(5)}, 10, 0.01, 10, 0.01, "c0", "m"
    )
    outcome = await engine({"c1": provider}).attempt(job(10, checkpoints=[done]), CONN)
    sent = [line["index"] for b in provider.batches for line in b.lines]
    assert sent == [str(i) for i in range(5, 10)]
    assert [line.text for line in outcome.lines[:5]] == [f"old{i}" for i in range(5)]
    assert outcome.tokens == 10  # only this attempt's spend


@pytest.mark.asyncio
async def test_quota_fails_the_attempt_with_its_checkpoints_and_is_never_partial():
    provider = ScriptedProvider(["ok", QuotaExhaustedError("daily", retry_after=600)])
    with pytest.raises(EngineError) as caught:
        await engine({"c1": provider}).attempt(job(15), CONN)
    error = caught.value
    assert error.kind is ErrorKind.QUOTA_EXHAUSTED and error.retry_after == 600
    assert set(error.checkpoints[0].lines) == {0, 1, 2, 3, 4}
    assert error.tokens == 10


@pytest.mark.asyncio
async def test_content_failures_give_a_partial_result_with_missing_positions():
    bad = InvalidResponseError("garbled")
    provider = ScriptedProvider(["ok", bad, bad])
    outcome = await engine({"c1": provider}).attempt(job(10), CONN)
    assert outcome.status == "partial" and outcome.missing == [5, 6, 7, 8, 9]
    assert [line.text for line in outcome.lines[5:]] == [f"l{i}" for i in range(5, 10)]  # source kept


@pytest.mark.asyncio
async def test_an_over_long_line_is_left_untranslated_not_fatal():
    def runaway(batch):
        from subtitle_translator.core.provider import TranslationResult

        return TranslationResult(
            [{"index": line["index"], "content": "x" * (MAX_LINE_CHARS + 1)} for line in batch.lines], "m"
        )

    outcome = await engine({"c1": ScriptedProvider([runaway, "ok"])}).attempt(job(10), CONN)
    assert outcome.status == "partial" and outcome.missing == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_a_zero_budget_sends_nothing_paid():
    provider = ScriptedProvider(price=0.01)
    with pytest.raises(EngineError) as caught:
        await engine({"c1": provider}).attempt(job(10, budget=0.0), CONN)
    assert caught.value.kind is ErrorKind.CANCELLED and caught.value.reason == "budget_exhausted"
    assert provider.batches == []


@pytest.mark.asyncio
async def test_rtl_marks_only_when_asked():
    target_job = TranslationJob(source=ENGLISH, target=HEBREW, lines=[Line(0, "Hi")])
    marked = await engine({"c1": ScriptedProvider()}).attempt(target_job, CONN)
    assert marked.lines[0].text.startswith("‫")
    plain_job = TranslationJob(source=ENGLISH, target=HEBREW, lines=[Line(0, "Hi")], add_rtl_marks=False)
    plain = await engine({"c1": ScriptedProvider()}).attempt(plain_job, CONN)
    assert "‫" not in plain.lines[0].text


@pytest.mark.asyncio
async def test_media_and_languages_reach_the_provider_and_the_connection_is_opened_once():
    provider = ScriptedProvider()
    backend = ScriptedBackend({"c1": provider})
    eng = TranslationEngine(registry_with(backend), BASE)
    media = MediaContext(title="Arrival", year=2016)
    await eng.attempt(job(3, media=media), CONN)
    await eng.attempt(job(3, media=media), CONN)
    assert provider.batches[0].media == media and provider.batches[0].target.name == "Hungarian"
    assert backend.opened == ["c1"]  # key health and learned sizes are kept


@pytest.mark.asyncio
async def test_an_unknown_back_end_type_is_invalid_config():
    with pytest.raises(EngineError) as caught:
        await engine({"c1": ScriptedProvider()}).attempt(job(3), Connection("c1", "no-such-type"))
    assert caught.value.kind is ErrorKind.INVALID_CONFIG


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,kind,reason",
    [
        (DeclinedError("no such pair", reason="language_unsupported"), ErrorKind.DECLINED, "language_unsupported"),
        (ValueError("a bug in the back end"), ErrorKind.INTERNAL, None),
    ],
)
async def test_a_declined_or_internal_batch_fails_the_attempt_and_is_never_partial(failure, kind, reason):
    provider = ScriptedProvider(["ok", failure])
    with pytest.raises(EngineError) as caught:
        await engine({"c1": provider}).attempt(job(10), CONN)
    assert caught.value.kind is kind and caught.value.reason == reason
    assert set(caught.value.checkpoints[0].lines) == {0, 1, 2, 3, 4}


@pytest.mark.asyncio
async def test_each_paid_request_is_held_before_it_is_sent_and_settled_by_its_usage():
    holds, settled = [], []

    async def on_reserve(amount, model):
        holds.append(amount)
        return f"hold-{len(holds)}"

    async def on_usage(usage):
        settled.extend(usage.reservations)

    provider = ScriptedProvider(price=0.02)
    await engine({"c1": provider}).attempt(job(10, budget=1.0), CONN, EngineEvents(on_usage=on_usage, on_reserve=on_reserve))
    assert holds == [0.02, 0.02] and settled == ["hold-1", "hold-2"]


@pytest.mark.asyncio
async def test_reconfigure_opens_every_connection_again_with_the_new_settings():
    provider = ScriptedProvider()
    backend = ScriptedBackend({"c1": provider})
    eng = TranslationEngine(registry_with(backend), BASE)
    await eng.attempt(job(3), CONN)
    await eng.reconfigure(CoreSettings(max_retries=1, retry_delay=0.0, request_timeout=9.0))
    await eng.attempt(job(3), CONN)
    assert backend.opened == ["c1", "c1"]
    assert [base.request_timeout for base in backend.bases] == [5.0, 9.0]


@pytest.mark.asyncio
async def test_reconfigure_closes_what_it_retired_once_no_attempt_uses_it():
    started, release = asyncio.Event(), asyncio.Event()

    class Slow(ScriptedProvider):
        async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
            started.set()
            await release.wait()
            return await super().translate_batch(batch, model, temperature, config_override)

    busy, idle = Slow(), ScriptedProvider()
    eng = TranslationEngine(registry_with(ScriptedBackend({"busy": busy, "idle": idle})), BASE)
    await eng.attempt(job(3), Connection("idle", "scripted"))
    running = asyncio.create_task(eng.attempt(job(3), Connection("busy", "scripted")))
    await started.wait()
    await eng.reconfigure(CoreSettings(max_retries=1, retry_delay=0.0, request_timeout=9.0))
    assert idle.closed is True and busy.closed is False  # a running job still uses busy
    release.set()
    assert (await running).status == "complete"
    assert busy.closed is True


@pytest.mark.asyncio
async def test_a_retired_provider_that_another_connection_uses_stays_open():
    # The standalone shares one OpenRouter provider between connections.
    started, release = asyncio.Event(), asyncio.Event()

    class FirstCallWaits(ScriptedProvider):
        waiting = True

        async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
            if self.waiting:
                self.waiting = False
                started.set()
                await release.wait()
            return await super().translate_batch(batch, model, temperature, config_override)

    shared = FirstCallWaits()
    eng = TranslationEngine(registry_with(ScriptedBackend({"a": shared, "b": shared})), BASE)
    running = asyncio.create_task(eng.attempt(job(3), Connection("b", "scripted")))
    await started.wait()
    await eng.reconfigure(CoreSettings(max_retries=1, retry_delay=0.0, request_timeout=9.0))
    await eng.attempt(job(3), Connection("a", "scripted"))  # opened after the change
    release.set()
    await running
    assert shared.closed is False  # a still uses it
    await eng.reconfigure(CoreSettings(max_retries=1, retry_delay=0.0, request_timeout=5.0))
    assert shared.closed is True


@pytest.mark.asyncio
async def test_each_batch_names_the_model_that_served_it():
    from subtitle_translator.core.engine import BatchModel
    from subtitle_translator.core.provider import TranslationResult

    def routed(batch):
        return TranslationResult(
            [{"index": item["index"], "content": f"R{item['content']}"} for item in batch.lines], "routed/model-b", 10, 0.01
        )

    checkpoints = []

    async def on_checkpoint(checkpoint):
        checkpoints.append(checkpoint)

    outcome = await engine({"c1": ScriptedProvider(["ok", routed])}).attempt(
        job(10), CONN, EngineEvents(on_checkpoint=on_checkpoint)
    )
    assert outcome.batches == [BatchModel((0, 1, 2, 3, 4), "scripted-model"), BatchModel((5, 6, 7, 8, 9), "routed/model-b")]
    assert [c.model for c in checkpoints] == ["scripted-model", "routed/model-b"]
