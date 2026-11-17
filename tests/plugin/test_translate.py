# tests/plugin/test_translate.py
"""translate runs one attempt on one connection and streams what Bazarr+ records."""

import asyncio

import pytest

from subtitle_translator.core.backends import BackendRegistry, RegisteredBackend
from subtitle_translator.core.checkpoints import Checkpoint, CheckpointLine, source_digest
from subtitle_translator.core.errors import QuotaExhaustedError
from subtitle_translator.plugin.contract import ContractError, TranslatorContract
from subtitle_translator.plugin.scope import PluginScope
from tests.core.engine_fakes import ScriptedBackend, ScriptedProvider


class Sink:
    def __init__(self):
        self.events = []

    def progress(self, done, total, message):
        self.events.append(("progress", done, total))

    def heartbeat(self):
        self.events.append(("heartbeat",))

    async def reserve(self, amount, model):
        self.events.append(("reserve", amount))
        return f"hold-{sum(1 for e in self.events if e[0] == 'reserve')}"

    async def usage(self, report):
        self.events.append(("usage", report["tokens"], report["total_tokens"]))
        self.settled = [*getattr(self, "settled", []), *report["reservations"]]

    async def checkpoint(self, data):
        self.events.append(("checkpoint", sorted(int(p) for p in data["lines"])))


def contract(provider):
    backend = ScriptedBackend({"c1": provider})
    backend.type_id = "scripted"
    result = TranslatorContract(PluginScope.builtin())
    result._registry = BackendRegistry([RegisteredBackend("scripted", backend, "lavx-ai-subtitle-translator", "0", None)])
    return result


def request(n=10, **extra):
    return {
        "job_id": "job-1",
        "source": {"code": "en", "name": "English"},
        "target": {"code": "he", "name": "Hebrew", "script": "Hebr", "rtl": True},
        "lines": [{"position": i, "text": f"<b>l{i}</b>"} for i in range(n)],
        "config": {"settings": {"max_retries": 1}, "connection": {"id": "c1", "type": "scripted", "enabled": True, "settings": {}}},
        "secrets": {},
        **extra,
    }


@pytest.mark.asyncio
async def test_a_clean_attempt_streams_usage_and_checkpoints_and_totals_match():
    sink = Sink()
    result = await contract(ScriptedProvider()).translate(request(), sink)
    assert result["status"] == "complete" and result["missing"] == []
    assert result["lines"][0]["text"] == "T<b>l0</b>" and "‫" not in result["lines"][0]["text"]
    usage = [e for e in sink.events if e[0] == "usage"]
    assert result["tokens"] == sum(e[1] for e in usage) == usage[-1][2]
    assert [e for e in sink.events if e[0] == "checkpoint"] == [("checkpoint", [0, 1, 2, 3, 4]), ("checkpoint", [5, 6, 7, 8, 9])]
    assert any(e[0] == "progress" for e in sink.events)
    assert [b["positions"] for b in result["batches"]] == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert {b["model"] for b in result["batches"]} == {"scripted-model"}


@pytest.mark.asyncio
async def test_a_budget_that_cannot_cover_the_worst_case_sends_nothing():
    provider = ScriptedProvider(price=0.02)
    with pytest.raises(ContractError) as caught:
        await contract(provider).translate(request(budget=0.01), Sink())
    assert caught.value.code == "cancelled" and caught.value.reason == "budget_exhausted"
    assert provider.batches == []


@pytest.mark.asyncio
async def test_quota_is_an_error_never_partial():
    with pytest.raises(ContractError) as caught:
        await contract(ScriptedProvider(["ok", QuotaExhaustedError("daily", retry_after=900)])).translate(request(), Sink())
    assert caught.value.code == "quota_exhausted" and caught.value.retry_after == 900


@pytest.mark.asyncio
async def test_checkpoints_from_another_connection_are_used_and_not_billed_again():
    earlier = Checkpoint({i: CheckpointLine(source_digest(f"<b>l{i}</b>"), f"old{i}") for i in range(5)},
                         40, 0.4, 40, 0.4, "other-connection", "other-model").to_dict()
    provider = ScriptedProvider()
    result = await contract(provider).translate(request(checkpoints=[earlier]), Sink())
    assert [line["index"] for b in provider.batches for line in b.lines] == [str(i) for i in range(5, 10)]
    assert result["tokens"] == 10 and result["lines"][0]["text"] == "old0"


@pytest.mark.asyncio
async def test_heartbeats_flow_while_a_request_is_slow():
    gate = asyncio.Event()

    async def slow(batch):
        await gate.wait()
        from subtitle_translator.core.provider import TranslationResult

        return TranslationResult([{"index": line["index"], "content": "x"} for line in batch.lines], "m")

    class Slow(ScriptedProvider):
        async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
            self.batches.append(batch)
            return await slow(batch)

    sink = Sink()
    task = asyncio.create_task(contract(Slow()).translate(request(3), sink, heartbeat_interval=0.01))
    await asyncio.sleep(0.05)
    gate.set()
    await task
    assert sum(1 for e in sink.events if e[0] == "heartbeat") >= 2


@pytest.mark.asyncio
async def test_cancel_stops_reporting():
    started = asyncio.Event()

    class Hanging(ScriptedProvider):
        async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
            started.set()
            await asyncio.Event().wait()

    sink = Sink()
    task = asyncio.create_task(contract(Hanging()).translate(request(10), sink, heartbeat_interval=0.01))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    count = len(sink.events)
    await asyncio.sleep(0.05)
    assert len(sink.events) == count
    assert not any(e[0] in ("usage", "checkpoint") for e in sink.events)


@pytest.mark.asyncio
async def test_a_malformed_request_is_invalid_config_not_a_crash():
    with pytest.raises(ContractError) as caught:
        await contract(ScriptedProvider()).translate({**request(), "lines": [{"position": -1, "text": "x"}]}, Sink())
    assert caught.value.code == "invalid_config" and "lines[0].position" in caught.value.field_errors


@pytest.mark.asyncio
async def test_each_paid_request_is_held_before_it_is_sent_and_its_usage_settles_the_hold():
    sink = Sink()
    await contract(ScriptedProvider(price=0.02)).translate(request(budget=1.0), sink)
    kinds = [e[0] for e in sink.events if e[0] in ("reserve", "usage")]
    assert kinds == ["reserve", "usage", "reserve", "usage"]
    assert sink.settled == ["hold-1", "hold-2"]


@pytest.mark.asyncio
async def test_the_host_cancel_event_stops_the_attempt_and_its_reports():
    started = asyncio.Event()

    class Hanging(ScriptedProvider):
        async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
            started.set()
            await asyncio.Event().wait()

    sink, cancel = Sink(), asyncio.Event()
    task = asyncio.create_task(
        contract(Hanging()).translate(request(10), sink, heartbeat_interval=0.01, cancelled=cancel)
    )
    await started.wait()
    cancel.set()
    with pytest.raises(ContractError) as caught:
        await asyncio.wait_for(task, 1)
    assert caught.value.code == "cancelled"
    count = len(sink.events)
    await asyncio.sleep(0.05)
    assert len(sink.events) == count
    assert not any(e[0] in ("usage", "checkpoint") for e in sink.events)


@pytest.mark.asyncio
async def test_a_job_cancelled_before_it_starts_sends_nothing():
    provider, cancel = ScriptedProvider(), asyncio.Event()
    cancel.set()
    with pytest.raises(ContractError) as caught:
        await contract(provider).translate(request(), Sink(), cancelled=cancel)
    assert caught.value.code == "cancelled" and provider.batches == []
