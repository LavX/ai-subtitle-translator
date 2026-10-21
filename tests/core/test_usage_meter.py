"""Per-call usage reports, the budget rule, and holds on requests in flight."""

import asyncio

import pytest

from subtitle_translator.core.usage import Usage, UsageMeter


@pytest.mark.asyncio
async def test_record_reports_each_billed_call_with_running_totals():
    seen: list[Usage] = []

    async def sink(usage: Usage) -> None:
        seen.append(usage)

    meter = UsageMeter(budget=None, sink=sink)
    await meter.record(10, 0.01, "m")
    await meter.record(0, 0.0, "m")  # nothing billed and nothing held: nothing reported
    await meter.record(5, 0.002, "m")
    assert [(u.tokens, u.total_tokens) for u in seen] == [(10, 10), (5, 15)]
    assert seen[-1].total_cost == pytest.approx(0.012)
    assert meter.calls == 2


def test_budget_rule():
    meter = UsageMeter(budget=0.10)
    meter.cost = 0.07
    assert meter.remaining == pytest.approx(0.03)
    assert meter.allows(0.02) is True
    assert meter.allows(0.05) is False  # a worst case that would exceed what is left
    assert meter.allows(0.0) is True  # a free request never exceeds it
    assert meter.allows(None) is True  # no price known, money left
    meter.reserved = 0.03
    assert meter.allows(None) is False  # what is left is held by requests in flight
    assert meter.allows(0.0) is True
    assert UsageMeter(budget=None).allows(99.0) is True


@pytest.mark.asyncio
async def test_parallel_reservations_never_promise_the_same_money():
    stored = []

    async def reserve_sink(amount, model):
        await asyncio.sleep(0)  # every hold yields while it is stored
        stored.append(amount)
        return f"r{len(stored)}"

    from subtitle_translator.core.usage import BudgetExhaustedError

    meter = UsageMeter(budget=0.12, reserve_sink=reserve_sink)
    results = await asyncio.gather(*(meter.reserve(0.05, "m") for _ in range(4)), return_exceptions=True)
    held = [r for r in results if not isinstance(r, BaseException)]
    assert len(held) == 2 and all(isinstance(r, BudgetExhaustedError) for r in results if r not in held)
    assert meter.reserved == pytest.approx(0.10) and stored == [0.05, 0.05]
    assert {r.id for r in held} == {"r1", "r2"}


@pytest.mark.asyncio
async def test_settling_a_call_releases_its_hold_and_names_it():
    seen = []

    async def sink(usage):
        seen.append(usage)

    async def reserve_sink(amount, model):
        return "hold-1"

    meter = UsageMeter(budget=1.0, sink=sink, reserve_sink=reserve_sink)
    hold = await meter.reserve(0.3, "m")
    assert meter.remaining == pytest.approx(0.7)
    await meter.record(0, 0.0, "m", reservations=[hold])  # an unbilled call still settles
    assert meter.reserved == 0.0 and meter.remaining == pytest.approx(1.0)
    assert seen[-1].reservations == ("hold-1",) and meter.calls == 0


@pytest.mark.asyncio
async def test_reserve_request_reads_the_meter_attached_to_the_request():
    from subtitle_translator.core.errors import ErrorKind, error_kind
    from subtitle_translator.core.models import TranslationConfig
    from subtitle_translator.core.usage import BudgetExhaustedError, reserve_request

    config = TranslationConfig().for_operation("job")
    await reserve_request(config, 5.0)  # no meter attached: nothing to enforce
    config._usage_meter = UsageMeter(budget=0.01)
    with pytest.raises(BudgetExhaustedError) as caught:
        await reserve_request(config, 0.02)
    assert error_kind(caught.value) is ErrorKind.CANCELLED and caught.value.reason == "budget_exhausted"
