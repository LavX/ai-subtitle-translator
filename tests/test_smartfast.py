"""Routing behavior at the endpoint-catalog HTTP boundary."""

import asyncio
import copy
import importlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError


def router_module():
    return importlib.import_module("subtitle_translator.providers.smartfast")


def endpoint(tag, prompt=0.1, completion=0.1, *, latency_seconds=None, throughput=None, **extra):
    """Accept USD/million and latency seconds; emit USD/token and milliseconds."""
    return {
        "tag": tag,
        "provider_name": tag.title(),
        "status": 0,
        "pricing": {
            "prompt": str(prompt / 1_000_000),
            "completion": str(completion / 1_000_000),
            "request": "0",
        },
        "latency_last_30m": {
            "p50": latency_seconds * 1000 if latency_seconds is not None else None
        },
        "throughput_last_30m": {"p50": throughput},
        "context_length": 32768,
        "max_prompt_tokens": 32768,
        "max_completion_tokens": 8192,
        "supported_parameters": ["temperature", "max_tokens", "reasoning"],
        **extra,
    }


class Catalog:
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.requests = []
        self.status = 200
        self.gate = None

    async def __call__(self, request):
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        return httpx.Response(
            self.status,
            json={"data": {"endpoints": copy.deepcopy(self.endpoints)}},
        )


def make_router(**kwargs):
    return router_module().SmartFastRouter(
        SimpleNamespace(openrouter_api_base="https://fixture.invalid/api/v1", request_timeout=30),
        **kwargs,
    )


def client_for(catalog):
    return httpx.AsyncClient(
        transport=httpx.MockTransport(catalog),
        base_url="https://fixture.invalid/api/v1",
    )


async def select(router, client, **kwargs):
    values = {
        "model": "fixture/model",
        "session_id": "opaque-session",
        "account_scope": "account-fingerprint",
        "input_tokens": 100,
        "output_tokens": 100,
        "context_key": "context-fingerprint",
    }
    values.update(kwargs)
    return await router.select(client, **values)


@pytest.mark.asyncio
async def test_median_cutoff_excludes_large_outlier_before_bootstrap():
    catalog = Catalog(
        [
            endpoint(tag, price, price)
            for tag, price in zip("abcde", [0.10, 0.11, 0.12, 0.13, 1.20], strict=True)
        ]
    )
    router = make_router()
    policy = router_module().SmartFastPolicy(maxPromptPrice=2, maxCompletionPrice=2)
    async with client_for(catalog) as client:
        decision = await select(router, client, policy=policy)
    assert decision.provider["only"] == ["a", "b", "c", "d"]
    assert decision.diagnostics["excluded"]["e"] == "price_outlier"
    assert decision.provider["max_price"] == {"prompt": 2, "completion": 2, "request": 0}
    assert catalog.requests[0].url.path == "/api/v1/models/fixture/model/endpoints"


@pytest.mark.asyncio
async def test_sparse_pool_uses_cheapest_price_and_preserves_boundary():
    catalog = Catalog([endpoint("a", 0.1), endpoint("b", 0.3, 0.3), endpoint("c", 0.301, 0.301)])
    router = make_router()
    async with client_for(catalog) as client:
        decision = await select(router, client)
    assert decision.provider["only"] == ["a", "b"]


@pytest.mark.asyncio
async def test_catalog_millisecond_latency_cannot_reverse_throughput_ranking():
    catalog = Catalog(
        [
            endpoint("low-latency", latency_last_30m={"p50": 100}, throughput=10),
            endpoint("high-throughput", latency_last_30m={"p50": 1000}, throughput=100),
        ]
    )
    router = make_router()
    async with client_for(catalog) as client:
        decision = await select(router, client, output_tokens=100)
    assert decision.provider["only"] == ["high-throughput"]
    assert decision.diagnostics["estimated_seconds"] == {
        "low-latency": 10.1,
        "high-throughput": 2,
    }
    assert decision.diagnostics["fast_pool"] == ["high-throughput"]


@pytest.mark.asyncio
async def test_catalog_millisecond_latency_preserves_twenty_percent_speed_boundary():
    catalog = Catalog(
        [
            endpoint("a", 0.12, latency_last_30m={"p50": 1000}, throughput=100),
            endpoint("b", 0.11, latency_last_30m={"p50": 1400}, throughput=100),
            endpoint("c", 0.10, latency_last_30m={"p50": 1401}, throughput=100),
        ]
    )
    router = make_router()
    async with client_for(catalog) as client:
        decision = await select(router, client, output_tokens=100)
    assert decision.diagnostics["estimated_seconds"] == pytest.approx(
        {"a": 2, "b": 2.4, "c": 2.401}
    )
    assert decision.diagnostics["fast_pool"] == ["a", "b"]
    assert decision.provider["only"] == ["b"]


@pytest.mark.asyncio
async def test_catalog_millisecond_latency_compares_with_local_observation_in_seconds():
    catalog = Catalog(
        [
            endpoint("observed", latency_last_30m={"p50": 2000}, throughput=100),
            endpoint("catalog", latency_last_30m={"p50": 1000}, throughput=100),
        ]
    )
    router = make_router()
    async with client_for(catalog) as client:
        initial = await select(router, client, only=["observed"], output_tokens=100)
        router.observe(initial, 5, completed("observed", output=100), True)
        router.release(initial)
        decision = await select(router, client, session_id="fresh-session", output_tokens=100)
    assert decision.diagnostics["estimated_seconds"] == {"observed": 5, "catalog": 2}
    assert decision.diagnostics["fast_pool"] == ["catalog"]
    assert decision.provider["only"] == ["catalog"]


@pytest.mark.asyncio
async def test_speed_band_includes_twelve_seconds_but_excludes_twelve_point_zero_one():
    catalog = Catalog(
        [
            endpoint("a", 0.12, latency_seconds=9, throughput=100),
            endpoint("b", 0.11, latency_seconds=11, throughput=100),
            endpoint("c", 0.10, latency_seconds=11.01, throughput=100),
        ]
    )
    router = make_router()
    async with client_for(catalog) as client:
        decision = await select(router, client)
    assert decision.diagnostics["estimated_seconds"] == {"a": 10, "b": 12, "c": 12.01}
    assert decision.diagnostics["fast_pool"] == ["a", "b"]
    assert decision.provider["only"] == ["b"]
    assert decision.endpoint == "b"


@pytest.mark.asyncio
async def test_missing_metrics_bootstrap_without_fabricated_speed_or_order():
    router = make_router()
    async with client_for(Catalog([endpoint("a"), endpoint("b")])) as client:
        decision = await select(router, client)
    assert decision.provider["only"] == ["a", "b"]
    assert decision.provider["sort"] == "throughput"
    assert "order" not in decision.provider
    assert decision.diagnostics["bootstrap"] is True
    assert decision.diagnostics["estimated_seconds"] == {"a": None, "b": None}
    assert decision.session_id == "opaque-session"
    assert decision.endpoint is None


@pytest.mark.asyncio
async def test_unknown_speed_cannot_replace_known_fast_endpoint():
    router = make_router()
    catalog = Catalog(
        [endpoint("unknown", 0.05), endpoint("known", 0.1, latency_seconds=1, throughput=10)]
    )
    async with client_for(catalog) as client:
        decision = await select(router, client)
    assert decision.provider["only"] == ["known"]
    assert decision.diagnostics["bootstrap"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pricing",
    [
        {"prompt": "0.00000101", "completion": "0", "request": "0"},
        {"prompt": "0", "completion": "0.00000301", "request": "0"},
        {"prompt": "0", "completion": "0", "request": "0.000001"},
        {"prompt": "NaN", "completion": "0", "request": "0"},
        {"prompt": "0", "completion": "Infinity", "request": "0"},
        {"prompt": "0", "completion": "0", "request": "-1"},
        {"prompt": None, "completion": "0", "request": "0"},
        {"prompt": "0", "completion": "0", "request": None},
    ],
)
async def test_invalid_or_over_cap_pricing_fails_before_completion(pricing):
    router = make_router()
    catalog = Catalog([endpoint("a", pricing=pricing)])
    async with client_for(catalog) as client:
        with pytest.raises(router_module().SmartFastRoutingError):
            await select(router, client)
    assert [(r.method, r.url.path) for r in catalog.requests] == [
        ("GET", "/api/v1/models/fixture/model/endpoints")
    ]


@pytest.mark.asyncio
async def test_free_variant_preserves_slug_and_demands_zero_prices():
    router = make_router()
    catalog = Catalog([endpoint("zero", 0, 0), endpoint("paid", 0.001, 0)])
    async with client_for(catalog) as client:
        decision = await select(router, client, model="fixture/model:free")
    assert catalog.requests[0].url.path == "/api/v1/models/fixture/model:free/endpoints"
    assert decision.provider["only"] == ["zero"]
    assert decision.provider["max_price"] == {"prompt": 0, "completion": 0, "request": 0}


@pytest.mark.asyncio
async def test_duplicate_tags_do_not_inflate_median_population():
    repeated = endpoint("b", 0.11, 0.11)
    catalog = Catalog(
        [
            endpoint("a", 0.1, 0.1),
            repeated,
            copy.deepcopy(repeated),
            endpoint("c", 0.12, 0.12),
            endpoint("d", 0.30, 0.30),
        ]
    )
    router = make_router()
    async with client_for(catalog) as client:
        decision = await select(router, client)
    assert decision.provider["only"] == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_conflicting_duplicate_tag_fails_closed_for_that_tag():
    catalog = Catalog([endpoint("a"), endpoint("b"), endpoint("b", status=1)])
    router = make_router()
    async with client_for(catalog) as client:
        decision = await select(router, client)
    assert decision.provider["only"] == ["a"]


@pytest.mark.asyncio
async def test_filters_intersect_and_do_not_enable_tiers_implicitly():
    catalog = Catalog(
        [
            endpoint("a/region"),
            endpoint("a/fast"),
            endpoint("a/flex"),
            endpoint("b"),
            endpoint("a/no-reason", supported_parameters=["max_tokens"]),
        ]
    )
    router = make_router()
    async with client_for(catalog) as client:
        decision = await select(
            router, client, only=["a"], ignore=["a/no-reason"], required_parameters=["reasoning"]
        )
        tier = await select(router, client, session_id="tier", only=["a/fast"])
    assert decision.provider["only"] == ["a/region"]
    assert tier.provider["only"] == ["a/fast"]


@pytest.mark.asyncio
async def test_capability_and_context_limits_filter_before_price_comparison():
    catalog = Catalog(
        [
            endpoint("valid"),
            endpoint("short", max_prompt_tokens=99),
            endpoint("output", max_completion_tokens=99),
            endpoint("context", context_length=199),
            endpoint("unsupported", supported_parameters=[]),
            endpoint("down", status=1),
        ]
    )
    router = make_router()
    async with client_for(catalog) as client:
        decision = await select(router, client, required_parameters=["reasoning"])
    assert decision.provider["only"] == ["valid"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("medianPremiumPercent", float("nan")),
        ("speedTolerancePercent", float("inf")),
        ("maxPromptPrice", -1),
        ("maxCompletionPrice", 1001),
        ("sparsePremiumMultiplier", 0.99),
        ("medianPremiumPercent", 1001),
        ("speedTolerancePercent", 1001),
        ("sparsePremiumMultiplier", 101),
    ],
)
def test_policy_rejects_nonfinite_negative_or_unbounded_values(field, value):
    with pytest.raises(ValidationError):
        router_module().SmartFastPolicy(**{field: value})


def completed(tag, *, output=100, cached=0, prompt=100, cost=0.00002):
    return {
        "provider": tag,
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": output,
            "cost": cost,
            "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 5},
        },
    }


def two_fast():
    return [
        endpoint("a", latency_seconds=1, throughput=100),
        endpoint("b", 0.104, 0.104, latency_seconds=1, throughput=100),
    ]


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(router_module(), "monotonic", lambda: now[0], raising=False)
    return now


@pytest.mark.asyncio
async def test_catalog_refresh_is_single_flight_and_first_session_admission_is_atomic():
    catalog = Catalog(two_fast())
    catalog.gate = asyncio.Event()
    router = make_router()
    async with client_for(catalog) as client:
        tasks = [asyncio.create_task(select(router, client)) for _ in range(8)]
        await asyncio.sleep(0)
        catalog.gate.set()
        decisions = await asyncio.gather(*tasks)
    assert len(catalog.requests) == 1
    assert {d.endpoint for d in decisions} == {"a"}
    for decision in decisions:
        router.release(decision)


@pytest.mark.asyncio
async def test_catalog_expires_at_300_seconds_and_failed_refresh_fails_closed(clock):
    catalog = Catalog([endpoint("a")])
    router = make_router()
    async with client_for(catalog) as client:
        first = await select(router, client)
        router.release(first)
        catalog.status = 503
        clock[0] += 299.99
        second = await select(router, client)
        router.release(second)
        assert len(catalog.requests) == 1
        clock[0] += 0.01
        with pytest.raises(router_module().SmartFastRoutingError):
            await select(router, client)
    assert len(catalog.requests) == 2


@pytest.mark.asyncio
async def test_new_sessions_balance_inflight_but_repeated_session_stays_pinned():
    router = make_router()
    async with client_for(Catalog(two_fast())) as client:
        first = await select(router, client, session_id="first")
        second = await select(router, client, session_id="second")
        repeated = await select(router, client, session_id="first")
        assert (first.endpoint, second.endpoint, repeated.endpoint) == ("a", "b", "a")
        router.release(first)
        router.release(repeated)
        router.release(first)
        third = await select(router, client, session_id="third")
        assert third.endpoint == "a"
        router.release(second)
        router.release(third)


@pytest.mark.asyncio
async def test_account_context_and_model_do_not_share_affinity_or_measurements():
    router = make_router()
    async with client_for(Catalog(two_fast())) as client:
        first = await select(router, client, only=["b"])
        router.observe(first, 100, completed("b"), True)
        router.release(first)
        other_account = await select(router, client, account_scope="other-account")
        other_context = await select(router, client, context_key="other-context")
        other_model = await select(router, client, model="fixture/other-model")
        assert [d.endpoint for d in (other_account, other_context, other_model)] == ["a", "a", "a"]
        assert all(
            d.diagnostics["estimated_seconds"]["b"] == 2
            for d in (other_account, other_context, other_model)
        )
        for decision in (other_account, other_context, other_model):
            router.release(decision)


@pytest.mark.asyncio
async def test_failure_quarantines_pin_and_retry_remains_price_bounded(clock):
    router = make_router()
    async with client_for(Catalog(two_fast())) as client:
        first = await select(router, client)
        router.observe(first, 1, {"error": {"code": 429, "message": "never retain this"}}, False)
        router.release(first)
        retry = await select(router, client)
        assert retry.endpoint == "b"
        assert retry.session_id == first.session_id
        assert retry.provider["max_price"] == {"prompt": 1, "completion": 3, "request": 0}
        assert retry.diagnostics["excluded"]["a"] == "quarantine"
        assert first.diagnostics["success"] is False
        assert first.diagnostics["failure_status"] == 429
        assert "never retain this" not in json.dumps(first.diagnostics)
        router.release(retry)
        clock[0] += 31
        fresh = await select(router, client, session_id="fresh")
        assert fresh.endpoint == "a"
        router.release(fresh)


@pytest.mark.asyncio
async def test_three_bad_successes_allow_reselection_but_one_or_two_keep_affinity():
    router = make_router()
    async with client_for(Catalog(two_fast())) as client:
        for _ in range(3):
            decision = await select(router, client)
            assert decision.endpoint == "a"
            router.observe(decision, 10, completed("a"), True)
            router.release(decision)
        replacement = await select(router, client)
        assert replacement.endpoint == "b"
        router.release(replacement)


@pytest.mark.asyncio
async def test_observed_response_time_is_normalized_by_actual_output_count():
    router = make_router()
    async with client_for(Catalog(two_fast())) as client:
        initial = await select(router, client, only=["a"])
        router.observe(initial, 10, completed("a", output=500), True)
        router.release(initial)
        next_batch = await select(router, client, output_tokens=200)
        assert next_batch.diagnostics["estimated_seconds"]["a"] == 4
        router.release(next_batch)


@pytest.mark.asyncio
async def test_actual_cache_hits_affect_only_warm_session_cost_and_speed():
    warm = endpoint("a", 0.12, 0.1, latency_seconds=1, throughput=100)
    warm["pricing"]["input_cache_read"] = "0.00000001"
    router = make_router()
    async with client_for(
        Catalog([warm, endpoint("b", latency_seconds=1, throughput=100)])
    ) as client:
        first = await select(router, client, only=["a"])
        assert first.diagnostics["effective_cost"]["a"] == pytest.approx(0.000022)
        router.observe(first, 1, completed("a", cached=80), True)
        router.release(first)
        repeated = await select(router, client)
        assert repeated.endpoint == "a"
        assert repeated.diagnostics["cold_cost"]["a"] == pytest.approx(0.000022)
        assert repeated.diagnostics["effective_cost"]["a"] == pytest.approx(0.0000132)
        assert repeated.diagnostics["estimated_seconds"]["a"] == 1
        router.release(repeated)
        other = await select(router, client, session_id="cold-session")
        assert other.endpoint == "b"
        assert other.diagnostics["effective_cost"]["a"] == pytest.approx(0.000022)
        assert other.diagnostics["estimated_seconds"]["a"] == 2
        assert first.diagnostics["cache_read_tokens"] == 80
        assert first.diagnostics["cache_write_tokens"] == 5
        assert first.diagnostics["reported_cost"] == 0.00002
        router.release(other)


@pytest.mark.asyncio
async def test_affinity_cannot_bypass_new_price_limit_or_manual_restriction():
    router = make_router()
    catalog = Catalog(
        [
            endpoint("a", 0.2, 0.1, latency_seconds=1, throughput=100),
            endpoint("b", 0.1, 0.1, latency_seconds=1, throughput=100),
        ]
    )
    async with client_for(catalog) as client:
        initial = await select(router, client, only=["a"])
        router.release(initial)
        tightened = await select(
            router, client, policy=router_module().SmartFastPolicy(maxPromptPrice=0.1)
        )
        assert tightened.endpoint == "b"
        router.release(tightened)
        manual = await select(router, client, only=["a"])
        assert manual.endpoint == "a"
        router.release(manual)


@pytest.mark.asyncio
async def test_bootstrap_attribution_learns_exact_endpoint_and_preserves_session():
    router = make_router()
    async with client_for(Catalog([endpoint("a"), endpoint("b")])) as client:
        initial = await select(router, client)
        router.observe(initial, 2, completed("b"), True)
        router.release(initial)
        next_batch = await select(router, client)
        assert next_batch.endpoint == "b"
        assert next_batch.diagnostics["bootstrap"] is False
        assert next_batch.session_id == initial.session_id
        router.release(next_batch)


@pytest.mark.asyncio
async def test_ambiguous_provider_attribution_does_not_invent_variant_metrics():
    catalog = Catalog(
        [endpoint("a/one", provider_name="Shared"), endpoint("a/two", provider_name="Shared")]
    )
    router = make_router()
    async with client_for(catalog) as client:
        initial = await select(router, client)
        router.observe(initial, 1, completed("Shared"), True)
        router.release(initial)
        second = await select(router, client)
        assert second.endpoint is None
        assert second.diagnostics["estimated_seconds"] == {"a/one": None, "a/two": None}
        assert initial.diagnostics["observed_endpoint"] is None
        router.release(second)


@pytest.mark.asyncio
async def test_base_tag_pin_blocks_catalog_variants_outside_eligible_pool():
    router = make_router()
    catalog = Catalog(
        [
            endpoint("a", latency_seconds=1, throughput=100),
            endpoint("a/expensive", 0.9, 0.9, latency_seconds=1, throughput=100),
        ]
    )
    async with client_for(catalog) as client:
        decision = await select(router, client)
    assert decision.provider["only"] == ["a"]
    assert decision.provider["ignore"] == ["a/expensive"]
    router.release(decision)


@pytest.mark.asyncio
async def test_cancelling_one_catalog_waiter_does_not_cancel_shared_refresh():
    router = make_router()
    catalog = Catalog(two_fast())
    catalog.gate = asyncio.Event()
    async with client_for(catalog) as client:
        first = asyncio.create_task(select(router, client, session_id="first"))
        second = asyncio.create_task(select(router, client, session_id="second"))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        catalog.gate.set()
        decision = await second
        assert decision.endpoint == "a"
        assert len(catalog.requests) == 1
        router.release(decision)


@pytest.mark.asyncio
async def test_state_maps_are_bounded_and_expire_without_retaining_raw_context(monkeypatch, clock):
    module = router_module()
    monkeypatch.setattr(module, "MAX_SESSIONS", 2, raising=False)
    monkeypatch.setattr(module, "MAX_OBSERVATIONS", 2, raising=False)
    monkeypatch.setattr(module, "MAX_CATALOGS", 2, raising=False)
    router = make_router()
    async with client_for(Catalog(two_fast())) as client:
        for number in range(5):
            decision = await select(
                router,
                client,
                model=f"fixture/model-{number}",
                session_id=f"session-{number}",
                context_key="private subtitle context",
                account_scope="private account",
            )
            router.observe(decision, 2, completed(decision.endpoint), True)
            router.release(decision)
        assert len(router._sessions) <= 2
        assert len(router._observations) <= 2
        assert len(router._catalogs) <= 2
        assert "private subtitle context" not in repr(router._sessions)
        assert "private account" not in repr(router._sessions)
        clock[0] += 1801
        decision = await select(router, client)
        assert len(router._sessions) == 1
        assert len(router._observations) == 0
        assert len(router._catalogs) == 1
        router.release(decision)


@pytest.mark.asyncio
async def test_active_capacity_fails_closed_and_release_restores_admission(monkeypatch):
    monkeypatch.setattr(router_module(), "MAX_ACTIVE", 1, raising=False)
    router = make_router()
    async with client_for(Catalog(two_fast())) as client:
        first = await select(router, client)
        with pytest.raises(router_module().SmartFastRoutingError):
            await select(router, client, session_id="second")
        router.release(first)
        second = await select(router, client, session_id="second")
        assert second.endpoint == "a"
        router.release(second)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
async def test_invalid_workload_cannot_disable_price_or_speed_filtering(value):
    router = make_router()
    catalog = Catalog(two_fast())
    async with client_for(catalog) as client:
        with pytest.raises(router_module().SmartFastRoutingError):
            await select(router, client, output_tokens=value)
    assert not catalog.requests


@pytest.mark.asyncio
async def test_omitted_request_fee_is_permitted_only_with_upstream_zero_fee_cap():
    record = endpoint("a")
    del record["pricing"]["request"]
    router = make_router()
    async with client_for(Catalog([record])) as client:
        decision = await select(router, client)
    assert decision.provider["only"] == ["a"]
    assert decision.provider["max_price"]["request"] == 0
    router.release(decision)


@pytest.mark.asyncio
async def test_long_context_override_is_strictly_above_threshold_and_rechecked_per_batch():
    record = endpoint("a", 0.1, 0.1)
    record["pricing"]["overrides"] = [{"min_prompt_tokens": 100, "prompt": "0.000002"}]
    router = make_router()
    async with client_for(Catalog([record])) as client:
        boundary = await select(router, client, input_tokens=100)
        router.release(boundary)
        with pytest.raises(router_module().SmartFastRoutingError):
            await select(router, client, input_tokens=101)


@pytest.mark.asyncio
async def test_matching_later_overrides_win_per_key_and_inherit_unspecified_prices():
    record = endpoint("a", 0.9, 0.1)
    record["pricing"]["overrides"] = [
        {"min_prompt_tokens": 10, "prompt": "0.000003", "completion": "0.0000002"},
        {"min_prompt_tokens": 20, "prompt": "0.0000001"},
    ]
    router = make_router()
    async with client_for(Catalog([record])) as client:
        decision = await select(router, client)
    assert decision.diagnostics["cold_cost"]["a"] == pytest.approx(0.00003)
    router.release(decision)


@pytest.mark.asyncio
async def test_time_pricing_crosses_midnight_and_rechecks_weekday_without_catalog_refresh(
    monkeypatch,
):
    record = endpoint("a", 0.1, 0.1)
    record["pricing"]["overrides"] = [
        {
            "utc_start": 1630,
            "utc_end": 30,
            "utc_days": ["monday"],
            "prompt": "0.000002",
        }
    ]
    current = [datetime(2026, 9, 7, 0, 29, tzinfo=UTC)]
    monkeypatch.setattr(router_module(), "_utc_now", lambda: current[0], raising=False)
    router = make_router()
    catalog = Catalog([record])
    async with client_for(catalog) as client:
        with pytest.raises(router_module().SmartFastRoutingError):
            await select(router, client)
        current[0] = datetime(2026, 9, 7, 0, 30, tzinfo=UTC)
        off_peak = await select(router, client)
        router.release(off_peak)
        current[0] = datetime(2026, 9, 7, 16, 30, tzinfo=UTC)
        with pytest.raises(router_module().SmartFastRoutingError):
            await select(router, client)
        current[0] = datetime(2026, 9, 8, 16, 30, tzinfo=UTC)
        tuesday = await select(router, client)
        router.release(tuesday)
    assert len(catalog.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"unknown_condition": 1, "prompt": "0.000002"},
        {"utc_start": 2460, "utc_end": 5, "prompt": "0.000002"},
        {"min_prompt_tokens": "NaN", "prompt": "0.000002"},
        {"utc_days": ["not-a-day"], "prompt": "0.000002"},
        {"prompt": None},
    ],
)
async def test_unusable_conditional_pricing_fails_closed(override):
    record = endpoint("a")
    record["pricing"]["overrides"] = [override]
    router = make_router()
    async with client_for(Catalog([record])) as client:
        with pytest.raises(router_module().SmartFastRoutingError):
            await select(router, client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("input_cache_read", "NaN"),
        ("input_cache_write", "Infinity"),
        ("input_cache_write", "0.000002"),
        ("input_cache_read", "0.000002"),
    ],
)
async def test_cache_token_prices_are_finite_and_within_prompt_cap(field, value):
    record = endpoint("a")
    record["pricing"][field] = value
    router = make_router()
    async with client_for(Catalog([record])) as client:
        with pytest.raises(router_module().SmartFastRoutingError):
            await select(router, client)


@pytest.mark.asyncio
async def test_cache_hits_can_warm_a_session_after_its_first_cold_response():
    record = endpoint("a")
    record["pricing"]["input_cache_read"] = "0.00000001"
    router = make_router()
    async with client_for(Catalog([record])) as client:
        for cached in [0, 80]:
            decision = await select(router, client)
            router.observe(decision, 2, completed("a", cached=cached), True)
            router.release(decision)
        warm = await select(router, client)
        assert warm.diagnostics["effective_cost"]["a"] == pytest.approx(0.0000128)
        router.release(warm)


@pytest.mark.asyncio
async def test_overflowing_speed_estimates_remain_unknown_and_diagnostics_are_finite():
    router = make_router()
    record = endpoint("a", latency_seconds=1e305, throughput=1e-320)
    async with client_for(Catalog([record])) as client:
        decision = await select(router, client)
        assert decision.diagnostics["estimated_seconds"]["a"] is None
        router.observe(decision, 1e308, completed("a", output=1e-320), True)
        router.release(decision)
        second = await select(router, client)
        assert second.diagnostics["bootstrap"] is True
        json.dumps(second.diagnostics, allow_nan=False)
        router.release(second)


@pytest.mark.asyncio
async def test_separate_reasoning_price_cannot_bypass_completion_cap():
    record = endpoint("a")
    record["pricing"]["internal_reasoning"] = "0.000004"
    router = make_router()
    async with client_for(Catalog([record])) as client:
        with pytest.raises(router_module().SmartFastRoutingError):
            await select(router, client, required_parameters=["reasoning"])


@pytest.mark.asyncio
async def test_cold_cost_conservatively_accounts_for_quoted_cache_writes():
    record = endpoint("a", 0.1, 0.1)
    record["pricing"]["input_cache_write"] = "0.000000125"
    router = make_router()
    async with client_for(Catalog([record])) as client:
        decision = await select(router, client)
        assert decision.diagnostics["cold_cost"]["a"] == pytest.approx(0.0000225)
        router.release(decision)


@pytest.mark.asyncio
async def test_quarantine_exhaustion_reports_earliest_eligible_expiry(clock):
    router = make_router()
    async with client_for(Catalog(two_fast())) as client:
        first = await select(router, client, only=["a"])
        router.observe(first, 1, {"error": {"code": 429}}, False)
        router.release(first)
        clock[0] += 5
        second = await select(router, client, only=["b"])
        router.observe(second, 1, {"error": {"code": 503}}, False)
        router.release(second)
        with pytest.raises(router_module().SmartFastRoutingError) as raised:
            await select(router, client)
        assert raised.value.retryable
        assert raised.value.retry_after == 25
        clock[0] += 25
        recovered = await select(router, client)
        assert recovered.endpoint == "a"
        assert recovered.provider["max_price"] == {"prompt": 1, "completion": 3, "request": 0}
        router.release(recovered)


@pytest.mark.asyncio
@pytest.mark.parametrize("restriction", ["price", "context", "capability", "only", "ignore"])
async def test_permanent_ineligibility_is_not_hidden_by_quarantine(clock, restriction):
    router = make_router()
    async with client_for(Catalog([endpoint("a", latency_seconds=1, throughput=100)])) as client:
        first = await select(router, client)
        router.observe(first, 1, {"error": {"code": 429}}, False)
        router.release(first)
        options = {
            "price": {"policy": router_module().SmartFastPolicy(maxPromptPrice=0)},
            "context": {"input_tokens": 32769},
            "capability": {"required_parameters": ["unsupported"]},
            "only": {"only": ["absent"]},
            "ignore": {"ignore": ["a"]},
        }[restriction]
        with pytest.raises(router_module().SmartFastRoutingError) as raised:
            await select(router, client, **options)
        assert not raised.value.retryable
        assert raised.value.retry_after is None


@pytest.mark.asyncio
async def test_quarantine_cannot_promote_a_price_outlier(clock):
    router = make_router()
    catalog = Catalog(
        [
            endpoint("a", latency_seconds=1, throughput=100),
            endpoint("expensive", 0.9, 0.9, latency_seconds=1, throughput=100),
        ]
    )
    async with client_for(catalog) as client:
        first = await select(router, client)
        assert first.endpoint == "a"
        router.observe(first, 1, {"error": {"code": 429}}, False)
        router.release(first)
        with pytest.raises(router_module().SmartFastRoutingError) as raised:
            await select(router, client)
        assert raised.value.retryable
        assert raised.value.retry_after == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_success", [True, False])
async def test_late_response_cannot_change_recovered_endpoint_health_or_speed(clock, stale_success):
    router = make_router()
    async with client_for(Catalog([endpoint("a", latency_seconds=1, throughput=100)])) as client:
        failed = await select(router, client)
        stale = await select(router, client)
        router.observe(failed, 1, {"error": {"code": 429}}, False)
        router.release(failed)
        clock[0] += 30
        recovered = await select(router, client)
        router.observe(recovered, 2, completed("a", cached=0), True)
        router.release(recovered)
        # This request began before the failure that invalidated its shared affinity.
        router.observe(stale, 0.01, completed("a", cached=80), stale_success)
        router.release(stale)
        next_batch = await select(router, client)
        assert next_batch.endpoint == "a"
        assert next_batch.diagnostics["estimated_seconds"]["a"] == 2
        assert stale.diagnostics["reported_cost"] == 0.00002
        router.release(next_batch)
