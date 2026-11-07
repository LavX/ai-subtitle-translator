"""Gemini: header keys, RetryInfo from the body, midnight Pacific, per-project limits, thinking rules."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from subtitle_translator.backends.gemini import (
    GEMINI_DEFAULT_BATCH_SIZE,
    GEMINI_DEFAULT_MODEL,
    GeminiBackendType,
    GeminiProvider,
    next_pacific_midnight,
    parse_google_error,
    reasoning_effort,
    resolve_model,
)
from subtitle_translator.core.backends import BackendContext, Connection
from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.errors import (
    AuthenticationError,
    ErrorKind,
    InvalidConfigError,
    QuotaExhaustedError,
    RateLimitError,
    error_kind,
)
from subtitle_translator.core.models import TranslationConfig
from subtitle_translator.core.provider import TranslationBatch
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.usage import UsageMeter

PT = ZoneInfo("America/Los_Angeles")
BATCH = TranslationBatch(lines=[{"index": "0", "content": "Hello"}], source_language="en", target_language="hu")
OK = {"choices": [{"message": {"content": json.dumps({"translations": [{"index": "0", "content": "Szia"}]})}}], "usage": {"total_tokens": 9}}


def quota_body(quota_id, delay="34s", wrap=False):
    body = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id, "quotaValue": "20"}]},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
            ],
        }
    }
    return [body] if wrap else body


class Clock:
    def __init__(self, when):
        self.now = when.timestamp()

    def __call__(self):
        return self.now


def gemini(handler, keys=("k1",), projects=(), clock=None, model="gemini-2.5-flash", reasoning="disabled"):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://g.test/v1beta/openai/")
    return GeminiProvider(
        list(keys), key_projects=list(projects), model=model, reasoning=reasoning,
        settings=CoreSettings(), client=client, clock=clock or Clock(datetime(2026, 9, 24, 10, 0, tzinfo=PT)),
    )


def test_parse_google_error_accepts_object_list_and_garbage():
    for wrap in (False, True):
        parsed = parse_google_error(quota_body("GenerateRequestsPerMinutePerProjectPerModel", wrap=wrap))
        assert parsed.retry_delay == 34.0 and parsed.status == "RESOURCE_EXHAUSTED"
        assert parsed.quota_ids == ("GenerateRequestsPerMinutePerProjectPerModel",)
    assert parse_google_error({"error": {"message": "x"}}).retry_delay is None
    assert parse_google_error("nonsense") is None and parse_google_error([]) is None


def test_next_pacific_midnight_across_dst():
    before = datetime(2026, 11, 1, 0, 30, tzinfo=PT).timestamp()  # the night DST ends
    assert datetime.fromtimestamp(next_pacific_midnight(before), PT) == datetime(2026, 11, 2, 0, 0, tzinfo=PT)
    late = datetime(2026, 9, 24, 23, 59, tzinfo=PT).timestamp()
    assert next_pacific_midnight(late) - late == pytest.approx(60)


@pytest.mark.asyncio
async def test_gemini_key_is_sent_in_a_header_never_the_url():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=OK)

    await gemini(handler, keys=["AIza-secret"]).translate_batch(BATCH)
    assert seen[0].headers["authorization"] == "Bearer AIza-secret"
    assert "AIza-secret" not in str(seen[0].url) and "key=" not in str(seen[0].url)


@pytest.mark.asyncio
async def test_retry_time_comes_from_retry_info_not_a_header():
    def handler(request):
        return httpx.Response(429, json=quota_body("GenerateRequestsPerMinutePerProjectPerModel", "17s"), headers={"retry-after": "999"})

    with pytest.raises(RateLimitError) as caught:
        await gemini(handler).translate_batch(BATCH)
    assert caught.value.retry_after == pytest.approx(17, abs=1)


@pytest.mark.asyncio
async def test_per_day_quota_waits_for_pacific_midnight_not_retry_delay():
    clock = Clock(datetime(2026, 9, 24, 22, 0, tzinfo=PT))

    def handler(request):
        return httpx.Response(429, json=quota_body("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "34s"))

    with pytest.raises(QuotaExhaustedError) as caught:
        await gemini(handler, clock=clock).translate_batch(BATCH)
    assert caught.value.retry_after == pytest.approx(2 * 3600, abs=5)
    assert error_kind(caught.value) is ErrorKind.QUOTA_EXHAUSTED


@pytest.mark.asyncio
async def test_keys_from_one_project_share_the_daily_limit():
    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        if request.headers["authorization"] in ("Bearer a", "Bearer b"):
            return httpx.Response(429, json=quota_body("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
        return httpx.Response(200, json=OK)

    p = gemini(handler, keys=["a", "b", "c"], projects=["shared", "shared", "other"])
    await p.translate_batch(BATCH)
    assert seen == ["Bearer a", "Bearer c"]  # b was never tried


@pytest.mark.asyncio
async def test_a_bad_key_moves_on_and_a_lone_bad_key_is_auth_required():
    def handler(request):
        if request.headers["authorization"] == "Bearer bad":
            return httpx.Response(400, json={"error": {"message": "API key not valid.", "status": "INVALID_ARGUMENT", "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]}})
        return httpx.Response(200, json=OK)

    assert (await gemini(handler, keys=["bad", "good"]).translate_batch(BATCH)).translations
    with pytest.raises(AuthenticationError):
        await gemini(handler, keys=["bad"]).translate_batch(BATCH)


def test_thinking_rules():
    assert reasoning_effort("gemini-2.5-flash", "disabled") == "none"
    assert reasoning_effort("gemini-3.6-flash", "disabled") == "low"  # 3.x cannot turn thinking off
    assert reasoning_effort("gemini-2.5-pro", "disabled") == "low"
    assert reasoning_effort("models/gemini-3.7-flash", "high") == "high"


def test_retired_models_are_mapped():
    assert resolve_model("gemini-2.0-flash") == GEMINI_DEFAULT_MODEL
    assert resolve_model("models/gemini-3-pro-preview") == "gemini-3.1-pro-preview"
    assert resolve_model("gemini-2.5-flash") == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_a_rejected_reasoning_effort_is_dropped_once():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "thinking level not supported for this model"}})
        return httpx.Response(200, json=OK)

    await gemini(handler, model="gemini-3.8-flash").translate_batch(BATCH)
    assert bodies[0]["reasoning_effort"] == "low" and "reasoning_effort" not in bodies[1]


def test_validate_normalizes_and_reports():
    backend = GeminiBackendType()
    good = backend.validate({"model": "gemini-2.0-flash"}, {"api_keys": ["k"]})
    assert good.ok and good.patch["model"] == GEMINI_DEFAULT_MODEL and good.patch["batch_size"] == GEMINI_DEFAULT_BATCH_SIZE
    bad = backend.validate({"batch_size": 5000, "reasoning": "max", "key_projects": ["a", "b"]}, {"api_keys": ["k"]})
    assert set(bad.errors) == {"batch_size", "reasoning", "key_projects"}
    assert backend.validate({}, {}).errors == {"api_keys": "API keys is required"}


def test_project_names_are_counted_only_against_keys_that_were_given():
    backend = GeminiBackendType()
    unseen = backend.validate({"key_projects": ["a", "b"]}, {})
    assert unseen.errors == {"api_keys": "API keys is required"} and unseen.patch["key_projects"] == ["a", "b"]
    assert backend.validate({"key_projects": ["a", "b"]}, {"api_keys": ["k1", "k2"]}).ok


@pytest.mark.asyncio
async def test_a_test_call_refuses_more_project_names_than_keys_before_sending():
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json=OK)

    connection = Connection("g", "gemini", settings={"key_projects": ["a", "b"]}, secrets={"api_keys": ["k"]})
    with pytest.raises(InvalidConfigError) as caught:
        await GeminiBackendType(transport=httpx.MockTransport(handler)).test(connection, CoreSettings(), BackendContext())
    assert caught.value.field_errors == {"key_projects": "There are more project names than keys"} and sent == []


def test_open_gives_one_batch_at_a_time_and_the_configured_batch_size():
    opened = GeminiBackendType().open(
        Connection("g", "gemini", settings={"model": "gemini-2.5-flash"}, secrets={"api_keys": ["k"]}),
        CoreSettings(openrouter_max_tokens=8000),
        BackendContext(),
    )
    assert opened.config.parallel_batches == 1 and opened.batch_size == GEMINI_DEFAULT_BATCH_SIZE
    assert opened.model == "gemini-2.5-flash" and opened.config.model == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_options_lists_generative_models_only():
    def handler(request):
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"data": [
            {"id": "models/gemini-2.5-flash"}, {"id": "models/gemini-embedding-001"},
            {"id": "models/gemini-3.8-flash-tts"}, {"id": "models/gemini-3.6-flash"},
        ]})

    choices = await GeminiBackendType(transport=httpx.MockTransport(handler)).options(
        "model", {}, {"api_keys": ["k"]}, BackendContext()
    )
    assert [c.value for c in choices] == ["gemini-2.5-flash", "gemini-3.6-flash"]


@pytest.mark.asyncio
async def test_the_gemini_payload_carries_only_its_own_fields():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=OK)

    await gemini(handler).translate_batch(BATCH)
    assert set(bodies[0]) == {"model", "messages", "temperature", "response_format", "reasoning_effort"}
    assert bodies[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_a_priced_gemini_request_is_held_before_it_is_sent():
    meter = UsageMeter(budget=1.0)
    held = []

    def handler(request):
        held.append(meter.reserved)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"translations": [{"index": "0", "content": "Szia"}]})}}], "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://g.test/v1beta/openai/")
    provider = GeminiProvider(
        ["k"], model="gemini-2.5-flash", price_input=1.0, price_output=2.0,
        settings=CoreSettings(), client=client, clock=Clock(datetime(2026, 9, 24, 10, 0, tzinfo=PT)),
    )
    config = TranslationConfig().for_operation()
    config._usage_meter = meter
    result = await provider.translate_batch(BATCH, config_override=config)
    assert held and held[0] > 0 and result.cost == pytest.approx(0.000028)


@pytest.mark.asyncio
async def test_a_priced_gemini_call_settles_at_the_computed_cost():
    meter = UsageMeter(budget=1.0)
    body = {"choices": [{"message": {"content": json.dumps({"translations": [{"index": "0", "content": "Szia"}]})}}], "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}}

    def handler(request):
        return httpx.Response(200, json=body)

    opened = GeminiBackendType(transport=httpx.MockTransport(handler)).open(
        Connection("g", "gemini", settings={"model": "gemini-2.5-flash", "price_input": 1.0, "price_output": 2.0}, secrets={"api_keys": ["k"]}),
        CoreSettings(request_timeout=5.0),
        BackendContext(),
    )
    processor = BatchProcessor(opened.provider, opened.settings, meter=meter)
    result = await processor.process_all_batches(BATCH.lines, "en", "hu", batch_size=10, model="gemini-2.5-flash")
    assert result.success
    assert meter.cost == pytest.approx(0.000028) and meter.reserved == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_a_priced_gemini_model_with_no_known_ceiling_refuses_only_with_a_budget():
    def handler(request):
        return httpx.Response(200, json=OK)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://g.test/v1beta/openai/")
    provider = GeminiProvider(
        ["k"], model="gemini-9-unknown", price_output=1.0, settings=CoreSettings(), client=client,
        clock=Clock(datetime(2026, 9, 24, 10, 0, tzinfo=PT)),
    )
    config = TranslationConfig().for_operation()
    config._usage_meter = UsageMeter(budget=1.0)
    with pytest.raises(InvalidConfigError) as caught:
        await provider.translate_batch(BATCH, config_override=config)
    assert "model" in caught.value.field_errors or "max_tokens" in caught.value.field_errors
    assert (await provider.translate_batch(BATCH)).translations == [{"index": "0", "content": "Szia"}]


def test_a_connection_with_its_own_request_timeout_applies_it_to_the_call_and_the_batch():
    def handler(request):
        return httpx.Response(200, json={})

    connection = Connection(
        "g", "gemini", settings={"model": "gemini-2.5-flash", "request_timeout": 900.0}, secrets={"api_keys": ["k"]}
    )
    opened = GeminiBackendType(transport=httpx.MockTransport(handler)).open(
        connection, CoreSettings(request_timeout=120.0), BackendContext()
    )
    assert opened.config.request_timeout == 900.0 and opened.settings.request_timeout == 900.0
    assert opened.provider.client.timeout.read == 900.0
    plain = GeminiBackendType(transport=httpx.MockTransport(handler)).open(
        Connection("g", "gemini", settings={"model": "gemini-2.5-flash"}, secrets={"api_keys": ["k"]}),
        CoreSettings(request_timeout=120.0),
        BackendContext(),
    )
    assert plain.config.request_timeout is None and plain.provider.client.timeout.read == 120.0
