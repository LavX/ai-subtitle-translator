"""A generic OpenAI-compatible back end: header keys, rotation, relaxation, typed transport errors."""

import json

import httpx
import pytest

from subtitle_translator.backends.openai_compat import OpenAICompatProvider
from subtitle_translator.core.errors import (
    AuthenticationError,
    DependencyUnavailableError,
    ProviderTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
)
from subtitle_translator.core.models import TranslationConfig
from subtitle_translator.core.provider import TranslationBatch
from subtitle_translator.core.settings import CoreSettings

BATCH = TranslationBatch(lines=[{"index": "0", "content": "Hello"}], source_language="en", target_language="es")
OK = {
    "choices": [{"message": {"content": json.dumps({"translations": [{"index": "0", "content": "Hola"}]})}}],
    "usage": {"total_tokens": 5},
}


def provider(handler, keys=(), **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://llm.local/v1/")
    return OpenAICompatProvider(
        "http://llm.local/v1", keys, settings=CoreSettings(request_timeout=2.0), client=client, default_model="local-model", **kw
    )


@pytest.mark.asyncio
async def test_no_keys_means_no_authorization_header():
    seen = {}

    def handler(request):
        seen.update(auth=request.headers.get("authorization"), path=request.url.path, body=json.loads(request.content))
        return httpx.Response(200, json=OK)

    result = await provider(handler).translate_batch(BATCH)
    assert seen["auth"] is None and seen["path"] == "/v1/chat/completions"
    assert seen["body"]["model"] == "local-model" and seen["body"]["response_format"] == {"type": "json_object"}
    assert result.translations == [{"index": "0", "content": "Hola"}]


@pytest.mark.asyncio
async def test_the_key_travels_in_a_header_and_never_in_the_url():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=OK)

    await provider(handler, keys=["sk-secret"]).translate_batch(BATCH)
    assert seen[0].headers["authorization"] == "Bearer sk-secret"
    assert "sk-secret" not in str(seen[0].url)


@pytest.mark.asyncio
async def test_config_override_key_wins_and_is_not_rotated():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow"}}, headers={"retry-after": "3"})

    p = provider(handler, keys=["ring"])
    with pytest.raises(RateLimitError):
        await p.translate_batch(BATCH, config_override=TranslationConfig(api_key="override"))
    assert p.keys.acquire("local-model") == "ring"  # the ring was not touched


@pytest.mark.asyncio
async def test_a_rate_limit_rotates_to_the_next_key():
    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        if request.headers["authorization"] == "Bearer k1":
            return httpx.Response(429, json={"error": {"message": "slow"}}, headers={"retry-after": "30"})
        return httpx.Response(200, json=OK)

    p = provider(handler, keys=["k1", "k2"])
    assert (await p.translate_batch(BATCH)).translations[0]["content"] == "Hola"
    await p.translate_batch(BATCH)
    assert seen == ["Bearer k1", "Bearer k2", "Bearer k2"]


@pytest.mark.asyncio
async def test_every_key_limited_raises_with_the_soonest_wait():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow"}}, headers={"retry-after": "12"})

    with pytest.raises(RateLimitError) as caught:
        await provider(handler, keys=["k1", "k2"]).translate_batch(BATCH)
    assert caught.value.retry_after == pytest.approx(12, abs=1)


@pytest.mark.asyncio
async def test_rejected_keys_are_set_aside_and_all_rejected_is_auth_required():
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "bad"}})

    p = provider(handler, keys=["k1", "k2"])
    with pytest.raises(AuthenticationError):
        await p.translate_batch(BATCH)
    assert p.keys.all_invalid()


@pytest.mark.asyncio
async def test_exhausted_quota_on_every_key_is_quota_exhausted():
    class Exhausting(OpenAICompatProvider):
        def classify(self, response, model):
            return QuotaExhaustedError("daily", retry_after=500) if response.status_code == 429 else None

    def handler(request):
        return httpx.Response(429, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://llm.local/v1/")
    p = Exhausting("http://llm.local/v1", ["k1", "k2"], client=client, default_model="m", settings=CoreSettings())
    with pytest.raises(QuotaExhaustedError) as caught:
        await p.translate_batch(BATCH)
    assert caught.value.retry_after == pytest.approx(500, abs=2)


@pytest.mark.asyncio
async def test_a_rejected_response_format_is_dropped_once():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return httpx.Response(200, json=OK)

    await provider(handler).translate_batch(BATCH)
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]


@pytest.mark.asyncio
async def test_transport_error_message_carries_no_key():
    def slow(request):
        raise httpx.ReadTimeout("late", request=request)

    def broken(request):
        raise httpx.ConnectError("connect failed for Bearer sk-secret", request=request)

    with pytest.raises(ProviderTimeoutError):
        await provider(slow, keys=["sk-secret"]).translate_batch(BATCH)
    with pytest.raises(DependencyUnavailableError) as caught:
        await provider(broken, keys=["sk-secret"]).translate_batch(BATCH)
    assert "sk-secret" not in caught.value.message and "sk-secret" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,expected,unknown",
    [
        (httpx.ConnectError, DependencyUnavailableError, False),
        (httpx.ReadError, DependencyUnavailableError, True),
        (httpx.RemoteProtocolError, DependencyUnavailableError, True),
        (httpx.ReadTimeout, ProviderTimeoutError, True),
    ],
)
async def test_only_a_failed_connect_proves_the_request_was_not_billed(failure, expected, unknown):
    def handler(request):
        raise failure(failure.__name__, request=request)

    with pytest.raises(expected) as caught:
        await provider(handler).translate_batch(BATCH)
    assert caught.value.billing_unknown is unknown


@pytest.mark.asyncio
async def test_health_and_models():
    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2", "name": "Two"}]})

    p = provider(handler)
    assert await p.health_check() is True
    assert await p.get_available_models() == [{"id": "m1", "name": "m1"}, {"id": "m2", "name": "Two"}]


@pytest.mark.asyncio
async def test_the_base_payload_is_the_one_gemini_builds_on():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=OK)

    await provider(handler).translate_batch(BATCH)
    assert set(bodies[0]) == {"model", "messages", "temperature", "response_format"}


@pytest.mark.asyncio
async def test_a_successful_override_key_translates_and_leaves_the_ring_alone(caplog):
    def handler(request):
        return httpx.Response(200, json=OK)

    p = provider(handler, keys=["ring-key"])
    with caplog.at_level("DEBUG"):
        result = await p.translate_batch(BATCH, config_override=TranslationConfig(api_key="sk-override-secret"))
    assert result.translations == [{"index": "0", "content": "Hola"}]
    assert "sk-override-secret" not in caplog.text
    assert "sk-override-secret" not in repr(result)
    assert p.keys.acquire("local-model") == "ring-key"


@pytest.mark.asyncio
async def test_a_rejected_key_gives_its_hold_back_before_the_next_key():
    from subtitle_translator.core.batch_processor import BatchProcessor
    from subtitle_translator.core.usage import UsageMeter

    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        if request.headers["authorization"] == "Bearer k1":
            return httpx.Response(401, json={"error": {"message": "bad"}})
        return httpx.Response(200, json=OK)

    p = provider(handler, keys=["k1", "k2"], price_input=1.0, price_output=2.0)
    p._output_limit = lambda model, payload: 1000
    one_call = p._worst_case("local-model", p._payload(BATCH, "local-model", None, None), None)
    meter = UsageMeter(budget=one_call * 1.5)  # room for one request in flight, not two
    result = await BatchProcessor(p, CoreSettings(), meter=meter).process_all_batches(
        BATCH.lines, "en", "es", batch_size=10, model="local-model"
    )
    assert seen == ["Bearer k1", "Bearer k2"] and result.success
    assert meter.reserved == pytest.approx(0.0)


def _priced(handler, **kw):
    p = provider(handler, price_input=1.0, price_output=2.0, **kw)
    p._output_limit = lambda model, payload: 1000
    return p


@pytest.mark.asyncio
async def test_a_paid_reply_that_is_not_a_translation_is_charged_for_its_tokens():
    from subtitle_translator.core.batch_processor import BatchProcessor
    from subtitle_translator.core.usage import UsageMeter

    garbled = {"choices": [{"message": {"content": "no json here"}}], "usage": {"total_tokens": 5000}}
    p = _priced(lambda request: httpx.Response(200, json=garbled))
    meter = UsageMeter(budget=1.0)
    result = await BatchProcessor(p, CoreSettings(max_retries=0), meter=meter).process_all_batches(
        BATCH.lines, "en", "es", batch_size=10, model="local-model"
    )
    assert not result.success
    assert meter.cost == pytest.approx(5000 * 2.0 / 1e6)  # the higher of the two prices
    assert meter.reserved == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_a_paid_reply_with_no_usage_and_no_content_keeps_its_hold():
    from subtitle_translator.core.batch_processor import BatchProcessor
    from subtitle_translator.core.usage import UsageMeter

    p = _priced(lambda request: httpx.Response(200, json=["not", "an", "object"]))
    meter = UsageMeter(budget=1.0)
    result = await BatchProcessor(p, CoreSettings(max_retries=0), meter=meter).process_all_batches(
        BATCH.lines, "en", "es", batch_size=10, model="local-model"
    )
    assert not result.success
    assert meter.reserved > 0  # it may have been billed, so the hold stays
