"""The OpenAI-compatible back end: presets, their request details, failures, and live models."""

import json

import httpx
import pytest

from subtitle_translator.backends.openai_compatible import PRESETS, OpenAICompatibleBackendType
from subtitle_translator.core.backends import BackendContext, Connection
from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.errors import (
    AuthenticationError,
    DependencyUnavailableError,
    ErrorKind,
    InvalidConfigError,
    ProviderTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
    error_kind,
)
from subtitle_translator.core.provider import TranslationBatch
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.usage import UsageMeter

BATCH = TranslationBatch(lines=[{"index": "0", "content": "Hello"}], source_language="en", target_language="hu")
OK = {
    "choices": [{"message": {"content": json.dumps({"translations": [{"index": "0", "content": "Szia"}]})}}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
}


def opened(handler, *, secrets=None, **settings):
    backend = OpenAICompatibleBackendType(transport=httpx.MockTransport(handler))
    settings.setdefault("model", "qwen3-8b")
    return backend.open(
        Connection("oc", "openai_compatible", settings=settings, secrets=secrets or {}),
        CoreSettings(request_timeout=5.0),
        BackendContext(),
    )


def recorder(response=None):
    seen = []

    def handler(request):
        seen.append(request)
        return response(request) if callable(response) else httpx.Response(200, json=OK)

    return seen, handler


def test_each_preset_fills_its_address_and_the_address_stays_editable():
    backend = OpenAICompatibleBackendType()
    preset = next(f for f in backend.fields() if f.key == "preset")
    fills = {c.value: c.fills["base_url"] for c in preset.choices}
    assert fills == {
        "openai": "https://api.openai.com/v1",
        "lmstudio": "http://localhost:1234/v1",
        "ollama": "http://localhost:11434/v1",
        "custom": "",
    }
    filled = backend.validate({"preset": "lmstudio", "model": "m"}, {})
    assert filled.ok and filled.patch["base_url"] == "http://localhost:1234/v1"
    moved = backend.validate({"preset": "lmstudio", "base_url": "http://192.168.1.20:1234/v1/", "model": "m"}, {})
    assert moved.patch["base_url"] == "http://192.168.1.20:1234/v1" and moved.patch["preset"] == "lmstudio"
    address = next(f for f in backend.fields() if f.key == "base_url")
    assert address.destination and not next(f for f in backend.fields() if f.key == "api_key").required


@pytest.mark.parametrize(
    "settings,field",
    [
        ({"preset": "custom", "model": "m"}, "base_url"),
        ({"base_url": "ftp://host/v1", "model": "m"}, "base_url"),
        ({"base_url": "http://user:secret@host/v1", "model": "m"}, "base_url"),
        ({"base_url": "http://host:port/v1", "model": "m"}, "base_url"),
        ({"base_url": "http://host/v1"}, "model"),
        ({"preset": "vllm", "base_url": "http://host/v1", "model": "m"}, "preset"),
        ({"base_url": "http://host/v1", "model": "m", "parallel_batches": 0}, "parallel_batches"),
        ({"base_url": "http://host/v1", "model": "m", "price_input": -1}, "price_input"),
    ],
)
def test_validate_names_the_field_at_fault(settings, field):
    assert field in OpenAICompatibleBackendType().validate(settings, {}).errors


def test_open_refuses_an_incomplete_connection():
    with pytest.raises(InvalidConfigError) as caught:
        OpenAICompatibleBackendType().open(Connection("oc", "openai_compatible"), CoreSettings(), BackendContext())
    assert set(caught.value.field_errors) == {"base_url", "model"}


def test_the_openai_preset_needs_a_price():
    _seen, handler = recorder()
    with pytest.raises(InvalidConfigError) as caught:
        opened(handler, preset="openai", model="gpt-5.6-mini", secrets={"api_key": "k"})
    assert "price_input" in caught.value.field_errors


@pytest.mark.asyncio
async def test_a_keyless_lm_studio_request():
    seen, handler = recorder()
    backend = opened(handler, preset="lmstudio")
    result = await backend.provider.translate_batch(BATCH, config_override=backend.config)
    request = seen[0]
    body = json.loads(request.content)
    assert "authorization" not in request.headers and str(request.url) == "http://localhost:1234/v1/chat/completions"
    assert body["model"] == "qwen3-8b" and body["temperature"] == 0.3 and body["max_tokens"] == 8000
    assert body["response_format"]["type"] == "json_schema" and "max_completion_tokens" not in body
    assert result.translations == [{"index": "0", "content": "Szia"}] and result.total_tokens == 24
    assert backend.batch_size == 100 and backend.config.parallel_batches == 1


@pytest.mark.asyncio
async def test_the_openai_preset_sends_a_bearer_key_and_max_completion_tokens():
    seen, handler = recorder()
    backend = opened(handler, preset="openai", model="gpt-5.6-mini", secrets={"api_key": "sk-live-secret"},
                     price_input=1.0, price_output=2.0)
    result = await backend.provider.translate_batch(BATCH, config_override=backend.config)
    body = json.loads(seen[0].content)
    assert seen[0].headers["authorization"] == "Bearer sk-live-secret" and "sk-live" not in str(seen[0].url)
    assert body["max_completion_tokens"] == 8000 and "max_tokens" not in body
    assert body["response_format"] == {"type": "json_object"}
    assert result.cost == pytest.approx(0.000028)


@pytest.mark.asyncio
async def test_ollama_uses_json_mode_and_max_tokens():
    seen, handler = recorder()
    backend = opened(handler, preset="ollama")
    await backend.provider.translate_batch(BATCH, config_override=backend.config)
    body = json.loads(seen[0].content)
    assert body["response_format"] == {"type": "json_object"} and body["max_tokens"] == 8000


@pytest.mark.asyncio
async def test_a_refused_temperature_is_dropped_and_remembered_for_that_model():
    def answer(request):
        if "temperature" in json.loads(request.content):
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0.3 with this model.", "param": "temperature"}})
        return httpx.Response(200, json=OK)

    seen, handler = recorder(answer)
    backend = opened(handler, preset="openai", model="o-reasoner", secrets={"api_key": "k"},
                     price_input=1.0, price_output=2.0)
    await backend.provider.translate_batch(BATCH, config_override=backend.config)
    await backend.provider.translate_batch(BATCH, config_override=backend.config)
    sent = ["temperature" in json.loads(r.content) for r in seen]
    assert sent == [True, False, False]  # one refusal, then never sent again


@pytest.mark.asyncio
async def test_no_preset_detail_reaches_a_plain_server():
    seen, handler = recorder()
    backend = opened(handler, base_url="http://10.0.0.5:8000/v1", max_tokens=0)
    await backend.provider.translate_batch(BATCH, config_override=backend.config)
    body = json.loads(seen[0].content)
    assert set(body) == {"model", "messages", "temperature", "response_format"}


@pytest.mark.asyncio
async def test_a_stopped_server_is_dependency_unavailable_and_names_no_key():
    def refused(request):
        raise httpx.ConnectError("connection refused for Bearer sk-live-secret", request=request)

    def silent(request):
        raise httpx.ReadTimeout("late", request=request)

    with pytest.raises(DependencyUnavailableError) as caught:
        backend = opened(refused, preset="lmstudio", secrets={"api_key": "sk-live-secret"})
        await backend.provider.translate_batch(BATCH, config_override=backend.config)
    assert error_kind(caught.value) is ErrorKind.DEPENDENCY_UNAVAILABLE
    assert caught.value.message == "LM Studio could not be reached (ConnectError)"
    with pytest.raises(ProviderTimeoutError) as late:
        backend = opened(silent, preset="ollama")
        await backend.provider.translate_batch(BATCH, config_override=backend.config)
    assert error_kind(late.value) is ErrorKind.DEPENDENCY_UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,body,headers,expected",
    [
        (401, {"error": {"message": "bad key"}}, {}, AuthenticationError),
        (403, {"error": {"message": "forbidden"}}, {}, AuthenticationError),
        (429, {"error": {"message": "slow down"}}, {"retry-after": "7"}, RateLimitError),
        (429, {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": "insufficient_quota"}}, {}, QuotaExhaustedError),
    ],
)
async def test_http_failures_land_on_the_shared_kinds(status, body, headers, expected):
    def answer(request):
        return httpx.Response(status, json=body, headers=headers)

    for secrets in ({}, {"api_key": "k"}):
        backend = opened(answer, preset="openai", secrets=secrets, price_input=1.0, price_output=2.0)
        with pytest.raises(expected):
            await backend.provider.translate_batch(BATCH, config_override=backend.config)


@pytest.mark.asyncio
async def test_options_list_the_loaded_models_or_nothing_for_free_text():
    def listing(request):
        assert request.url.path == "/v1/models" and "authorization" not in request.headers
        return httpx.Response(200, json={"object": "list", "data": [{"id": "qwen3-8b"}, {"id": "gemma-4-12b"}]})

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    def locked(request):
        return httpx.Response(401, json={"error": {"message": "needs a key"}})

    settings = {"preset": "lmstudio"}
    listed = await OpenAICompatibleBackendType(transport=httpx.MockTransport(listing)).options("model", settings, {}, BackendContext())
    assert [c.value for c in listed] == ["qwen3-8b", "gemma-4-12b"]
    for handler in (refused, locked):
        assert await OpenAICompatibleBackendType(transport=httpx.MockTransport(handler)).options(
            "model", settings, {}, BackendContext()
        ) == []
    assert await OpenAICompatibleBackendType().options("model", {"preset": "custom"}, {}, BackendContext()) == []


def test_the_presets_are_the_four_the_design_names():
    assert list(PRESETS) == ["openai", "lmstudio", "ollama", "custom"]


@pytest.mark.asyncio
async def test_a_total_only_usage_uses_the_higher_price():
    body = {
        "choices": [{"message": {"content": json.dumps({"translations": [{"index": "0", "content": "Szia"}]})}}],
        "usage": {"total_tokens": 10},
    }
    seen, handler = recorder(lambda request: httpx.Response(200, json=body))
    backend = opened(handler, preset="openai", model="gpt-5.6-mini", secrets={"api_key": "k"},
                     price_input=1.0, price_output=2.0)
    result = await backend.provider.translate_batch(BATCH, config_override=backend.config)
    assert result.cost == pytest.approx(0.00002)


@pytest.mark.asyncio
async def test_a_priced_request_is_held_before_it_is_sent():
    meter = UsageMeter(budget=1.0)
    held_at_send = []

    def handler(request):
        held_at_send.append(meter.reserved)
        return httpx.Response(200, json=OK)

    backend = opened(handler, preset="openai", model="gpt-5.6-mini", secrets={"api_key": "k"},
                     price_input=1.0, price_output=2.0)
    config = backend.config.for_operation()
    config._usage_meter = meter
    result = await backend.provider.translate_batch(BATCH, config_override=config)
    assert held_at_send and held_at_send[0] > 0
    assert meter.reserved == pytest.approx(held_at_send[0])  # nothing settles outside a call scope
    assert result.cost == pytest.approx(0.000028)


@pytest.mark.asyncio
async def test_a_priced_call_settles_at_the_computed_cost():
    meter = UsageMeter(budget=1.0)
    _seen, handler = recorder()
    backend = opened(handler, preset="openai", model="gpt-5.6-mini", secrets={"api_key": "k"},
                     price_input=1.0, price_output=2.0)
    processor = BatchProcessor(backend.provider, backend.settings, meter=meter)
    result = await processor.process_all_batches(
        BATCH.lines, "en", "hu", batch_size=10, model="gpt-5.6-mini"
    )
    assert result.success
    assert meter.cost == pytest.approx(0.000028) and meter.reserved == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_no_prices_hold_nothing():
    _seen, handler = recorder()
    backend = opened(handler, preset="lmstudio")
    assert backend.provider.prices_requests is False


@pytest.mark.asyncio
async def test_an_unknown_output_bound_refuses_only_with_a_budget():
    _seen, handler = recorder()
    backend = opened(handler, base_url="http://host/v1", max_tokens=0, price_output=1.0)
    config = backend.config.for_operation()
    config._usage_meter = UsageMeter(budget=1.0)
    with pytest.raises(InvalidConfigError) as caught:
        await backend.provider.translate_batch(BATCH, config_override=config)
    assert "max_tokens" in caught.value.field_errors
    sent = await backend.provider.translate_batch(BATCH, config_override=backend.config)
    assert sent.translations == [{"index": "0", "content": "Szia"}]
