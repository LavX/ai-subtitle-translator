"""Official SDK dispatch with real request building and fixture HTTP responses."""

import asyncio
import json
import logging

import httpx
import pytest
from openrouter.chat import Chat

from subtitle_translator.api.models import TranslationConfig
from subtitle_translator.providers.base import (
    InvalidResponseError,
    ProviderTimeoutError,
    RateLimitError,
    TranslationProviderError,
)
from tests.test_smartfast_integration import batch, endpoint
from tests.test_smartfast_integration import environment as environment


@pytest.fixture
def sdk_calls(monkeypatch):
    calls = []
    original = Chat.send_async

    async def record(self, **kwargs):
        calls.append(kwargs)
        return await original(self, **kwargs)

    monkeypatch.setattr(Chat, "send_async", record)
    return calls


async def test_completion_uses_official_async_sdk_and_preserves_free_wire_contract(
    environment, sdk_calls, monkeypatch, caplog
):
    wire, provider, _ = environment
    wire.endpoints = [endpoint("a", 0)]
    wire.endpoints[0]["supported_parameters"].remove("response_format")
    monkeypatch.setenv("OPENROUTER_DEBUG", "true")
    caplog.set_level(logging.DEBUG)
    config = TranslationConfig(
        apiKey="synthetic-per-request-key",
        requestTimeout=600,
        reasoning={"effort": "none"},
    )
    config._smartfast_session_id = "stable-job-session"
    pool = provider.client
    result = await provider.translate_batch(
        batch(), model="fixture/model:free:smartfast", config_override=config
    )
    assert len(sdk_calls) == 1
    assert sdk_calls[0]["retries"] is None
    assert sdk_calls[0]["timeout_ms"] == 600000
    body = wire.completions[0]
    assert body["model"] == "fixture/model:free"
    assert "response_format" not in body
    assert "usage" not in body
    assert body["reasoning"] == {"effort": "none"}
    assert body["session_id"] == "stable-job-session"
    assert body["provider"]["max_price"] == {"prompt": "0", "completion": "0", "request": "0"}
    request = wire.requests[-1]
    assert request.headers["Authorization"] == "Bearer synthetic-per-request-key"
    assert request.headers["HTTP-Referer"] == provider.settings.openrouter_headers["HTTP-Referer"]
    assert request.extensions["timeout"] == dict.fromkeys(["connect", "read", "write", "pool"], 600)
    assert result.raw_response["provider"] == "A"
    assert result.total_tokens == 1050
    assert result.cost == 0.001
    assert not pool.is_closed
    assert provider.client is pool
    assert pool.headers["Authorization"] == "Bearer synthetic-default-key"
    assert "synthetic-per-request-key" not in caplog.text
    assert "synthetic-default-key" not in caplog.text


@pytest.mark.parametrize(
    "mode,expected,retry_after",
    [
        ("http429", RateLimitError, 0),
        ("http503", TranslationProviderError, 0),
        ("error", RateLimitError, None),
        ("no_choices", InvalidResponseError, None),
        ("bad_choices", InvalidResponseError, None),
        ("bad_message", InvalidResponseError, None),
        ("bad_content", InvalidResponseError, None),
    ],
)
async def test_sdk_keeps_raw_error_usage_and_existing_classification_without_hidden_retries(
    environment, sdk_calls, mode, expected, retry_after
):
    wire, provider, _ = environment
    wire.reply_modes = [mode]
    with pytest.raises(expected) as raised:
        await provider.translate_batch(batch(), model="fixture/model:smartfast")
    assert len(sdk_calls) == len(wire.completions) == 1
    assert raised.value.tokens_used == 1050
    assert raised.value.cost == 0.001
    assert raised.value.retry_after == retry_after
    assert raised.value.routing_diagnostics["reported_cost"] == 0.001


async def test_sdk_keeps_malformed_top_level_response_for_existing_parser(environment, sdk_calls):
    wire, provider, _ = environment

    async def malformed(request):
        response = await wire(request)
        if request.method == "POST":
            return httpx.Response(200, text="<html>invalid provider response</html>")
        return response

    provider._client = httpx.AsyncClient(
        base_url=provider.settings.openrouter_api_base,
        transport=httpx.MockTransport(malformed),
    )
    with pytest.raises(InvalidResponseError, match="Invalid JSON response"):
        await provider.translate_batch(batch(), model="fixture/model:smartfast")
    assert len(sdk_calls) == len(wire.completions) == 1


async def test_sdk_preserves_raw_usage_extensions_and_repaired_translation_output(
    environment, sdk_calls
):
    wire, provider, _ = environment
    wire.reply_modes = ["trailing_comma"]
    result = await provider.translate_batch(batch(), model="fixture/model:smartfast")
    assert len(sdk_calls) == 1
    assert result.note is not None
    assert len(result.translations) == 2
    assert result.raw_response["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}
    assert result.raw_response["usage"]["cost"] == 0.001
    assert result.routing_diagnostics["success"] is False
    assert result.routing_diagnostics["healthy"] is True


async def test_sdk_deadline_cancels_active_request_without_closing_shared_pool(
    environment, sdk_calls
):
    wire, provider, settings = environment
    wire.completion_gate = asyncio.Event()
    settings.request_timeout = 0.2
    pool = provider.client
    with pytest.raises(ProviderTimeoutError):
        await provider.translate_batch(
            batch(),
            model="fixture/model:smartfast",
        )
    assert len(sdk_calls) == 1
    assert wire.disconnected.is_set()
    assert not pool.is_closed
    assert provider._smartfast_router._active == {}


async def test_sdk_concurrent_keys_share_pool_without_cross_request_authentication(
    environment, sdk_calls
):
    wire, provider, _ = environment
    pool = provider.client
    await asyncio.gather(
        *(
            provider.translate_batch(
                batch(),
                model="fixture/model:smartfast",
                config_override=TranslationConfig(apiKey=key),
            )
            for key in ["synthetic-first-key", "synthetic-second-key"]
        )
    )
    assert len(sdk_calls) == 2
    requests = [request for request in wire.requests if request.method == "POST"]
    assert {request.headers["Authorization"] for request in requests} == {
        "Bearer synthetic-first-key",
        "Bearer synthetic-second-key",
    }
    assert len({json.loads(request.content)["session_id"] for request in requests}) == 2
    assert provider.client is pool
    assert not pool.is_closed


@pytest.mark.parametrize(
    "model,configuration,expected",
    [
        ("anthropic/claude-haiku-4.5", {"maxTokens": 2500}, {"max_tokens": 2500}),
        ("google/gemini-2.5-flash-preview-09-2025", {"maxTokens": 2500}, {"max_tokens": 2500}),
        ("x-ai/grok-4.1", {"enabled": True}, {"enabled": True}),
    ],
)
async def test_sdk_preserves_existing_token_budget_and_boolean_reasoning(
    environment, sdk_calls, model, configuration, expected
):
    wire, provider, _ = environment
    wire.models = [{"id": model, "supported_parameters": ["temperature", "reasoning"]}]
    result = await provider.translate_batch(
        batch(),
        model=model + ":smartfast",
        config_override=TranslationConfig(reasoning=configuration),
    )
    assert len(sdk_calls) == 1
    assert len(result.translations) == 2
    body = wire.completions[0]
    assert body["reasoning"] == expected
    assert "response_format" not in body
    assert body["provider"]["only"] == ["a"]


async def test_async_completion_does_not_construct_an_unused_sync_http_pool(
    environment, sdk_calls, monkeypatch
):
    wire, provider, _ = environment

    def forbid_sync_client(*args, **kwargs):
        raise AssertionError("Async completion must not build an unused synchronous TLS pool")

    monkeypatch.setattr(httpx, "Client", forbid_sync_client)
    result = await provider.translate_batch(batch(), model="fixture/model:smartfast")
    assert len(sdk_calls) == len(wire.completions) == 1
    assert len(result.translations) == 2
    assert not provider.client.is_closed
