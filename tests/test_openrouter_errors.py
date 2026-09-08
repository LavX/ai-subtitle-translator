"""Provider error envelopes must not train smaller translation batches."""

import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.batch_sizing import get_batch_size_resolver
from subtitle_translator.providers.base import (
    AuthenticationError,
    InvalidResponseError,
    ProviderTimeoutError,
    RateLimitError,
    TranslationProviderError,
)
from subtitle_translator.providers.openrouter import OpenRouterProvider
from tests.test_batch_reliability import response, settings


@pytest.mark.asyncio
@pytest.mark.parametrize("http_status", [200, None])
@pytest.mark.parametrize(
    "code,expected,retryable",
    [
        (401, AuthenticationError, False),
        (402, TranslationProviderError, False),
        (403, TranslationProviderError, False),
        (408, ProviderTimeoutError, True),
        (429, RateLimitError, True),
        (500, TranslationProviderError, True),
        (503, TranslationProviderError, True),
    ],
)
async def test_error_code_and_usage_survive_http_envelope(http_status, code, expected, retryable):
    provider = OpenRouterProvider(settings())
    with pytest.raises(expected) as caught:
        await provider._process_response(
            httpx.Response(
                http_status or code,
                json={
                    "error": {"code": code, "message": "synthetic provider error"},
                    "usage": {"total_tokens": 7, "cost": 0.007},
                },
            ),
            "fixture/model",
        )
    error = caught.value
    assert type(error) is expected
    assert error.status_code == code and error.retryable is retryable
    assert error.tokens_used == 7 and error.cost == pytest.approx(0.007)


@pytest.mark.asyncio
@pytest.mark.parametrize("detail", [None, {"code": 503, "message": "synthetic interruption"}])
async def test_error_finish_cannot_be_success_even_with_parseable_content(detail):
    provider = OpenRouterProvider(settings())
    choice = {
        "message": {
            "content": json.dumps({"translations": [{"index": "1", "content": "translated"}]})
        },
        "finish_reason": "error",
    }
    if detail is not None:
        choice["error"] = detail
    with pytest.raises(TranslationProviderError) as caught:
        await provider._process_response(
            httpx.Response(
                200, json={"choices": [choice], "usage": {"total_tokens": 7, "cost": 0.007}}
            ),
            "fixture/model",
        )
    assert type(caught.value) is TranslationProviderError
    assert caught.value.retryable and caught.value.tokens_used == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["later", "NaN", "inf", "-1", "2", "date"])
async def test_retry_after_is_safe_and_supports_http_dates(header):
    provider = OpenRouterProvider(settings())
    value = (
        format_datetime(datetime.now(UTC) + timedelta(seconds=60), usegmt=True)
        if header == "date"
        else header
    )
    with pytest.raises(RateLimitError) as caught:
        await provider._process_response(
            httpx.Response(429, headers={"Retry-After": value}, json={}), "fixture/model"
        )
    delay = caught.value.retry_after
    if header == "date":
        assert 55 < delay <= 60
    elif header == "2":
        assert delay == 2
    else:
        assert delay is None


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [429, 503, 401, 402])
async def test_embedded_provider_error_preserves_batch_size_and_known_cost(code):
    resolver = get_batch_size_resolver()
    resolver.reset()
    provider = OpenRouterProvider(settings())
    provider._model_params_fetched = True
    sent = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        sent.append(len(lines))
        if len(sent) == 1:
            return httpx.Response(
                200,
                headers={"Retry-After": "0.001"},
                json={
                    "error": {"code": code, "message": "temporarily unavailable"},
                    "usage": {"total_tokens": 7, "cost": 0.007},
                },
            )
        return response([{**line, "content": "translated"} for line in lines])

    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://synthetic.invalid"
    )
    messages = []
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": str(i), "content": "source"} for i in range(10)],
            "en",
            "hu",
            model="fixture/envelope",
            batch_size=10,
            progress_callback=lambda progress: messages.append(progress.message),
        )
        retryable = code in (429, 503)
        assert sent == ([10, 10] if retryable else [10])
        assert result.success is retryable
        assert result.total_tokens == (17 if retryable else 7)
        assert result.progress.total_cost == pytest.approx(0.017 if retryable else 0.007)
        assert resolver.limit_planned_size("fixture/envelope", 10) == 10
        if code == 429:
            assert any("rate limited" in message for message in messages)
        assert not any("invalid response" in message for message in messages)
    finally:
        await provider.close()
        resolver.reset()


@pytest.mark.asyncio
async def test_empty_choices_without_provider_error_remains_invalid_translation():
    with pytest.raises(InvalidResponseError):
        await OpenRouterProvider(settings())._process_response(
            httpx.Response(200, json={"choices": []}), "fixture/model"
        )
