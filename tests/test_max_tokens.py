"""The OPENROUTER_MAX_TOKENS output budget that goes out with every request."""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from subtitle_translator.api.models import ReasoningConfig, TranslationConfig
from subtitle_translator.providers.base import TranslationBatch
from subtitle_translator.providers.openrouter import (
    MAX_TOKENS_REASONING_MODELS,
    OpenRouterProvider,
)

PLAIN = "deepseek/deepseek-v4-flash"
REASONER = MAX_TOKENS_REASONING_MODELS[0]


def _make_settings(max_tokens=8000):
    settings = MagicMock()
    settings.openrouter_api_key = "sk-test-key-123"
    settings.openrouter_api_base = "https://openrouter.ai/api/v1"
    settings.openrouter_default_model = PLAIN
    settings.openrouter_temperature = 0.3
    settings.openrouter_max_tokens = max_tokens
    settings.request_timeout = 120.0
    settings.openrouter_headers = {"Authorization": "Bearer sk-test-key-123"}
    settings.get_openrouter_headers = lambda api_key_override=None: {
        "Authorization": f"Bearer {api_key_override or settings.openrouter_api_key}"
    }
    settings.max_retries = 0
    settings.retry_delay = 0
    return settings


def _provider(max_tokens=8000, supported=None, ceiling=None):
    provider = OpenRouterProvider(settings=_make_settings(max_tokens))
    # Stand in for the catalog fetch so no test touches the network.
    provider._model_params_fetched = True
    for model in (PLAIN, REASONER):
        provider._model_params_cache[model] = (
            list(supported) if supported is not None else ["temperature", "max_tokens"]
        )
        if ceiling is not None:
            provider._model_max_output_cache[model] = ceiling
    return provider


def _ok_response(model, finish_reason=None):
    choice: dict = {"message": {"content": '{"translations": [{"index": 0, "content": "Szia"}]}'}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return httpx.Response(
        200,
        json={
            "id": "gen-1",
            "model": model,
            "choices": [choice],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


async def _send(provider, config=None, model=None, finish_reason=None):
    batch = TranslationBatch(
        lines=[{"index": 0, "content": "Hi"}],
        target_language="hu",
        source_language="en",
    )
    requests = []

    async def respond(request):
        requests.append(request)
        return _ok_response(model or PLAIN, finish_reason)

    async with httpx.AsyncClient(
        base_url=provider.settings.openrouter_api_base,
        transport=httpx.MockTransport(respond),
    ) as client:
        provider._client = client
        await provider.translate_batch(batch, model=model, config_override=config)
    assert len(requests) == 1
    return json.loads(requests[0].content)


@pytest.mark.asyncio
class TestConfiguredBudget:
    async def test_configured_budget_is_sent(self):
        payload = await _send(_provider())
        assert payload["max_tokens"] == 8000

    async def test_a_custom_budget_is_honored(self):
        payload = await _send(_provider(max_tokens=3000))
        assert payload["max_tokens"] == 3000

    async def test_zero_lets_the_provider_decide(self):
        payload = await _send(_provider(max_tokens=0))
        assert "max_tokens" not in payload

    async def test_a_negative_budget_is_treated_as_unset(self):
        payload = await _send(_provider(max_tokens=-1))
        assert "max_tokens" not in payload

    async def test_a_per_request_key_still_carries_the_budget(self):
        config = TranslationConfig(apiKey="sk-request-key")
        payload = await _send(_provider(), config=config)
        assert payload["max_tokens"] == 8000


@pytest.mark.asyncio
class TestReasoningHeadroom:
    async def test_reasoning_budget_is_added_on_top(self):
        """max_tokens must stay strictly above the reasoning budget it contains."""
        config = TranslationConfig(reasoning=ReasoningConfig(maxTokens=2000))
        payload = await _send(_provider(), config=config, model=REASONER)
        assert payload["reasoning"] == {"max_tokens": 2000}
        assert payload["max_tokens"] == 10000

    async def test_default_reasoning_budget_is_added_on_top(self):
        config = TranslationConfig(reasoning=ReasoningConfig(enabled=True))
        payload = await _send(_provider(), config=config, model=REASONER)
        assert payload["reasoning"] == {"max_tokens": 2000}
        assert payload["max_tokens"] == 10000

    async def test_effort_reasoning_leaves_the_budget_alone(self):
        config = TranslationConfig(reasoning=ReasoningConfig(effort="high"))
        payload = await _send(_provider(), config=config, model=REASONER)
        assert payload["max_tokens"] == 8000


@pytest.mark.asyncio
class TestModelCeiling:
    async def test_budget_is_clamped_to_the_model_output_ceiling(self):
        payload = await _send(_provider(max_tokens=64000, ceiling=4096))
        assert payload["max_tokens"] == 4096

    async def test_a_budget_under_the_ceiling_is_left_alone(self):
        payload = await _send(_provider(max_tokens=8000, ceiling=65536))
        assert payload["max_tokens"] == 8000

    async def test_omitted_when_the_model_does_not_take_it(self):
        payload = await _send(_provider(supported=["temperature"]))
        assert "max_tokens" not in payload

    async def test_sent_when_the_catalog_has_no_entry_for_the_model(self):
        provider = _provider()
        provider._model_params_cache.clear()
        payload = await _send(provider)
        assert payload["max_tokens"] == 8000


@pytest.mark.asyncio
class TestTruncationWarning:
    async def test_a_reply_cut_by_the_budget_names_the_budget(self, caplog):
        with caplog.at_level("WARNING"):
            await _send(_provider(max_tokens=1500), finish_reason="length")
        assert "1500-token output budget" in caplog.text
        assert "OPENROUTER_MAX_TOKENS" in caplog.text

    async def test_a_reply_cut_without_a_budget_names_the_model_ceiling(self, caplog):
        with caplog.at_level("WARNING"):
            await _send(_provider(max_tokens=0), finish_reason="length")
        assert "its own output ceiling" in caplog.text

    async def test_a_complete_reply_warns_about_nothing(self, caplog):
        with caplog.at_level("WARNING"):
            await _send(_provider(), finish_reason="stop")
        assert "truncated" not in caplog.text
