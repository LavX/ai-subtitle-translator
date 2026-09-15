"""The OPENROUTER_MAX_TOKENS output budget that goes out with every request."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from subtitle_translator.api.models import ReasoningConfig, TranslationConfig
from subtitle_translator.providers.base import TranslationBatch
from subtitle_translator.providers.openrouter import (
    EFFORT_REASONING_MODELS,
    ENABLED_REASONING_MODELS,
    MAX_TOKENS_REASONING_MODELS,
    THINKING_VARIANT_MODELS,
    OpenRouterProvider,
)

PLAIN = "deepseek/deepseek-v4-flash"
REASONER = MAX_TOKENS_REASONING_MODELS[0]
EFFORT_MODEL = EFFORT_REASONING_MODELS[0]
ENABLED_MODEL = ENABLED_REASONING_MODELS[0]
THINKING_MODEL = THINKING_VARIANT_MODELS[0]
MODELS = (PLAIN, REASONER, EFFORT_MODEL, ENABLED_MODEL, THINKING_MODEL)


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


def _provider(max_tokens=8000, supported=None, ceiling=None, reasoning=None):
    provider = OpenRouterProvider(settings=_make_settings(max_tokens))
    # Stand in for the catalog fetch so no test touches the network.
    provider._model_params_fetched = True
    for model in MODELS:
        provider._model_params_cache[model] = (
            list(supported) if supported is not None else ["temperature", "max_tokens"]
        )
        if ceiling is not None:
            provider._model_max_output_cache[model] = ceiling
        if reasoning is not None:
            provider._model_reasoning_cache[model] = dict(reasoning)
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
    """Reasoning is spent out of max_tokens, so the budget has to make room for it."""

    async def test_a_declared_reasoning_budget_is_added_on_top(self):
        config = TranslationConfig(reasoning=ReasoningConfig(maxTokens=2000))
        payload = await _send(_provider(), config=config, model=REASONER)
        assert payload["reasoning"] == {"max_tokens": 2000}
        assert payload["max_tokens"] == 10000

    async def test_the_default_reasoning_budget_is_added_on_top(self):
        config = TranslationConfig(reasoning=ReasoningConfig(enabled=True))
        payload = await _send(_provider(), config=config, model=REASONER)
        assert payload["reasoning"] == {"max_tokens": 2000}
        assert payload["max_tokens"] == 10000

    async def test_an_effort_level_buys_a_share_of_the_budget(self):
        """High effort spends about 80% of max_tokens, so 8000 for the answer needs 5x."""
        config = TranslationConfig(reasoning=ReasoningConfig(effort="high"))
        payload = await _send(_provider(), config=config, model=EFFORT_MODEL)
        assert payload["reasoning"] == {"effort": "high"}
        assert payload["max_tokens"] == 40000

    async def test_a_lower_effort_buys_a_smaller_share(self):
        config = TranslationConfig(reasoning=ReasoningConfig(effort="low"))
        payload = await _send(_provider(), config=config, model=EFFORT_MODEL)
        assert payload["max_tokens"] == 10000

    async def test_reasoning_switched_on_without_an_effort_gets_the_medium_share(self):
        config = TranslationConfig(reasoning=ReasoningConfig(enabled=True))
        payload = await _send(_provider(), config=config, model=ENABLED_MODEL)
        assert payload["reasoning"] == {"enabled": True}
        assert payload["max_tokens"] == 16000

    async def test_a_thinking_variant_gets_room_although_it_declares_nothing(self):
        config = TranslationConfig(reasoning=ReasoningConfig(enabled=True), useThinkingVariant=True)
        payload = await _send(_provider(), config=config, model=THINKING_MODEL)
        assert payload["model"].endswith(":thinking")
        assert "reasoning" not in payload
        assert payload["max_tokens"] == 16000

    async def test_a_mandatory_reasoner_gets_room_without_being_asked(self):
        """Over a hundred catalog models think whether or not the request says so."""
        provider = _provider(reasoning={"mandatory": True, "default_effort": "high"})
        payload = await _send(provider)
        assert "reasoning" not in payload
        assert payload["max_tokens"] == 40000

    async def test_an_optional_reasoner_left_off_keeps_the_plain_budget(self):
        provider = _provider(reasoning={"mandatory": False, "default_effort": "high"})
        payload = await _send(provider)
        assert payload["max_tokens"] == 8000

    async def test_reasoning_turned_off_keeps_the_plain_budget(self):
        config = TranslationConfig(reasoning=ReasoningConfig(effort="none"))
        payload = await _send(_provider(), config=config, model=EFFORT_MODEL)
        assert payload["max_tokens"] == 8000


@pytest.mark.asyncio
class TestModelCeiling:
    async def test_budget_is_clamped_to_the_published_ceiling(self):
        payload = await _send(_provider(max_tokens=64000, ceiling=4096))
        assert payload["max_tokens"] == 4096

    async def test_a_budget_under_the_ceiling_is_left_alone(self):
        payload = await _send(_provider(max_tokens=8000, ceiling=65536))
        assert payload["max_tokens"] == 8000

    async def test_an_implausible_ceiling_is_ignored(self):
        """A one-token ceiling is a broken catalog, not a budget worth honouring."""
        payload = await _send(_provider(max_tokens=8000, ceiling=1))
        assert payload["max_tokens"] == 8000

    async def test_no_budget_is_sent_when_the_ceiling_cannot_hold_the_reasoning(self, caplog):
        """Clamping here would ask for a reply the reasoning leaves no room to write."""
        config = TranslationConfig(reasoning=ReasoningConfig(maxTokens=16000))
        with caplog.at_level("WARNING"):
            payload = await _send(_provider(ceiling=16384), config=config, model=REASONER)
        assert "max_tokens" not in payload
        assert payload["reasoning"] == {"max_tokens": 16000}
        assert "no answer room" in caplog.text

    async def test_omitted_when_the_model_takes_no_budget_parameter(self, caplog):
        with caplog.at_level("INFO"):
            payload = await _send(_provider(supported=["temperature"]))
        assert "max_tokens" not in payload
        assert "accepts no output budget parameter" in caplog.text

    async def test_sent_under_the_newer_name_when_that_is_what_the_model_takes(self):
        payload = await _send(_provider(supported=["temperature", "max_completion_tokens"]))
        assert "max_tokens" not in payload
        assert payload["max_completion_tokens"] == 8000

    async def test_sent_when_the_catalog_has_no_entry_for_the_model(self):
        provider = _provider()
        provider._model_params_cache.clear()
        payload = await _send(provider)
        assert payload["max_tokens"] == 8000


async def _load_catalog(provider, body):
    """Run the catalog fetch against a stand-in body, as the fetch builds its own client."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = body
    with patch("subtitle_translator.providers.openrouter.httpx.AsyncClient") as MockClient:
        context = AsyncMock()
        context.get.return_value = response
        MockClient.return_value.__aenter__ = AsyncMock(return_value=context)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        await provider._ensure_model_params_cache()


@pytest.mark.asyncio
class TestCatalogCeilingCache:
    async def test_a_rejected_catalog_leaves_no_ceiling_behind(self):
        """A body with no usable parameters is not a catalog, so nothing it said binds."""
        provider = OpenRouterProvider(settings=_make_settings())
        await _load_catalog(
            provider, {"data": [{"id": PLAIN, "top_provider": {"max_completion_tokens": 4096}}]}
        )

        assert provider._model_params_fetched is False
        assert provider._model_max_output_cache == {}

    async def test_an_accepted_catalog_records_the_ceiling(self):
        provider = OpenRouterProvider(settings=_make_settings())
        await _load_catalog(
            provider,
            {
                "data": [
                    {
                        "id": PLAIN,
                        "supported_parameters": ["max_tokens"],
                        "top_provider": {"max_completion_tokens": 4096},
                    }
                ]
            },
        )

        assert provider._model_params_fetched is True
        assert provider._model_max_output_cache == {PLAIN: 4096}


@pytest.mark.asyncio
class TestTruncationWarning:
    async def test_a_reply_cut_by_the_budget_names_the_budget(self, caplog):
        with caplog.at_level("WARNING"):
            await _send(_provider(max_tokens=1500), finish_reason="length")
        assert "1500-token output budget" in caplog.text
        assert "Raise OPENROUTER_MAX_TOKENS" in caplog.text

    async def test_a_reply_cut_at_the_ceiling_does_not_advise_raising_the_budget(self, caplog):
        """The clamp would swallow the increase, so telling the user to raise it lies."""
        with caplog.at_level("WARNING"):
            await _send(_provider(max_tokens=64000, ceiling=4096), finish_reason="length")
        assert "4096-token output budget" in caplog.text
        assert "will not lift it" in caplog.text

    async def test_a_reply_cut_without_a_budget_names_the_model_ceiling(self, caplog):
        with caplog.at_level("WARNING"):
            await _send(_provider(max_tokens=0), finish_reason="length")
        assert "own output ceiling" in caplog.text

    async def test_a_complete_reply_warns_about_nothing(self, caplog):
        with caplog.at_level("WARNING"):
            await _send(_provider(), finish_reason="stop")
        assert "cut short" not in caplog.text
