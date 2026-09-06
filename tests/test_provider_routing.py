"""OpenRouter provider routing: sort values, the :nitro/:floor shortcuts and typed suffixes."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from subtitle_translator.api.models import ProviderConfig, TranslationConfig
from subtitle_translator.providers.base import TranslationBatch
from subtitle_translator.providers.openrouter import (
    EXCELLENT_MODELS,
    THINKING_VARIANT_MODELS,
    OpenRouterProvider,
    split_routing_suffix,
)

PLAIN = "deepseek/deepseek-v4-flash"


def _make_settings():
    settings = MagicMock()
    settings.openrouter_api_key = "sk-test-key-123"
    settings.openrouter_api_base = "https://openrouter.ai/api/v1"
    settings.openrouter_default_model = PLAIN
    settings.openrouter_temperature = 0.3
    settings.openrouter_max_tokens = 8000
    settings.request_timeout = 120.0
    settings.openrouter_headers = {"Authorization": "Bearer sk-test-key-123"}
    settings.max_retries = 0
    settings.retry_delay = 0
    return settings


def _provider():
    return OpenRouterProvider(settings=_make_settings())


def _ok_response(model):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "id": "gen-1",
        "model": model,
        "choices": [
            {
                "message": {
                    "content": '{"translations": [{"index": 0, "content": "Szia"}]}',
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    resp.text = ""
    return resp


async def _send(provider, config, model=None):
    batch = TranslationBatch(
        lines=[{"index": 0, "content": "Hi"}],
        target_language="hu",
        source_language="en",
    )
    client = AsyncMock()
    client.post.return_value = _ok_response(model or PLAIN)
    client.is_closed = False
    provider._client = client
    await provider.translate_batch(batch, model=model, config_override=config)
    call = client.post.call_args
    return call.kwargs.get("json") or call[1].get("json")


class TestSplitRoutingSuffix:
    def test_plain_slug_has_no_suffix(self):
        assert split_routing_suffix(PLAIN) == (PLAIN, None)

    def test_floor_suffix_is_split_off(self):
        assert split_routing_suffix(f"{PLAIN}:floor") == (PLAIN, "floor")

    def test_nitro_suffix_is_split_off(self):
        assert split_routing_suffix(f"{PLAIN}:nitro") == (PLAIN, "nitro")

    def test_other_variants_are_not_routing(self):
        assert split_routing_suffix(f"{PLAIN}:thinking") == (f"{PLAIN}:thinking", None)
        assert split_routing_suffix(f"{PLAIN}:free") == (f"{PLAIN}:free", None)


class TestProviderConfigSort:
    @pytest.mark.parametrize(
        "value", ["default", "throughput", "price", "latency", "nitro", "floor"]
    )
    def test_known_values_accepted(self, value):
        assert ProviderConfig(sort=value).sort == value

    def test_case_and_whitespace_normalised(self):
        assert ProviderConfig(sort=" Floor ").sort == "floor"

    def test_unknown_value_rejected(self):
        with pytest.raises(ValidationError):
            ProviderConfig(sort="cheapest")


class TestBuildProviderPayload:
    def test_no_config_keeps_throughput_default(self):
        model, payload = _provider()._build_provider_payload(None, PLAIN)
        assert model == PLAIN
        assert payload == {"provider": {"sort": "throughput"}}

    def test_typed_floor_suffix_is_honoured_without_a_competing_sort(self):
        model, payload = _provider()._build_provider_payload(None, f"{PLAIN}:floor")
        assert model == f"{PLAIN}:floor"
        assert payload == {}

    def test_typed_suffix_wins_over_configured_sort(self):
        config = TranslationConfig(provider=ProviderConfig(sort="price"))
        model, payload = _provider()._build_provider_payload(config, f"{PLAIN}:nitro")
        assert model == f"{PLAIN}:nitro"
        assert payload == {}

    def test_floor_appends_the_shortcut(self):
        config = TranslationConfig(provider=ProviderConfig(sort="floor"))
        model, payload = _provider()._build_provider_payload(config, PLAIN)
        assert model == f"{PLAIN}:floor"
        assert "sort" not in payload.get("provider", {})

    def test_nitro_appends_the_shortcut(self):
        config = TranslationConfig(provider=ProviderConfig(sort="nitro"))
        model, payload = _provider()._build_provider_payload(config, PLAIN)
        assert model == f"{PLAIN}:nitro"
        assert "sort" not in payload.get("provider", {})

    def test_floor_falls_back_to_price_sort_on_a_variant_slug(self):
        config = TranslationConfig(provider=ProviderConfig(sort="floor"))
        model, payload = _provider()._build_provider_payload(config, f"{PLAIN}:thinking")
        assert model == f"{PLAIN}:thinking"
        assert payload == {"provider": {"sort": "price"}}

    def test_nitro_falls_back_to_throughput_sort_on_a_variant_slug(self):
        config = TranslationConfig(provider=ProviderConfig(sort="nitro"))
        model, payload = _provider()._build_provider_payload(config, f"{PLAIN}:free")
        assert model == f"{PLAIN}:free"
        assert payload == {"provider": {"sort": "throughput"}}

    @pytest.mark.parametrize("value", ["price", "latency", "throughput"])
    def test_plain_sorts_go_out_as_provider_sort(self, value):
        config = TranslationConfig(provider=ProviderConfig(sort=value))
        model, payload = _provider()._build_provider_payload(config, PLAIN)
        assert model == PLAIN
        assert payload == {"provider": {"sort": value}}

    def test_default_sends_no_sort(self):
        config = TranslationConfig(provider=ProviderConfig(sort="default"))
        model, payload = _provider()._build_provider_payload(config, PLAIN)
        assert model == PLAIN
        assert payload == {}

    def test_default_keeps_the_other_routing_fields(self):
        config = TranslationConfig(
            provider=ProviderConfig(sort="default", only=["deepinfra"], allow_fallbacks=False)
        )
        _, payload = _provider()._build_provider_payload(config, PLAIN)
        assert payload == {"provider": {"only": ["deepinfra"], "allow_fallbacks": False}}

    def test_floor_keeps_the_provider_order(self):
        config = TranslationConfig(provider=ProviderConfig(sort="floor", order=["deepinfra"]))
        model, payload = _provider()._build_provider_payload(config, PLAIN)
        assert model == f"{PLAIN}:floor"
        assert payload["provider"] == {"order": ["deepinfra"]}

    def test_order_without_sort_still_adds_no_sort(self):
        config = TranslationConfig(provider=ProviderConfig(order=["deepinfra"]))
        model, payload = _provider()._build_provider_payload(config, PLAIN)
        assert model == PLAIN
        assert "sort" not in payload["provider"]


class TestRequestBody:
    async def test_floor_setting_reaches_the_model_slug(self):
        config = TranslationConfig(model=PLAIN, provider=ProviderConfig(sort="floor"))
        payload = await _send(_provider(), config)
        assert payload["model"] == f"{PLAIN}:floor"
        assert "provider" not in payload

    async def test_typed_floor_slug_is_sent_unchanged(self):
        config = TranslationConfig(model=f"{PLAIN}:floor")
        payload = await _send(_provider(), config, model=f"{PLAIN}:floor")
        assert payload["model"] == f"{PLAIN}:floor"
        assert "provider" not in payload

    async def test_nothing_configured_still_sorts_by_throughput(self):
        payload = await _send(_provider(), TranslationConfig(model=PLAIN))
        assert payload["model"] == PLAIN
        assert payload["provider"] == {"sort": "throughput"}

    async def test_thinking_variant_on_a_typed_floor_slug_falls_back_to_price_sort(self):
        base = THINKING_VARIANT_MODELS[0]
        config = TranslationConfig(model=f"{base}:floor", use_thinking_variant=True)
        payload = await _send(_provider(), config, model=f"{base}:floor")
        assert payload["model"] == f"{base}:thinking"
        assert payload["provider"] == {"sort": "price"}


class TestMetadataLookupsIgnoreRoutingSuffix:
    async def test_reasoning_type_resolves_through_a_floor_suffix(self):
        base = THINKING_VARIANT_MODELS[0]
        assert await _provider()._get_reasoning_type(f"{base}:floor") == "thinking_variant"

    def test_model_metadata_resolves_through_a_nitro_suffix(self):
        base = EXCELLENT_MODELS[0]["id"]
        assert _provider().get_model_metadata(f"{base}:nitro") == _provider().get_model_metadata(
            base
        )
