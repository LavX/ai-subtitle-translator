"""Comprehensive tests for OpenRouterProvider."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from subtitle_translator.api.models import (
    ProviderConfig,
    ReasoningConfig,
    TranslationConfig,
)
from subtitle_translator.providers.base import (
    AuthenticationError,
    InvalidResponseError,
    RateLimitError,
    TranslationBatch,
    TranslationProviderError,
    TranslationResult,
)
from subtitle_translator.providers.openrouter import (
    EFFORT_REASONING_MODELS,
    ENABLED_REASONING_MODELS,
    EXCELLENT_MODELS,
    GOOD_MODELS,
    MAX_TOKENS_REASONING_MODELS,
    POOR_MODELS,
    RECOMMENDED_MODELS,
    THINKING_VARIANT_MODELS,
    OpenRouterProvider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    """Build a mock Settings object with sensible defaults."""
    defaults = {
        "openrouter_api_key": "sk-test-key-123",
        "openrouter_api_base": "https://openrouter.ai/api/v1",
        "openrouter_default_model": "meta-llama/llama-4-maverick",
        "openrouter_temperature": 0.3,
        "openrouter_max_tokens": 8000,
        "request_timeout": 120.0,
        "openrouter_headers": {
            "Authorization": "Bearer sk-test-key-123",
            "Content-Type": "application/json",
            "X-Title": "ai-subtitle-translator",
        },
        "app_name": "ai-subtitle-translator",
        "app_url": "https://lavx.hu",
    }
    defaults.update(overrides)

    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)

    # get_openrouter_headers method
    def _get_headers(api_key_override=None):
        key = api_key_override or defaults["openrouter_api_key"]
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": defaults["app_name"],
        }

    settings.get_openrouter_headers = _get_headers
    return settings


def _make_batch(
    lines=None,
    source_language="English",
    target_language="Hungarian",
):
    if lines is None:
        lines = [
            {"index": "0", "content": "Hello, world!"},
            {"index": "1", "content": "How are you?"},
        ]
    return TranslationBatch(
        lines=lines,
        source_language=source_language,
        target_language=target_language,
    )


def _ok_response_json(translations=None, model="meta-llama/llama-4-maverick"):
    """Build a typical successful OpenRouter chat/completions JSON body."""
    if translations is None:
        translations = [
            {"index": "0", "content": "Szia, vilag!"},
            {"index": "1", "content": "Hogy vagy?"},
        ]
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(translations, ensure_ascii=False),
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.001,
        },
        "model": model,
    }


def _prompt_example(prompt):
    """The JSON example the system prompt shows the model, parsed."""
    start = prompt.index("{", prompt.index("STEP 1"))
    example, _ = json.JSONDecoder().raw_decode(prompt, start)
    return example


def _mock_response(status_code=200, json_data=None, text="", headers=None):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text or (json.dumps(json_data) if json_data else "")
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = json.JSONDecodeError("err", "", 0)
    return resp


# ===========================================================================
# Tests
# ===========================================================================


class TestInit:
    def test_uses_provided_settings(self):
        settings = _make_settings(openrouter_api_key="custom-key")
        provider = OpenRouterProvider(settings=settings)
        assert provider.settings is settings

    @patch("subtitle_translator.providers.openrouter.get_settings")
    def test_uses_global_settings_when_none(self, mock_get):
        mock_get.return_value = _make_settings()
        provider = OpenRouterProvider()
        mock_get.assert_called_once()
        assert provider.settings is mock_get.return_value

    def test_provider_name(self):
        provider = OpenRouterProvider(settings=_make_settings())
        assert provider.provider_name == "openrouter"

    def test_initial_state(self):
        provider = OpenRouterProvider(settings=_make_settings())
        assert provider._client is None
        assert provider._model_params_cache == {}
        assert provider._model_params_fetched is False


class TestClientProperty:
    def test_creates_client_lazily(self):
        provider = OpenRouterProvider(settings=_make_settings())
        assert provider._client is None
        client = provider.client
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

    def test_reuses_existing_client(self):
        provider = OpenRouterProvider(settings=_make_settings())
        c1 = provider.client
        c2 = provider.client
        assert c1 is c2

    def test_recreates_closed_client(self):
        provider = OpenRouterProvider(settings=_make_settings())
        provider.client  # noqa: B018
        # Simulate closed client
        provider._client = MagicMock(is_closed=True)
        c2 = provider.client
        assert c2 is not provider._client or not c2.is_closed


class TestClose:
    async def test_close_open_client(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_client = AsyncMock()
        mock_client.is_closed = False
        provider._client = mock_client

        await provider.close()
        mock_client.aclose.assert_awaited_once()
        assert provider._client is None

    async def test_close_already_closed(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_client = AsyncMock()
        mock_client.is_closed = True
        provider._client = mock_client

        await provider.close()
        mock_client.aclose.assert_not_awaited()

    async def test_close_when_no_client(self):
        provider = OpenRouterProvider(settings=_make_settings())
        await provider.close()  # Should not raise


class TestHealthCheck:
    async def test_success(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_client = AsyncMock()
        mock_resp = MagicMock(status_code=200)
        mock_client.get.return_value = mock_resp
        provider._client = mock_client
        mock_client.is_closed = False

        assert await provider.health_check() is True
        mock_client.get.assert_awaited_once_with("/models", timeout=10.0)

    async def test_failure_non_200(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_client = AsyncMock()
        mock_client.get.return_value = MagicMock(status_code=500)
        provider._client = mock_client
        mock_client.is_closed = False

        assert await provider.health_check() is False

    async def test_failure_exception(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("down")
        provider._client = mock_client
        mock_client.is_closed = False

        assert await provider.health_check() is False

    async def test_no_api_key(self):
        settings = _make_settings(openrouter_api_key="")
        provider = OpenRouterProvider(settings=settings)
        assert await provider.health_check() is False


class TestGetAvailableModels:
    async def test_returns_sorted_list(self):
        provider = OpenRouterProvider(settings=_make_settings())
        models = await provider.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    async def test_default_model_first(self):
        default_id = EXCELLENT_MODELS[0]["id"]
        provider = OpenRouterProvider(settings=_make_settings(openrouter_default_model=default_id))
        models = await provider.get_available_models()
        assert models[0]["is_default"] is True
        assert models[0]["id"] == default_id

    async def test_contains_excellent_and_good(self):
        provider = OpenRouterProvider(settings=_make_settings())
        models = await provider.get_available_models()
        ids = {m["id"] for m in models}
        for m in EXCELLENT_MODELS:
            assert m["id"] in ids
        for m in GOOD_MODELS:
            assert m["id"] in ids


class TestGetModelMetadata:
    def test_known_model(self):
        provider = OpenRouterProvider(settings=_make_settings())
        meta = provider.get_model_metadata(EXCELLENT_MODELS[0]["id"])
        assert meta is not None
        assert meta["id"] == EXCELLENT_MODELS[0]["id"]

    def test_unknown_model(self):
        provider = OpenRouterProvider(settings=_make_settings())
        assert provider.get_model_metadata("nonexistent/model") is None

    def test_poor_model_included(self):
        provider = OpenRouterProvider(settings=_make_settings())
        if POOR_MODELS:
            meta = provider.get_model_metadata(POOR_MODELS[0]["id"])
            assert meta is not None

    def test_cache_built_once(self):
        provider = OpenRouterProvider(settings=_make_settings())
        provider.get_model_metadata("anything")
        assert hasattr(provider, "_model_metadata_cache")
        cache_ref = provider._model_metadata_cache
        provider.get_model_metadata("anything")
        assert provider._model_metadata_cache is cache_ref


class TestGetBestModelForLanguage:
    def test_hungarian_hu(self):
        provider = OpenRouterProvider(settings=_make_settings())
        result = provider.get_best_model_for_language("hu")
        assert result == EXCELLENT_MODELS[0]["id"]

    def test_hungarian_full_name(self):
        provider = OpenRouterProvider(settings=_make_settings())
        result = provider.get_best_model_for_language("hungarian")
        assert result == EXCELLENT_MODELS[0]["id"]

    def test_other_language_returns_reasoning_model(self):
        provider = OpenRouterProvider(settings=_make_settings())
        result = provider.get_best_model_for_language("de")
        # Should return a model that supports reasoning
        assert result is not None
        found = False
        for m in RECOMMENDED_MODELS:
            if m["id"] == result:
                found = True
                break
        assert found

    def test_case_insensitive(self):
        provider = OpenRouterProvider(settings=_make_settings())
        assert provider.get_best_model_for_language("HU") == EXCELLENT_MODELS[0]["id"]


class TestGetHungarianRecommendations:
    def test_structure(self):
        provider = OpenRouterProvider(settings=_make_settings())
        recs = provider.get_hungarian_recommendations()
        assert "best" in recs
        assert "also_good" in recs
        assert "avoid" in recs
        assert "recommended_config" in recs
        assert "testing_reference" in recs

    def test_avoid_list_from_poor(self):
        provider = OpenRouterProvider(settings=_make_settings())
        recs = provider.get_hungarian_recommendations()
        for model in POOR_MODELS:
            assert model["id"] in recs["avoid"]


class TestTranslateBatch:
    """Tests for the main translate_batch method."""

    async def test_success_default_settings(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        resp_json = _ok_response_json()
        mock_resp = _mock_response(200, resp_json)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        result = await provider.translate_batch(batch)
        assert isinstance(result, TranslationResult)
        assert len(result.translations) == 2
        assert result.model_used == "meta-llama/llama-4-maverick"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.cost == 0.001
        mock_client.post.assert_awaited_once()

    async def test_success_with_config_override_model_and_temp(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        config = TranslationConfig(model="openai/gpt-5", temperature=0.7)
        resp_json = _ok_response_json(model="openai/gpt-5")
        mock_resp = _mock_response(200, resp_json)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        result = await provider.translate_batch(batch, config_override=config)
        assert result.model_used == "openai/gpt-5"
        # Verify payload temperature
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["temperature"] == 0.7

    async def test_success_with_config_override_api_key(self):
        """When config_override has api_key, a separate httpx client is used."""
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        config = TranslationConfig(api_key="sk-override-key")
        resp_json = _ok_response_json()
        mock_resp = _mock_response(200, resp_json)

        with patch("subtitle_translator.providers.openrouter.httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.post.return_value = mock_resp
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await provider.translate_batch(batch, config_override=config)
            assert isinstance(result, TranslationResult)
            # Verify the override client was created with the right headers
            MockClient.assert_called_once()
            call_kwargs = MockClient.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert "sk-override-key" in headers["Authorization"]

    async def test_no_api_key_raises(self):
        settings = _make_settings(openrouter_api_key="")
        provider = OpenRouterProvider(settings=settings)
        batch = _make_batch()

        with pytest.raises(TranslationProviderError, match="API key not configured"):
            await provider.translate_batch(batch)

    async def test_auth_error_401(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        mock_resp = _mock_response(401, json_data={"error": {"message": "invalid key"}})

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        with pytest.raises(AuthenticationError):
            await provider.translate_batch(batch)

    async def test_rate_limit_429_with_retry_after(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        mock_resp = _mock_response(
            429,
            json_data={"error": {"message": "rate limited"}},
            headers={"retry-after": "30"},
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        with pytest.raises(RateLimitError) as exc_info:
            await provider.translate_batch(batch)
        assert exc_info.value.retry_after == 30.0

    async def test_rate_limit_429_without_retry_after(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        mock_resp = _mock_response(429, json_data={"error": {"message": "rate limited"}})

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        with pytest.raises(RateLimitError) as exc_info:
            await provider.translate_batch(batch)
        assert exc_info.value.retry_after is None

    async def test_server_error_500_retryable(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        mock_resp = _mock_response(500, text="Internal Server Error")

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        with pytest.raises(TranslationProviderError) as exc_info:
            await provider.translate_batch(batch)
        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 500

    async def test_client_error_400_non_retryable(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        mock_resp = _mock_response(400, json_data={"error": {"message": "bad request"}})

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        with pytest.raises(TranslationProviderError) as exc_info:
            await provider.translate_batch(batch)
        assert exc_info.value.retryable is False
        assert exc_info.value.status_code == 400

    async def test_timeout_raises_retryable(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ReadTimeout("timed out")
        mock_client.is_closed = False
        provider._client = mock_client

        with pytest.raises(TranslationProviderError) as exc_info:
            await provider.translate_batch(batch)
        assert exc_info.value.retryable is True

    async def test_network_error_raises_retryable(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        mock_client.is_closed = False
        provider._client = mock_client

        with pytest.raises(TranslationProviderError) as exc_info:
            await provider.translate_batch(batch)
        assert exc_info.value.retryable is True

    async def test_anthropic_model_uses_cache_control(self):
        """Anthropic models should get cache_control in system message."""
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        config = TranslationConfig(model="anthropic/claude-haiku-4.5")
        resp_json = _ok_response_json(model="anthropic/claude-haiku-4.5")
        mock_resp = _mock_response(200, resp_json)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        await provider.translate_batch(batch, config_override=config)
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        system_msg = payload["messages"][0]
        assert system_msg["role"] == "system"
        # Anthropic format uses list of content blocks with cache_control
        assert isinstance(system_msg["content"], list)
        assert system_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    async def test_non_anthropic_model_simple_system(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        resp_json = _ok_response_json()
        mock_resp = _mock_response(200, resp_json)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        await provider.translate_batch(batch)
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        system_msg = payload["messages"][0]
        assert isinstance(system_msg["content"], str)

    async def test_json_object_format_when_no_reasoning(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        resp_json = _ok_response_json()
        mock_resp = _mock_response(200, resp_json)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        await provider.translate_batch(batch)
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["response_format"] == {"type": "json_object"}
        # JSON mode only allows an object at the top level, so the example the same
        # request shows the model has to be that object: asked for a bare array under
        # JSON mode, models answered with a single translated line per batch.
        example = _prompt_example(payload["messages"][0]["content"])
        assert isinstance(example, dict)
        assert list(example) == ["translations"]

    def test_prompt_example_is_what_the_parser_expects(self):
        provider = OpenRouterProvider(settings=_make_settings())
        example = _prompt_example(provider.build_system_prompt("Dutch", "English"))
        assert [item["index"] for item in example["translations"]] == ["0", "1"]
        parsed = provider._parse_translations(json.dumps(example))
        assert [item["index"] for item in parsed] == ["0", "1"]
        assert all(set(item) == {"index", "content"} for item in parsed)

    async def test_no_json_object_format_with_reasoning(self):
        provider = OpenRouterProvider(settings=_make_settings())
        batch = _make_batch()
        config = TranslationConfig(
            model="anthropic/claude-haiku-4.5",
            reasoning=ReasoningConfig(enabled=True),
        )
        resp_json = _ok_response_json(model="anthropic/claude-haiku-4.5")
        mock_resp = _mock_response(200, resp_json)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        provider._client = mock_client

        await provider.translate_batch(batch, config_override=config)
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "response_format" not in payload


class TestProcessResponse:
    async def test_success(self):
        provider = OpenRouterProvider(settings=_make_settings())
        resp_json = _ok_response_json()
        mock_resp = _mock_response(200, resp_json)

        result = await provider._process_response(mock_resp, "test-model")
        assert len(result.translations) == 2
        assert result.translations[0]["content"] == "Szia, vilag!"
        assert result.prompt_tokens == 100
        assert result.cost == 0.001

    async def test_401_raises_auth(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_resp = _mock_response(401)
        with pytest.raises(AuthenticationError):
            await provider._process_response(mock_resp, "m")

    async def test_429_with_retry_after_header(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_resp = _mock_response(429, headers={"retry-after": "5"})
        with pytest.raises(RateLimitError) as exc_info:
            await provider._process_response(mock_resp, "m")
        assert exc_info.value.retry_after == 5.0

    async def test_429_no_retry_after(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_resp = _mock_response(429)
        with pytest.raises(RateLimitError) as exc_info:
            await provider._process_response(mock_resp, "m")
        assert exc_info.value.retry_after is None

    async def test_500_retryable(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_resp = _mock_response(500)
        with pytest.raises(TranslationProviderError) as exc_info:
            await provider._process_response(mock_resp, "m")
        assert exc_info.value.retryable is True

    async def test_400_non_retryable(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_resp = _mock_response(400, json_data={"error": {"message": "bad"}})
        with pytest.raises(TranslationProviderError) as exc_info:
            await provider._process_response(mock_resp, "m")
        assert exc_info.value.retryable is False

    async def test_400_plain_text_error(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_resp = _mock_response(400, text="plain error")
        # json() raises on plain text
        mock_resp.json.side_effect = ValueError("not json")
        with pytest.raises(TranslationProviderError):
            await provider._process_response(mock_resp, "m")

    async def test_invalid_json_body(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.text = "not json at all"
        mock_resp.json.side_effect = json.JSONDecodeError("err", "", 0)

        with pytest.raises(InvalidResponseError, match="Invalid JSON"):
            await provider._process_response(mock_resp, "m")

    async def test_no_choices(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_resp = _mock_response(200, {"usage": {}, "choices": []})
        with pytest.raises(InvalidResponseError, match="No choices"):
            await provider._process_response(mock_resp, "m")

    async def test_empty_content(self):
        provider = OpenRouterProvider(settings=_make_settings())
        data = {
            "choices": [{"message": {"content": ""}}],
            "usage": {},
        }
        mock_resp = _mock_response(200, data)
        with pytest.raises(InvalidResponseError, match="Empty content"):
            await provider._process_response(mock_resp, "m")

    async def test_wrapped_translations(self):
        """Response wrapped in {"translations": [...]}."""
        provider = OpenRouterProvider(settings=_make_settings())
        inner = [{"index": "0", "content": "Szia"}]
        wrapper = {"translations": inner}
        data = {
            "choices": [{"message": {"content": json.dumps(wrapper)}}],
            "usage": {},
        }
        mock_resp = _mock_response(200, data)
        result = await provider._process_response(mock_resp, "m")
        assert len(result.translations) == 1
        assert result.translations[0]["content"] == "Szia"

    async def test_single_object_response(self):
        """Response is a single {"index": "0", "content": "..."} instead of array."""
        provider = OpenRouterProvider(settings=_make_settings())
        single = {"index": "0", "content": "Szia"}
        data = {
            "choices": [{"message": {"content": json.dumps(single)}}],
            "usage": {},
        }
        mock_resp = _mock_response(200, data)
        result = await provider._process_response(mock_resp, "m")
        assert len(result.translations) == 1

    async def test_alternative_key_names(self):
        """Keys like 'text', 'idx', 'translation' should be accepted."""
        provider = OpenRouterProvider(settings=_make_settings())
        items = [
            {"idx": "0", "text": "Szia"},
            {"position": "1", "translation": "Hogy vagy?"},
        ]
        data = {
            "choices": [{"message": {"content": json.dumps(items)}}],
            "usage": {},
        }
        mock_resp = _mock_response(200, data)
        result = await provider._process_response(mock_resp, "m")
        assert len(result.translations) == 2

    async def test_cost_parsing_invalid(self):
        provider = OpenRouterProvider(settings=_make_settings())
        data = _ok_response_json()
        data["usage"]["cost"] = "not-a-number"
        mock_resp = _mock_response(200, data)
        result = await provider._process_response(mock_resp, "m")
        assert result.cost is None

    async def test_json_in_markdown_code_block(self):
        """Content wrapped in ```json ... ```."""
        provider = OpenRouterProvider(settings=_make_settings())
        inner = json.dumps([{"index": "0", "content": "Szia"}])
        content = f"```json\n{inner}\n```"
        data = {
            "choices": [{"message": {"content": content}}],
            "usage": {},
        }
        mock_resp = _mock_response(200, data)
        result = await provider._process_response(mock_resp, "m")
        assert len(result.translations) == 1

    async def test_json_array_embedded_in_text(self):
        """Content has text before/after a JSON array."""
        provider = OpenRouterProvider(settings=_make_settings())
        inner = json.dumps([{"index": "0", "content": "Szia"}])
        content = f"Here are the translations: {inner} done."
        data = {
            "choices": [{"message": {"content": content}}],
            "usage": {},
        }
        mock_resp = _mock_response(200, data)
        result = await provider._process_response(mock_resp, "m")
        assert len(result.translations) == 1


class TestValidateAndWarnUnchanged:
    def test_no_warning_when_different(self, caplog):
        provider = OpenRouterProvider(settings=_make_settings())
        original = [
            {"index": "0", "content": "Hello"},
            {"index": "1", "content": "World"},
        ]
        translations = [
            {"index": "0", "content": "Szia"},
            {"index": "1", "content": "Vilag"},
        ]
        with caplog.at_level(logging.WARNING):
            provider._validate_and_warn_unchanged(original, translations)
        # No WARNING about safety check
        assert "SAFETY CHECK" not in caplog.text

    def test_warning_when_50_percent_unchanged(self, caplog):
        provider = OpenRouterProvider(settings=_make_settings())
        original = [
            {"index": "0", "content": "Hello"},
            {"index": "1", "content": "World"},
        ]
        translations = [
            {"index": "0", "content": "Hello"},  # unchanged
            {"index": "1", "content": "World"},  # unchanged
        ]
        with caplog.at_level(logging.WARNING):
            provider._validate_and_warn_unchanged(original, translations)
        assert "SAFETY CHECK" in caplog.text

    def test_info_when_20_percent_unchanged(self, caplog):
        provider = OpenRouterProvider(settings=_make_settings())
        original = [
            {"index": "0", "content": "Hello"},
            {"index": "1", "content": "World"},
            {"index": "2", "content": "Test"},
            {"index": "3", "content": "Foo"},
        ]
        translations = [
            {"index": "0", "content": "Hello"},  # unchanged (25%)
            {"index": "1", "content": "Vilag"},
            {"index": "2", "content": "Teszt"},
            {"index": "3", "content": "Fu"},
        ]
        with caplog.at_level(logging.INFO):
            provider._validate_and_warn_unchanged(original, translations)
        assert "identical to original" in caplog.text

    def test_empty_translations(self):
        provider = OpenRouterProvider(settings=_make_settings())
        # Should not raise
        provider._validate_and_warn_unchanged([], [])


class TestBuildReasoningPayload:
    """Tests for _build_reasoning_payload."""

    async def test_no_config_override(self):
        provider = OpenRouterProvider(settings=_make_settings())
        model, params = await provider._build_reasoning_payload("some/model", None)
        assert model == "some/model"
        assert params == {}

    async def test_effort_reasoning(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=EFFORT_REASONING_MODELS[0],
            reasoning=ReasoningConfig(effort="high"),
        )
        model, params = await provider._build_reasoning_payload(EFFORT_REASONING_MODELS[0], config)
        assert params == {"reasoning": {"effort": "high"}}

    async def test_effort_none_disables(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=EFFORT_REASONING_MODELS[0],
            reasoning=ReasoningConfig(effort="none"),
        )
        model, params = await provider._build_reasoning_payload(EFFORT_REASONING_MODELS[0], config)
        assert params == {}

    async def test_invalid_effort_ignored(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=EFFORT_REASONING_MODELS[0],
            reasoning=ReasoningConfig(effort="superduper"),
        )
        model, params = await provider._build_reasoning_payload(EFFORT_REASONING_MODELS[0], config)
        # Invalid effort should be warned and not included
        assert "effort" not in params.get("reasoning", {})

    async def test_max_tokens_reasoning(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=MAX_TOKENS_REASONING_MODELS[0],
            reasoning=ReasoningConfig(max_tokens=4000),
        )
        model, params = await provider._build_reasoning_payload(
            MAX_TOKENS_REASONING_MODELS[0], config
        )
        assert params == {"reasoning": {"max_tokens": 4000}}

    async def test_max_tokens_default_when_enabled(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=MAX_TOKENS_REASONING_MODELS[0],
            reasoning=ReasoningConfig(enabled=True),
        )
        model, params = await provider._build_reasoning_payload(
            MAX_TOKENS_REASONING_MODELS[0], config
        )
        assert params == {"reasoning": {"max_tokens": 2000}}

    async def test_enabled_reasoning(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=ENABLED_REASONING_MODELS[0],
            reasoning=ReasoningConfig(enabled=True),
        )
        model, params = await provider._build_reasoning_payload(ENABLED_REASONING_MODELS[0], config)
        assert params == {"reasoning": {"enabled": True}}

    async def test_enabled_reasoning_false(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=ENABLED_REASONING_MODELS[0],
            reasoning=ReasoningConfig(enabled=False),
        )
        model, params = await provider._build_reasoning_payload(ENABLED_REASONING_MODELS[0], config)
        assert params == {"reasoning": {"enabled": False}}

    async def test_thinking_variant_via_use_thinking(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=THINKING_VARIANT_MODELS[0],
            use_thinking_variant=True,
        )
        model, params = await provider._build_reasoning_payload(THINKING_VARIANT_MODELS[0], config)
        assert model == f"{THINKING_VARIANT_MODELS[0]}:thinking"
        assert params == {}

    async def test_thinking_variant_already_suffixed(self):
        provider = OpenRouterProvider(settings=_make_settings())
        model_id = f"{THINKING_VARIANT_MODELS[0]}:thinking"
        config = TranslationConfig(
            model=model_id,
            use_thinking_variant=True,
        )
        model, params = await provider._build_reasoning_payload(model_id, config)
        # Should not double-append :thinking
        assert model == model_id

    async def test_thinking_variant_via_reasoning_enabled(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=THINKING_VARIANT_MODELS[0],
            reasoning=ReasoningConfig(enabled=True),
        )
        model, params = await provider._build_reasoning_payload(THINKING_VARIANT_MODELS[0], config)
        assert model.endswith(":thinking")

    async def test_no_reasoning_support_warns(self, caplog):
        provider = OpenRouterProvider(settings=_make_settings())
        # Use a model that does NOT support reasoning
        provider._model_params_fetched = True  # skip API call
        config = TranslationConfig(
            model="meta-llama/llama-4-maverick",
            reasoning=ReasoningConfig(enabled=True),
        )
        with caplog.at_level(logging.WARNING):
            model, params = await provider._build_reasoning_payload(
                "meta-llama/llama-4-maverick", config
            )
        assert params == {}
        assert "does not support reasoning" in caplog.text

    async def test_effort_maps_max_tokens_to_effort(self):
        """When user passes max_tokens on an effort-type model, map it to effort level."""
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=EFFORT_REASONING_MODELS[0],
            reasoning=ReasoningConfig(max_tokens=5000),
        )
        model, params = await provider._build_reasoning_payload(EFFORT_REASONING_MODELS[0], config)
        assert params["reasoning"]["effort"] == "high"

    async def test_effort_maps_max_tokens_medium(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=EFFORT_REASONING_MODELS[0],
            reasoning=ReasoningConfig(max_tokens=2500),
        )
        _, params = await provider._build_reasoning_payload(EFFORT_REASONING_MODELS[0], config)
        assert params["reasoning"]["effort"] == "medium"

    async def test_effort_maps_max_tokens_low(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=EFFORT_REASONING_MODELS[0],
            reasoning=ReasoningConfig(max_tokens=500),
        )
        _, params = await provider._build_reasoning_payload(EFFORT_REASONING_MODELS[0], config)
        assert params["reasoning"]["effort"] == "low"

    async def test_effort_enabled_default_medium(self):
        """If effort model and only enabled=True, default to medium."""
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            model=EFFORT_REASONING_MODELS[0],
            reasoning=ReasoningConfig(enabled=True),
        )
        _, params = await provider._build_reasoning_payload(EFFORT_REASONING_MODELS[0], config)
        assert params["reasoning"]["effort"] == "medium"


class TestBuildProviderPayload:
    def test_default_no_config(self):
        provider = OpenRouterProvider(settings=_make_settings())
        _, result = provider._build_provider_payload(None, "some/model")
        assert result == {"provider": {"sort": "throughput"}}

    def test_default_no_provider_in_config(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(model="some/model")
        _, result = provider._build_provider_payload(config, "some/model")
        assert result == {"provider": {"sort": "throughput"}}

    def test_custom_sort(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            provider=ProviderConfig(sort="price"),
        )
        _, result = provider._build_provider_payload(config, "some/model")
        assert result["provider"]["sort"] == "price"

    def test_order(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            provider=ProviderConfig(order=["deepinfra", "together"]),
        )
        _, result = provider._build_provider_payload(config, "some/model")
        assert result["provider"]["order"] == ["deepinfra", "together"]

    def test_allow_fallbacks(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            provider=ProviderConfig(allow_fallbacks=False),
        )
        _, result = provider._build_provider_payload(config, "some/model")
        assert result["provider"]["allow_fallbacks"] is False

    def test_only_and_ignore(self):
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            provider=ProviderConfig(only=["deepinfra"], ignore=["together"]),
        )
        _, result = provider._build_provider_payload(config, "some/model")
        assert result["provider"]["only"] == ["deepinfra"]
        assert result["provider"]["ignore"] == ["together"]

    def test_order_without_sort_defaults_no_throughput(self):
        """When order is specified, sort should not default to throughput."""
        provider = OpenRouterProvider(settings=_make_settings())
        config = TranslationConfig(
            provider=ProviderConfig(order=["deepinfra"]),
        )
        _, result = provider._build_provider_payload(config, "some/model")
        assert "sort" not in result["provider"]


class TestEnsureModelParamsCache:
    async def test_fetches_and_caches(self):
        provider = OpenRouterProvider(settings=_make_settings())
        api_data = {
            "data": [
                {"id": "model-a", "supported_parameters": ["reasoning", "temperature"]},
                {"id": "model-b", "supported_parameters": ["temperature"]},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_data

        with patch("subtitle_translator.providers.openrouter.httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.get.return_value = mock_resp
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider._ensure_model_params_cache()

        assert provider._model_params_fetched is True
        assert provider._model_params_cache["model-a"] == ["reasoning", "temperature"]
        assert provider._model_params_cache["model-b"] == ["temperature"]

    async def test_skips_if_already_fetched(self):
        provider = OpenRouterProvider(settings=_make_settings())
        provider._model_params_fetched = True

        with patch("subtitle_translator.providers.openrouter.httpx.AsyncClient") as MockClient:
            await provider._ensure_model_params_cache()
            MockClient.assert_not_called()

    async def test_handles_api_failure(self):
        provider = OpenRouterProvider(settings=_make_settings())

        with patch("subtitle_translator.providers.openrouter.httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.get.side_effect = httpx.ConnectError("down")
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider._ensure_model_params_cache()

        # Should mark as fetched even on failure (so it doesn't retry endlessly)
        assert provider._model_params_fetched is True
        assert provider._model_params_cache == {}

    async def test_handles_non_200(self):
        provider = OpenRouterProvider(settings=_make_settings())
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch("subtitle_translator.providers.openrouter.httpx.AsyncClient") as MockClient:
            mock_ctx = AsyncMock()
            mock_ctx.get.return_value = mock_resp
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider._ensure_model_params_cache()

        assert provider._model_params_fetched is True
        assert provider._model_params_cache == {}


class TestGetReasoningType:
    async def test_thinking_variant_model(self):
        provider = OpenRouterProvider(settings=_make_settings())
        provider._model_params_fetched = True
        result = await provider._get_reasoning_type(THINKING_VARIANT_MODELS[0])
        assert result == "thinking_variant"

    async def test_thinking_variant_with_suffix(self):
        provider = OpenRouterProvider(settings=_make_settings())
        provider._model_params_fetched = True
        result = await provider._get_reasoning_type(f"{THINKING_VARIANT_MODELS[0]}:thinking")
        assert result == "thinking_variant"

    async def test_enabled_model(self):
        provider = OpenRouterProvider(settings=_make_settings())
        provider._model_params_fetched = True
        result = await provider._get_reasoning_type(ENABLED_REASONING_MODELS[0])
        assert result == "enabled"

    async def test_effort_model(self):
        provider = OpenRouterProvider(settings=_make_settings())
        provider._model_params_fetched = True
        result = await provider._get_reasoning_type(EFFORT_REASONING_MODELS[0])
        assert result == "effort"

    async def test_max_tokens_model(self):
        provider = OpenRouterProvider(settings=_make_settings())
        provider._model_params_fetched = True
        result = await provider._get_reasoning_type(MAX_TOKENS_REASONING_MODELS[0])
        assert result == "max_tokens"

    async def test_no_reasoning_support(self):
        provider = OpenRouterProvider(settings=_make_settings())
        provider._model_params_fetched = True
        # meta-llama/llama-4-maverick has supports_reasoning=False in RECOMMENDED_MODELS
        result = await provider._get_reasoning_type("meta-llama/llama-4-maverick")
        assert result is None

    async def test_dynamic_lookup_reasoning_param(self):
        """Unknown model with 'reasoning' in supported_parameters gets effort type."""
        provider = OpenRouterProvider(settings=_make_settings())
        provider._model_params_fetched = True
        provider._model_params_cache["custom/model"] = ["reasoning", "temperature"]
        result = await provider._get_reasoning_type("custom/model")
        assert result == "effort"

    async def test_dynamic_lookup_no_reasoning(self):
        provider = OpenRouterProvider(settings=_make_settings())
        provider._model_params_fetched = True
        provider._model_params_cache["custom/model"] = ["temperature"]
        result = await provider._get_reasoning_type("custom/model")
        assert result is None

    async def test_triggers_cache_fetch_for_unknown_model(self):
        provider = OpenRouterProvider(settings=_make_settings())
        # Not yet fetched
        assert provider._model_params_fetched is False

        with patch.object(
            provider, "_ensure_model_params_cache", new_callable=AsyncMock
        ) as mock_ensure:
            await provider._get_reasoning_type("totally-unknown/model")
            mock_ensure.assert_awaited_once()


class TestParseTranslations:
    """Test _parse_translations indirectly via _process_response or directly."""

    def test_valid_array(self):
        provider = OpenRouterProvider(settings=_make_settings())
        content = json.dumps(
            [
                {"index": "0", "content": "Szia"},
                {"index": "1", "content": "Vilag"},
            ]
        )
        result = provider._parse_translations(content)
        assert len(result) == 2

    def test_wrapped_in_translations_key(self):
        provider = OpenRouterProvider(settings=_make_settings())
        content = json.dumps({"translations": [{"index": "0", "content": "Szia"}]})
        result = provider._parse_translations(content)
        assert len(result) == 1

    def test_wrapped_in_results_key(self):
        provider = OpenRouterProvider(settings=_make_settings())
        content = json.dumps({"results": [{"index": "0", "content": "Szia"}]})
        result = provider._parse_translations(content)
        assert len(result) == 1

    def test_wrapped_in_data_key(self):
        provider = OpenRouterProvider(settings=_make_settings())
        content = json.dumps({"data": [{"index": "0", "content": "Szia"}]})
        result = provider._parse_translations(content)
        assert len(result) == 1

    def test_single_object(self):
        provider = OpenRouterProvider(settings=_make_settings())
        content = json.dumps({"index": "0", "content": "Szia"})
        result = provider._parse_translations(content)
        assert len(result) == 1

    def test_unexpected_dict_structure(self):
        provider = OpenRouterProvider(settings=_make_settings())
        content = json.dumps({"foo": "bar"})
        with pytest.raises(InvalidResponseError, match="Unexpected response structure"):
            provider._parse_translations(content)

    def test_not_a_list_result(self):
        provider = OpenRouterProvider(settings=_make_settings())
        content = json.dumps("just a string")
        with pytest.raises(InvalidResponseError, match="Expected list"):
            provider._parse_translations(content)

    def test_skips_non_dict_items(self, caplog):
        provider = OpenRouterProvider(settings=_make_settings())
        content = json.dumps(
            [
                {"index": "0", "content": "Szia"},
                "not a dict",
                {"index": "1", "content": "Vilag"},
            ]
        )
        with caplog.at_level(logging.WARNING):
            result = provider._parse_translations(content)
        assert len(result) == 2

    def test_markdown_code_block(self):
        provider = OpenRouterProvider(settings=_make_settings())
        inner = json.dumps([{"index": "0", "content": "Szia"}])
        content = f"```json\n{inner}\n```"
        result = provider._parse_translations(content)
        assert len(result) == 1

    def test_embedded_array_in_text(self):
        provider = OpenRouterProvider(settings=_make_settings())
        inner = json.dumps([{"index": "0", "content": "Szia"}])
        content = f"Here you go: {inner} enjoy"
        result = provider._parse_translations(content)
        assert len(result) == 1

    def test_completely_unparseable(self):
        provider = OpenRouterProvider(settings=_make_settings())
        with pytest.raises(InvalidResponseError, match="Failed to parse JSON"):
            provider._parse_translations("this is not json and has no arrays")

    def test_alternative_keys_idx_text(self):
        provider = OpenRouterProvider(settings=_make_settings())
        content = json.dumps([{"idx": "0", "text": "Szia"}])
        result = provider._parse_translations(content)
        assert len(result) == 1
        assert result[0]["index"] == "0"
        assert result[0]["content"] == "Szia"

    def test_alternative_keys_position_translation(self):
        provider = OpenRouterProvider(settings=_make_settings())
        content = json.dumps([{"position": "0", "translation": "Szia"}])
        result = provider._parse_translations(content)
        assert len(result) == 1

    def test_duplicate_json_keys(self):
        """Malformed JSON with duplicate keys at top level of an object."""
        provider = OpenRouterProvider(settings=_make_settings())
        # This simulates: {"index":"0","content":"A","index":"1","content":"B"}
        # json.loads with our hook should recover two items
        raw = '{"index":"0","content":"Szia","index":"1","content":"Vilag"}'
        result = provider._parse_translations(raw)
        assert len(result) == 2


class TestFactoryFunction:
    async def test_get_openrouter_provider(self):
        from subtitle_translator.providers.openrouter import get_openrouter_provider

        with patch("subtitle_translator.providers.openrouter.get_settings") as mock_gs:
            mock_gs.return_value = _make_settings()
            provider = await get_openrouter_provider()
            assert isinstance(provider, OpenRouterProvider)
