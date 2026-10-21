"""Response handling is shared by every chat-completions back end."""

import httpx
import pytest

from subtitle_translator.backends.chat_completions import ChatCompletionsProvider
from subtitle_translator.backends.openrouter import OpenRouterProvider
from subtitle_translator.core.errors import AuthenticationError, RateLimitError
from subtitle_translator.core.provider import TranslationResult


class Local(ChatCompletionsProvider):
    error_label = "Local"

    @property
    def provider_name(self):
        return "local"

    async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
        raise NotImplementedError

    async def health_check(self):
        return True

    async def close(self):
        pass

    async def get_available_models(self):
        return []


def response(status, body, headers=None):
    return httpx.Response(
        status,
        json=body,
        headers=headers,
        request=httpx.Request("POST", "http://x/chat/completions"),
    )


@pytest.mark.asyncio
async def test_error_messages_use_the_subclass_label():
    with pytest.raises(AuthenticationError, match="Invalid Local API key"):
        await Local()._process_response(response(401, {"error": {"message": "no"}}), "m")


@pytest.mark.asyncio
async def test_rate_limit_carries_retry_after():
    with pytest.raises(RateLimitError) as info:
        await Local()._process_response(
            response(429, {"error": {"message": "slow"}}, {"retry-after": "7"}), "m"
        )
    assert info.value.retry_after == 7.0


@pytest.mark.asyncio
async def test_success_parses_translations_and_usage():
    body = {
        "choices": [
            {"message": {"content": '{"translations": [{"index": "0", "content": "Hola"}]}'}}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "cost": 0.001},
    }
    result = await Local()._process_response(response(200, body), "m")
    assert isinstance(result, TranslationResult)
    assert result.translations == [{"index": "0", "content": "Hola"}]
    assert result.total_tokens == 5 and result.cost == 0.001


def test_openrouter_is_a_chat_completions_provider():
    assert issubclass(OpenRouterProvider, ChatCompletionsProvider)
    assert OpenRouterProvider.error_label == "OpenRouter"
