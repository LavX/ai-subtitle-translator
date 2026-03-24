"""Tests for crypto.py missing lines and providers/base.py missing lines."""

import pytest

from subtitle_translator.crypto import decrypt, derive_auth_token, generate_key, _parse_hex_key
from subtitle_translator.providers.base import (
    TranslationProvider,
    TranslationProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidResponseError,
    TranslationBatch,
)


# ---------------------------------------------------------------------------
# crypto.py tests
# ---------------------------------------------------------------------------


class TestDecryptInvalidBase64:
    """Lines 54-55: decrypt() with invalid base64 payload."""

    def test_invalid_base64_raises_valueerror(self):
        key = generate_key()
        with pytest.raises(ValueError, match="not valid base64"):
            decrypt("enc:!!!not-base64!!!", key)


class TestDecryptTooShort:
    """Line 58: decrypt() with data too short for nonce extraction."""

    def test_short_payload_raises_valueerror(self):
        import base64

        key = generate_key()
        # 4 bytes is shorter than the 12-byte nonce
        short_data = base64.b64encode(b"\x00" * 4).decode()
        with pytest.raises(ValueError, match="too short"):
            decrypt(f"enc:{short_data}", key)


class TestDeriveAuthToken:
    """Lines 78-80: derive_auth_token() function."""

    def test_returns_hex_string(self):
        key = generate_key()
        token = derive_auth_token(key)
        assert isinstance(token, str)
        # SHA-256 hex digest is 64 characters
        assert len(token) == 64
        # Should be valid hex
        int(token, 16)

    def test_deterministic(self):
        key = generate_key()
        assert derive_auth_token(key) == derive_auth_token(key)

    def test_different_keys_produce_different_tokens(self):
        key1 = generate_key()
        key2 = generate_key()
        assert derive_auth_token(key1) != derive_auth_token(key2)


class TestParseHexKeyInvalidChars:
    """Lines 128-129: _parse_hex_key() with invalid hex characters."""

    def test_non_hex_chars_raises_valueerror(self):
        # 64 characters but not valid hex (contains 'z' and 'g')
        bad_hex = "zzzzzzzz" * 8  # 64 chars, invalid hex
        with pytest.raises(ValueError, match="not valid hex"):
            _parse_hex_key(bad_hex, source="test")


# ---------------------------------------------------------------------------
# providers/base.py tests
# ---------------------------------------------------------------------------


class ConcreteProvider(TranslationProvider):
    """Minimal concrete subclass for testing non-abstract methods."""

    @property
    def provider_name(self):
        return "test"

    async def translate_batch(self, *a, **kw):
        pass

    async def health_check(self):
        return True

    async def close(self):
        pass

    async def get_available_models(self):
        return []


class TestTranslationProviderError:
    """Line 64: TranslationProviderError constructor attributes."""

    def test_attributes(self):
        err = TranslationProviderError(
            "something broke", provider="openai", retryable=True, status_code=500
        )
        assert str(err) == "something broke"
        assert err.message == "something broke"
        assert err.provider == "openai"
        assert err.retryable is True
        assert err.status_code == 500

    def test_defaults(self):
        err = TranslationProviderError("fail")
        assert err.provider == "unknown"
        assert err.retryable is False
        assert err.status_code is None


class TestRateLimitError:
    """Line 82: RateLimitError with retry_after attribute."""

    def test_retry_after(self):
        err = RateLimitError("slow down", provider="deepl", retry_after=30.0)
        assert err.retry_after == 30.0
        assert err.retryable is True
        assert err.status_code == 429
        assert err.provider == "deepl"

    def test_retry_after_default_none(self):
        err = RateLimitError("slow down")
        assert err.retry_after is None


class TestAuthenticationError:
    """Line 107: AuthenticationError with status_code=401."""

    def test_status_code_401(self):
        err = AuthenticationError("bad key", provider="openai")
        assert err.status_code == 401
        assert err.retryable is False
        assert err.provider == "openai"


class TestInvalidResponseError:
    """Lines 117, 122: InvalidResponseError with raw_response attribute."""

    def test_raw_response(self):
        err = InvalidResponseError("bad json", provider="openai", raw_response='{"garbage": true}')
        assert err.raw_response == '{"garbage": true}'
        assert err.retryable is True
        assert err.provider == "openai"

    def test_raw_response_default_none(self):
        err = InvalidResponseError("bad json")
        assert err.raw_response is None


class TestBuildSystemPrompt:
    """Lines 132, 144, 165-173: build_system_prompt with and without context."""

    def test_without_context(self):
        provider = ConcreteProvider()
        prompt = provider.build_system_prompt("Hungarian", "English")
        assert "Hungarian" in prompt
        assert "English" in prompt
        # No media title or type should appear
        assert "Media title:" not in prompt
        assert "Media type:" not in prompt

    def test_with_context_title_and_media_type(self):
        provider = ConcreteProvider()
        prompt = provider.build_system_prompt(
            "Hungarian", "English", "Breaking Bad", "Episode"
        )
        assert "Hungarian" in prompt
        assert "English" in prompt
        assert "Breaking Bad" in prompt
        assert "Episode" in prompt
        assert "Media title: Breaking Bad" in prompt
        assert "Media type: Episode" in prompt

    def test_with_title_only(self):
        provider = ConcreteProvider()
        prompt = provider.build_system_prompt("German", "English", context_title="Inception")
        assert "Media title: Inception" in prompt
        assert "Media type:" not in prompt

    def test_with_media_type_only(self):
        provider = ConcreteProvider()
        prompt = provider.build_system_prompt("German", "English", context_media_type="Movie")
        assert "Media title:" not in prompt
        assert "Media type: Movie" in prompt


class TestFormatInputForTranslation:
    """Lines 239-241: format_input_for_translation JSON formatting."""

    def test_json_output(self):
        import json

        provider = ConcreteProvider()
        lines = [
            {"index": "0", "content": "Hello"},
            {"index": "1", "content": "World"},
        ]
        result = provider.format_input_for_translation(lines)
        parsed = json.loads(result)
        assert parsed == lines

    def test_unicode_preserved(self):
        import json

        provider = ConcreteProvider()
        lines = [{"index": "0", "content": "Szeretem a csokoladét"}]
        result = provider.format_input_for_translation(lines)
        # ensure_ascii=False means unicode chars are kept as-is
        assert "csokoladét" in result
        parsed = json.loads(result)
        assert parsed == lines


class TestGetModelMetadata:
    """Line 144: get_model_metadata default returns None."""

    def test_returns_none(self):
        provider = ConcreteProvider()
        assert provider.get_model_metadata("gpt-4") is None
        assert provider.get_model_metadata("anything") is None
