"""Tests for OpenRouterProvider._parse_translations JSON parsing."""

import pytest
from unittest.mock import patch

from subtitle_translator.providers.openrouter import OpenRouterProvider


@pytest.fixture
def provider():
    """Create provider instance for testing."""
    with patch("subtitle_translator.providers.openrouter.get_settings") as mock_settings:
        settings = mock_settings.return_value
        settings.openrouter_api_key = "test-key"
        settings.openrouter_model = "test/model"
        settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        settings.target_language = "Hungarian"
        settings.batch_size = 10
        settings.openrouter_timeout = 60
        settings.openrouter_max_retries = 3
        settings.openrouter_rate_limit_delay = 1.0
        settings.parallel_batch_size = 3
        p = OpenRouterProvider()
        return p


class TestParseTranslationsValidJSON:
    """Tests for well-formed JSON responses."""

    def test_valid_json_array(self, provider):
        result = provider._parse_translations('[{"index":"0","content":"Szia"}]')
        assert result == [{"index": "0", "content": "Szia"}]

    def test_valid_json_array_multiple(self, provider):
        content = '[{"index":"0","content":"Szia"},{"index":"1","content":"Világ"}]'
        result = provider._parse_translations(content)
        assert len(result) == 2
        assert result[0] == {"index": "0", "content": "Szia"}
        assert result[1] == {"index": "1", "content": "Világ"}

    def test_single_dict_wrapped_in_list(self, provider):
        result = provider._parse_translations('{"index":"0","content":"Szia"}')
        assert result == [{"index": "0", "content": "Szia"}]

    def test_wrapped_in_translations_key(self, provider):
        content = '{"translations":[{"index":"0","content":"Szia"}]}'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Szia"}]

    def test_alternative_key_names(self, provider):
        content = '[{"idx":"0","text":"Szia"}]'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Szia"}]

    def test_integer_index_values(self, provider):
        content = '[{"index":0,"content":"Szia"},{"index":1,"content":"Világ"}]'
        result = provider._parse_translations(content)
        assert result[0] == {"index": "0", "content": "Szia"}
        assert result[1] == {"index": "1", "content": "Világ"}

    def test_content_with_escaped_characters(self, provider):
        content = '[{"index":"0","content":"Line1\\nLine2"}]'
        result = provider._parse_translations(content)
        assert result[0]["content"] == "Line1\nLine2"

    def test_content_with_unicode(self, provider):
        content = '[{"index":"0","content":"Héllo wörld"}]'
        result = provider._parse_translations(content)
        assert result[0]["content"] == "Héllo wörld"


class TestParseTranslationsDuplicateKeys:
    """Tests for malformed JSON with duplicate keys (the bug this PR fixes)."""

    def test_duplicate_keys_two_pairs(self, provider):
        """The core bug: json.loads() silently keeps only the last duplicate key."""
        content = '{"index":"0","content":"Szia","index":"1","content":"Világ"}'
        result = provider._parse_translations(content)
        assert len(result) == 2
        assert result[0] == {"index": "0", "content": "Szia"}
        assert result[1] == {"index": "1", "content": "Világ"}

    def test_duplicate_keys_many_pairs(self, provider):
        """Realistic batch size with duplicate keys."""
        pairs = ",".join(
            f'"index":"{i}","content":"Line {i}"' for i in range(10)
        )
        content = "{" + pairs + "}"
        result = provider._parse_translations(content)
        assert len(result) == 10
        for i in range(10):
            assert result[i] == {"index": str(i), "content": f"Line {i}"}

    def test_duplicate_keys_with_integer_index(self, provider):
        """Duplicate keys where index is an integer, not a string."""
        content = '{"index":0,"content":"Szia","index":1,"content":"Világ"}'
        result = provider._parse_translations(content)
        assert len(result) == 2
        assert result[0] == {"index": "0", "content": "Szia"}
        assert result[1] == {"index": "1", "content": "Világ"}

    def test_duplicate_keys_with_escaped_content(self, provider):
        """Duplicate keys where content has escape sequences."""
        content = '{"index":"0","content":"She said \\"hello\\"","index":"1","content":"OK"}'
        result = provider._parse_translations(content)
        assert len(result) == 2
        assert result[0]["content"] == 'She said "hello"'
        assert result[1]["content"] == "OK"

    def test_duplicate_keys_with_newlines(self, provider):
        content = '{"index":"0","content":"Line1\\nLine2","index":"1","content":"Line3"}'
        result = provider._parse_translations(content)
        assert len(result) == 2
        assert result[0]["content"] == "Line1\nLine2"

    def test_single_pair_no_duplicates(self, provider):
        """A single object with no duplicate keys should still work."""
        content = '{"index":"0","content":"Szia"}'
        result = provider._parse_translations(content)
        assert len(result) == 1
        assert result[0] == {"index": "0", "content": "Szia"}


class TestParseTranslationsFallbacks:
    """Tests for markdown and other fallback parsing."""

    def test_markdown_code_block(self, provider):
        content = '```json\n[{"index":"0","content":"Szia"}]\n```'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Szia"}]

    def test_markdown_code_block_no_lang(self, provider):
        content = '```\n[{"index":"0","content":"Szia"}]\n```'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Szia"}]

    def test_json_array_in_surrounding_text(self, provider):
        content = 'Here are the translations: [{"index":"0","content":"Szia"}] Hope that helps!'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Szia"}]

    def test_invalid_json_raises_error(self, provider):
        from subtitle_translator.providers.base import InvalidResponseError
        with pytest.raises(InvalidResponseError):
            provider._parse_translations("this is not json at all")
