"""Tests for OpenRouterProvider._parse_translations JSON parsing."""

from unittest.mock import patch

import pytest

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


class TestParseTranslationsControlCharacters:
    """Models can return unescaped control characters in multi-line cue text."""

    def test_raw_newline_preserved(self, provider):
        content = '[{"index":"0","content":"Line1\nLine2"}]'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Line1\nLine2"}]

    def test_raw_tab_preserved(self, provider):
        content = '[{"index":"0","content":"Hello\tworld"}]'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Hello\tworld"}]

    def test_raw_newline_with_duplicate_keys(self, provider):
        content = '{"index":"0","content":"Line1\nLine2","index":"1","content":"Line3"}'
        result = provider._parse_translations(content)
        assert result == [
            {"index": "0", "content": "Line1\nLine2"},
            {"index": "1", "content": "Line3"},
        ]

    def test_markdown_fence_with_raw_newline(self, provider):
        content = '```json\n[{"index":"0","content":"Line1\nLine2"}]\n```'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Line1\nLine2"}]


class TestParseTranslationsOtherControlCharacters:
    """Permissive parsing is for line breaks and tabs, not for arbitrary control bytes."""

    def test_other_control_characters_are_stripped(self, provider):
        content = '[{"index":"0","content":"Sz\x00ia\x1b[0m \x08ok"}]'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Szia[0m ok"}]

    def test_carriage_return_is_preserved(self, provider):
        content = '[{"index":"0","content":"Line1\r\nLine2"}]'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Line1\r\nLine2"}]


class TestParseTranslationsInvalidEscapes:
    """Models escape characters JSON does not allow escaping, most often an apostrophe."""

    def test_escaped_apostrophe_loses_the_backslash(self, provider):
        content = '[{"index":"0","content":"Don\\\'t stop"}]'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Don't stop"}]

    def test_escaped_backslash_pair_is_preserved(self, provider):
        content = '[{"index":"0","content":"C:\\\\path"}]'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "C:\\path"}]

    def test_valid_escapes_are_untouched(self, provider):
        content = '[{"index":"0","content":"Line1\\nLine2 \\"quoted\\" a\\/b \\u00e9"}]'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": 'Line1\nLine2 "quoted" a/b \u00e9'}]

    def test_invalid_escape_with_raw_newline(self, provider):
        content = '[{"index":"0","content":"Don\\\'t\nstop"}]'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "Don't\nstop"}]

    def test_escaped_backslash_before_u_is_preserved(self, provider):
        # An escaped backslash followed by a plain "u" is not a broken Unicode escape;
        # reading its second half as one corrupted the text before.
        content = '[{"index":"0","content":"C:\\\\users\\\\me"}]'
        result = provider._parse_translations(content)
        assert result == [{"index": "0", "content": "C:\\users\\me"}]


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
        pairs = ",".join(f'"index":"{i}","content":"Line {i}"' for i in range(10))
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

    def test_broken_unicode_escape_sanitized(self, provider):
        """Models sometimes produce invalid \\uXXXX escapes that break json.loads."""
        # Simulate the exact error from the logs: Invalid \uXXXX escape
        content = '[{"index":"0","content":"Hello \\uZZZZ world"},{"index":"1","content":"Good"}]'
        result = provider._parse_translations(content)
        assert len(result) == 2
        assert result[0]["index"] == "0"
        assert result[1]["content"] == "Good"

    def test_truncated_unicode_escape_sanitized(self, provider):
        """Truncated \\u sequence at end of string."""
        content = '[{"index":"0","content":"Test \\u00e"},{"index":"1","content":"OK"}]'
        result = provider._parse_translations(content)
        assert len(result) >= 1

    def test_valid_unicode_escape_preserved(self, provider):
        """Valid \\uXXXX escapes must not be stripped."""
        content = '[{"index":"0","content":"caf\\u00e9"}]'
        result = provider._parse_translations(content)
        assert result[0]["content"] == "caf\u00e9"

    def test_invalid_json_raises_error(self, provider):
        from subtitle_translator.providers.base import InvalidResponseError

        with pytest.raises(InvalidResponseError):
            provider._parse_translations("this is not json at all")
