"""Tests for subtitle_translator.config module."""

import os
from datetime import timedelta
from unittest.mock import patch

import pytest

from subtitle_translator.config import (
    Settings,
    clear_runtime_overrides,
    get_runtime_overrides,
    get_settings,
    reset_settings,
    update_runtime_config,
)


@pytest.fixture(autouse=True)
def clean_settings():
    """Reset global state before and after each test."""
    reset_settings()
    yield
    reset_settings()


# ---------------------------------------------------------------------------
# Settings.get_openrouter_headers
# ---------------------------------------------------------------------------


class TestGetOpenrouterHeaders:
    def test_uses_configured_key_by_default(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "configured-key")
        s = Settings()
        headers = s.get_openrouter_headers()
        assert headers["Authorization"] == "Bearer configured-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["X-Title"] == s.app_name

    def test_api_key_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "configured-key")
        s = Settings()
        headers = s.get_openrouter_headers(api_key_override="override-key")
        assert headers["Authorization"] == "Bearer override-key"

    def test_includes_http_referer_when_app_url_set(self):
        s = Settings(app_url="https://example.com")
        headers = s.get_openrouter_headers()
        assert headers["HTTP-Referer"] == "https://example.com"

    def test_omits_http_referer_when_app_url_none(self):
        s = Settings(app_url=None)
        headers = s.get_openrouter_headers()
        assert "HTTP-Referer" not in headers

    def test_omits_http_referer_when_app_url_empty(self):
        s = Settings(app_url="")
        headers = s.get_openrouter_headers()
        assert "HTTP-Referer" not in headers


# ---------------------------------------------------------------------------
# Settings.openrouter_headers (legacy property)
# ---------------------------------------------------------------------------


class TestOpenrouterHeadersProperty:
    def test_returns_same_as_get_openrouter_headers(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "prop-key")
        s = Settings()
        assert s.openrouter_headers == s.get_openrouter_headers()
        assert s.openrouter_headers["Authorization"] == "Bearer prop-key"


# ---------------------------------------------------------------------------
# Settings.is_rtl_language
# ---------------------------------------------------------------------------


class TestIsRtlLanguage:
    @pytest.mark.parametrize("code", ["ar", "he", "fa", "ur", "yi", "ps", "sd", "ku"])
    def test_rtl_base_codes(self, code):
        s = Settings()
        assert s.is_rtl_language(code) is True

    def test_hyphenated_code(self):
        s = Settings()
        assert s.is_rtl_language("ar-SA") is True

    def test_underscore_code(self):
        s = Settings()
        assert s.is_rtl_language("fa_IR") is True

    def test_case_insensitive(self):
        s = Settings()
        assert s.is_rtl_language("AR") is True
        assert s.is_rtl_language("He") is True

    def test_non_rtl_language(self):
        s = Settings()
        assert s.is_rtl_language("en") is False
        assert s.is_rtl_language("fr-FR") is False


# ---------------------------------------------------------------------------
# Settings.job_queue_ttl
# ---------------------------------------------------------------------------


class TestJobQueueTtl:
    def test_returns_timedelta(self):
        s = Settings(job_queue_ttl_hours=3)
        assert s.job_queue_ttl == timedelta(hours=3)

    def test_default_value(self):
        s = Settings()
        assert s.job_queue_ttl == timedelta(hours=1)


# ---------------------------------------------------------------------------
# get_settings
# ---------------------------------------------------------------------------


class TestGetSettings:
    def test_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_returns_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_applies_runtime_overrides(self):
        update_runtime_config("batch_size", 42)
        s = get_settings()
        assert s.batch_size == 42

    def test_caches_overridden_settings(self):
        update_runtime_config("batch_size", 42)
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


# ---------------------------------------------------------------------------
# update_runtime_config
# ---------------------------------------------------------------------------


class TestUpdateRuntimeConfig:
    def test_valid_key_updates(self):
        update_runtime_config("batch_size", 50)
        s = get_settings()
        assert s.batch_size == 50

    def test_invalid_key_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid configuration key"):
            update_runtime_config("nonexistent_key", "value")

    def test_clears_cached_overridden_settings(self):
        update_runtime_config("batch_size", 10)
        s1 = get_settings()
        update_runtime_config("batch_size", 20)
        s2 = get_settings()
        assert s1 is not s2
        assert s2.batch_size == 20


# ---------------------------------------------------------------------------
# get_runtime_overrides
# ---------------------------------------------------------------------------


class TestGetRuntimeOverrides:
    def test_masks_key_fields(self):
        update_runtime_config("openrouter_api_key", "secret-value")
        overrides = get_runtime_overrides()
        assert overrides["openrouter_api_key"] == "***"

    def test_masks_empty_sensitive_value_as_none(self):
        update_runtime_config("openrouter_api_key", "")
        overrides = get_runtime_overrides()
        assert overrides["openrouter_api_key"] is None

    def test_masks_encryption_key(self):
        update_runtime_config("encryption_key", "hex-key-value")
        overrides = get_runtime_overrides()
        assert overrides["encryption_key"] == "***"

    def test_masks_admin_api_key(self):
        update_runtime_config("admin_api_key", "admin-secret")
        overrides = get_runtime_overrides()
        assert overrides["admin_api_key"] == "***"

    def test_returns_plain_value_for_non_sensitive(self):
        update_runtime_config("batch_size", 99)
        overrides = get_runtime_overrides()
        assert overrides["batch_size"] == 99

    def test_empty_when_no_overrides(self):
        overrides = get_runtime_overrides()
        assert overrides == {}


# ---------------------------------------------------------------------------
# clear_runtime_overrides
# ---------------------------------------------------------------------------


class TestClearRuntimeOverrides:
    def test_clears_overrides(self):
        update_runtime_config("batch_size", 77)
        clear_runtime_overrides()
        overrides = get_runtime_overrides()
        assert overrides == {}

    def test_settings_revert_to_defaults(self):
        default_batch = get_settings().batch_size
        update_runtime_config("batch_size", 77)
        assert get_settings().batch_size == 77
        clear_runtime_overrides()
        assert get_settings().batch_size == default_batch


# ---------------------------------------------------------------------------
# reset_settings
# ---------------------------------------------------------------------------


class TestResetSettings:
    def test_clears_singleton(self):
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2

    def test_clears_overrides(self):
        update_runtime_config("batch_size", 55)
        reset_settings()
        overrides = get_runtime_overrides()
        assert overrides == {}

    def test_clears_overridden_settings(self):
        update_runtime_config("batch_size", 55)
        _ = get_settings()  # populate _overridden_settings cache
        reset_settings()
        s = get_settings()
        assert s.batch_size != 55 or s.batch_size == Settings().batch_size
