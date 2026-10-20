"""CoreSettings is the env-free subset of Settings that the core reads."""

import os

import pytest
from pydantic import ValidationError

from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.batch_sizing import BatchSizeResolver
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.translator import SubtitleTranslator
from subtitle_translator.server.config import Settings


def test_core_fields_are_a_subset_of_settings_with_equal_defaults(monkeypatch):
    # Settings reads the process environment; the defaults are what this test compares,
    # so an exported OPENROUTER_API_KEY or BATCH_SIZE must not leak in.
    monkeypatch.setattr(os, "environ", {})
    core_defaults = CoreSettings().model_dump()
    server_defaults = Settings(_env_file=None).model_dump()
    for name, value in core_defaults.items():
        assert name in server_defaults, name
        assert server_defaults[name] == value, name


def test_settings_core_copies_values():
    core = Settings(_env_file=None, batch_size=7, openrouter_api_key="k").core()
    assert isinstance(core, CoreSettings)
    assert core.batch_size == 7 and core.openrouter_api_key == "k"


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        CoreSettings(job_queue_max_concurrent=3)


def test_headers_and_rtl():
    core = CoreSettings(openrouter_api_key="abc", app_url=None)
    assert core.get_openrouter_headers()["Authorization"] == "Bearer abc"
    assert "HTTP-Referer" not in core.get_openrouter_headers()
    assert core.get_openrouter_headers(api_key_override="z")["Authorization"] == "Bearer z"
    assert core.is_rtl_language("he-IL") is True and core.is_rtl_language("en") is False


def test_each_processor_has_its_own_resolver_unless_one_is_shared():
    settings = CoreSettings(batch_size=50)
    first = BatchProcessor(provider=object(), settings=settings)
    second = BatchProcessor(provider=object(), settings=settings)
    assert first.resolver is not second.resolver
    shared = BatchSizeResolver(settings)
    assert BatchProcessor(provider=object(), settings=settings, resolver=shared).resolver is shared


def test_a_translator_shares_one_resolver_across_its_jobs():
    translator = SubtitleTranslator(provider=object(), settings=CoreSettings())
    assert isinstance(translator.resolver, BatchSizeResolver)


def test_a_translator_without_provider_says_so():
    with pytest.raises(RuntimeError, match="needs a provider"):
        _ = SubtitleTranslator(settings=CoreSettings()).provider
