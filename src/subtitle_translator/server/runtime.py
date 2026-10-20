"""Process-wide translator for the standalone service. Reads the environment-backed Settings."""

import asyncio

from subtitle_translator.backends.openrouter import OpenRouterProvider
from subtitle_translator.core.translator import SubtitleTranslator
from subtitle_translator.server.config import Settings, get_settings

_translator: SubtitleTranslator | None = None
_translator_source: Settings | None = None
_translator_lock = asyncio.Lock()


async def get_translator() -> SubtitleTranslator:
    """The shared translator, rebuilt when runtime configuration changes the settings.

    The provider's connection pool and the learned batch sizes survive a rebuild; only
    the settings snapshot changes, as in 2.1.1.
    """
    global _translator, _translator_source
    async with _translator_lock:
        settings = get_settings()
        if _translator is None:
            core = settings.core()
            _translator = SubtitleTranslator(provider=OpenRouterProvider(core), settings=core)
            _translator_source = settings
        elif _translator_source is not settings:
            core = settings.core()
            provider = _translator.provider
            if isinstance(provider, OpenRouterProvider):
                provider = provider.with_settings(core)
            rebuilt = SubtitleTranslator(provider=provider, settings=core)
            rebuilt.resolver = _translator.resolver
            _translator, _translator_source = rebuilt, settings
    return _translator


async def close_translator() -> None:
    """Close the shared translator."""
    global _translator, _translator_source
    if _translator is not None:
        await _translator.close()
        _translator = None
        _translator_source = None
