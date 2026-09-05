"""Extended tests for translator.py to cover missing lines."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from subtitle_translator.api.models import (
    SubtitleLine,
    TranslateContentRequest,
    TranslationConfig,
)
from subtitle_translator.core.batch_processor import (
    BatchProcessingResult,
    BatchProgress,
    BatchResult,
)
from subtitle_translator.core.translator import (
    SubtitleTranslator,
    close_translator,
    get_translator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    settings = MagicMock()
    settings.batch_size = 100
    settings.max_retries = 3
    settings.retry_delay = 0.1
    settings.openrouter_default_model = "test-model"
    settings.is_rtl_language = MagicMock(return_value=False)
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def _make_provider():
    provider = AsyncMock()
    provider.provider_name = "mock"
    provider.close = AsyncMock()
    provider.get_available_models = AsyncMock(return_value=[{"id": "m1"}])
    provider.health_check = AsyncMock(return_value=True)
    provider.get_model_metadata = MagicMock(return_value=None)
    return provider


def _make_batch_result(success=True, batch_index=0, translations=None, error=None):
    return BatchResult(
        batch_index=batch_index,
        success=success,
        translations=translations or [],
        tokens_used=50 if success else 0,
        error=error,
    )


SIMPLE_SRT = """\
1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:00:08,000
How are you?
"""


# ---------------------------------------------------------------------------
# provider property: lazy creation (line 67)
# ---------------------------------------------------------------------------


class TestProviderLazyCreation:
    def test_provider_creates_openrouter_when_none(self):
        """Accessing .provider without injecting one creates OpenRouterProvider."""
        settings = _make_settings()
        translator = SubtitleTranslator(provider=None, settings=settings)

        with patch("subtitle_translator.core.translator.OpenRouterProvider") as MockOR:
            mock_instance = MagicMock()
            MockOR.return_value = mock_instance

            result = translator.provider

            MockOR.assert_called_once_with(settings)
            assert result is mock_instance

    def test_provider_reuses_existing(self):
        """If a provider was injected, the property returns it directly."""
        provider = _make_provider()
        translator = SubtitleTranslator(provider=provider, settings=_make_settings())
        assert translator.provider is provider


# ---------------------------------------------------------------------------
# translate_content: partial failure WITH translations (lines 121-135)
# ---------------------------------------------------------------------------


class TestTranslateContentPartialFailure:
    @pytest.mark.asyncio
    async def test_partial_failure_with_translations(self):
        provider = _make_provider()
        settings = _make_settings()
        translator = SubtitleTranslator(provider=provider, settings=settings)

        request = TranslateContentRequest(
            sourceLanguage="en",
            targetLanguage="hu",
            lines=[
                SubtitleLine(position=1, line="Hello"),
                SubtitleLine(position=2, line="World"),
            ],
        )

        partial_translations = [{"index": "1", "content": "Szia"}]

        mock_result = BatchProcessingResult(
            all_translations=partial_translations,
            total_tokens=50,
            model_used="test-model",
            batch_results=[
                _make_batch_result(success=True, batch_index=0, translations=partial_translations),
                _make_batch_result(success=False, batch_index=1, error="API error"),
            ],
            progress=BatchProgress(total_batches=2, completed_batches=1, failed_batches=1),
        )

        with patch("subtitle_translator.core.translator.BatchProcessor") as MockBP:
            MockBP.return_value.process_all_batches = AsyncMock(return_value=mock_result)

            result = await translator.translate_content(request)

        assert result.success is False
        assert "Partial failure" in result.error
        assert len(result.lines) > 0
        assert result.tokens_used == 50

    # translate_content: partial failure with NO translations (lines 136-143)
    @pytest.mark.asyncio
    async def test_total_failure_no_translations(self):
        provider = _make_provider()
        settings = _make_settings()
        translator = SubtitleTranslator(provider=provider, settings=settings)

        request = TranslateContentRequest(
            sourceLanguage="en",
            targetLanguage="hu",
            lines=[SubtitleLine(position=1, line="Hello")],
        )

        # The failed attempts were billed; the batch processor reports their usage.
        mock_result = BatchProcessingResult(
            all_translations=[],
            total_tokens=30,
            model_used="test-model",
            batch_results=[
                _make_batch_result(success=False, batch_index=0, error="Total failure"),
            ],
            progress=BatchProgress(total_batches=1, failed_batches=1),
        )

        with patch("subtitle_translator.core.translator.BatchProcessor") as MockBP:
            MockBP.return_value.process_all_batches = AsyncMock(return_value=mock_result)

            result = await translator.translate_content(request)

        assert result.success is False
        assert result.lines == []
        assert result.tokens_used == 30
        assert "Total failure" in result.error


# ---------------------------------------------------------------------------
# translate_content: exception path (lines 156-158)
# ---------------------------------------------------------------------------


class TestTranslateContentException:
    @pytest.mark.asyncio
    async def test_exception_returns_error_result(self):
        provider = _make_provider()
        settings = _make_settings()
        translator = SubtitleTranslator(provider=provider, settings=settings)

        request = TranslateContentRequest(
            sourceLanguage="en",
            targetLanguage="hu",
            lines=[SubtitleLine(position=1, line="Hello")],
        )

        with patch("subtitle_translator.core.translator.BatchProcessor") as MockBP:
            MockBP.return_value.process_all_batches = AsyncMock(
                side_effect=RuntimeError("connection died")
            )

            result = await translator.translate_content(request)

        assert result.success is False
        assert result.lines == []
        assert "connection died" in result.error


# ---------------------------------------------------------------------------
# translate_file: model from config_override (line 192)
# ---------------------------------------------------------------------------


class TestTranslateFileConfigOverride:
    @pytest.mark.asyncio
    async def test_model_from_config_override(self):
        provider = _make_provider()
        settings = _make_settings()
        translator = SubtitleTranslator(provider=provider, settings=settings)

        config = TranslationConfig(model="override-model")

        # Force an exception so we can inspect the model_used in the error result
        # without needing to mock the full happy path.
        with patch("subtitle_translator.core.translator.BatchProcessor") as MockBP:
            MockBP.side_effect = RuntimeError("boom")

            result = await translator.translate_file(
                content=SIMPLE_SRT,
                source_language="en",
                target_language="es",
                config_override=config,
            )

        assert result.model_used == "override-model"
        assert result.success is False


# ---------------------------------------------------------------------------
# translate_file: empty entries (line 200)
# ---------------------------------------------------------------------------


class TestTranslateFileEmptyEntries:
    @pytest.mark.asyncio
    async def test_empty_srt_returns_early(self):
        provider = _make_provider()
        settings = _make_settings()
        translator = SubtitleTranslator(provider=provider, settings=settings)

        with patch.object(translator, "_srt_parser") as mock_parser:
            mock_parser.parse.return_value = []

            result = await translator.translate_file(
                content="",
                source_language="en",
                target_language="es",
            )

        assert result.success is True
        assert result.subtitle_count == 0


# ---------------------------------------------------------------------------
# translate_file: partial failure paths (lines 224-244)
# ---------------------------------------------------------------------------


class TestTranslateFilePartialFailure:
    @pytest.mark.asyncio
    async def test_partial_failure_with_translations(self):
        provider = _make_provider()
        settings = _make_settings()
        translator = SubtitleTranslator(provider=provider, settings=settings)

        partial_translations = [{"index": "1", "content": "Hola mundo"}]

        mock_result = BatchProcessingResult(
            all_translations=partial_translations,
            total_tokens=80,
            model_used="test-model",
            batch_results=[
                _make_batch_result(success=True, batch_index=0, translations=partial_translations),
                _make_batch_result(success=False, batch_index=1, error="timeout"),
            ],
            progress=BatchProgress(total_batches=2, completed_batches=1, failed_batches=1),
        )

        with patch("subtitle_translator.core.translator.BatchProcessor") as MockBP:
            MockBP.return_value.process_all_batches = AsyncMock(return_value=mock_result)

            result = await translator.translate_file(
                content=SIMPLE_SRT,
                source_language="en",
                target_language="es",
            )

        assert result.success is False
        assert "Partial failure" in result.error
        assert result.subtitle_count == 2
        assert result.tokens_used == 80
        assert result.content  # non-empty partial content

    @pytest.mark.asyncio
    async def test_total_failure_no_translations(self):
        provider = _make_provider()
        settings = _make_settings()
        translator = SubtitleTranslator(provider=provider, settings=settings)

        mock_result = BatchProcessingResult(
            all_translations=[],
            total_tokens=30,
            model_used="test-model",
            batch_results=[
                _make_batch_result(success=False, batch_index=0, error="catastrophic"),
            ],
            progress=BatchProgress(total_batches=1, failed_batches=1),
        )

        with patch("subtitle_translator.core.translator.BatchProcessor") as MockBP:
            MockBP.return_value.process_all_batches = AsyncMock(return_value=mock_result)

            result = await translator.translate_file(
                content=SIMPLE_SRT,
                source_language="en",
                target_language="es",
            )

        assert result.success is False
        assert result.tokens_used == 30
        assert result.content == ""
        assert "catastrophic" in result.error


# ---------------------------------------------------------------------------
# _add_rtl_markers delegation (line 294)
# ---------------------------------------------------------------------------


class TestAddRtlMarkers:
    def test_add_rtl_markers_delegates(self):
        translator = SubtitleTranslator(provider=_make_provider(), settings=_make_settings())
        result = translator._add_rtl_markers("some text")
        # add_rtl_markers wraps text with RTL embedding characters
        assert "\u202b" in result or "\u200f" in result


# ---------------------------------------------------------------------------
# get_available_models and health_check delegation (lines 298, 331)
# ---------------------------------------------------------------------------


class TestDelegationMethods:
    @pytest.mark.asyncio
    async def test_get_available_models(self):
        provider = _make_provider()
        translator = SubtitleTranslator(provider=provider, settings=_make_settings())

        models = await translator.get_available_models()

        provider.get_available_models.assert_awaited_once()
        assert models == [{"id": "m1"}]

    @pytest.mark.asyncio
    async def test_health_check(self):
        provider = _make_provider()
        translator = SubtitleTranslator(provider=provider, settings=_make_settings())

        ok = await translator.health_check()

        provider.health_check.assert_awaited_once()
        assert ok is True


# ---------------------------------------------------------------------------
# close_translator global helper (lines 356-358)
# ---------------------------------------------------------------------------


class TestCloseTranslator:
    @pytest.mark.asyncio
    async def test_close_translator_resets_instance(self):
        """close_translator() should close the provider and clear the global."""
        import subtitle_translator.core.translator as mod

        mock_translator = AsyncMock(spec=SubtitleTranslator)
        mock_translator.close = AsyncMock()

        # Inject a fake global instance
        mod._translator_instance = mock_translator

        await close_translator()

        mock_translator.close.assert_awaited_once()
        assert mod._translator_instance is None

    @pytest.mark.asyncio
    async def test_close_translator_noop_when_none(self):
        """close_translator() is safe to call when no instance exists."""
        import subtitle_translator.core.translator as mod

        mod._translator_instance = None
        await close_translator()  # should not raise
        assert mod._translator_instance is None


# ---------------------------------------------------------------------------
# map_translations_to_lines RTL branch (line 331)
# ---------------------------------------------------------------------------


class TestMapTranslationsRTL:
    @pytest.mark.asyncio
    async def test_content_translation_with_rtl_language(self):
        """translate_content with an RTL target language triggers add_rtl_markers."""
        provider = _make_provider()
        settings = _make_settings()
        settings.is_rtl_language = MagicMock(return_value=True)
        translator = SubtitleTranslator(provider=provider, settings=settings)

        request = TranslateContentRequest(
            sourceLanguage="en",
            targetLanguage="ar",
            lines=[SubtitleLine(position=1, line="Hello")],
        )

        mock_result = BatchProcessingResult(
            all_translations=[{"index": "1", "content": "مرحبا"}],
            total_tokens=30,
            model_used="test-model",
            batch_results=[
                _make_batch_result(
                    success=True, batch_index=0, translations=[{"index": "1", "content": "مرحبا"}]
                ),
            ],
            progress=BatchProgress(total_batches=1, completed_batches=1),
        )

        with patch("subtitle_translator.core.translator.BatchProcessor") as MockBP:
            MockBP.return_value.process_all_batches = AsyncMock(return_value=mock_result)

            result = await translator.translate_content(request)

        assert result.success is True
        # RTL markers should be present in the translated text
        assert "\u202b" in result.lines[0].line or "\u200f" in result.lines[0].line


# ---------------------------------------------------------------------------
# get_translator global helper (lines 346-350)
# ---------------------------------------------------------------------------


class TestGetTranslator:
    @pytest.mark.asyncio
    async def test_get_translator_creates_instance(self):
        """get_translator() creates a SubtitleTranslator when none exists."""
        import subtitle_translator.core.translator as mod

        mod._translator_instance = None

        with patch("subtitle_translator.core.translator.SubtitleTranslator") as MockST:
            mock_inst = MagicMock()
            MockST.return_value = mock_inst

            result = await get_translator()

            MockST.assert_called_once()
            assert result is mock_inst

        # Clean up
        mod._translator_instance = None

    @pytest.mark.asyncio
    async def test_get_translator_returns_existing(self):
        """get_translator() returns the existing instance if already set."""
        import subtitle_translator.core.translator as mod

        sentinel = MagicMock()
        mod._translator_instance = sentinel

        result = await get_translator()
        assert result is sentinel

        # Clean up
        mod._translator_instance = None
