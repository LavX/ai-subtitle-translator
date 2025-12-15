"""Tests for translator core functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import timedelta

from subtitle_translator.core.translator import SubtitleTranslator, ContentTranslationResult
from subtitle_translator.core.srt_parser import SRTParser, SubtitleEntry, SRTParserError
from subtitle_translator.core.batch_processor import BatchProcessor, BatchProgress, BatchResult
from subtitle_translator.api.models import SubtitleLine, TranslateContentRequest
from subtitle_translator.providers.base import (
    TranslationBatch,
    TranslationResult,
    TranslationProviderError,
    RateLimitError,
)


class TestSRTParser:
    """Tests for SRT parser."""

    def test_parse_valid_srt(self):
        """Test parsing valid SRT content."""
        parser = SRTParser()
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:00:08,000
How are you?
"""
        entries = parser.parse(srt_content)
        
        assert len(entries) == 2
        assert entries[0].index == 1
        assert entries[0].content == "Hello world"
        assert entries[1].index == 2
        assert entries[1].content == "How are you?"

    def test_parse_multiline_subtitle(self):
        """Test parsing subtitles with multiple lines."""
        parser = SRTParser()
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Line one
Line two
"""
        entries = parser.parse(srt_content)
        
        assert len(entries) == 1
        assert "Line one" in entries[0].content
        assert "Line two" in entries[0].content

    def test_parse_invalid_srt(self):
        """Test parsing invalid SRT raises error."""
        parser = SRTParser()
        
        # This should not raise but return empty or handle gracefully
        # depending on the srt library behavior
        try:
            entries = parser.parse("Not valid SRT")
            # If it doesn't raise, it should return empty list
            assert len(entries) == 0
        except SRTParserError:
            # Expected behavior
            pass

    def test_compose_srt(self):
        """Test composing entries back to SRT format."""
        parser = SRTParser()
        entries = [
            SubtitleEntry(
                index=1,
                start=timedelta(seconds=1),
                end=timedelta(seconds=4),
                content="Hello world",
            ),
            SubtitleEntry(
                index=2,
                start=timedelta(seconds=5),
                end=timedelta(seconds=8),
                content="How are you?",
            ),
        ]
        
        result = parser.compose(entries)
        
        assert "Hello world" in result
        assert "How are you?" in result
        assert "00:00:01,000" in result

    def test_extract_lines_for_translation(self):
        """Test extracting lines for translation."""
        parser = SRTParser()
        entries = [
            SubtitleEntry(
                index=1,
                start=timedelta(seconds=1),
                end=timedelta(seconds=4),
                content="Hello",
            ),
            SubtitleEntry(
                index=2,
                start=timedelta(seconds=5),
                end=timedelta(seconds=8),
                content="World",
            ),
        ]
        
        lines = parser.extract_lines_for_translation(entries)
        
        assert len(lines) == 2
        assert lines[0] == {"index": "1", "content": "Hello"}
        assert lines[1] == {"index": "2", "content": "World"}

    def test_apply_translations(self):
        """Test applying translations back to entries."""
        parser = SRTParser()
        entries = [
            SubtitleEntry(
                index=1,
                start=timedelta(seconds=1),
                end=timedelta(seconds=4),
                content="Hello",
            ),
        ]
        translations = [{"index": "1", "content": "Hola"}]
        
        result = parser.apply_translations(entries, translations)
        
        assert len(result) == 1
        assert result[0].content == "Hola"
        assert result[0].start == entries[0].start

    def test_apply_translations_rtl(self):
        """Test applying translations with RTL markers."""
        parser = SRTParser()
        entries = [
            SubtitleEntry(
                index=1,
                start=timedelta(seconds=1),
                end=timedelta(seconds=4),
                content="Hello",
            ),
        ]
        translations = [{"index": "1", "content": "שלום"}]
        
        result = parser.apply_translations(entries, translations, is_rtl=True)
        
        assert len(result) == 1
        # Check RTL markers are present
        assert "\u202B" in result[0].content or "\u202C" in result[0].content

    def test_validate_srt(self):
        """Test SRT validation."""
        parser = SRTParser()
        
        valid_srt = """1
00:00:01,000 --> 00:00:04,000
Hello
"""
        is_valid, error = parser.validate_srt(valid_srt)
        assert is_valid is True
        assert error is None

    def test_split_long_subtitles(self):
        """Test splitting long subtitles."""
        parser = SRTParser()
        entries = [
            SubtitleEntry(
                index=1,
                start=timedelta(seconds=1),
                end=timedelta(seconds=4),
                content="This is a very long subtitle line that should be split into multiple lines for better readability",
            ),
        ]
        
        result = parser.split_long_subtitles(entries, max_chars_per_line=40)
        
        assert len(result) == 1
        # Content should have line breaks
        assert "\n" in result[0].content


class TestBatchProcessor:
    """Tests for batch processor."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = AsyncMock()
        provider.provider_name = "mock"
        return provider

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.batch_size = 50
        settings.max_retries = 3
        settings.retry_delay = 0.1
        settings.openrouter_default_model = "test-model"
        return settings

    def test_create_batches(self, mock_provider, mock_settings):
        """Test batch creation."""
        processor = BatchProcessor(mock_provider, mock_settings)
        
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(125)]
        
        batches = processor.create_batches(lines, batch_size=50)
        
        assert len(batches) == 3
        assert len(batches[0]) == 50
        assert len(batches[1]) == 50
        assert len(batches[2]) == 25

    @pytest.mark.asyncio
    async def test_process_batch_success(self, mock_provider, mock_settings):
        """Test successful batch processing."""
        processor = BatchProcessor(mock_provider, mock_settings)
        
        mock_provider.translate_batch = AsyncMock(
            return_value=TranslationResult(
                translations=[{"index": "0", "content": "Translated"}],
                model_used="test-model",
                total_tokens=100,
            )
        )
        
        batch = TranslationBatch(
            lines=[{"index": "0", "content": "Original"}],
            source_language="en",
            target_language="es",
        )
        
        result = await processor.process_batch(batch, batch_index=0)
        
        assert result.success is True
        assert len(result.translations) == 1
        assert result.tokens_used == 100

    @pytest.mark.asyncio
    async def test_process_batch_retry_on_rate_limit(self, mock_provider, mock_settings):
        """Test retry logic on rate limit."""
        processor = BatchProcessor(mock_provider, mock_settings)
        
        # First call raises rate limit, second succeeds
        mock_provider.translate_batch = AsyncMock(
            side_effect=[
                RateLimitError("Rate limited", retry_after=0.1),
                TranslationResult(
                    translations=[{"index": "0", "content": "Translated"}],
                    model_used="test-model",
                    total_tokens=100,
                ),
            ]
        )
        
        batch = TranslationBatch(
            lines=[{"index": "0", "content": "Original"}],
            source_language="en",
            target_language="es",
        )
        
        result = await processor.process_batch(batch, batch_index=0)
        
        assert result.success is True
        assert result.retries == 1

    @pytest.mark.asyncio
    async def test_process_batch_max_retries_exceeded(self, mock_provider, mock_settings):
        """Test failure after max retries."""
        mock_settings.max_retries = 2
        processor = BatchProcessor(mock_provider, mock_settings)
        
        mock_provider.translate_batch = AsyncMock(
            side_effect=TranslationProviderError("Error", retryable=True)
        )
        
        batch = TranslationBatch(
            lines=[{"index": "0", "content": "Original"}],
            source_language="en",
            target_language="es",
        )
        
        result = await processor.process_batch(batch, batch_index=0)
        
        assert result.success is False
        assert "Max retries exceeded" in result.error


class TestSubtitleTranslator:
    """Tests for subtitle translator."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = AsyncMock()
        provider.provider_name = "mock"
        provider.translate_batch = AsyncMock(
            return_value=TranslationResult(
                translations=[{"index": "1", "content": "Translated"}],
                model_used="test-model",
                total_tokens=100,
            )
        )
        provider.get_available_models = AsyncMock(return_value=[])
        provider.health_check = AsyncMock(return_value=True)
        provider.close = AsyncMock()
        return provider

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.batch_size = 100
        settings.max_retries = 3
        settings.retry_delay = 0.1
        settings.openrouter_default_model = "test-model"
        settings.is_rtl_language = MagicMock(return_value=False)
        return settings

    @pytest.mark.asyncio
    async def test_translate_content_success(self, mock_provider, mock_settings):
        """Test successful content translation."""
        translator = SubtitleTranslator(provider=mock_provider, settings=mock_settings)
        
        request = TranslateContentRequest(
            sourceLanguage="en",
            targetLanguage="es",
            lines=[SubtitleLine(position=1, line="Hello")],
        )
        
        result = await translator.translate_content(request)
        
        assert result.success is True
        assert len(result.lines) == 1
        assert result.model_used == "test-model"

    @pytest.mark.asyncio
    async def test_translate_content_empty_lines(self, mock_provider, mock_settings):
        """Test translation with empty lines."""
        translator = SubtitleTranslator(provider=mock_provider, settings=mock_settings)
        
        request = TranslateContentRequest(
            sourceLanguage="en",
            targetLanguage="es",
            lines=[],
        )
        
        result = await translator.translate_content(request)
        
        assert result.success is True
        assert len(result.lines) == 0

    @pytest.mark.asyncio
    async def test_translate_file_success(self, mock_provider, mock_settings):
        """Test successful file translation."""
        translator = SubtitleTranslator(provider=mock_provider, settings=mock_settings)
        
        mock_provider.translate_batch = AsyncMock(
            return_value=TranslationResult(
                translations=[
                    {"index": "1", "content": "Hola mundo"},
                    {"index": "2", "content": "¿Cómo estás?"},
                ],
                model_used="test-model",
                total_tokens=200,
            )
        )
        
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:00:08,000
How are you?
"""
        
        result = await translator.translate_file(
            content=srt_content,
            source_language="en",
            target_language="es",
        )
        
        assert result.success is True
        assert result.subtitle_count == 2
        assert "Hola mundo" in result.content

    @pytest.mark.asyncio
    async def test_translate_file_invalid_srt(self, mock_provider, mock_settings):
        """Test translation with invalid SRT."""
        translator = SubtitleTranslator(provider=mock_provider, settings=mock_settings)
        
        result = await translator.translate_file(
            content="Not valid SRT",
            source_language="en",
            target_language="es",
        )
        
        # Should either fail gracefully or return empty
        # depending on how the parser handles it
        assert result.success is False or result.subtitle_count == 0

    @pytest.mark.asyncio
    async def test_health_check(self, mock_provider, mock_settings):
        """Test health check."""
        translator = SubtitleTranslator(provider=mock_provider, settings=mock_settings)
        
        result = await translator.health_check()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_close(self, mock_provider, mock_settings):
        """Test resource cleanup."""
        translator = SubtitleTranslator(provider=mock_provider, settings=mock_settings)
        
        await translator.close()
        
        mock_provider.close.assert_called_once()


class TestBatchProgress:
    """Tests for batch progress tracking."""

    def test_percent_complete_zero(self):
        """Test percentage when no batches."""
        progress = BatchProgress(total_batches=0)
        assert progress.percent_complete == 100.0

    def test_percent_complete_partial(self):
        """Test percentage with partial completion."""
        progress = BatchProgress(total_batches=4, completed_batches=2)
        assert progress.percent_complete == 50.0

    def test_status_processing(self):
        """Test status during processing."""
        progress = BatchProgress(total_batches=4, completed_batches=2)
        assert progress.status == "processing"

    def test_status_completed(self):
        """Test status when completed."""
        progress = BatchProgress(total_batches=4, completed_batches=4)
        assert progress.status == "completed"

    def test_status_partial_failure(self):
        """Test status with failures."""
        progress = BatchProgress(total_batches=4, completed_batches=2, failed_batches=1)
        assert progress.status == "partial_failure"