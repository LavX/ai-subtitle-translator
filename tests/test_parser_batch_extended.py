"""Extended tests for srt_parser.py and batch_processor.py to cover missing lines."""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from subtitle_translator.core.srt_parser import (
    SRTParser,
    SRTParserError,
    add_rtl_markers,
    get_srt_parser,
)
from subtitle_translator.core.batch_processor import (
    BatchProcessor,
    BatchProgress,
    get_batch_processor,
)
from subtitle_translator.providers.base import (
    InvalidResponseError,
    TranslationBatch,
    TranslationResult,
    TranslationProviderError,
)
from subtitle_translator.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    defaults = dict(
        openrouter_api_key="test-key",
        batch_size=10,
        max_retries=2,
        retry_delay=0.01,
        parallel_batches_per_job=1,
        openrouter_default_model="test-model",
    )
    defaults.update(overrides)
    return Settings(**defaults)


VALID_SRT = (
    "1\n"
    "00:00:01,000 --> 00:00:02,000\n"
    "Hello world\n\n"
    "2\n"
    "00:00:03,000 --> 00:00:04,000\n"
    "Goodbye world\n"
)


# ===================================================================
# SRT Parser tests
# ===================================================================


class TestSRTParserGenericException:
    """Lines 79-80: parse() generic (non-SRTParseError) exception path."""

    def test_parse_unexpected_error(self):
        parser = SRTParser()
        with patch("srt.parse", side_effect=TypeError("boom")):
            with pytest.raises(SRTParserError, match="Unexpected error parsing SRT"):
                parser.parse("anything")


class TestValidateSrt:
    """Lines 166-171: validate_srt() branches."""

    def test_valid_srt(self):
        parser = SRTParser()
        is_valid, error = parser.validate_srt(VALID_SRT)
        assert is_valid is True
        assert error is None

    def test_empty_srt(self):
        parser = SRTParser()
        is_valid, error = parser.validate_srt("")
        assert is_valid is False
        assert error == "No subtitles found in content"

    def test_invalid_srt_parse_error(self):
        parser = SRTParser()
        with patch("srt.parse", side_effect=__import__("srt").SRTParseError(0, 0, "")):
            is_valid, error = parser.validate_srt("bad content")
            assert is_valid is False
            assert "Invalid SRT format" in error

    def test_validate_generic_error(self):
        parser = SRTParser()
        with patch("srt.parse", side_effect=RuntimeError("something")):
            is_valid, error = parser.validate_srt("bad content")
            assert is_valid is False
            assert "Validation error" in error


class TestGetSubtitleCount:
    """Lines 183-186: get_subtitle_count() success and failure."""

    def test_success(self):
        parser = SRTParser()
        assert parser.get_subtitle_count(VALID_SRT) == 2

    def test_failure_returns_zero(self):
        parser = SRTParser()
        with patch("srt.parse", side_effect=Exception("fail")):
            assert parser.get_subtitle_count("bad") == 0


class TestAddRtlMarkers:
    """Line 267: module-level add_rtl_markers with empty line branch."""

    def test_marks_non_empty_lines(self):
        result = add_rtl_markers("hello")
        assert result.startswith("\u202b")
        assert result.endswith("\u202c")

    def test_preserves_empty_lines(self):
        result = add_rtl_markers("hello\n\ngoodbye")
        lines = result.split("\n")
        assert lines[0].startswith("\u202b")
        assert lines[1] == ""  # empty line preserved without markers
        assert lines[2].startswith("\u202b")


class TestGetSrtParser:
    """Line 274: get_srt_parser() factory."""

    def test_returns_parser_instance(self):
        parser = get_srt_parser()
        assert isinstance(parser, SRTParser)


# ===================================================================
# Batch Processor tests
# ===================================================================


def _mock_provider(translate_result=None, side_effect=None):
    provider = AsyncMock()
    if side_effect:
        provider.translate_batch = AsyncMock(side_effect=side_effect)
    elif translate_result:
        provider.translate_batch = AsyncMock(return_value=translate_result)
    else:
        provider.translate_batch = AsyncMock(
            return_value=TranslationResult(
                translations=[{"index": "0", "content": "Hola"}],
                model_used="test",
                total_tokens=10,
                cost=0.001,
            )
        )
    provider.get_model_metadata = MagicMock(return_value=None)
    return provider


def _make_batch(lines=None, src="en", tgt="es"):
    if lines is None:
        lines = [{"index": "0", "content": "Hello"}]
    return TranslationBatch(
        lines=lines,
        source_language=src,
        target_language=tgt,
    )


class TestProcessBatchAdaptiveRetryCountMismatch:
    """Lines 195, 244-249: _is_adaptive_retry=True with count mismatch returns failure directly."""

    @pytest.mark.asyncio
    async def test_adaptive_retry_count_mismatch_returns_failure(self):
        """When _is_adaptive_retry=True and count mismatches, return failure instead of retrying."""
        provider = _mock_provider(
            translate_result=TranslationResult(
                translations=[{"index": "0", "content": "Hola"}],  # only 1
                model_used="test",
                total_tokens=5,
            )
        )
        settings = _make_settings()
        processor = BatchProcessor(provider, settings)

        batch = _make_batch(
            lines=[
                {"index": "0", "content": "Hello"},
                {"index": "1", "content": "World"},
            ]
        )

        result = await processor.process_batch(
            batch, batch_index=0, _is_adaptive_retry=True
        )

        assert result.success is False
        assert "Partial translations" in result.error


class TestProcessBatchInvalidResponseAtFloor:
    """Lines 274-276: InvalidResponseError when at MIN_BATCH_SIZE triggers normal retry."""

    @pytest.mark.asyncio
    async def test_invalid_response_retries_at_floor(self):
        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise InvalidResponseError("bad json", provider="test")
            return TranslationResult(
                translations=[{"index": "0", "content": "Hola"}],
                model_used="test",
                total_tokens=10,
            )

        provider = _mock_provider(side_effect=_side_effect)
        settings = _make_settings(max_retries=3, retry_delay=0.001)
        processor = BatchProcessor(provider, settings)

        # Use _is_adaptive_retry=True so can_adaptive is False,
        # forcing the "at floor" retry path.
        batch = _make_batch()
        result = await processor.process_batch(
            batch, batch_index=0, _is_adaptive_retry=True
        )

        assert result.success is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_invalid_response_exhausts_retries_at_floor(self):
        """InvalidResponseError at floor exhausts retries and returns failure."""
        provider = _mock_provider(
            side_effect=InvalidResponseError("bad json", provider="test")
        )
        settings = _make_settings(max_retries=2, retry_delay=0.001)
        processor = BatchProcessor(provider, settings)

        batch = _make_batch()
        result = await processor.process_batch(
            batch, batch_index=0, _is_adaptive_retry=True
        )

        assert result.success is False
        assert result.error == "bad json"


class TestProcessBatchUnexpectedException:
    """Lines 274-276: unexpected (non-provider) exception returns immediate failure."""

    @pytest.mark.asyncio
    async def test_unexpected_exception(self):
        provider = _mock_provider(side_effect=RuntimeError("kaboom"))
        settings = _make_settings()
        processor = BatchProcessor(provider, settings)

        batch = _make_batch()
        result = await processor.process_batch(batch, batch_index=0)

        assert result.success is False
        assert "kaboom" in result.error


class TestRetryWithSmallerBatchesSubBatchFailure:
    """Line 345: sub-batch failure within _retry_with_smaller_batches."""

    @pytest.mark.asyncio
    async def test_sub_batch_failure_propagates(self):
        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call triggers adaptive retry (InvalidResponseError on big batch).
            # Sub-batch calls also fail.
            raise InvalidResponseError("bad", provider="test")

        provider = _mock_provider(side_effect=_side_effect)
        settings = _make_settings(max_retries=1, retry_delay=0.001, batch_size=20)
        processor = BatchProcessor(provider, settings)

        # Build a batch larger than MIN_BATCH_SIZE so can_adaptive=True initially
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(10)]
        batch = _make_batch(lines=lines)

        result = await processor.process_batch(batch, batch_index=0)

        assert result.success is False
        assert "Adaptive retry failed" in result.error


class TestProcessAllBatchesEdgeCases:
    """Lines 402, 410, 436, 453, 487-488, 493: process_all_batches edge cases."""

    @pytest.mark.asyncio
    async def test_config_override_model_and_parallel(self):
        """Lines 402, 410: config_override.model and parallel_batches used."""
        from subtitle_translator.api.models import TranslationConfig

        provider = _mock_provider()
        settings = _make_settings()
        processor = BatchProcessor(provider, settings)

        config = TranslationConfig(model="override-model", parallel_batches=2)
        lines = [{"index": "0", "content": "Hello"}]

        result = await processor.process_all_batches(
            lines, "en", "es", config_override=config
        )

        assert result.model_used == "override-model"
        assert result.success

    @pytest.mark.asyncio
    async def test_progress_callback_called(self):
        """Line 436, 493: progress_callback fires on start and per-batch completion."""
        provider = _mock_provider()
        settings = _make_settings()
        processor = BatchProcessor(provider, settings)

        progress_updates = []
        lines = [{"index": "0", "content": "Hello"}]

        await processor.process_all_batches(
            lines, "en", "es",
            progress_callback=lambda p: progress_updates.append(p.percent_complete),
        )

        # At least initial callback + one completion callback
        assert len(progress_updates) >= 2

    @pytest.mark.asyncio
    async def test_stagger_delay_for_parallel_batches(self):
        """Line 453: stagger > 0 triggers asyncio.sleep in _run_batch."""
        provider = _mock_provider()
        settings = _make_settings(parallel_batches_per_job=3)
        processor = BatchProcessor(provider, settings)

        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(3)]
        # Need 3 separate batches to trigger stagger
        result = await processor.process_all_batches(
            lines, "en", "es", batch_size=1
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_batch_failure_increments_failed_count(self):
        """Lines 487-488: failed batch increments progress.failed_batches."""
        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail first batch")
            return TranslationResult(
                translations=[{"index": "1", "content": "Mundo"}],
                model_used="test",
                total_tokens=5,
            )

        provider = _mock_provider(side_effect=_side_effect)
        settings = _make_settings(batch_size=1)
        processor = BatchProcessor(provider, settings)

        lines = [
            {"index": "0", "content": "Hello"},
            {"index": "1", "content": "World"},
        ]

        result = await processor.process_all_batches(lines, "en", "es")

        assert result.progress.failed_batches >= 1
        assert not result.success


class TestProcessBatchesStream:
    """Lines 537-570, 575: process_batches_stream generator."""

    @pytest.mark.asyncio
    async def test_stream_yields_results(self):
        provider = _mock_provider(
            translate_result=TranslationResult(
                translations=[{"index": "0", "content": "Hola"}],
                model_used="test",
                total_tokens=10,
                cost=0.001,
            )
        )
        settings = _make_settings()
        processor = BatchProcessor(provider, settings)

        lines = [{"index": "0", "content": "Hello"}]
        results = []
        async for batch_result, progress in processor.process_batches_stream(
            lines, "en", "es"
        ):
            results.append((batch_result, progress))

        assert len(results) == 1
        assert results[0][0].success is True
        assert results[0][1].completed_batches == 1
        assert results[0][1].percent_complete == 100.0

    @pytest.mark.asyncio
    async def test_stream_multiple_batches(self):
        provider = _mock_provider()
        settings = _make_settings(batch_size=1)
        processor = BatchProcessor(provider, settings)

        lines = [
            {"index": "0", "content": "Hello"},
            {"index": "1", "content": "World"},
        ]

        results = []
        async for batch_result, progress in processor.process_batches_stream(
            lines, "en", "es"
        ):
            # Capture a snapshot since progress is mutated in place
            results.append((batch_result, progress.completed_batches))

        assert len(results) == 2
        # Both batches complete; progress object is shared so check ordering
        assert results[0][1] == 1
        assert results[1][1] == 2

    @pytest.mark.asyncio
    async def test_stream_failed_batch_increments_failures(self):
        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("stream fail")
            return TranslationResult(
                translations=[{"index": "1", "content": "Mundo"}],
                model_used="test",
                total_tokens=5,
            )

        provider = _mock_provider(side_effect=_side_effect)
        settings = _make_settings(batch_size=1)
        processor = BatchProcessor(provider, settings)

        lines = [
            {"index": "0", "content": "Hello"},
            {"index": "1", "content": "World"},
        ]

        results = []
        async for batch_result, progress in processor.process_batches_stream(
            lines, "en", "es"
        ):
            results.append((batch_result, progress))

        assert len(results) == 2
        assert results[0][0].success is False
        assert results[0][1].failed_batches == 1
        assert results[1][0].success is True


class TestGetBatchProcessor:
    """Line 575: get_batch_processor factory."""

    def test_returns_processor_instance(self):
        provider = AsyncMock()
        processor = get_batch_processor(provider)
        assert isinstance(processor, BatchProcessor)
