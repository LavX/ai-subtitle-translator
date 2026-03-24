"""Tests for adaptive batch sizing."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.batch_sizing import (
    MIN_BATCH_SIZE,
    BatchSizeResolver,
    get_batch_size_resolver,
)
from subtitle_translator.providers.base import (
    InvalidResponseError,
    TranslationBatch,
    TranslationProviderError,
    TranslationResult,
)


class TestBatchSizeResolverResolve:
    """Tests for BatchSizeResolver.resolve()."""

    @pytest.fixture(autouse=True)
    def resolver(self):
        self.resolver = BatchSizeResolver()
        self.resolver._settings = MagicMock()
        self.resolver._settings.batch_size = 100

    def test_resolve_unknown_model_returns_global_default(self):
        """Unknown model with no metadata falls back to global batch_size."""
        result = self.resolver.resolve("unknown/model")
        assert result == 100

    def test_resolve_with_max_batch_size_override(self):
        """Model metadata max_batch_size takes precedence over heuristic."""
        result = self.resolver.resolve("small/model", max_batch_size=20)
        assert result == 20

    def test_resolve_max_batch_size_capped_by_global(self):
        """max_batch_size is capped by global batch_size."""
        result = self.resolver.resolve("big/model", max_batch_size=200)
        assert result == 100

    def test_resolve_with_context_length_heuristic(self):
        """Heuristic: context_length // 800, capped by global."""
        # 32768 // 800 = 40
        result = self.resolver.resolve("medium/model", context_length=32768)
        assert result == 40

    def test_resolve_heuristic_capped_by_global(self):
        """Large context_length is capped by global batch_size."""
        # 1048576 // 800 = 1310, but capped at 100
        result = self.resolver.resolve("large/model", context_length=1048576)
        assert result == 100

    def test_resolve_heuristic_floored_at_min(self):
        """Very small context_length floors at MIN_BATCH_SIZE."""
        # 2048 // 800 = 2, but floored at 5
        result = self.resolver.resolve("tiny/model", context_length=2048)
        assert result == MIN_BATCH_SIZE

    def test_resolve_learned_size_takes_priority(self):
        """Learned size from cache takes priority over everything."""
        self.resolver._learned_sizes["test/model"] = 25
        result = self.resolver.resolve("test/model", context_length=1048576, max_batch_size=50)
        assert result == 25

    def test_resolve_max_batch_size_over_context_length(self):
        """max_batch_size takes priority over context_length heuristic."""
        result = self.resolver.resolve("model", context_length=32768, max_batch_size=10)
        assert result == 10


class TestBatchSizeResolverRecordFailure:
    """Tests for BatchSizeResolver.record_failure()."""

    @pytest.fixture(autouse=True)
    def resolver(self):
        self.resolver = BatchSizeResolver()
        self.resolver._settings = MagicMock()
        self.resolver._settings.batch_size = 100

    def test_first_failure_halves_batch_size(self):
        new_size = self.resolver.record_failure("model/a", 100)
        assert new_size == 50

    def test_failure_stored_in_learned_cache(self):
        self.resolver.record_failure("model/a", 100)
        assert self.resolver._learned_sizes["model/a"] == 50

    def test_second_failure_halves_learned_size(self):
        self.resolver.record_failure("model/a", 100)  # learned: 50
        new_size = self.resolver.record_failure("model/a", 100)  # halves 50, not 100
        assert new_size == 25

    def test_failure_floors_at_min_batch_size(self):
        self.resolver.record_failure("model/a", 8)  # 8 // 2 = 4, floored to 5
        assert self.resolver._learned_sizes["model/a"] == MIN_BATCH_SIZE

    def test_cascading_failures_to_floor(self):
        """Repeated failures cascade: 100 -> 50 -> 25 -> 12 -> 6 -> 5."""
        sizes = []
        for _ in range(6):
            size = self.resolver.record_failure("model/a", 100)
            sizes.append(size)
        assert sizes == [50, 25, 12, 6, 5, 5]

    def test_resolve_returns_learned_after_failure(self):
        self.resolver.record_failure("model/a", 80)
        result = self.resolver.resolve("model/a", context_length=1048576, max_batch_size=100)
        assert result == 40  # learned size, not metadata

    def test_reset_clears_learned(self):
        self.resolver.record_failure("model/a", 100)
        self.resolver.reset()
        assert self.resolver._learned_sizes == {}


class TestBatchProcessorAdaptiveSizing:
    """Tests for adaptive batch sizing in BatchProcessor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.provider = MagicMock()
        self.provider.get_model_metadata.return_value = {"context_length": 32768}
        self.settings = MagicMock()
        self.settings.batch_size = 100
        self.settings.max_retries = 3
        self.settings.retry_delay = 0.01
        self.settings.openrouter_default_model = "default/model"
        self.processor = BatchProcessor(self.provider, self.settings)
        get_batch_size_resolver().reset()
        get_batch_size_resolver()._settings = self.settings

    def test_create_batches_with_model_uses_resolver(self):
        """When model is provided, batch size comes from resolver."""
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(100)]
        # context_length 32768 // 800 = 40
        batches = self.processor.create_batches(lines, model="medium/model")
        assert len(batches) == 3  # 40 + 40 + 20
        assert len(batches[0]) == 40
        assert len(batches[2]) == 20

    def test_create_batches_explicit_size_overrides_model(self):
        """Explicit batch_size always wins over model-based resolution."""
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(100)]
        batches = self.processor.create_batches(lines, batch_size=25, model="medium/model")
        assert len(batches) == 4  # 25 * 4

    def test_create_batches_no_model_uses_global(self):
        """Without model, uses global batch_size as before."""
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(100)]
        batches = self.processor.create_batches(lines)
        assert len(batches) == 1  # 100 lines, batch_size=100

    def test_create_batches_unknown_model_uses_global(self):
        """Unknown model (no metadata) falls back to global batch_size."""
        self.provider.get_model_metadata.return_value = None
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(100)]
        batches = self.processor.create_batches(lines, model="unknown/model")
        assert len(batches) == 1


class TestAdaptiveRetry:
    """Tests for adaptive retry when batches fail."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.provider = MagicMock()
        self.provider.get_model_metadata.return_value = None
        self.settings = MagicMock()
        self.settings.batch_size = 100
        self.settings.max_retries = 2
        self.settings.retry_delay = 0.01
        self.settings.openrouter_default_model = "test/model"
        self.processor = BatchProcessor(self.provider, self.settings)
        get_batch_size_resolver().reset()
        get_batch_size_resolver()._settings = self.settings

    @pytest.mark.asyncio
    async def test_invalid_response_triggers_adaptive_retry(self):
        """InvalidResponseError should trigger adaptive retry with smaller batches."""
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(10)]

        call_count = 0

        async def mock_translate(batch, **kwargs):
            nonlocal call_count
            call_count += 1
            if len(batch.lines) > 5:
                raise InvalidResponseError("Truncated response")
            return TranslationResult(
                translations=[
                    {"index": line["index"], "content": f"T-{line['content']}"}
                    for line in batch.lines
                ],
                model_used="test/model",
                total_tokens=50,
            )

        self.provider.translate_batch = mock_translate
        batch = TranslationBatch(lines=lines, source_language="en", target_language="hu")
        result = await self.processor.process_batch(batch, batch_index=0, model="test/model")

        assert result.success is True
        assert len(result.translations) == 10
        assert get_batch_size_resolver()._learned_sizes["test/model"] == 5

    @pytest.mark.asyncio
    async def test_count_mismatch_triggers_adaptive_retry(self):
        """When model returns fewer translations than input, trigger adaptive retry."""
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(10)]

        async def mock_translate(batch, **kwargs):
            if len(batch.lines) > 5:
                # Return only half the translations (count mismatch)
                return TranslationResult(
                    translations=[{"index": "0", "content": "Only one"}],
                    model_used="test/model",
                    total_tokens=50,
                )
            return TranslationResult(
                translations=[
                    {"index": line["index"], "content": f"T-{line['content']}"}
                    for line in batch.lines
                ],
                model_used="test/model",
                total_tokens=50,
            )

        self.provider.translate_batch = mock_translate
        batch = TranslationBatch(lines=lines, source_language="en", target_language="hu")
        result = await self.processor.process_batch(batch, batch_index=0, model="test/model")

        assert result.success is True
        assert len(result.translations) == 10

    @pytest.mark.asyncio
    async def test_adaptive_retry_at_floor_gives_up(self):
        """When batch is already at MIN_BATCH_SIZE, don't try adaptive retry."""
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(5)]

        self.provider.translate_batch = AsyncMock(side_effect=InvalidResponseError("Always fails"))

        batch = TranslationBatch(lines=lines, source_language="en", target_language="hu")
        result = await self.processor.process_batch(batch, batch_index=0, model="test/model")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_timeout_after_retries_triggers_adaptive(self):
        """Timeout errors trigger adaptive retry after exhausting normal retries."""
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(10)]

        call_count = 0

        async def mock_translate(batch, **kwargs):
            nonlocal call_count
            call_count += 1
            if len(batch.lines) > 5:
                raise TranslationProviderError("Request timeout", retryable=True)
            return TranslationResult(
                translations=[
                    {"index": line["index"], "content": f"T-{line['content']}"}
                    for line in batch.lines
                ],
                model_used="test/model",
                total_tokens=50,
            )

        self.provider.translate_batch = mock_translate
        batch = TranslationBatch(lines=lines, source_language="en", target_language="hu")
        result = await self.processor.process_batch(batch, batch_index=0, model="test/model")

        assert result.success is True
        assert len(result.translations) == 10
