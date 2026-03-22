"""Tests for adaptive batch sizing."""

import pytest
from unittest.mock import MagicMock

from subtitle_translator.core.batch_sizing import (
    BatchSizeResolver,
    get_batch_size_resolver,
    MIN_BATCH_SIZE,
    TOKENS_PER_LINE_ESTIMATE,
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
