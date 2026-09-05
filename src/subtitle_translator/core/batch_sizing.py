"""Adaptive batch sizing for different model capabilities."""

import logging

from subtitle_translator.config import get_settings

logger = logging.getLogger(__name__)

TOKENS_PER_LINE_ESTIMATE = 800
MIN_BATCH_SIZE = 5


class BatchSizeResolver:
    """Determines optimal batch size per model through learned cache, metadata, and heuristics."""

    def __init__(self) -> None:
        self._learned_sizes: dict[str, int] = {}
        self._ceilings: dict[str, int] = {}
        self._success_counts: dict[str, int] = {}
        self._settings = get_settings()

    def resolve(
        self,
        model_id: str,
        context_length: int | None = None,
        max_batch_size: int | None = None,
    ) -> int:
        global_max = self._settings.batch_size

        if model_id in self._learned_sizes:
            return self._learned_sizes[model_id]

        if max_batch_size is not None:
            return min(max_batch_size, global_max)

        if context_length is not None:
            heuristic = context_length // TOKENS_PER_LINE_ESTIMATE
            return min(global_max, max(MIN_BATCH_SIZE, heuristic))

        return global_max

    def record_failure(self, model_id: str, failed_batch_size: int) -> int:
        # Halve the smaller of the cached size and the size that actually failed: with
        # parallel batches the cache can already have grown while an older, smaller
        # batch was still in flight, and its retry must end up below the failed size.
        base = min(self._learned_sizes.get(model_id, failed_batch_size), failed_batch_size)
        new_size = max(MIN_BATCH_SIZE, base // 2)
        self._learned_sizes[model_id] = new_size
        self._ceilings.setdefault(model_id, failed_batch_size)
        self._success_counts[model_id] = 0
        logger.warning(
            f"Adaptive batch sizing: {model_id} failed at size {failed_batch_size}, "
            f"learned safe size: {new_size}"
        )
        return new_size

    def record_floor_failure(self, model_id: str) -> None:
        """A failure that could not be split any further keeps the learned size but
        clears the success streak, so growing back still takes consecutive successes."""
        if model_id in self._learned_sizes:
            self._success_counts[model_id] = 0

    def record_success(self, model_id: str, batch_size: int) -> None:
        learned_size = self._learned_sizes.get(model_id)
        if learned_size is None or batch_size < learned_size:
            return

        self._success_counts[model_id] += 1
        if self._success_counts[model_id] < 3:
            return

        ceiling = self._ceilings[model_id]
        new_size = min(learned_size * 2, ceiling)
        self._learned_sizes[model_id] = new_size
        self._success_counts[model_id] = 0
        logger.info(
            f"Adaptive batch sizing: {model_id} grew from {learned_size} to {new_size} "
            "after 3 consecutive successes"
        )
        if new_size >= ceiling:
            del self._learned_sizes[model_id]
            del self._success_counts[model_id]

    def reset(self) -> None:
        self._learned_sizes.clear()
        self._ceilings.clear()
        self._success_counts.clear()


_resolver_instance: BatchSizeResolver | None = None


def get_batch_size_resolver() -> BatchSizeResolver:
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = BatchSizeResolver()
    return _resolver_instance
