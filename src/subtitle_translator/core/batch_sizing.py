"""Adaptive batch sizing for different model capabilities."""

import logging
from typing import Optional

from subtitle_translator.config import get_settings

logger = logging.getLogger(__name__)

TOKENS_PER_LINE_ESTIMATE = 800
MIN_BATCH_SIZE = 5


class BatchSizeResolver:
    """Determines optimal batch size per model through learned cache, metadata, and heuristics."""

    def __init__(self) -> None:
        self._learned_sizes: dict[str, int] = {}
        self._settings = get_settings()

    def resolve(
        self,
        model_id: str,
        context_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
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
        if model_id in self._learned_sizes:
            base = self._learned_sizes[model_id]
        else:
            base = failed_batch_size
        new_size = max(MIN_BATCH_SIZE, base // 2)
        self._learned_sizes[model_id] = new_size
        logger.warning(
            f"Adaptive batch sizing: {model_id} failed at size {failed_batch_size}, "
            f"learned safe size: {new_size}"
        )
        return new_size

    def record_success(self, model_id: str, batch_size: int) -> None:
        pass

    def reset(self) -> None:
        self._learned_sizes.clear()


_resolver_instance: Optional[BatchSizeResolver] = None


def get_batch_size_resolver() -> BatchSizeResolver:
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = BatchSizeResolver()
    return _resolver_instance
