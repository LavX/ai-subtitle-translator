"""Adaptive batch sizing for different model capabilities."""

import logging

from subtitle_translator.config import get_settings

logger = logging.getLogger(__name__)

TOKENS_PER_LINE_ESTIMATE = 800
MIN_BATCH_SIZE = 5

# What one translated cue costs in the reply, counting the JSON that carries it.
# Measured over a 1355-cue film into Hungarian: 100-cue batches came back between
# 6200 and 8000 completion tokens, so 62 to 80 per cue. The estimate sits at the
# top of that range because guessing high only costs an extra request, while
# guessing low costs a whole truncated reply that is billed and thrown away.
OUTPUT_TOKENS_PER_LINE_ESTIMATE = 100


class BatchSizeResolver:
    """Determines optimal batch size per model through learned cache, metadata, and heuristics."""

    def __init__(self) -> None:
        self._learned_sizes: dict[str, int] = {}
        self._ceilings: dict[str, int] = {}
        self._success_counts: dict[str, int] = {}
        self._settings = get_settings()

    def _budget_cap(self, output_ceiling: int | None = None) -> int | None:
        """How many cues the configured output budget can answer for, if it is set.

        Without this the first batch of every job is planned as if the reply had the
        model's whole output ceiling to write in. When OPENROUTER_MAX_TOKENS says
        otherwise the reply is cut short, and the batch is only resized after that
        truncated attempt has been billed. The budget is knowable up front, so the
        first batch may as well fit it.
        """
        budget = getattr(self._settings, "openrouter_max_tokens", None)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            # Unset means no budget is sent, so only the model's own ceiling applies.
            budget = None
        if isinstance(output_ceiling, bool) or not isinstance(output_ceiling, int):
            output_ceiling = None
        elif output_ceiling <= 0:
            output_ceiling = None

        # The provider trims the budget to the model's ceiling before sending, so the
        # room the reply really gets is the smaller of the two. Planning from the
        # configured budget alone hands a low-ceiling model a batch it can never
        # answer, and the cut-short reply is billed before the sizer starts halving.
        room = (
            min(x for x in (budget, output_ceiling) if x is not None)
            if (budget is not None or output_ceiling is not None)
            else None
        )
        if room is None:
            return None
        return max(MIN_BATCH_SIZE, room // OUTPUT_TOKENS_PER_LINE_ESTIMATE)

    def resolve(
        self,
        model_id: str,
        context_length: int | None = None,
        max_batch_size: int | None = None,
        output_ceiling: int | None = None,
    ) -> int:
        global_max = self._settings.batch_size

        # A size learned from a real failure is evidence about this model; the
        # estimates below are not, so it is never widened by them.
        if model_id in self._learned_sizes:
            return self._learned_sizes[model_id]

        limits = [global_max]
        budget_cap = self._budget_cap(output_ceiling)
        if budget_cap is not None:
            limits.append(budget_cap)

        if max_batch_size is not None:
            limits.append(max_batch_size)
        elif context_length is not None:
            limits.append(max(MIN_BATCH_SIZE, context_length // TOKENS_PER_LINE_ESTIMATE))

        return min(limits)

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

    def limit_planned_size(self, model_id: str, planned_size: int) -> int:
        """Apply only learned reductions to an already planned request."""
        return min(planned_size, self._learned_sizes.get(model_id, planned_size))

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
            # The ceiling belonged to this recovery cycle. A later failure at the restored
            # size starts a new one and must set its own, or it would be capped below it.
            del self._learned_sizes[model_id]
            del self._success_counts[model_id]
            self._ceilings.pop(model_id, None)

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
