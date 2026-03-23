# Adaptive Batch Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically determine optimal batch sizes per model, using metadata overrides, context-length heuristics, and adaptive retry with in-memory learning.

**Architecture:** New `BatchSizeResolver` singleton resolves batch size through 3 layers (learned cache → metadata override → heuristic → global default). `BatchProcessor.process_batch()` detects sizing failures and triggers adaptive retry, halving the batch and caching the learned size for future requests.

**Tech Stack:** Python 3.11+, pytest, asyncio, pydantic

**Spec:** `docs/superpowers/specs/2026-03-22-adaptive-batch-sizing-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/subtitle_translator/core/batch_sizing.py` | **New.** `BatchSizeResolver` class, constants, singleton accessor |
| `src/subtitle_translator/providers/base.py` | Add abstract `get_model_metadata()` to `TranslationProvider` |
| `src/subtitle_translator/providers/openrouter.py` | Implement `get_model_metadata()`, add `max_batch_size` to model dicts |
| `src/subtitle_translator/core/batch_processor.py` | Use resolver in `create_batches()`, add adaptive retry in `process_batch()` |
| `tests/test_batch_sizing.py` | **New.** Tests for `BatchSizeResolver` |
| `tests/test_translator.py` | Add/update tests for adaptive retry in `BatchProcessor` |

---

### Task 1: BatchSizeResolver — resolve() logic

**Files:**
- Create: `src/subtitle_translator/core/batch_sizing.py`
- Create: `tests/test_batch_sizing.py`

- [ ] **Step 1: Write failing tests for resolve()**

In `tests/test_batch_sizing.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_batch_sizing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'subtitle_translator.core.batch_sizing'`

- [ ] **Step 3: Implement BatchSizeResolver**

Create `src/subtitle_translator/core/batch_sizing.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_batch_sizing.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/subtitle_translator/core/batch_sizing.py tests/test_batch_sizing.py
git commit -m "feat: add BatchSizeResolver with resolve logic and tests"
```

---

### Task 2: BatchSizeResolver — record_failure() logic

**Files:**
- Modify: `tests/test_batch_sizing.py`
- (Implementation already in `batch_sizing.py` from Task 1)

- [ ] **Step 1: Write failing tests for record_failure()**

Append to `tests/test_batch_sizing.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_batch_sizing.py -v`
Expected: All 15 tests PASS (implementation already done in Task 1)

- [ ] **Step 3: Commit**

```bash
git add tests/test_batch_sizing.py
git commit -m "test: add record_failure tests for BatchSizeResolver"
```

---

### Task 3: Provider — get_model_metadata()

**Files:**
- Modify: `src/subtitle_translator/providers/base.py:74-131` (add abstract method)
- Modify: `src/subtitle_translator/providers/openrouter.py:288+` (implement method)
- Modify: `src/subtitle_translator/providers/openrouter.py:35-219` (add `max_batch_size` to small-context models)

- [ ] **Step 1: Add abstract method to TranslationProvider**

In `src/subtitle_translator/providers/base.py`, add after the `get_available_models` method (after line 131):

```python
    def get_model_metadata(self, model_id: str) -> Optional[dict]:
        """
        Get metadata for a specific model by ID.

        Args:
            model_id: The model identifier

        Returns:
            Model metadata dict or None if not found
        """
        return None
```

Note: This is a concrete method with a default return of `None`, not abstract. This avoids breaking any existing provider implementations and is fine because the resolver handles `None` gracefully (falls back to global default).

- [ ] **Step 2: Implement get_model_metadata in OpenRouterProvider**

In `src/subtitle_translator/providers/openrouter.py`, add a cached lookup. Add after the `get_available_models` method (around line 372):

```python
    def get_model_metadata(self, model_id: str) -> Optional[dict]:
        if not hasattr(self, "_model_metadata_cache"):
            self._model_metadata_cache: dict[str, dict] = {}
            for model_list in [EXCELLENT_MODELS, EXCELLENT_FREE_MODELS, GOOD_MODELS, POOR_MODELS]:
                for model in model_list:
                    self._model_metadata_cache[model["id"]] = model
        return self._model_metadata_cache.get(model_id)
```

- [ ] **Step 3: Add max_batch_size to small-context model dicts**

In `src/subtitle_translator/providers/openrouter.py`, add `"max_batch_size"` to model dicts in `POOR_MODELS` that have small context windows:

- `google/gemma` (context_length 8192): add `"max_batch_size": 10`
- `meta-llama/llama-3.1-8b` (context_length 131072): no change needed, heuristic gives 163 → capped at 100

No changes needed for EXCELLENT_MODELS or EXCELLENT_FREE_MODELS — they all have large contexts.

Also add a comment block near the model lists documenting the override convention for future model additions:
```python
# Batch size override rules (from adaptive-batch-sizing spec):
# - context_length <= 16384: add "max_batch_size": 10
# - context_length <= 65536: add "max_batch_size": 30
# - larger context models: no override needed (heuristic handles it)
```

- [ ] **Step 4: Run existing tests to verify nothing broke**

Run: `python -m pytest tests/ -v`
Expected: Same pass/fail count as before (45 pass, 9 pre-existing async failures)

- [ ] **Step 5: Commit**

```bash
git add src/subtitle_translator/providers/base.py src/subtitle_translator/providers/openrouter.py
git commit -m "feat: add get_model_metadata to providers with max_batch_size support"
```

---

### Task 4: BatchProcessor — adaptive create_batches()

**Files:**
- Modify: `src/subtitle_translator/core/batch_processor.py:102-123`
- Modify: `tests/test_translator.py` (update existing `test_create_batches`)

- [ ] **Step 1: Write test for model-aware batch creation**

Add to `tests/test_batch_sizing.py`:

```python
from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.batch_sizing import get_batch_size_resolver
from subtitle_translator.providers.base import TranslationBatch


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_batch_sizing.py::TestBatchProcessorAdaptiveSizing -v`
Expected: FAIL — `create_batches()` doesn't accept `model` parameter yet

- [ ] **Step 3: Update create_batches() to accept model parameter**

In `src/subtitle_translator/core/batch_processor.py`, modify `create_batches()` (line 102):

```python
    def create_batches(
        self,
        lines: list[dict[str, str]],
        batch_size: Optional[int] = None,
        model: Optional[str] = None,
    ) -> list[list[dict[str, str]]]:
        if batch_size:
            size = batch_size
        elif model:
            from subtitle_translator.core.batch_sizing import get_batch_size_resolver
            metadata = self.provider.get_model_metadata(model)
            resolver = get_batch_size_resolver()
            size = resolver.resolve(
                model,
                context_length=metadata.get("context_length") if metadata else None,
                max_batch_size=metadata.get("max_batch_size") if metadata else None,
            )
        else:
            size = self.settings.batch_size

        batches = []
        for i in range(0, len(lines), size):
            batches.append(lines[i : i + size])
        return batches
```

- [ ] **Step 4: Update process_all_batches and process_batches_stream to pass model**

In `process_all_batches()` (line 290), change:
```python
batches = self.create_batches(lines, batch_size)
```
to:
```python
batches = self.create_batches(lines, batch_size, model=model_to_use)
```

In `process_batches_stream()` (line 398), change:
```python
batches = self.create_batches(lines, batch_size)
```
to:
```python
batches = self.create_batches(lines, batch_size, model=model)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_batch_sizing.py tests/test_translator.py -v -k "not asyncio"`
Expected: All new and existing sync tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/subtitle_translator/core/batch_processor.py tests/test_batch_sizing.py
git commit -m "feat: integrate adaptive batch sizing into create_batches"
```

---

### Task 5: BatchProcessor — adaptive retry on InvalidResponseError

**Files:**
- Modify: `src/subtitle_translator/core/batch_processor.py:125-207`
- Modify: `tests/test_batch_sizing.py`

- [ ] **Step 1: Write tests for adaptive retry**

Add to `tests/test_batch_sizing.py`:

```python
from unittest.mock import AsyncMock
from subtitle_translator.providers.base import (
    InvalidResponseError,
    TranslationProviderError,
    TranslationResult,
)


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
                translations=[{"index": l["index"], "content": f"T-{l['content']}"} for l in batch.lines],
                model_used="test/model",
                total_tokens=50,
            )

        self.provider.translate_batch = mock_translate
        batch = TranslationBatch(
            lines=lines, source_language="en", target_language="hu"
        )
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
                translations=[{"index": l["index"], "content": f"T-{l['content']}"} for l in batch.lines],
                model_used="test/model",
                total_tokens=50,
            )

        self.provider.translate_batch = mock_translate
        batch = TranslationBatch(
            lines=lines, source_language="en", target_language="hu"
        )
        result = await self.processor.process_batch(batch, batch_index=0, model="test/model")

        assert result.success is True
        assert len(result.translations) == 10

    @pytest.mark.asyncio
    async def test_adaptive_retry_at_floor_gives_up(self):
        """When batch is already at MIN_BATCH_SIZE, don't try adaptive retry."""
        lines = [{"index": str(i), "content": f"Line {i}"} for i in range(5)]

        self.provider.translate_batch = AsyncMock(
            side_effect=InvalidResponseError("Always fails")
        )

        batch = TranslationBatch(
            lines=lines, source_language="en", target_language="hu"
        )
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
                translations=[{"index": l["index"], "content": f"T-{l['content']}"} for l in batch.lines],
                model_used="test/model",
                total_tokens=50,
            )

        self.provider.translate_batch = mock_translate
        batch = TranslationBatch(
            lines=lines, source_language="en", target_language="hu"
        )
        result = await self.processor.process_batch(batch, batch_index=0, model="test/model")

        assert result.success is True
        assert len(result.translations) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_batch_sizing.py::TestAdaptiveRetry -v`
Expected: FAIL — adaptive retry not implemented yet

- [ ] **Step 3: Implement adaptive retry in process_batch()**

In `src/subtitle_translator/core/batch_processor.py`, replace `process_batch()` method (lines 125-207):

```python
    async def process_batch(
        self,
        batch: TranslationBatch,
        batch_index: int,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        config_override: Optional["TranslationConfig"] = None,
        _is_adaptive_retry: bool = False,
    ) -> BatchResult:
        from subtitle_translator.core.batch_sizing import get_batch_size_resolver, MIN_BATCH_SIZE

        retries = 0
        last_error: Optional[str] = None
        is_timeout = False
        can_adaptive = not _is_adaptive_retry and len(batch.lines) > MIN_BATCH_SIZE

        while retries <= self.settings.max_retries:
            try:
                result = await self.provider.translate_batch(
                    batch, model=model, temperature=temperature, config_override=config_override
                )

                # Count mismatch check
                if len(result.translations) < len(batch.lines):
                    logger.warning(
                        f"Batch {batch_index}: got {len(result.translations)}/{len(batch.lines)} translations"
                    )
                    if can_adaptive:
                        return await self._retry_with_smaller_batches(
                            batch, batch_index, model, temperature, config_override
                        )
                    # In sub-batch retry or at floor — treat as failure
                    return BatchResult(
                        batch_index=batch_index, success=False,
                        error=f"Partial translations: expected {len(batch.lines)}, got {len(result.translations)}",
                        retries=retries,
                    )

                model_id = model or self.settings.openrouter_default_model
                get_batch_size_resolver().record_success(model_id, len(batch.lines))

                return BatchResult(
                    batch_index=batch_index,
                    success=True,
                    translations=result.translations,
                    tokens_used=result.total_tokens or 0,
                    retries=retries,
                )

            except InvalidResponseError as e:
                if can_adaptive:
                    return await self._retry_with_smaller_batches(
                        batch, batch_index, model, temperature, config_override
                    )
                # At floor or already in adaptive retry — use normal retry
                if retries < self.settings.max_retries:
                    retries += 1
                    last_error = e.message
                    await asyncio.sleep(self.settings.retry_delay * (2 ** (retries - 1)))
                else:
                    return BatchResult(
                        batch_index=batch_index, success=False,
                        error=e.message, retries=retries,
                    )

            except RateLimitError as e:
                delay = e.retry_after or (self.settings.retry_delay * (2 ** retries))
                logger.warning(
                    f"Rate limit hit on batch {batch_index}, waiting {delay}s (retry {retries + 1})"
                )
                await asyncio.sleep(delay)
                retries += 1
                last_error = str(e)

            except TranslationProviderError as e:
                is_timeout = "timeout" in e.message.lower()
                if e.retryable and retries < self.settings.max_retries:
                    delay = self.settings.retry_delay * (2 ** retries)
                    logger.warning(
                        f"Retryable error on batch {batch_index}: {e.message}, "
                        f"waiting {delay}s (retry {retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    retries += 1
                    last_error = e.message
                else:
                    logger.error(f"Non-retryable error on batch {batch_index}: {e.message}")
                    last_error = e.message
                    break

            except Exception as e:
                logger.error(f"Unexpected error on batch {batch_index}: {str(e)}")
                return BatchResult(
                    batch_index=batch_index, success=False,
                    error=str(e), retries=retries,
                )

        # Max retries exceeded
        if is_timeout and can_adaptive:
            return await self._retry_with_smaller_batches(
                batch, batch_index, model, temperature, config_override
            )

        return BatchResult(
            batch_index=batch_index, success=False,
            error=f"Max retries exceeded. Last error: {last_error}",
            retries=retries,
        )
```

- [ ] **Step 4: Implement _retry_with_smaller_batches()**

Add to `BatchProcessor` class in `batch_processor.py`, after `process_batch()`:

```python
    async def _retry_with_smaller_batches(
        self,
        batch: TranslationBatch,
        batch_index: int,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        config_override: Optional["TranslationConfig"] = None,
    ) -> BatchResult:
        from subtitle_translator.core.batch_sizing import get_batch_size_resolver

        model_id = model or self.settings.openrouter_default_model
        resolver = get_batch_size_resolver()
        new_size = resolver.record_failure(model_id, len(batch.lines))

        logger.warning(
            f"Batch {batch_index}: adaptive retry with size {new_size} (was {len(batch.lines)})"
        )

        sub_batches = [batch.lines[i:i + new_size] for i in range(0, len(batch.lines), new_size)]

        all_translations: list[dict[str, str]] = []
        total_tokens = 0
        total_retries = 0

        for sub_batch_lines in sub_batches:
            sub_batch = TranslationBatch(
                lines=sub_batch_lines,
                source_language=batch.source_language,
                target_language=batch.target_language,
                context_title=batch.context_title,
                context_media_type=batch.context_media_type,
            )
            sub_result = await self.process_batch(
                sub_batch, batch_index, model, temperature, config_override,
                _is_adaptive_retry=True,
            )
            if not sub_result.success:
                return BatchResult(
                    batch_index=batch_index, success=False,
                    translations=all_translations,
                    error=f"Adaptive retry failed at size {new_size}: {sub_result.error}",
                    retries=total_retries,
                )
            all_translations.extend(sub_result.translations)
            total_tokens += sub_result.tokens_used
            total_retries += sub_result.retries

        return BatchResult(
            batch_index=batch_index, success=True,
            translations=all_translations, tokens_used=total_tokens,
            retries=total_retries,
        )
```

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/test_batch_sizing.py tests/test_translator.py -v`
Expected: All batch sizing tests PASS. Existing sync tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/subtitle_translator/core/batch_processor.py tests/test_batch_sizing.py
git commit -m "feat: add adaptive retry with batch halving on failures"
```

---

### Task 6: Integration verification

**Files:**
- No new files — run full test suite and verify

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest tests/ -v`
Expected: All new tests PASS. Pre-existing test results unchanged (45+ pass, 9 async failures).

- [ ] **Step 2: Verify the singleton and import chain works**

Run: `python -c "from subtitle_translator.core.batch_sizing import get_batch_size_resolver; r = get_batch_size_resolver(); print(f'Resolver ready, default batch size: {r._settings.batch_size}')"`
Expected: Prints resolver info without errors.

- [ ] **Step 3: Commit any remaining fixes if needed**

Only if step 1 or 2 revealed issues.
