# Adaptive Batch Sizing

## Problem

The translator uses a fixed batch size (default 100 lines) for all models. Some models can't handle large batches — they truncate output, return garbage, or timeout. Different models fail in different ways and at different thresholds.

## Solution

A three-layer `BatchSizeResolver` that determines the optimal batch size per model:

1. **Per-model metadata override** — hardcoded `max_batch_size` in model dicts
2. **Context-length heuristic** — estimates safe size from the model's context window
3. **Adaptive retry with in-memory cache** — on failure, halves batch size, caches the learned safe size for future jobs

## Components

### `BatchSizeResolver` — new class in `core/batch_sizing.py`

Stateful singleton that manages batch size resolution and learning.

**Constants:**

```python
TOKENS_PER_LINE_ESTIMATE = 800  # prompt overhead + source + translation + JSON structure
MIN_BATCH_SIZE = 5              # floor to prevent single-line batches
```

**Interface:**

```python
class BatchSizeResolver:
    def resolve(self, model_id: str, context_length: Optional[int] = None, max_batch_size: Optional[int] = None) -> int
    def record_failure(self, model_id: str, failed_batch_size: int) -> int
    def record_success(self, model_id: str, batch_size: int) -> None
    def reset(self) -> None  # for testing
```

`resolve()` takes model metadata values directly (context_length, max_batch_size) rather than a provider reference. The caller is responsible for looking up the metadata. This avoids coupling the resolver to any specific provider implementation.

**Resolution order in `resolve()`:**

1. Check `_learned_sizes` cache (dict[str, int]) — if model has a learned safe size, return it
2. If `max_batch_size` argument provided (from model metadata), return `min(max_batch_size, settings.batch_size)`
3. If `context_length` argument provided, compute heuristic: `min(settings.batch_size, max(MIN_BATCH_SIZE, context_length // TOKENS_PER_LINE_ESTIMATE))`
4. Fall back to global `settings.batch_size` (for unknown models with no metadata at all)

Step 3 uses 800 tokens per line as a conservative estimate covering: prompt template (~200 tokens amortized), source line (~50 tokens), translated line (~50 tokens), and JSON structure overhead. The `// 800` divisor means a 32K context model gets batch size 40, a 128K model gets 100+ (capped by global default), and an 8K model gets 10.

For unknown models (step 4), adaptive retry is the only protection — the first failure will trigger learning.

**`record_failure(model_id, failed_batch_size) -> int`:**

- If model already has a learned size in cache, halve _that_ value: `new_size = max(MIN_BATCH_SIZE, _learned_sizes[model_id] // 2)`
- Otherwise halve the failed size: `new_size = max(MIN_BATCH_SIZE, failed_batch_size // 2)`
- Stores in `_learned_sizes[model_id] = new_size`
- Returns `new_size` for immediate use
- Logs at WARNING level: model ID, failed size, new size

This handles concurrent failures correctly — if two parallel batches fail simultaneously, the second `record_failure` call will halve the already-learned size from the first, giving a cascading reduction rather than a no-op.

**`record_success(model_id, batch_size)`:**

- No-op for now. Exists as a hook if we later want to try growing batch sizes.

**Singleton access:**

```python
_resolver_instance: Optional[BatchSizeResolver] = None

def get_batch_size_resolver() -> BatchSizeResolver:
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = BatchSizeResolver()
    return _resolver_instance
```

No lock needed — Python's GIL makes this safe, and the class is sync-only (no async init).

### Model Metadata Changes — `providers/openrouter.py`

Add optional `max_batch_size` field to model dicts for known problematic models:

- Models with `context_length <= 16384`: add `"max_batch_size": 10`
- Models with `context_length <= 65536`: add `"max_batch_size": 30`
- Models that are known champions with large contexts: no override needed (heuristic or global default works)

Only add overrides where the heuristic would be wrong or where battle royale data shows a specific limit.

Add a `get_model_metadata(model_id: str) -> Optional[dict]` method to `OpenRouterProvider` that returns the model dict for a given ID. Implementation: build a `dict[str, dict]` lookup from the model lists on first call, cache it. Also add this as an abstract method on `TranslationProvider` in `base.py`.

### `BatchProcessor` Changes — `core/batch_processor.py`

**`create_batches()`:**

Change signature to accept `model` parameter. Look up model metadata via the provider and pass values to the resolver.

```python
def create_batches(self, lines, batch_size=None, model=None) -> list[list[dict]]:
    if batch_size:
        size = batch_size  # explicit override always wins
    elif model:
        metadata = self.provider.get_model_metadata(model)
        resolver = get_batch_size_resolver()
        size = resolver.resolve(
            model,
            context_length=metadata.get("context_length") if metadata else None,
            max_batch_size=metadata.get("max_batch_size") if metadata else None,
        )
    else:
        size = self.settings.batch_size
    ...
```

**`process_batch()` — adaptive retry interaction with existing retry loop:**

The existing retry loop handles transient errors (rate limits, retryable provider errors). Adaptive batch sizing is a _different_ mechanism that should trigger _instead of_ continuing retries at the same size, because retrying the same batch size won't help if the model can't handle it.

Behavior:
1. On `InvalidResponseError`: immediately exit the retry loop and trigger adaptive retry (don't waste retries at the bad size)
2. On count mismatch (successful parse but `len(translations) < len(batch.lines)`): treat as a sizing failure. Check this inside `process_batch` after `translate_batch()` returns, before returning the successful `BatchResult`. Raise a new `InvalidResponseError` with a descriptive message to enter the adaptive path.
3. On timeout (`TranslationProviderError` where the error message contains "timeout" or similar): after exhausting normal retries, trigger adaptive retry as a last resort instead of giving up

```python
async def process_batch(self, batch, batch_index, model=None, ...) -> BatchResult:
    retries = 0
    last_error = None
    is_timeout = False

    while retries <= self.settings.max_retries:
        try:
            result = await self.provider.translate_batch(batch, ...)

            # Count mismatch check — adaptive sizing trigger
            if len(result.translations) < len(batch.lines):
                logger.warning(f"Batch {batch_index}: got {len(result.translations)}/{len(batch.lines)} translations")
                raise InvalidResponseError(
                    f"Partial translations: expected {len(batch.lines)}, got {len(result.translations)}"
                )

            resolver = get_batch_size_resolver()
            resolver.record_success(model or self.settings.openrouter_default_model, len(batch.lines))
            return BatchResult(success=True, ...)

        except InvalidResponseError as e:
            # Batch size issue — skip remaining retries, go straight to adaptive retry
            return await self._retry_with_smaller_batches(batch, batch_index, model, ...)

        except RateLimitError as e:
            # existing backoff logic, unchanged
            ...

        except TranslationProviderError as e:
            is_timeout = "timeout" in e.message.lower()
            # existing retry logic, unchanged
            ...

    # Max retries exceeded
    if is_timeout:
        # Timeout after all retries — try adaptive sizing as last resort
        return await self._retry_with_smaller_batches(batch, batch_index, model, ...)

    return BatchResult(success=False, error=f"Max retries exceeded: {last_error}", ...)
```

**`_retry_with_smaller_batches()` — new private method:**

Returns a single `BatchResult` that merges all sub-batch translations. `success=True` only if ALL sub-batches succeed.

```python
async def _retry_with_smaller_batches(
    self, batch, batch_index, model, temperature, config_override
) -> BatchResult:
    model_id = model or self.settings.openrouter_default_model
    resolver = get_batch_size_resolver()
    new_size = resolver.record_failure(model_id, len(batch.lines))

    logger.warning(f"Batch {batch_index}: adaptive retry with size {new_size} (was {len(batch.lines)})")

    # Re-split the failed batch
    sub_batches = [batch.lines[i:i+new_size] for i in range(0, len(batch.lines), new_size)]

    all_translations = []
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
        # Recursive call — if this also fails, it will halve again down to MIN_BATCH_SIZE
        sub_result = await self.process_batch(
            sub_batch, batch_index, model, temperature, config_override
        )
        if not sub_result.success:
            return BatchResult(
                batch_index=batch_index, success=False,
                translations=all_translations,  # partial results
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

Recursion depth is bounded: batch size halves each time with a floor of `MIN_BATCH_SIZE=5`. A batch of 100 can halve at most ~4 times (100→50→25→12→6→5 floor). At the floor, failures go through normal retries and return `BatchResult(success=False)`.

To prevent infinite recursion between `process_batch` and `_retry_with_smaller_batches`, add a `_is_adaptive_retry: bool = False` parameter to `process_batch`. When `True`, `InvalidResponseError` triggers normal retry behavior instead of adaptive retry. `_retry_with_smaller_batches` passes `_is_adaptive_retry=True` when calling `process_batch` for sub-batches. At the floor size (`len(batch.lines) <= MIN_BATCH_SIZE`), also skip adaptive retry.

**`process_all_batches()`:**

Pass `model` (resolved from config_override or parameter) through to `create_batches()`:

```python
batches = self.create_batches(lines, batch_size, model=model_to_use)
```

**`process_batches_stream()`:**

Same change — pass `model` through to `create_batches()`:

```python
batches = self.create_batches(lines, batch_size, model=model)
```

### Failure Detection Criteria

Adaptive sizing triggers on these existing error types:

| Error Type | Trigger? | Reason |
|---|---|---|
| `InvalidResponseError` | Yes | Truncated/garbage output — likely context overflow |
| Count mismatch after successful parse | Yes | Model skipped/forgot lines |
| `TranslationProviderError` (timeout) after max retries | Yes | Model too slow for batch size |
| `RateLimitError` | No | Already handled by backoff, not a sizing issue |
| `AuthenticationError` | No | Not a sizing issue |
| Network errors | No | Not a sizing issue |

## Files Changed

| File | Change |
|---|---|
| `core/batch_sizing.py` | New file — `BatchSizeResolver` class, constants, singleton accessor |
| `core/batch_processor.py` | Use resolver in `create_batches()`, add adaptive retry in `process_batch()`, add `_retry_with_smaller_batches()`, update `process_batches_stream()` |
| `providers/openrouter.py` | Add `max_batch_size` to model metadata dicts, add `get_model_metadata()` method |
| `providers/base.py` | Add abstract `get_model_metadata()` method to `TranslationProvider` |

## Not In Scope

- Persistent storage for learned sizes — in-memory cache resets on restart, which is fine
- Automatic batch size growth/recovery — no upward adjustment after learning a smaller size
- Per-language batch size tuning — not needed, batch sizing is model-dependent
- API contract changes — adaptive sizing is internal and transparent to callers
- Changes to `config.py` — global `batch_size` remains as the upper bound default
