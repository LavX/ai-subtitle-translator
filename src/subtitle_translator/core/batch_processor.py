"""Batch processing logic for subtitle translation."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from subtitle_translator.config import Settings, get_settings
from subtitle_translator.providers.base import (
    InvalidResponseError,
    RateLimitError,
    TranslationBatch,
    TranslationProvider,
    TranslationProviderError,
)

if TYPE_CHECKING:
    from subtitle_translator.api.models import TranslationConfig

logger = logging.getLogger(__name__)

# Debug logger for detailed request/response logging
debug_logger = logging.getLogger(f"{__name__}.debug")


@dataclass
class BatchProgress:
    """Progress tracking for batch processing."""

    total_batches: int
    completed_batches: int = 0
    total_lines: int = 0
    completed_lines: int = 0
    failed_batches: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    @property
    def percent_complete(self) -> float:
        """Calculate percentage of completion."""
        if self.total_batches == 0:
            return 100.0
        return (self.completed_batches / self.total_batches) * 100

    @property
    def status(self) -> str:
        """Get current status string."""
        if self.completed_batches == self.total_batches:
            return "completed"
        elif self.failed_batches > 0:
            return "partial_failure"
        return "processing"


@dataclass
class BatchResult:
    """Result of processing a single batch."""

    batch_index: int
    success: bool
    translations: list[dict[str, str]] = field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    error: str | None = None
    retries: int = 0


@dataclass
class BatchProcessingResult:
    """Complete result of batch processing."""

    all_translations: list[dict[str, str]]
    total_tokens: int
    model_used: str
    batch_results: list[BatchResult]
    progress: BatchProgress

    @property
    def success(self) -> bool:
        """Check if all batches succeeded."""
        return all(r.success for r in self.batch_results)


class BatchProcessor:
    """Handles batch processing of subtitle translations."""

    def __init__(
        self,
        provider: TranslationProvider,
        settings: Settings | None = None,
    ):
        """
        Initialize batch processor.

        Args:
            provider: Translation provider to use
            settings: Optional settings instance
        """
        self.provider = provider
        self.settings = settings or get_settings()

    def create_batches(
        self,
        lines: list[dict[str, str]],
        batch_size: int | None = None,
        model: str | None = None,
    ) -> list[list[dict[str, str]]]:
        """
        Split lines into batches.

        Args:
            lines: List of {"index": "X", "content": "text"} dictionaries
            batch_size: Optional batch size override
            model: Optional model ID for adaptive batch sizing

        Returns:
            List of batches, each being a list of line dictionaries
        """
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

    async def process_batch(
        self,
        batch: TranslationBatch,
        batch_index: int,
        model: str | None = None,
        temperature: float | None = None,
        config_override: Optional["TranslationConfig"] = None,
        _is_adaptive_retry: bool = False,
        _rate_limit_lock: asyncio.Lock | None = None,
    ) -> BatchResult:
        """
        Process a single batch with retry logic.

        Args:
            batch: The batch to translate
            batch_index: Index of this batch
            model: Optional model override
            temperature: Optional temperature override
            config_override: Optional per-request configuration override
            _is_adaptive_retry: Internal flag to prevent infinite recursion

        Returns:
            BatchResult with translations or error
        """
        from subtitle_translator.core.batch_sizing import MIN_BATCH_SIZE, get_batch_size_resolver

        retries = 0
        last_error: str | None = None
        is_timeout = False
        can_adaptive = not _is_adaptive_retry and len(batch.lines) > MIN_BATCH_SIZE
        # Rate limits get extra retries (3 more than normal errors)
        max_retries_with_rate_limit = self.settings.max_retries + 3
        model_id = (
            (config_override.model if config_override and config_override.model else None)
            or model
            or self.settings.openrouter_default_model
        )

        def _note_unsplittable_failure() -> None:
            # Without a split there is no record_failure call. A sub-batch of an adaptive
            # retry above the floor still failed at its size, and the cache may already
            # have grown past it on earlier sub-batches, so it is recorded like any other
            # failure; anything else only interrupts the streak that grows the size back.
            resolver = get_batch_size_resolver()
            if _is_adaptive_retry and len(batch.lines) > MIN_BATCH_SIZE:
                resolver.record_failure(model_id, len(batch.lines))
            else:
                resolver.record_floor_failure(model_id)

        while retries < max_retries_with_rate_limit:
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
                            batch,
                            batch_index,
                            model,
                            temperature,
                            config_override,
                            _rate_limit_lock,
                        )
                    # In sub-batch retry or at floor - treat as failure
                    _note_unsplittable_failure()
                    return BatchResult(
                        batch_index=batch_index,
                        success=False,
                        error=f"Partial translations: expected {len(batch.lines)}, got {len(result.translations)}",
                        retries=retries,
                    )

                get_batch_size_resolver().record_success(model_id, len(batch.lines))

                return BatchResult(
                    batch_index=batch_index,
                    success=True,
                    translations=result.translations,
                    tokens_used=result.total_tokens or 0,
                    cost=result.cost or 0.0,
                    retries=retries,
                )

            except InvalidResponseError as e:
                if can_adaptive:
                    return await self._retry_with_smaller_batches(
                        batch, batch_index, model, temperature, config_override, _rate_limit_lock
                    )
                # At floor or already in adaptive retry - use normal retry
                if retries < self.settings.max_retries:
                    retries += 1
                    last_error = e.message
                    await asyncio.sleep(self.settings.retry_delay * (2 ** (retries - 1)))
                else:
                    _note_unsplittable_failure()
                    return BatchResult(
                        batch_index=batch_index,
                        success=False,
                        error=e.message,
                        retries=retries,
                    )

            except RateLimitError as e:
                # Rate limits: start at 5s, exponential backoff, cap at 30s.
                # Use lock to serialize retries so parallel batches don't all
                # hammer the API simultaneously after a 429.
                rate_limit_base_delay = 5.0
                delay = e.retry_after or min(rate_limit_base_delay * (2**retries), 30.0)
                retries += 1
                if _rate_limit_lock:
                    async with _rate_limit_lock:
                        logger.warning(
                            f"Rate limit (429) on batch {batch_index}, waiting {delay:.0f}s "
                            f"(retry {retries}/{max_retries_with_rate_limit})"
                        )
                        await asyncio.sleep(delay)
                else:
                    logger.warning(
                        f"Rate limit (429) on batch {batch_index}, waiting {delay:.0f}s "
                        f"(retry {retries}/{max_retries_with_rate_limit})"
                    )
                    await asyncio.sleep(delay)
                last_error = str(e)

            except TranslationProviderError as e:
                is_timeout = "timeout" in e.message.lower()
                if e.retryable and retries < self.settings.max_retries:
                    delay = self.settings.retry_delay * (2**retries)
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
                _note_unsplittable_failure()
                return BatchResult(
                    batch_index=batch_index,
                    success=False,
                    error=str(e),
                    retries=retries,
                )

        # Max retries exceeded
        if is_timeout and can_adaptive:
            return await self._retry_with_smaller_batches(
                batch, batch_index, model, temperature, config_override, _rate_limit_lock
            )

        _note_unsplittable_failure()
        return BatchResult(
            batch_index=batch_index,
            success=False,
            error=f"Max retries exceeded. Last error: {last_error}",
            retries=retries,
        )

    async def _retry_with_smaller_batches(
        self,
        batch: TranslationBatch,
        batch_index: int,
        model: str | None = None,
        temperature: float | None = None,
        config_override: Optional["TranslationConfig"] = None,
        _rate_limit_lock: asyncio.Lock | None = None,
    ) -> BatchResult:
        """Retry a failed batch by splitting it into smaller sub-batches."""
        from subtitle_translator.core.batch_sizing import get_batch_size_resolver

        model_id = (
            (config_override.model if config_override and config_override.model else None)
            or model
            or self.settings.openrouter_default_model
        )
        resolver = get_batch_size_resolver()
        new_size = resolver.record_failure(model_id, len(batch.lines))

        logger.warning(
            f"Batch {batch_index}: adaptive retry with size {new_size} (was {len(batch.lines)})"
        )

        sub_batches = [batch.lines[i : i + new_size] for i in range(0, len(batch.lines), new_size)]

        all_translations: list[dict[str, str]] = []
        total_tokens = 0
        total_cost = 0.0
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
                sub_batch,
                batch_index,
                model,
                temperature,
                config_override,
                _is_adaptive_retry=True,
                _rate_limit_lock=_rate_limit_lock,
            )
            if not sub_result.success:
                return BatchResult(
                    batch_index=batch_index,
                    success=False,
                    translations=all_translations,
                    error=f"Adaptive retry failed at size {new_size}: {sub_result.error}",
                    retries=total_retries,
                )
            all_translations.extend(sub_result.translations)
            total_tokens += sub_result.tokens_used
            total_cost += sub_result.cost
            total_retries += sub_result.retries

        return BatchResult(
            batch_index=batch_index,
            success=True,
            translations=all_translations,
            tokens_used=total_tokens,
            cost=total_cost,
            retries=total_retries,
        )

    async def process_all_batches(
        self,
        lines: list[dict[str, str]],
        source_language: str,
        target_language: str,
        context_title: str | None = None,
        context_media_type: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        batch_size: int | None = None,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        config_override: Optional["TranslationConfig"] = None,
    ) -> BatchProcessingResult:
        """
        Process all batches with parallel processing support.

        Batches are processed in parallel groups based on the parallel_batches_per_job
        setting or per-request config override.

        Args:
            lines: List of subtitle lines to translate
            source_language: Source language code
            target_language: Target language code
            context_title: Optional media title for context
            context_media_type: Optional media type (Episode/Movie)
            model: Optional model override
            temperature: Optional temperature override
            batch_size: Optional batch size override
            progress_callback: Optional callback for progress updates
            config_override: Optional per-request configuration override

        Returns:
            BatchProcessingResult with all translations
        """
        # Determine model to use (config override takes precedence)
        if config_override and config_override.model:
            model_to_use = config_override.model
        else:
            model_to_use = model or self.settings.openrouter_default_model

        batches = self.create_batches(lines, batch_size, model=model_to_use)

        # Determine parallel batch count (config override takes precedence)
        if config_override and config_override.parallel_batches:
            parallel_count = config_override.parallel_batches
        else:
            parallel_count = self.settings.parallel_batches_per_job

        progress = BatchProgress(
            total_batches=len(batches),
            total_lines=len(lines),
        )

        batch_results: list[BatchResult] = []
        all_translations: list[dict[str, str]] = []

        # Create indexed batches for tracking
        indexed_batches = list(enumerate(batches))

        logger.info(
            f"Processing {len(batches)} batches with {parallel_count} parallel batches per group - "
            f"source={source_language}, target={target_language}, "
            f"model={model_to_use}, temperature={temperature or 'default'}"
        )

        # Shared lock so rate-limited retries don't all fire simultaneously
        rate_limit_lock = asyncio.Lock()

        # Fire initial progress so job shows totalBatches immediately
        if progress_callback:
            progress_callback(progress)

        # Process batches in parallel groups, updating progress per batch
        for group_start in range(0, len(indexed_batches), parallel_count):
            batch_group = indexed_batches[group_start : group_start + parallel_count]

            group_indices = [idx for idx, _ in batch_group]
            logger.info(
                f"Processing parallel batch group: batches {group_indices} "
                f"({len(batch_group)} batches in parallel)"
            )

            # Create tasks with staggered start to avoid hitting rate limits
            async def _run_batch(
                bi: int, bl: list, stagger: float
            ) -> tuple[int, list, BatchResult]:
                if stagger > 0:
                    await asyncio.sleep(stagger)
                b = TranslationBatch(
                    lines=bl,
                    source_language=source_language,
                    target_language=target_language,
                    context_title=context_title,
                    context_media_type=context_media_type,
                )
                r = await self.process_batch(
                    b,
                    batch_index=bi,
                    model=model,
                    temperature=temperature,
                    config_override=config_override,
                    _rate_limit_lock=rate_limit_lock,
                )
                return bi, bl, r

            tasks = [
                asyncio.ensure_future(_run_batch(bi, bl, i * 0.5))
                for i, (bi, bl) in enumerate(batch_group)
            ]

            # Process results as each batch completes (not waiting for all)
            for coro in asyncio.as_completed(tasks):
                batch_index, batch_lines, result = await coro
                batch_results.append(result)

                if result.success:
                    all_translations.extend(result.translations)
                    progress.completed_lines += len(batch_lines)
                    progress.total_tokens += result.tokens_used
                    progress.total_cost += result.cost
                else:
                    progress.failed_batches += 1
                    logger.error(f"Batch {batch_index + 1} failed: {result.error}")

                progress.completed_batches += 1

                if progress_callback:
                    progress_callback(progress)

        # Sort batch_results by batch_index to maintain order
        batch_results.sort(key=lambda r: r.batch_index)

        return BatchProcessingResult(
            all_translations=all_translations,
            total_tokens=progress.total_tokens,
            model_used=model_to_use,
            batch_results=batch_results,
            progress=progress,
        )

    async def process_batches_stream(
        self,
        lines: list[dict[str, str]],
        source_language: str,
        target_language: str,
        context_title: str | None = None,
        context_media_type: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        batch_size: int | None = None,
        config_override: Optional["TranslationConfig"] = None,
    ) -> AsyncGenerator[tuple[BatchResult, BatchProgress], None]:
        """
        Process batches and yield results as they complete.

        This is useful for streaming progress updates.

        Args:
            lines: List of subtitle lines to translate
            source_language: Source language code
            target_language: Target language code
            context_title: Optional media title for context
            context_media_type: Optional media type (Episode/Movie)
            model: Optional model override
            temperature: Optional temperature override
            batch_size: Optional batch size override
            config_override: Optional per-request configuration override

        Yields:
            Tuples of (BatchResult, BatchProgress) for each completed batch
        """
        batches = self.create_batches(lines, batch_size, model=model)

        progress = BatchProgress(
            total_batches=len(batches),
            total_lines=len(lines),
        )

        for i, batch_lines in enumerate(batches):
            batch = TranslationBatch(
                lines=batch_lines,
                source_language=source_language,
                target_language=target_language,
                context_title=context_title,
                context_media_type=context_media_type,
            )

            result = await self.process_batch(
                batch,
                batch_index=i,
                model=model,
                temperature=temperature,
                config_override=config_override,
            )

            if result.success:
                progress.completed_lines += len(batch_lines)
                progress.total_tokens += result.tokens_used
                progress.total_cost += result.cost
            else:
                progress.failed_batches += 1

            progress.completed_batches += 1

            yield result, progress


def get_batch_processor(provider: TranslationProvider) -> BatchProcessor:
    """Factory function to get a batch processor instance."""
    return BatchProcessor(provider)
