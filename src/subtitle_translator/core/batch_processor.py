"""Batch processing logic for subtitle translation."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, List, Optional, TYPE_CHECKING

from subtitle_translator.config import Settings, get_settings
from subtitle_translator.providers.base import (
    InvalidResponseError,
    RateLimitError,
    TranslationBatch,
    TranslationProvider,
    TranslationProviderError,
    TranslationResult,
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
    error: Optional[str] = None
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
        settings: Optional[Settings] = None,
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
        batch_size: Optional[int] = None,
    ) -> list[list[dict[str, str]]]:
        """
        Split lines into batches.

        Args:
            lines: List of {"index": "X", "content": "text"} dictionaries
            batch_size: Optional batch size override

        Returns:
            List of batches, each being a list of line dictionaries
        """
        size = batch_size or self.settings.batch_size
        batches = []
        
        for i in range(0, len(lines), size):
            batches.append(lines[i : i + size])
        
        return batches

    async def process_batch(
        self,
        batch: TranslationBatch,
        batch_index: int,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        config_override: Optional["TranslationConfig"] = None,
    ) -> BatchResult:
        """
        Process a single batch with retry logic.

        Args:
            batch: The batch to translate
            batch_index: Index of this batch
            model: Optional model override
            temperature: Optional temperature override
            config_override: Optional per-request configuration override

        Returns:
            BatchResult with translations or error
        """
        retries = 0
        last_error: Optional[str] = None

        while retries <= self.settings.max_retries:
            try:
                result = await self.provider.translate_batch(
                    batch, model=model, temperature=temperature, config_override=config_override
                )
                
                return BatchResult(
                    batch_index=batch_index,
                    success=True,
                    translations=result.translations,
                    tokens_used=result.total_tokens or 0,
                    retries=retries,
                )
                
            except RateLimitError as e:
                # Special handling for rate limits
                delay = e.retry_after or (self.settings.retry_delay * (2 ** retries))
                logger.warning(
                    f"Rate limit hit on batch {batch_index}, waiting {delay}s (retry {retries + 1})"
                )
                await asyncio.sleep(delay)
                retries += 1
                last_error = str(e)
                
            except TranslationProviderError as e:
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
                    return BatchResult(
                        batch_index=batch_index,
                        success=False,
                        error=e.message,
                        retries=retries,
                    )
                    
            except Exception as e:
                logger.error(f"Unexpected error on batch {batch_index}: {str(e)}")
                return BatchResult(
                    batch_index=batch_index,
                    success=False,
                    error=str(e),
                    retries=retries,
                )

        # Max retries exceeded
        return BatchResult(
            batch_index=batch_index,
            success=False,
            error=f"Max retries exceeded. Last error: {last_error}",
            retries=retries,
        )

    async def _process_batch_group(
        self,
        batch_group: List[tuple[int, list[dict[str, str]]]],
        source_language: str,
        target_language: str,
        context_title: Optional[str],
        context_media_type: Optional[str],
        model: Optional[str],
        temperature: Optional[float],
        config_override: Optional["TranslationConfig"],
    ) -> List[BatchResult]:
        """
        Process a group of batches in parallel.

        Args:
            batch_group: List of (batch_index, batch_lines) tuples
            source_language: Source language code
            target_language: Target language code
            context_title: Optional media title for context
            context_media_type: Optional media type (Episode/Movie)
            model: Optional model override
            temperature: Optional temperature override
            config_override: Optional per-request configuration override

        Returns:
            List of BatchResults in the same order as input
        """
        tasks = []
        for batch_index, batch_lines in batch_group:
            batch = TranslationBatch(
                lines=batch_lines,
                source_language=source_language,
                target_language=target_language,
                context_title=context_title,
                context_media_type=context_media_type,
            )
            task = self.process_batch(
                batch,
                batch_index=batch_index,
                model=model,
                temperature=temperature,
                config_override=config_override,
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

    async def process_all_batches(
        self,
        lines: list[dict[str, str]],
        source_language: str,
        target_language: str,
        context_title: Optional[str] = None,
        context_media_type: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
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
        batches = self.create_batches(lines, batch_size)
        
        # Determine model to use (config override takes precedence)
        if config_override and config_override.model:
            model_to_use = config_override.model
        else:
            model_to_use = model or self.settings.openrouter_default_model
        
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
        
        logger.info(f"Processing {len(batches)} batches with {parallel_count} parallel batches per group - "
                   f"source={source_language}, target={target_language}, "
                   f"model={model_to_use}, temperature={temperature or 'default'}")

        # Process batches in parallel groups
        for group_start in range(0, len(indexed_batches), parallel_count):
            batch_group = indexed_batches[group_start:group_start + parallel_count]
            
            group_indices = [idx for idx, _ in batch_group]
            logger.info(f"Processing parallel batch group: batches {group_indices} "
                       f"({len(batch_group)} batches in parallel)")
            
            # Process batch group in parallel
            group_results = await self._process_batch_group(
                batch_group,
                source_language=source_language,
                target_language=target_language,
                context_title=context_title,
                context_media_type=context_media_type,
                model=model,
                temperature=temperature,
                config_override=config_override,
            )
            
            # Process results from this group
            for (batch_index, batch_lines), result in zip(batch_group, group_results):
                batch_results.append(result)
                
                if result.success:
                    all_translations.extend(result.translations)
                    progress.completed_lines += len(batch_lines)
                    progress.total_tokens += result.tokens_used
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
        context_title: Optional[str] = None,
        context_media_type: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        batch_size: Optional[int] = None,
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
        batches = self.create_batches(lines, batch_size)
        
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
                batch, batch_index=i, model=model, temperature=temperature, config_override=config_override
            )

            if result.success:
                progress.completed_lines += len(batch_lines)
                progress.total_tokens += result.tokens_used
            else:
                progress.failed_batches += 1

            progress.completed_batches += 1

            yield result, progress


def get_batch_processor(provider: TranslationProvider) -> BatchProcessor:
    """Factory function to get a batch processor instance."""
    return BatchProcessor(provider)