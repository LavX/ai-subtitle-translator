"""Batch processing logic for subtitle translation."""

import asyncio
import logging
from collections import Counter, defaultdict, deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from subtitle_translator.config import Settings, get_settings
from subtitle_translator.providers.base import (
    InvalidResponseError,
    ProviderTimeoutError,
    RateLimitError,
    TranslationBatch,
    TranslationProvider,
    TranslationProviderError,
)

if TYPE_CHECKING:
    from subtitle_translator.api.models import TranslationConfig

logger = logging.getLogger(__name__)

# How many times a stalled request above the floor is reissued unchanged before
# the batch falls back to smaller requests. Each retry costs a full request timeout,
# so the batch deadline grows by the same amount and the last window is always kept
# for the fallback: a batch never spends its whole budget on identical attempts.
SAME_SIZE_TIMEOUT_RETRIES = 1

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
    message: str = ""

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
    timed_out: bool = False


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


def summarize_batch_failure(result: BatchProcessingResult, total_batches: int | None) -> str:
    """Describe attempted failures and any work stopped before an attempt."""
    failed_batches = [batch for batch in result.batch_results if not batch.success]
    error = "; ".join(batch.error or "Unknown error" for batch in failed_batches)
    attempted = len(result.batch_results)
    total = total_batches if total_batches is not None else attempted
    not_attempted = max(0, total - attempted)
    if not_attempted:
        reason = result.progress.message
        stop_reason = f"{reason.rstrip('.')}. " if reason else ""
        return (
            f"{attempted} of {total} batches attempted; {len(failed_batches)} failed; "
            f"{not_attempted} not attempted. {stop_reason}{error}"
        )
    if result.all_translations:
        return f"{len(failed_batches)} of {attempted} batches failed: {error}"
    return f"All {len(failed_batches)} batches failed: {error}"


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
        _deadline: float | None = None,
        _activity_callback: Callable[[str], None] | None = None,
        _check_learned_size: bool = True,
        _prior_retries: int = 0,
        _prior_attempts: int = 0,
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
        timeout_retries = 0
        last_error: str | None = None
        loop = asyncio.get_running_loop()
        request_timeout = (
            config_override.request_timeout
            if config_override and config_override.request_timeout is not None
            else self.settings.request_timeout
        )
        if _deadline is None:
            _deadline = loop.time() + (2 + SAME_SIZE_TIMEOUT_RETRIES) * request_timeout

        def activity(message: str) -> None:
            if _activity_callback:
                _activity_callback(f"Batch {batch_index + 1}: {message}")

        async def retry_wait(delay: float) -> None:
            await asyncio.sleep(min(delay, max(0, _deadline - loop.time())))

        # Usage of an attempt that is retried is still billed; it is carried into the
        # eventual result so the job totals stay honest.
        spent_tokens = 0
        spent_cost = 0.0
        non_timeout_failure = False
        retained: dict[str, dict[str, str]] = {}
        requested = {str(line["index"]) for line in batch.lines}

        def _billed(outcome: BatchResult, add_retries: bool = False) -> BatchResult:
            # Whatever the outcome, the attempts made before it were billed.
            outcome.tokens_used += spent_tokens
            outcome.cost += spent_cost
            if add_retries:
                outcome.retries += retries
            outcome.timed_out = outcome.timed_out and not non_timeout_failure
            merged = dict(retained)
            merged.update(
                {str(t["index"]): t for t in outcome.translations if str(t["index"]) in requested}
            )
            outcome.translations = list(merged.values())
            return outcome

        can_adaptive = not _is_adaptive_retry and len(batch.lines) > MIN_BATCH_SIZE
        # Rate limits get extra retries (3 more than normal errors)
        max_retries = max(0, self.settings.max_retries - _prior_retries)
        max_retries_with_rate_limit = self.settings.max_retries + 3 - _prior_retries
        model_id = (
            (config_override.model if config_override and config_override.model else None)
            or model
            or self.settings.openrouter_default_model
        )

        def _note_unsplittable_failure(size_related: bool) -> None:
            # Without a split there is no record_failure call. A sub-batch of an adaptive
            # retry above the floor that failed on an invalid or incomplete response or a
            # timeout still failed at its size, and the cache may already have grown past
            # it on earlier sub-batches, so it is recorded like any other failure. A rate
            # limit, an authentication error or a local exception says nothing about the
            # size; those only interrupt the streak that grows the size back.
            resolver = get_batch_size_resolver()
            if size_related and _is_adaptive_retry and len(batch.lines) > MIN_BATCH_SIZE:
                resolver.record_failure(model_id, len(batch.lines))
            else:
                resolver.record_floor_failure(model_id)

        while retries < max_retries_with_rate_limit:
            if loop.time() >= _deadline:
                activity("timeout budget exhausted; stopping this batch")
                return _billed(
                    BatchResult(
                        batch_index=batch_index,
                        success=False,
                        timed_out=True,
                        error="Batch timeout budget exhausted",
                        retries=retries,
                    )
                )
            # Backoff yields to other roots, which may lower the learned cap.
            # Split only to a strictly smaller size and keep the spent retry budget.
            if _check_learned_size or retries:
                learned_size = get_batch_size_resolver().limit_planned_size(
                    model_id, len(batch.lines)
                )
                if learned_size < len(batch.lines):
                    activity(f"using learned limit of {learned_size} lines per request")
                    return _billed(
                        await self._process_sub_batches(
                            batch,
                            batch_index,
                            learned_size,
                            model,
                            temperature,
                            config_override,
                            _rate_limit_lock,
                            _deadline,
                            _activity_callback,
                            _is_adaptive_retry=_is_adaptive_retry,
                            _prior_retries=_prior_retries + retries,
                            _prior_attempts=_prior_attempts + retries,
                        ),
                        add_retries=True,
                    )
            # Publish only service-generated activity, without provider response text.
            # Recovery keeps working on the same batch, so the attempt number carries the
            # attempts already spent on it. Reporting the local counter restarted at one on
            # every shrink and hid batches that had already burnt several request budgets.
            activity(
                f"request in progress for {len(batch.lines)} lines "
                f"(attempt {_prior_attempts + retries + 1})"
            )
            try:
                try:
                    async with asyncio.timeout_at(_deadline):
                        result = await self.provider.translate_batch(
                            batch,
                            model=model,
                            temperature=temperature,
                            config_override=config_override,
                        )
                except TimeoutError as e:
                    raise ProviderTimeoutError("Batch timeout budget exhausted") from e

                # Every requested position has to come back. A response with the right
                # count but substituted or repeated indices leaves lines untranslated and
                # used to pass as a success that also grew the learned size.
                retained.update(
                    {
                        str(t["index"]): t
                        for t in result.translations
                        if str(t["index"]) in requested
                    }
                )
                # Coverage is cumulative over the attempts of this batch: a retry that
                # brings the cues the previous reply left out completes the batch.
                covered = len(retained)
                if covered < len(requested):
                    non_timeout_failure = True
                    logger.warning(
                        f"Batch {batch_index + 1}: got {covered}/{len(requested)} translations"
                        + (f" ({result.note})" if result.note else "")
                    )
                    spent_tokens += result.total_tokens or 0
                    spent_cost += result.cost or 0.0
                    if can_adaptive:
                        return _billed(
                            await self._retry_with_smaller_batches(
                                batch,
                                batch_index,
                                model,
                                temperature,
                                config_override,
                                _rate_limit_lock,
                                _deadline,
                                _activity_callback,
                                _prior_retries=_prior_retries + retries,
                                _prior_attempts=_prior_attempts + retries + 1,
                            ),
                            add_retries=True,
                        )
                    # A sub-batch or a floor batch cannot split again. A partial reply
                    # is as transient as an unparsable one (a model that answers five
                    # lines with one usually answers all five on the next attempt), so
                    # it gets the same retries before it counts as a failure; a floor
                    # batch used to fail on the first partial reply, which turned one
                    # bad answer into "all 72 batches failed".
                    last_error = f"Partial translations: expected {len(requested)}, got {covered}"
                    if result.note:
                        last_error += f" ({result.note})"
                    if retries < max_retries:
                        retries += 1
                        activity(f"retry {retries} after incomplete or invalid response")
                        await retry_wait(self.settings.retry_delay * (2 ** (retries - 1)))
                        continue
                    _note_unsplittable_failure(size_related=True)
                    return _billed(
                        BatchResult(
                            batch_index=batch_index,
                            success=False,
                            error=last_error,
                            retries=retries,
                        )
                    )

                get_batch_size_resolver().record_success(model_id, len(batch.lines))

                return BatchResult(
                    batch_index=batch_index,
                    success=True,
                    translations=list(retained.values()),
                    tokens_used=spent_tokens + (result.total_tokens or 0),
                    cost=spent_cost + (result.cost or 0.0),
                    retries=retries,
                )

            except InvalidResponseError as e:
                non_timeout_failure = True
                spent_tokens += e.tokens_used
                spent_cost += e.cost
                if can_adaptive:
                    return _billed(
                        await self._retry_with_smaller_batches(
                            batch,
                            batch_index,
                            model,
                            temperature,
                            config_override,
                            _rate_limit_lock,
                            _deadline,
                            _activity_callback,
                            _prior_retries=_prior_retries + retries,
                            _prior_attempts=_prior_attempts + retries + 1,
                        ),
                        add_retries=True,
                    )
                # At floor or already in adaptive retry - use normal retry
                if retries < max_retries:
                    retries += 1
                    last_error = e.message
                    activity(f"retry {retries} after invalid response")
                    await retry_wait(self.settings.retry_delay * (2 ** (retries - 1)))
                else:
                    _note_unsplittable_failure(size_related=True)
                    return _billed(
                        BatchResult(
                            batch_index=batch_index,
                            success=False,
                            error=e.message,
                            retries=retries,
                        )
                    )

            except RateLimitError as e:
                non_timeout_failure = True
                spent_tokens += e.tokens_used
                spent_cost += e.cost
                # Rate limits: start at 5s, exponential backoff, cap at 30s.
                # Use lock to serialize retries so parallel batches don't all
                # hammer the API simultaneously after a 429.
                rate_limit_base_delay = 5.0
                delay = (
                    e.retry_after
                    if e.retry_after is not None
                    else min(rate_limit_base_delay * (2**retries), 30.0)
                )
                retries += 1
                activity(f"rate limited; retry {retries} after {delay:g}s backoff")
                try:
                    async with asyncio.timeout_at(_deadline):
                        if _rate_limit_lock:
                            async with _rate_limit_lock:
                                await retry_wait(delay)
                        else:
                            await retry_wait(delay)
                except TimeoutError:
                    activity("rate-limit retry budget exhausted; stopping this batch")
                    return _billed(
                        BatchResult(
                            batch_index=batch_index,
                            success=False,
                            error="Rate-limit retry budget exhausted",
                            retries=retries,
                        )
                    )
                last_error = str(e)

            except ProviderTimeoutError as e:
                spent_tokens += e.tokens_used
                spent_cost += e.cost
                exhausted = loop.time() >= _deadline
                # A stalled request carries no information about batch size. Measured
                # against the live deployment, requests intermittently never return at
                # any size: a 5-line request timed out after 600s in the same window a
                # 100-line request finished in 28s. Shrinking spends another full
                # request budget per rung and cannot reach a size that works, so a
                # timeout retries the same request instead, within the batch deadline.
                # Only above the floor: at the floor a stall keeps its reviewed early
                # stop. And only while more than one attempt's worth of budget remains,
                # so the smaller-request fallback always keeps its own window.
                if (
                    not exhausted
                    and can_adaptive
                    and timeout_retries < SAME_SIZE_TIMEOUT_RETRIES
                    and (_deadline - loop.time()) > request_timeout
                ):
                    timeout_retries += 1
                    retries += 1
                    last_error = e.message
                    activity("request timed out; retrying at the same size")
                    continue
                activity("timeout budget exhausted" if exhausted else "request timed out")
                if can_adaptive and not exhausted:

                    def recovery_activity(message: str) -> None:
                        if _activity_callback:
                            _activity_callback(f"{message} (recovering after timeout)")

                    # Smaller requests remain the last resort for a batch that keeps
                    # stalling, but the split must not be recorded as a size failure.
                    return _billed(
                        await self._retry_with_smaller_batches(
                            batch,
                            batch_index,
                            model,
                            temperature,
                            config_override,
                            _rate_limit_lock,
                            _deadline,
                            recovery_activity,
                            _prior_retries=_prior_retries + retries,
                            _prior_attempts=_prior_attempts + retries + 1,
                            _record_size_failure=False,
                        ),
                        add_retries=True,
                    )
                # Never teach the size resolver from a stall: doing so dropped every
                # later batch of the job to the floor after a single stalled request.
                _note_unsplittable_failure(size_related=False)
                return _billed(
                    BatchResult(
                        batch_index=batch_index,
                        success=False,
                        timed_out=True,
                        error="Batch timeout budget exhausted" if exhausted else e.message,
                        retries=retries,
                    )
                )

            except TranslationProviderError as e:
                non_timeout_failure = True
                spent_tokens += e.tokens_used
                spent_cost += e.cost
                if e.retryable and retries < max_retries:
                    delay = (
                        e.retry_after
                        if e.retry_after is not None
                        else self.settings.retry_delay * (2**retries)
                    )
                    logger.warning(
                        f"Retryable error on batch {batch_index}: {e.message}, "
                        f"waiting {delay}s (retry {retries + 1})"
                    )
                    activity(f"provider error; retry {retries + 1} after {delay:g}s backoff")
                    await retry_wait(delay)
                    retries += 1
                    last_error = e.message
                else:
                    logger.error(f"Non-retryable error on batch {batch_index}: {e.message}")
                    last_error = e.message
                    break

            except Exception as e:
                logger.error(f"Unexpected error on batch {batch_index}: {str(e)}")
                _note_unsplittable_failure(size_related=False)
                return _billed(
                    BatchResult(
                        batch_index=batch_index,
                        success=False,
                        error=str(e),
                        retries=retries,
                    )
                )

        _note_unsplittable_failure(size_related=False)
        return _billed(
            BatchResult(
                batch_index=batch_index,
                success=False,
                error=f"Max retries exceeded. Last error: {last_error}",
                retries=retries,
            )
        )

    async def _retry_with_smaller_batches(
        self,
        batch: TranslationBatch,
        batch_index: int,
        model: str | None = None,
        temperature: float | None = None,
        config_override: Optional["TranslationConfig"] = None,
        _rate_limit_lock: asyncio.Lock | None = None,
        _deadline: float | None = None,
        _activity_callback: Callable[[str], None] | None = None,
        _prior_retries: int = 0,
        _prior_attempts: int = 0,
        _record_size_failure: bool = True,
    ) -> BatchResult:
        """Retry a failed batch by splitting it into smaller sub-batches."""
        from subtitle_translator.core.batch_sizing import MIN_BATCH_SIZE, get_batch_size_resolver

        model_id = (
            (config_override.model if config_override and config_override.model else None)
            or model
            or self.settings.openrouter_default_model
        )
        resolver = get_batch_size_resolver()
        if _record_size_failure:
            new_size = resolver.record_failure(model_id, len(batch.lines))
        else:
            # Split without teaching the resolver. A stalled request says nothing about
            # size, and recording it capped every later batch of the job at the floor.
            new_size = max(MIN_BATCH_SIZE, len(batch.lines) // 2)

        logger.warning(
            f"Batch {batch_index + 1}: adaptive retry with size {new_size} (was {len(batch.lines)})"
        )

        if _activity_callback:
            _activity_callback(
                f"Batch {batch_index + 1}: recovering with smaller {new_size}-line requests"
            )

        return await self._process_sub_batches(
            batch,
            batch_index,
            new_size,
            model,
            temperature,
            config_override,
            _rate_limit_lock,
            _deadline,
            _activity_callback,
            _is_adaptive_retry=True,
            _prior_retries=_prior_retries,
            _prior_attempts=_prior_attempts,
        )

    async def _process_sub_batches(
        self,
        batch: TranslationBatch,
        batch_index: int,
        batch_size: int,
        model: str | None = None,
        temperature: float | None = None,
        config_override: Optional["TranslationConfig"] = None,
        _rate_limit_lock: asyncio.Lock | None = None,
        _deadline: float | None = None,
        _activity_callback: Callable[[str], None] | None = None,
        _is_adaptive_retry: bool = False,
        _prior_retries: int = 0,
        _prior_attempts: int = 0,
    ) -> BatchResult:
        """Dispatch smaller requests within the original root's budget and result."""
        from subtitle_translator.core.batch_sizing import get_batch_size_resolver

        model_id = (
            (config_override.model if config_override and config_override.model else None)
            or model
            or self.settings.openrouter_default_model
        )
        resolver = get_batch_size_resolver()

        all_translations: list[dict[str, str]] = []
        total_tokens = 0
        total_cost = 0.0
        total_retries = 0
        failures: list[str] = []
        failed_timed_out = True
        loop = asyncio.get_running_loop()

        offset = 0
        while offset < len(batch.lines):
            if failures and loop.time() >= _deadline:
                # A failed child already spent the budget; do not queue instant timeouts.
                remaining = batch.lines[offset:]
                failures.append(
                    f"cues {remaining[0]['index']}-{remaining[-1]['index']} not attempted "
                    "(batch budget exhausted)"
                )
                break
            # Another in-flight request may lower the cap between these children.
            # Resizing pending work is not a failure and does not spend a retry.
            size = resolver.limit_planned_size(model_id, batch_size)
            sub_batch_lines = batch.lines[offset : offset + size]
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
                _is_adaptive_retry=_is_adaptive_retry,
                _rate_limit_lock=_rate_limit_lock,
                _deadline=_deadline,
                _activity_callback=_activity_callback,
                # This loop owns scheduling. A child may recover from one actual
                # failure, but cannot recursively split just to apply a learned cap.
                _check_learned_size=False,
                _prior_retries=_prior_retries,
                _prior_attempts=_prior_attempts,
            )
            all_translations.extend(sub_result.translations)
            total_tokens += sub_result.tokens_used
            total_cost += sub_result.cost
            total_retries += sub_result.retries
            offset += len(sub_batch_lines)
            if not sub_result.success:
                # One bad answer must not cost the lines behind it: the siblings are
                # still attempted, and the failure names the cues it covers so the
                # result can say which lines were left in the source language.
                failed_timed_out = failed_timed_out and sub_result.timed_out
                failures.append(
                    f"cues {sub_batch_lines[0]['index']}-{sub_batch_lines[-1]['index']} "
                    f"at size {len(sub_batch_lines)}: {sub_result.error}"
                )

        if failures:
            failure_context = "Adaptive retry" if _is_adaptive_retry else "Smaller batch"
            return BatchResult(
                batch_index=batch_index,
                success=False,
                translations=all_translations,
                tokens_used=total_tokens,
                cost=total_cost,
                error=f"{failure_context} failed for {'; '.join(failures)}",
                retries=total_retries,
                timed_out=failed_timed_out,
            )

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
        translated_indices: set[str] = set()

        # Create indexed batches for tracking
        indexed_batches = list(enumerate(batches))

        # Shared lock so rate-limited retries don't all fire simultaneously
        rate_limit_lock = asyncio.Lock()

        # Fire initial progress so job shows totalBatches immediately
        if progress_callback:
            progress_callback(progress)

        def activity(message: str) -> None:
            progress.message = message
            if progress_callback:
                progress_callback(progress)

        active_messages: dict[int, str] = {}

        async def _run_batch(bi: int, bl: list, stagger: float) -> tuple[int, list, BatchResult]:
            # The root budget starts when the request actually starts, after pacing.
            if stagger > 0:
                await asyncio.sleep(stagger)
            b = TranslationBatch(
                lines=bl,
                source_language=source_language,
                target_language=target_language,
                context_title=context_title,
                context_media_type=context_media_type,
            )

            def batch_activity(message: str) -> None:
                active_messages[bi] = message
                activity(message)

            r = await self.process_batch(
                b,
                batch_index=bi,
                model=model,
                temperature=temperature,
                config_override=config_override,
                _rate_limit_lock=rate_limit_lock,
                _activity_callback=batch_activity,
            )
            return bi, bl, r

        # Roots are admitted on a rolling basis: at most parallel_count in flight,
        # refilled in index order as soon as a slot frees, so one stalled request
        # does not idle the other slots. The fixed cohorts of parallel_count roots
        # remain the unit of the early stop: a cohort whose every result timed out
        # without output stops the job, and the cohort after it is only admitted
        # once one of its results has proved that cannot happen.
        pending = deque(indexed_batches)
        # A request may repeat a position; a translation is applied to every line
        # carrying it, so completed lines count lines, not distinct positions.
        line_counts = Counter(str(line["index"]) for line in lines)
        running: dict[asyncio.Task, tuple[int, list]] = {}
        cohort_sizes = [
            len(indexed_batches[i : i + parallel_count])
            for i in range(0, len(indexed_batches), parallel_count)
        ]
        cohort_done: dict[int, int] = defaultdict(int)
        cohort_disproved: set[int] = set()
        stopped = False

        logger.info(
            f"Processing {len(batches)} batches, up to {parallel_count} in flight - "
            f"source={source_language}, target={target_language}, "
            f"model={model_to_use}, temperature={temperature or 'default'}"
        )

        try:
            while True:
                admitted = 0
                while pending and len(running) < parallel_count and not stopped:
                    batch_index, batch_lines = pending[0]
                    cohort = batch_index // parallel_count
                    if cohort and (cohort - 1) not in cohort_disproved:
                        break
                    pending.popleft()
                    # Roots admitted together are staggered by half a second so they do
                    # not hit the provider as a burst; a single refill of a freed slot
                    # replaces a request that just finished and starts at once.
                    stagger = admitted * 0.5
                    admitted += 1
                    logger.info(f"Batch {batch_index + 1} admitted ({len(running) + 1} in flight)")
                    task = asyncio.ensure_future(_run_batch(batch_index, batch_lines, stagger))
                    running[task] = (batch_index, batch_lines)
                if not running:
                    break
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    running.pop(task)
                    batch_index, batch_lines, result = task.result()
                    batch_results.append(result)
                    progress.total_tokens += result.tokens_used
                    progress.total_cost += result.cost
                    requested = {str(line["index"]) for line in batch_lines}
                    for translation in result.translations:
                        index = str(translation["index"])
                        if index in requested and index not in translated_indices:
                            all_translations.append(translation)
                            translated_indices.add(index)
                    progress.completed_lines = sum(
                        line_counts[index] for index in translated_indices
                    )
                    if not result.success:
                        progress.failed_batches += 1
                        logger.error(f"Batch {batch_index + 1} failed: {result.error}")
                    progress.completed_batches += 1
                    active_messages.pop(batch_index, None)
                    progress.message = next(reversed(active_messages.values()), "")
                    if progress_callback:
                        progress_callback(progress)

                    cohort = batch_index // parallel_count
                    cohort_done[cohort] += 1
                    if not (result.timed_out and not result.translations):
                        cohort_disproved.add(cohort)
                    elif (
                        cohort_done[cohort] == cohort_sizes[cohort]
                        and cohort not in cohort_disproved
                    ):
                        stopped = True
        finally:
            # Parent cancellation and callback errors must finish child cleanup
            # before workers release the provider or persistent progress store.
            for task in running:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*running, return_exceptions=True)

        if stopped and pending:
            activity("Provider requests timed out without usable output; remaining batches stopped")

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

        translated_indices: set[str] = set()
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

            progress.total_tokens += result.tokens_used
            progress.total_cost += result.cost
            requested = {str(line["index"]) for line in batch_lines}
            translated_indices.update(
                str(t["index"]) for t in result.translations if str(t["index"]) in requested
            )
            progress.completed_lines = len(translated_indices)
            if not result.success:
                progress.failed_batches += 1

            progress.completed_batches += 1

            yield result, progress


def get_batch_processor(provider: TranslationProvider) -> BatchProcessor:
    """Factory function to get a batch processor instance."""
    return BatchProcessor(provider)
