"""Worker functions for processing translation jobs."""

import logging
from typing import Any

from subtitle_translator.api.models import TranslateContentRequest, TranslationConfig
from subtitle_translator.core.batch_processor import (
    BatchProcessor,
    BatchProgress,
    summarize_batch_failure,
)
from subtitle_translator.core.translator import (
    SubtitleTranslator,
    get_translator,
    map_translations_to_lines,
)
from subtitle_translator.queue.job_manager import JobManager, JobType

logger = logging.getLogger(__name__)


async def process_content_translation_job(
    job_manager: JobManager,
    job_id: str,
    translator: SubtitleTranslator,
) -> None:
    """
    Process a content translation job.

    Args:
        job_manager: The job manager instance
        job_id: The job ID to process
        translator: The subtitle translator instance
    """
    job = job_manager.get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} not found")
        return

    try:
        # Parse request data
        request = TranslateContentRequest(**job.request_data)
        model = (
            (request.config.model if request.config else None)
            or request.model
            or translator.settings.openrouter_default_model
        )
        job_manager.set_job_model(job_id, model)

        if not request.lines:
            job_manager.set_job_completed(job_id, {"lines": [], "model_used": "", "tokens_used": 0})
            return

        # Log request metadata (without actual text content)
        logger.info(
            f"Job {job_id}: Processing translation request - "
            f"source={request.sourceLanguage}, target={request.targetLanguage}, "
            f"lines={len(request.lines)}, title='{request.title or 'N/A'}', "
            f"mediaType='{request.mediaType or 'N/A'}', model='{request.model or 'default'}', "
            f"temperature={request.temperature}"
        )

        # Log incoming request config (without sensitive data)
        raw_config = job.request_data.get("config")
        if raw_config:
            safe_config = {}
            for key, value in request.config.model_dump(exclude_none=True).items():
                if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower():
                    safe_config[key] = "***" if value else None
                else:
                    safe_config[key] = value
            logger.info(f"Job {job_id}: Request config: {safe_config}")

        # Extract config override from request
        config_override = _extract_config_override(request.config, job.request_data)

        # Restore API key from job-level storage (stripped from request_data for security)
        if job.api_key_override:
            if config_override is None:
                config_override = TranslationConfig(api_key=job.api_key_override)
            elif not config_override.api_key:
                config_override.api_key = job.api_key_override

        # Log API key source (no key material)
        if config_override and config_override.api_key:
            logger.debug(f"Job {job_id}: Using per-request API key")
        else:
            logger.debug(f"Job {job_id}: Using default API key")

        # Log reasoning and provider settings if configured
        if config_override and (config_override.reasoning or config_override.use_thinking_variant):
            logger.info(
                f"Job {job_id}: Reasoning config - enabled={bool(config_override.reasoning)}, "
                f"use_thinking_variant={bool(config_override.use_thinking_variant)}"
            )
        if config_override and config_override.provider:
            logger.info(f"Job {job_id}: Provider config configured")

        # Convert request lines to internal format
        lines = [{"index": str(line.position), "content": line.line} for line in request.lines]

        # Reconstruct the same opaque session after queue recovery.
        config_override = (config_override or TranslationConfig()).for_operation(job_id)

        # Create batch processor
        processor = BatchProcessor(translator.provider, translator.settings)

        # Define progress callback
        def progress_callback(progress: BatchProgress) -> None:
            percent = int(progress.percent_complete)
            failed_info = f", {progress.failed_batches} failed" if progress.failed_batches else ""
            message = progress.message or (
                f"Translated {progress.completed_lines}/{progress.total_lines} lines "
                f"({progress.completed_batches}/{progress.total_batches} batches{failed_info})"
            )
            job_manager.update_progress(
                job_id,
                percent,
                message,
                total_batches=progress.total_batches,
                completed_batches=progress.completed_batches,
                completed_lines=progress.completed_lines,
                tokens_used=progress.total_tokens,
                total_cost=progress.total_cost,
            )

        # Process all batches
        result = await processor.process_all_batches(
            lines=lines,
            source_language=request.sourceLanguage,
            target_language=request.targetLanguage,
            context_title=request.title,
            context_media_type=request.mediaType,
            model=request.model,
            temperature=request.temperature,
            progress_callback=progress_callback,
            config_override=config_override,
        )

        if not result.success:
            error_summary = summarize_batch_failure(result, job.total_batches)

            # Report as partial if we have some results, otherwise fail
            if result.all_translations:
                translated_lines = map_translations_to_lines(
                    request.lines,
                    result.all_translations,
                    request.targetLanguage,
                    translator.settings,
                )
                # Count request lines, not distinct positions: a request may repeat a
                # position and the translation is applied to every line carrying it,
                # while a returned index nobody asked for is not a line. The set of
                # returned indices keeps a duplicated reply from counting twice.
                total_lines = len(request.lines)
                returned_indices = {str(t["index"]) for t in result.all_translations}
                translated_count = sum(
                    1 for line in request.lines if str(line.position) in returned_indices
                )
                job_manager.set_job_partial(
                    job_id,
                    {
                        "lines": [line.model_dump() for line in translated_lines],
                        "model_used": result.model_used,
                        "tokens_used": result.total_tokens,
                    },
                    error=f"{translated_count}/{total_lines} lines translated. {error_summary}",
                )
            else:
                job_manager.set_job_failed(
                    job_id,
                    error_summary,
                )
            return

        # Map translations back to SubtitleLine format
        translated_lines = map_translations_to_lines(
            request.lines,
            result.all_translations,
            request.targetLanguage,
            translator.settings,
        )

        job_manager.set_job_completed(
            job_id,
            {
                "lines": [line.model_dump() for line in translated_lines],
                "model_used": result.model_used,
                "tokens_used": result.total_tokens,
            },
        )

    except Exception as e:
        logger.exception(f"Content translation job {job_id} failed: {e}")
        job_manager.set_job_failed(job_id, str(e))


async def process_file_translation_job(
    job_manager: JobManager,
    job_id: str,
    translator: SubtitleTranslator,
) -> None:
    """
    Process a file translation job.

    Args:
        job_manager: The job manager instance
        job_id: The job ID to process
        translator: The subtitle translator instance
    """
    job = job_manager.get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} not found")
        return

    try:
        # Extract request parameters
        request_data = job.request_data
        content = request_data.get("content", "")
        source_language = request_data.get("sourceLanguage", "")
        target_language = request_data.get("targetLanguage", "")
        title = request_data.get("title")
        model = request_data.get("model")
        temperature = request_data.get("temperature")

        # Log file translation request metadata (without actual content)
        content_length = len(content) if content else 0
        logger.info(
            f"Job {job_id}: Processing file translation - "
            f"source={source_language}, target={target_language}, "
            f"content_length={content_length}, title='{title or 'N/A'}', "
            f"model='{model or 'default'}', temperature={temperature or 'default'}"
        )

        # Extract config override from request data
        config_override = _extract_config_override_from_dict(
            request_data.get("config"),
            fallback_model=model or translator.settings.openrouter_default_model,
        )
        model_to_use = (
            (config_override.model if config_override else None)
            or model
            or translator.settings.openrouter_default_model
        )
        job_manager.set_job_model(job_id, model_to_use)

        # Restore API key from job-level storage (stripped from request_data for security)
        if job.api_key_override:
            if config_override is None:
                config_override = TranslationConfig(api_key=job.api_key_override)
            elif not config_override.api_key:
                config_override.api_key = job.api_key_override

        # Log request config for file translation
        if request_data.get("config"):
            safe_config = {}
            for key, value in (
                (config_override or TranslationConfig()).model_dump(exclude_none=True).items()
            ):
                if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower():
                    safe_config[key] = "***" if value else None
                else:
                    safe_config[key] = value
            logger.info(f"Job {job_id}: File translation config: {safe_config}")

        # Log API key source (no key material)
        if config_override and config_override.api_key:
            logger.debug(f"Job {job_id}: Using per-request API key")
        else:
            logger.debug(f"Job {job_id}: Using default API key")

        if not content or not content.strip():
            job_manager.set_job_failed(job_id, "SRT content is required")
            return

        # Parse and validate SRT content
        try:
            entries = translator._srt_parser.parse(content)
        except Exception as parse_error:
            job_manager.set_job_failed(job_id, f"Invalid SRT content: {parse_error}")
            return
        if not entries:
            job_manager.set_job_completed(
                job_id,
                {
                    "content": content,
                    "model_used": model or translator.settings.openrouter_default_model,
                    "tokens_used": 0,
                    "subtitle_count": 0,
                },
            )
            return

        # Extract lines for translation
        lines = translator._srt_parser.extract_lines_for_translation(entries)

        # Update total_lines now that we know the actual count
        if job_id in job_manager.jobs:
            job_manager.jobs[job_id].total_lines = len(lines)

        # Reconstruct the same opaque session after queue recovery.
        config_override = (config_override or TranslationConfig()).for_operation(job_id)

        # Create batch processor
        processor = BatchProcessor(translator.provider, translator.settings)

        # Define progress callback
        def progress_callback(progress: BatchProgress) -> None:
            percent = int(progress.percent_complete)
            failed_info = f", {progress.failed_batches} failed" if progress.failed_batches else ""
            message = progress.message or (
                f"Translated {progress.completed_lines}/{progress.total_lines} lines "
                f"({progress.completed_batches}/{progress.total_batches} batches{failed_info})"
            )
            job_manager.update_progress(
                job_id,
                percent,
                message,
                total_batches=progress.total_batches,
                completed_batches=progress.completed_batches,
                completed_lines=progress.completed_lines,
                tokens_used=progress.total_tokens,
                total_cost=progress.total_cost,
            )

        # Process all batches
        result = await processor.process_all_batches(
            lines=lines,
            source_language=source_language,
            target_language=target_language,
            context_title=title,
            model=model,
            temperature=temperature,
            progress_callback=progress_callback,
            config_override=config_override,
        )

        if not result.success:
            error_summary = summarize_batch_failure(result, job.total_batches)

            if result.all_translations:
                is_rtl = translator.settings.is_rtl_language(target_language)
                translated_entries = translator._srt_parser.apply_translations(
                    entries, result.all_translations, is_rtl=is_rtl
                )
                translated_content = translator._srt_parser.compose(translated_entries)
                # Count entries, not distinct cue numbers: a file may repeat a number
                # and the translation is applied to every entry carrying it.
                returned_indices = {str(t["index"]) for t in result.all_translations}
                translated_count = sum(
                    1 for line in lines if str(line["index"]) in returned_indices
                )
                total_count = len(lines)

                job_manager.set_job_partial(
                    job_id,
                    {
                        "content": translated_content,
                        "model_used": result.model_used,
                        "tokens_used": result.total_tokens,
                        "subtitle_count": len(entries),
                    },
                    error=f"{translated_count}/{total_count} lines translated. {error_summary}",
                )
            else:
                job_manager.set_job_failed(
                    job_id,
                    error_summary,
                )
            return

        # Check if target language is RTL
        is_rtl = translator.settings.is_rtl_language(target_language)

        # Apply translations back to entries
        translated_entries = translator._srt_parser.apply_translations(
            entries, result.all_translations, is_rtl=is_rtl
        )

        # Optionally split long subtitles
        translated_entries = translator._srt_parser.split_long_subtitles(translated_entries)

        # Compose back to SRT format
        translated_content = translator._srt_parser.compose(translated_entries)

        job_manager.set_job_completed(
            job_id,
            {
                "content": translated_content,
                "model_used": result.model_used,
                "tokens_used": result.total_tokens,
                "subtitle_count": len(entries),
            },
        )

    except Exception as e:
        logger.exception(f"File translation job {job_id} failed: {e}")
        job_manager.set_job_failed(job_id, str(e))


def _extract_config_override(
    config: TranslationConfig | None,
    request_data: dict[str, Any],
) -> TranslationConfig | None:
    """
    Extract TranslationConfig from request, handling both parsed model and raw dict.

    Args:
        config: Parsed TranslationConfig if available
        request_data: Raw request data dictionary

    Returns:
        TranslationConfig if present, None otherwise
    """
    if config is not None:
        return config

    # Try to extract from raw request data
    return _extract_config_override_from_dict(
        request_data.get("config"), fallback_model=request_data.get("model")
    )


def _extract_config_override_from_dict(
    config_dict: dict[str, Any] | None,
    *,
    fallback_model: str | None = None,
) -> TranslationConfig | None:
    """
    Extract TranslationConfig from a dictionary.

    Args:
        config_dict: Raw config dictionary from request
        fallback_model: Top-level model or settings default used without a config model

    Returns:
        TranslationConfig if valid dict provided, None otherwise
    """
    if config_dict is None:
        return None

    configured_model = config_dict.get("model") if isinstance(config_dict, dict) else None
    # Invalid options must not erase a requested SmartFast mode or restore a
    # SmartFast fallback while silently dropping explicit provider restrictions.
    smartfast_model = any(
        isinstance(model, str) and "smartfast" in model.rsplit("/", 1)[-1].split(":")[1:]
        for model in (configured_model, fallback_model)
    )
    if not isinstance(config_dict, dict):
        if smartfast_model:
            raise ValueError("Invalid SmartFast routing configuration")
        return None

    # Check if any fields are present
    if not any(
        key in config_dict
        for key in [
            "apiKey",
            "api_key",
            "model",
            "temperature",
            "requestTimeout",
            "request_timeout",
            "maxConcurrentJobs",
            "max_concurrent_jobs",
            "reasoning",
            "provider",
            "serviceTier",
            "service_tier",
            "parallelBatches",
            "parallel_batches",
        ]
    ):
        return None

    try:
        return TranslationConfig(**config_dict)
    except Exception:
        provider_config = config_dict.get("provider")
        if smartfast_model or (
            isinstance(provider_config, dict)
            and (
                str(provider_config.get("sort", "")).strip().lower() == "smartfast"
                or "smartFast" in provider_config
                or "smart_fast" in provider_config
            )
        ):
            raise ValueError("Invalid SmartFast routing configuration") from None
        logger.warning("Failed to validate stored translation configuration")
        return None


async def job_worker_handler(
    job_manager: JobManager,
    job_id: str,
    job_type: JobType,
) -> None:
    """
    Main worker handler that routes jobs to appropriate processors.

    This function is called by the job manager's worker threads.

    Args:
        job_manager: The job manager instance
        job_id: The job ID to process
        job_type: The type of job
    """
    job = job_manager.get_job(job_id)
    if job is not None and "_ui_owner" in job.request_data:
        if not job.api_key_override or job.api_key_override.startswith("enc:"):
            job_manager.set_job_failed(
                job_id, "OpenRouter key is unavailable after restart. Submit this file again."
            )
            return

    # Get translator instance
    translator = await get_translator()

    if job_type == JobType.TRANSLATE_CONTENT:
        await process_content_translation_job(job_manager, job_id, translator)
    elif job_type == JobType.TRANSLATE_FILE:
        await process_file_translation_job(job_manager, job_id, translator)
    else:
        job_manager.set_job_failed(job_id, f"Unknown job type: {job_type}")
