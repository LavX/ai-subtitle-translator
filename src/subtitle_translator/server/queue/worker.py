"""Worker functions for processing translation jobs."""

import logging
from collections.abc import Callable
from typing import Any

from subtitle_translator.core.engine import Line, Progress
from subtitle_translator.core.models import SubtitleLine
from subtitle_translator.core.srt_parser import SRTParser
from subtitle_translator.core.translator import SubtitleTranslator
from subtitle_translator.server.api.models import TranslateContentRequest, TranslationConfig
from subtitle_translator.server.queue.job_manager import JobManager, JobType
from subtitle_translator.server.runtime import EngineTranslator, get_translator
from subtitle_translator.server.translate import run_lines, runtime_for

logger = logging.getLogger(__name__)


async def process_content_translation_job(
    job_manager: JobManager,
    job_id: str,
    translator: SubtitleTranslator | EngineTranslator,
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
        runtime = runtime_for(translator)
        config_override = _extract_config_override(request.config, job.request_data)

        # Restore API key from job-level storage (stripped from request_data for security)
        if job.api_key_override:
            if config_override is None:
                config_override = TranslationConfig(api_key=job.api_key_override)
            elif not config_override.api_key:
                config_override.api_key = job.api_key_override

        connections, pinned = runtime.plan(
            config_override, model=request.model, temperature=request.temperature
        )
        job_manager.set_job_model(job_id, runtime.planned_model(connections, pinned))

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

        lines = [Line(line.position, line.line) for line in request.lines]
        result = await run_lines(
            runtime,
            lines,
            source=request.sourceLanguage,
            target=request.targetLanguage,
            title=request.title,
            media_type=request.mediaType,
            config=config_override,
            model=request.model,
            temperature=request.temperature,
            job_id=job_id,
            on_progress=_progress_reporter(job_manager, job_id),
        )
        payload = {
            "lines": [
                SubtitleLine(position=item.position, line=item.text).model_dump()
                for item in result.lines
            ],
            "model_used": result.model_used,
            "tokens_used": result.tokens,
        }
        if result.status == "completed":
            job_manager.set_job_completed(job_id, payload)
        elif result.status == "partial":
            job_manager.set_job_partial(
                job_id,
                payload,
                error=f"{result.translated}/{len(lines)} lines translated. {result.error}",
            )
        else:
            job_manager.set_job_failed(job_id, result.error or "Translation failed")

    except Exception as e:
        logger.exception(f"Content translation job {job_id} failed: {e}")
        job_manager.set_job_failed(job_id, str(e))


async def process_file_translation_job(
    job_manager: JobManager,
    job_id: str,
    translator: SubtitleTranslator | EngineTranslator,
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
        runtime = runtime_for(translator)
        config_override = _extract_config_override_from_dict(
            request_data.get("config"),
            fallback_model=model or runtime.core.openrouter_default_model,
        )

        # Restore API key from job-level storage (stripped from request_data for security)
        if job.api_key_override:
            if config_override is None:
                config_override = TranslationConfig(api_key=job.api_key_override)
            elif not config_override.api_key:
                config_override.api_key = job.api_key_override

        job_manager.set_job_model(
            job_id,
            runtime.planned_model(
                *runtime.plan(config_override, model=model, temperature=temperature)
            ),
        )

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
            parser = SRTParser()
            entries = parser.parse(content)
        except Exception as parse_error:
            job_manager.set_job_failed(job_id, f"Invalid SRT content: {parse_error}")
            return
        if not entries:
            job_manager.set_job_completed(
                job_id,
                {
                    "content": content,
                    "model_used": model or runtime.core.openrouter_default_model,
                    "tokens_used": 0,
                    "subtitle_count": 0,
                },
            )
            return

        entries = parser.with_positions(entries)
        extracted = parser.extract_lines_for_translation(entries)
        lines = [Line(int(item["index"]), item["content"]) for item in extracted]

        # Update total_lines now that we know the actual count
        if job_id in job_manager.jobs:
            job_manager.jobs[job_id].total_lines = len(lines)

        result = await run_lines(
            runtime,
            lines,
            source=source_language,
            target=target_language,
            title=title,
            config=config_override,
            model=model,
            temperature=temperature,
            job_id=job_id,
            add_rtl_marks=False,  # apply_translations marks RTL, as in 2.1.1
            on_progress=_progress_reporter(job_manager, job_id),
        )
        if result.status == "failed":
            job_manager.set_job_failed(job_id, result.error or "Translation failed")
            return
        translations = [
            {"index": str(line.position), "content": line.text}
            for line in result.lines
            if line.position in result.done
        ]
        translated_entries = parser.apply_translations(
            entries, translations, is_rtl=runtime.core.is_rtl_language(target_language)
        )
        if result.status == "completed":
            translated_entries = parser.split_long_subtitles(translated_entries)
        payload = {
            "content": parser.compose(translated_entries),
            "model_used": result.model_used,
            "tokens_used": result.tokens,
            "subtitle_count": len(entries),
        }
        if result.status == "completed":
            job_manager.set_job_completed(job_id, payload)
        else:
            job_manager.set_job_partial(
                job_id,
                payload,
                error=f"{result.translated}/{len(lines)} lines translated. {result.error}",
            )

    except Exception as e:
        logger.exception(f"File translation job {job_id} failed: {e}")
        job_manager.set_job_failed(job_id, str(e))


def _progress_reporter(job_manager: JobManager, job_id: str) -> Callable[[Progress], None]:
    def report(progress: Progress) -> None:
        percent = (
            int(100 * progress.done_batches / progress.total_batches)
            if progress.total_batches
            else 100
        )
        failed = f", {progress.failed_batches} failed" if progress.failed_batches else ""
        message = progress.message or (
            f"Translated {progress.done_lines}/{progress.total_lines} lines "
            f"({progress.done_batches}/{progress.total_batches} batches{failed})"
        )
        job_manager.update_progress(
            job_id,
            percent,
            message,
            total_batches=progress.total_batches,
            completed_batches=progress.done_batches,
            completed_lines=progress.done_lines,
            tokens_used=progress.tokens,
            total_cost=progress.cost,
        )

    return report


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
            "backend",
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
