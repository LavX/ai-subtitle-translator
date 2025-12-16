"""Worker functions for processing translation jobs."""

import logging
from typing import Any, Dict, List, Optional

from subtitle_translator.api.models import SubtitleLine, TranslateContentRequest, TranslationConfig
from subtitle_translator.core.batch_processor import BatchProgress, BatchProcessor
from subtitle_translator.core.translator import SubtitleTranslator, get_translator
from subtitle_translator.queue.job_manager import JobManager, JobStatus, JobType

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
        
        if not request.lines:
            job_manager.set_job_completed(job_id, {"lines": [], "model_used": "", "tokens_used": 0})
            return
        
        # Log incoming request data for debugging
        raw_config = job.request_data.get("config")
        logger.info(f"Job {job_id}: Raw config from request_data: {raw_config}")
        
        # Extract config override from request
        config_override = _extract_config_override(request.config, job.request_data)
        
        # Log API key tracking (masked)
        if config_override and config_override.api_key:
            masked_key = config_override.api_key[:10] + "..." if len(config_override.api_key) > 10 else "***"
            logger.info(f"Job {job_id}: Using API key from config: {masked_key}")
        else:
            logger.info(f"Job {job_id}: No API key in config_override, will use env default")
        
        # Convert request lines to internal format
        lines = [
            {"index": str(line.position), "content": line.line}
            for line in request.lines
        ]
        
        # Create batch processor
        processor = BatchProcessor(translator.provider, translator.settings)
        
        # Define progress callback
        def progress_callback(progress: BatchProgress) -> None:
            percent = int(progress.percent_complete)
            message = f"Processing batch {progress.completed_batches}/{progress.total_batches}"
            job_manager.update_progress(job_id, percent, message)
        
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
            failed_batches = [r for r in result.batch_results if not r.success]
            error_msg = "; ".join(r.error or "Unknown error" for r in failed_batches)
            
            # Check if we have partial results
            if result.all_translations:
                translated_lines = _map_translations_to_lines(
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
                        "partial": True,
                        "warning": f"Partial failure: {error_msg}",
                    },
                )
            else:
                job_manager.set_job_failed(job_id, error_msg)
            return
        
        # Map translations back to SubtitleLine format
        translated_lines = _map_translations_to_lines(
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
        
        # Extract config override from request data
        config_override = _extract_config_override_from_dict(request_data.get("config"))
        
        # Debug log for API key tracking (masked)
        if config_override and config_override.api_key:
            masked_key = config_override.api_key[:10] + "..." if len(config_override.api_key) > 10 else "***"
            logger.debug(f"Job {job_id}: Using API key from config: {masked_key}")
        else:
            logger.warning(f"Job {job_id}: No API key in config_override, will use env default")
        
        if not content or not content.strip():
            job_manager.set_job_failed(job_id, "SRT content is required")
            return
        
        # Validate and parse SRT
        is_valid, error = translator._srt_parser.validate_srt(content)
        if not is_valid:
            job_manager.set_job_failed(job_id, f"Invalid SRT content: {error}")
            return
        
        # Parse SRT content
        entries = translator._srt_parser.parse(content)
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
        
        # Create batch processor
        processor = BatchProcessor(translator.provider, translator.settings)
        
        # Define progress callback
        def progress_callback(progress: BatchProgress) -> None:
            percent = int(progress.percent_complete)
            message = f"Processing batch {progress.completed_batches}/{progress.total_batches}"
            job_manager.update_progress(job_id, percent, message)
        
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
            failed_batches = [r for r in result.batch_results if not r.success]
            error_msg = "; ".join(r.error or "Unknown error" for r in failed_batches)
            
            # Check if we have partial results
            if result.all_translations:
                is_rtl = translator.settings.is_rtl_language(target_language)
                translated_entries = translator._srt_parser.apply_translations(
                    entries, result.all_translations, is_rtl=is_rtl
                )
                translated_content = translator._srt_parser.compose(translated_entries)
                
                job_manager.set_job_completed(
                    job_id,
                    {
                        "content": translated_content,
                        "model_used": result.model_used,
                        "tokens_used": result.total_tokens,
                        "subtitle_count": len(entries),
                        "partial": True,
                        "warning": f"Partial failure: {error_msg}",
                    },
                )
            else:
                job_manager.set_job_failed(job_id, error_msg)
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


def _map_translations_to_lines(
    original_lines: List[SubtitleLine],
    translations: List[Dict[str, str]],
    target_language: str,
    settings: Any,
) -> List[SubtitleLine]:
    """
    Map translated content back to SubtitleLine format.
    
    Args:
        original_lines: Original subtitle lines
        translations: Translated content
        target_language: Target language for RTL handling
        settings: Settings instance
        
    Returns:
        List of SubtitleLine with translated content
    """
    # Build translation map
    translation_map = {t["index"]: t["content"] for t in translations}
    
    # Check if RTL markers needed
    is_rtl = settings.is_rtl_language(target_language)
    
    result = []
    for line in original_lines:
        translated_text = translation_map.get(str(line.position), line.line)
        
        if is_rtl:
            translated_text = _add_rtl_markers(translated_text)
        
        result.append(SubtitleLine(position=line.position, line=translated_text))
    
    return result


def _add_rtl_markers(text: str) -> str:
    """Add RTL markers to text for proper display."""
    RLE = "\u202B"  # RIGHT-TO-LEFT EMBEDDING
    PDF = "\u202C"  # POP DIRECTIONAL FORMATTING
    
    lines = text.split("\n")
    marked_lines = []
    
    for line in lines:
        if line.strip():
            marked_lines.append(f"{RLE}{line}{PDF}")
        else:
            marked_lines.append(line)
    
    return "\n".join(marked_lines)


def _extract_config_override(
    config: Optional[TranslationConfig],
    request_data: Dict[str, Any],
) -> Optional[TranslationConfig]:
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
    return _extract_config_override_from_dict(request_data.get("config"))


def _extract_config_override_from_dict(
    config_dict: Optional[Dict[str, Any]],
) -> Optional[TranslationConfig]:
    """
    Extract TranslationConfig from a dictionary.
    
    Args:
        config_dict: Raw config dictionary from request
        
    Returns:
        TranslationConfig if valid dict provided, None otherwise
    """
    if config_dict is None:
        return None
    
    if not isinstance(config_dict, dict):
        return None
    
    # Check if any fields are present
    if not any(key in config_dict for key in ["apiKey", "api_key", "model", "temperature", "maxConcurrentJobs", "max_concurrent_jobs", "reasoning", "provider"]):
        return None
    
    try:
        return TranslationConfig(**config_dict)
    except Exception as e:
        # Log validation failure for debugging
        logger.warning(f"Failed to create TranslationConfig from dict: {e}")
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
    # Get translator instance
    translator = await get_translator()
    
    if job_type == JobType.TRANSLATE_CONTENT:
        await process_content_translation_job(job_manager, job_id, translator)
    elif job_type == JobType.TRANSLATE_FILE:
        await process_file_translation_job(job_manager, job_id, translator)
    else:
        job_manager.set_job_failed(job_id, f"Unknown job type: {job_type}")