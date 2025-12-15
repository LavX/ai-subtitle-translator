"""FastAPI API endpoints for subtitle translation."""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from subtitle_translator.api.models import (
    ErrorResponse,
    HealthResponse,
    JobDeleteResponse,
    JobListResponse,
    JobStatusResponse,
    JobSubmitResponse,
    ModelInfo,
    ModelsResponse,
    SubtitleLine,
    TranslateContentRequest,
    TranslateContentResponse,
    TranslateFileRequest,
    TranslateFileResponse,
)
from subtitle_translator.config import get_settings
from subtitle_translator.core.translator import SubtitleTranslator, get_translator
from subtitle_translator.queue.job_manager import JobStatus, JobType, job_manager

logger = logging.getLogger(__name__)

# Create routers
health_router = APIRouter(tags=["Health"])
api_router = APIRouter(prefix="/api/v1", tags=["Translation"])
jobs_router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])


# Dependency for getting translator
async def get_translator_dependency() -> SubtitleTranslator:
    """Dependency injection for translator."""
    return await get_translator()


TranslatorDep = Annotated[SubtitleTranslator, Depends(get_translator_dependency)]


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the service is healthy and properly configured.",
)
async def health_check(translator: TranslatorDep) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service status and configuration state.
    """
    settings = get_settings()
    openrouter_configured = bool(settings.openrouter_api_key)
    
    # Optionally check if OpenRouter is actually reachable
    is_healthy = True
    if openrouter_configured:
        try:
            is_healthy = await translator.health_check()
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            is_healthy = False

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        version="1.0.0",
        openrouter_configured=openrouter_configured,
    )


@api_router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List Available Models",
    description="Get a list of recommended LLM models for subtitle translation.",
)
async def list_models(translator: TranslatorDep) -> ModelsResponse:
    """
    List available models for translation.
    
    Returns a list of recommended models with their capabilities and pricing.
    """
    settings = get_settings()
    models_data = await translator.get_available_models()
    
    models = [
        ModelInfo(
            id=m["id"],
            name=m["name"],
            description=m.get("description"),
            context_length=m.get("context_length"),
            pricing=m.get("pricing"),
            is_default=m.get("is_default", False),
        )
        for m in models_data
    ]
    
    return ModelsResponse(
        models=models,
        default_model=settings.openrouter_default_model,
    )


@api_router.post(
    "/translate/content",
    response_model=TranslateContentResponse,
    summary="Translate Subtitle Content",
    description=(
        "Translate subtitle lines. Compatible with Lingarr API format for "
        "seamless Bazarr integration."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Translation failed"},
    },
)
async def translate_content(
    request: TranslateContentRequest,
    translator: TranslatorDep,
) -> TranslateContentResponse:
    """
    Translate subtitle content.
    
    This endpoint accepts subtitle lines and returns translated lines.
    It is compatible with the Lingarr API format for Bazarr integration.
    
    The lines are processed in batches to handle large subtitle files efficiently.
    """
    # Validate that API key is configured
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "configuration_error",
                "message": "OpenRouter API key is not configured",
            },
        )

    # Validate request
    if not request.lines:
        return TranslateContentResponse(
            lines=[],
            model_used=request.model or settings.openrouter_default_model,
            tokens_used=0,
        )

    # Perform translation
    result = await translator.translate_content(request)

    if not result.success:
        # Return partial results if available, otherwise raise error
        if result.lines:
            logger.warning(f"Partial translation success: {result.error}")
            return TranslateContentResponse(
                lines=result.lines,
                model_used=result.model_used,
                tokens_used=result.tokens_used,
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "translation_failed",
                "message": result.error or "Unknown translation error",
            },
        )

    return TranslateContentResponse(
        lines=result.lines,
        model_used=result.model_used,
        tokens_used=result.tokens_used,
    )


@api_router.post(
    "/translate/file",
    response_model=TranslateFileResponse,
    summary="Translate SRT File",
    description="Translate an entire SRT subtitle file.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid SRT content"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Translation failed"},
    },
)
async def translate_file(
    request: TranslateFileRequest,
    translator: TranslatorDep,
) -> TranslateFileResponse:
    """
    Translate an entire SRT file.
    
    This endpoint accepts raw SRT content and returns the translated SRT file.
    It handles:
    - SRT parsing and validation
    - Batch processing for large files
    - RTL language support with directional markers
    - Line length optimization
    """
    # Validate that API key is configured
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "configuration_error",
                "message": "OpenRouter API key is not configured",
            },
        )

    # Validate request
    if not request.content or not request.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "message": "SRT content is required",
            },
        )

    # Perform translation
    result = await translator.translate_file(
        content=request.content,
        source_language=request.sourceLanguage,
        target_language=request.targetLanguage,
        title=request.title,
        model=request.model,
        temperature=request.temperature,
    )

    if not result.success:
        # Return partial results if available
        if result.content:
            logger.warning(f"Partial file translation success: {result.error}")
            return TranslateFileResponse(
                content=result.content,
                model_used=result.model_used,
                tokens_used=result.tokens_used,
                subtitle_count=result.subtitle_count,
            )
        
        # Check for specific error types
        if "Invalid SRT" in (result.error or ""):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_srt",
                    "message": result.error,
                },
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "translation_failed",
                "message": result.error or "Unknown translation error",
            },
        )

    return TranslateFileResponse(
        content=result.content,
        model_used=result.model_used,
        tokens_used=result.tokens_used,
        subtitle_count=result.subtitle_count,
    )


# ============================================================================
# Job Queue Endpoints
# ============================================================================


@jobs_router.post(
    "/translate/content",
    response_model=JobSubmitResponse,
    summary="Submit Content Translation Job",
    description=(
        "Submit a subtitle content translation job to the queue. "
        "Returns immediately with a job ID for status tracking."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "Job queue full"},
    },
)
async def submit_translate_content_job(
    request: TranslateContentRequest,
) -> JobSubmitResponse:
    """
    Submit a content translation job to the async queue.
    
    This endpoint accepts the same request format as /translate/content
    but processes it asynchronously. Use the returned jobId to poll
    for status and results.
    """
    # Validate that API key is configured
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "configuration_error",
                "message": "OpenRouter API key is not configured",
            },
        )
    
    try:
        job_id = await job_manager.submit_job(
            request_data=request.model_dump(),
            job_type=JobType.TRANSLATE_CONTENT,
        )
        
        position = job_manager.get_queue_position(job_id)
        
        return JobSubmitResponse(
            jobId=job_id,
            status="queued",
            position=position,
        )
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "queue_full",
                "message": str(e),
            },
        )


@jobs_router.post(
    "/translate/file",
    response_model=JobSubmitResponse,
    summary="Submit File Translation Job",
    description=(
        "Submit an SRT file translation job to the queue. "
        "Returns immediately with a job ID for status tracking."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        429: {"model": ErrorResponse, "description": "Job queue full"},
    },
)
async def submit_translate_file_job(
    request: TranslateFileRequest,
) -> JobSubmitResponse:
    """
    Submit a file translation job to the async queue.
    
    This endpoint accepts the same request format as /translate/file
    but processes it asynchronously. Use the returned jobId to poll
    for status and results.
    """
    # Validate that API key is configured
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "configuration_error",
                "message": "OpenRouter API key is not configured",
            },
        )
    
    # Validate request
    if not request.content or not request.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "message": "SRT content is required",
            },
        )
    
    try:
        job_id = await job_manager.submit_job(
            request_data=request.model_dump(),
            job_type=JobType.TRANSLATE_FILE,
        )
        
        position = job_manager.get_queue_position(job_id)
        
        return JobSubmitResponse(
            jobId=job_id,
            status="queued",
            position=position,
        )
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "queue_full",
                "message": str(e),
            },
        )


@jobs_router.get(
    "",
    response_model=JobListResponse,
    summary="List Jobs",
    description="List all jobs with optional status filtering.",
)
async def list_jobs(
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        description="Filter by status: queued, processing, completed, failed, cancelled",
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of jobs to return",
    ),
) -> JobListResponse:
    """
    List all jobs in the queue.
    
    Use the status query parameter to filter by job status.
    Results are sorted by creation time (newest first).
    """
    # Parse status filter
    job_status = None
    if status_filter:
        try:
            job_status = JobStatus(status_filter.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_status",
                    "message": f"Invalid status: {status_filter}. "
                    f"Valid values: {', '.join(s.value for s in JobStatus)}",
                },
            )
    
    jobs = job_manager.list_jobs(status_filter=job_status, limit=limit)
    stats = job_manager.get_stats()
    
    job_responses = []
    for job in jobs:
        job_responses.append(
            JobStatusResponse(
                jobId=job.id,
                jobType=job.job_type.value,
                status=job.status.value,
                progress=job.progress,
                message=job.message or None,
                createdAt=job.created_at,
                startedAt=job.started_at,
                completedAt=job.completed_at,
                result=job.result if job.status == JobStatus.COMPLETED else None,
                error=job.error if job.status == JobStatus.FAILED else None,
            )
        )
    
    return JobListResponse(
        jobs=job_responses,
        total=stats["total"],
        processing=stats["processing"],
        queued=stats["queued"],
    )


@jobs_router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get Job Status",
    description="Get the status and result of a specific job.",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get the status of a specific job.
    
    When the job is completed, the result field will contain
    the translation output. When failed, the error field will
    contain the error message.
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "job_not_found",
                "message": f"Job {job_id} not found",
            },
        )
    
    return JobStatusResponse(
        jobId=job.id,
        jobType=job.job_type.value,
        status=job.status.value,
        progress=job.progress,
        message=job.message or None,
        createdAt=job.created_at,
        startedAt=job.started_at,
        completedAt=job.completed_at,
        result=job.result if job.status == JobStatus.COMPLETED else None,
        error=job.error if job.status == JobStatus.FAILED else None,
    )


@jobs_router.delete(
    "/{job_id}",
    response_model=JobDeleteResponse,
    summary="Cancel/Delete Job",
    description=(
        "Cancel a queued job or delete a completed/failed job. "
        "Processing jobs cannot be cancelled."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
async def cancel_or_delete_job(job_id: str) -> JobDeleteResponse:
    """
    Cancel a queued job or delete a completed job.
    
    - Queued jobs will be cancelled
    - Completed/failed/cancelled jobs will be deleted
    - Processing jobs cannot be cancelled (returns current status)
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "job_not_found",
                "message": f"Job {job_id} not found",
            },
        )
    
    # Try to cancel if queued
    if job.status == JobStatus.QUEUED:
        job_manager.cancel_job(job_id)
        return JobDeleteResponse(
            jobId=job_id,
            status=JobStatus.CANCELLED.value,
            message="Job cancelled successfully",
        )
    
    # Try to delete if completed/failed/cancelled
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        job_manager.delete_job(job_id)
        return JobDeleteResponse(
            jobId=job_id,
            status="deleted",
            message="Job deleted successfully",
        )
    
    # Processing jobs cannot be cancelled
    return JobDeleteResponse(
        jobId=job_id,
        status=job.status.value,
        message="Cannot cancel job that is currently processing",
    )