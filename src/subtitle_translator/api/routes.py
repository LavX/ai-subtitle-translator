"""FastAPI API endpoints for subtitle translation."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from subtitle_translator.api.models import (
    ErrorResponse,
    HealthResponse,
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

logger = logging.getLogger(__name__)

# Create routers
health_router = APIRouter(tags=["Health"])
api_router = APIRouter(prefix="/api/v1", tags=["Translation"])


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