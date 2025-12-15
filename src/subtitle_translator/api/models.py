"""Pydantic request/response models for the API."""

from typing import Optional

from pydantic import BaseModel, Field


class SubtitleLine(BaseModel):
    """A single subtitle line with position and text content."""

    position: int = Field(..., description="Line position/index in the subtitle file")
    line: str = Field(..., description="The subtitle text content")


class TranslateContentRequest(BaseModel):
    """
    Request model for translating subtitle content.
    
    Compatible with Lingarr API format for seamless Bazarr integration.
    """

    arrMediaId: Optional[int] = Field(
        default=None, description="Media ID from Sonarr/Radarr (optional)"
    )
    title: Optional[str] = Field(
        default=None, description="Title of the media (helps translation context)"
    )
    sourceLanguage: str = Field(..., description="Source language code (e.g., 'en', 'English')")
    targetLanguage: str = Field(..., description="Target language code (e.g., 'es', 'Spanish')")
    mediaType: Optional[str] = Field(
        default=None, description="Type of media: 'Episode' or 'Movie'"
    )
    lines: list[SubtitleLine] = Field(..., description="List of subtitle lines to translate")
    model: Optional[str] = Field(
        default=None, description="Override default LLM model for translation"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Override default temperature (0.0-2.0)"
    )


class TranslateContentResponse(BaseModel):
    """Response model for translated subtitle content."""

    lines: list[SubtitleLine] = Field(..., description="Translated subtitle lines")
    model_used: str = Field(..., description="The LLM model used for translation")
    tokens_used: Optional[int] = Field(
        default=None, description="Total tokens consumed (if available)"
    )


class TranslateFileRequest(BaseModel):
    """Request model for translating an entire SRT file."""

    content: str = Field(..., description="Complete SRT file content as string")
    sourceLanguage: str = Field(..., description="Source language code")
    targetLanguage: str = Field(..., description="Target language code")
    title: Optional[str] = Field(
        default=None, description="Title of the media (helps translation context)"
    )
    model: Optional[str] = Field(
        default=None, description="Override default LLM model for translation"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Override default temperature"
    )


class TranslateFileResponse(BaseModel):
    """Response model for translated SRT file."""

    content: str = Field(..., description="Translated SRT file content")
    model_used: str = Field(..., description="The LLM model used for translation")
    tokens_used: Optional[int] = Field(
        default=None, description="Total tokens consumed (if available)"
    )
    subtitle_count: int = Field(..., description="Number of subtitles translated")


class ModelInfo(BaseModel):
    """Information about an available LLM model."""

    id: str = Field(..., description="Model identifier for API calls")
    name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = Field(default=None, description="Model description")
    context_length: Optional[int] = Field(
        default=None, description="Maximum context length in tokens"
    )
    pricing: Optional[dict] = Field(
        default=None, description="Pricing information (prompt/completion per token)"
    )
    is_default: bool = Field(default=False, description="Whether this is the default model")


class ModelsResponse(BaseModel):
    """Response model for listing available models."""

    models: list[ModelInfo] = Field(..., description="List of available/recommended models")
    default_model: str = Field(..., description="The default model ID")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status ('healthy' or 'unhealthy')")
    version: str = Field(..., description="Service version")
    openrouter_configured: bool = Field(
        ..., description="Whether OpenRouter API key is configured"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")


class TranslationProgress(BaseModel):
    """Progress information for ongoing translation."""

    total_batches: int = Field(..., description="Total number of batches to process")
    completed_batches: int = Field(..., description="Number of completed batches")
    total_lines: int = Field(..., description="Total number of lines to translate")
    completed_lines: int = Field(..., description="Number of lines translated")
    percent_complete: float = Field(..., description="Percentage of completion (0-100)")
    status: str = Field(..., description="Current status: 'processing', 'completed', 'failed'")