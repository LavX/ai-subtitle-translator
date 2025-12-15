"""Pydantic request/response models for the API."""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class SubtitleLine(BaseModel):
    """A single subtitle line with position and text content."""

    position: int = Field(..., description="Line position/index in the subtitle file")
    line: str = Field(..., description="The subtitle text content")


class ReasoningConfig(BaseModel):
    """Configuration for model reasoning/thinking capabilities."""
    
    enabled: Optional[bool] = Field(
        default=None,
        description="Enable reasoning (default medium effort). Not all models support this."
    )
    effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort level: 'xhigh', 'high', 'medium', 'low', 'minimal', 'none'"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        alias="maxTokens",
        ge=100,
        le=32000,
        description="Max tokens for reasoning (alternative to effort)"
    )
    
    model_config = {"populate_by_name": True}


class ProviderConfig(BaseModel):
    """Configuration for OpenRouter provider routing."""
    
    order: Optional[List[str]] = Field(
        default=None,
        description="List of provider slugs to try in order (e.g., ['exacto', 'deepinfra'])"
    )
    allow_fallbacks: Optional[bool] = Field(
        default=True,
        alias="allowFallbacks",
        description="Whether to allow fallbacks to other providers"
    )
    sort: Optional[str] = Field(
        default=None,
        description="Sort providers by: 'price', 'throughput', or 'latency'"
    )
    only: Optional[List[str]] = Field(
        default=None,
        description="List of provider slugs to allow exclusively"
    )
    ignore: Optional[List[str]] = Field(
        default=None,
        description="List of provider slugs to skip"
    )
    
    model_config = {"populate_by_name": True}


class TranslationConfig(BaseModel):
    """Per-request configuration that can override defaults."""
    
    api_key: Optional[str] = Field(
        default=None,
        alias="apiKey",
        description="OpenRouter API key (overrides environment variable)"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use for translation"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)"
    )
    max_concurrent_jobs: Optional[int] = Field(
        default=None,
        alias="maxConcurrentJobs",
        ge=1,
        le=10,
        description="Max concurrent workers (only via PUT /config)"
    )
    reasoning: Optional[ReasoningConfig] = Field(
        default=None,
        description="Reasoning/thinking configuration (only supported by some models)"
    )
    use_thinking_variant: Optional[bool] = Field(
        default=None,
        alias="useThinkingVariant",
        description="Append :thinking to model ID for extended reasoning (DeepSeek, Qwen)"
    )
    provider: Optional[ProviderConfig] = Field(
        default=None,
        description="OpenRouter provider routing configuration"
    )
    
    model_config = {"populate_by_name": True}


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
    config: Optional[TranslationConfig] = Field(
        default=None, description="Per-request configuration overrides"
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
    config: Optional[TranslationConfig] = Field(
        default=None, description="Per-request configuration overrides"
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


# Job Queue Models

class JobSubmitResponse(BaseModel):
    """Response model for job submission."""

    jobId: str = Field(..., description="Unique job identifier (UUID)")
    status: str = Field(default="queued", description="Initial job status")
    position: Optional[int] = Field(default=None, description="Position in queue (1-based)")


class JobStatusResponse(BaseModel):
    """Response model for job status."""

    jobId: str = Field(..., description="Unique job identifier (UUID)")
    jobType: Optional[str] = Field(default=None, description="Type of job (translate_content, translate_file)")
    status: str = Field(..., description="Job status: queued, processing, completed, failed, cancelled")
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage (0-100)")
    message: Optional[str] = Field(default=None, description="Status message")
    createdAt: datetime = Field(..., description="Job creation timestamp")
    startedAt: Optional[datetime] = Field(default=None, description="Processing start timestamp")
    completedAt: Optional[datetime] = Field(default=None, description="Completion timestamp")
    result: Optional[Any] = Field(default=None, description="Translation result (only when completed)")
    error: Optional[str] = Field(default=None, description="Error message (only when failed)")


class JobListResponse(BaseModel):
    """Response model for listing jobs."""

    jobs: List[JobStatusResponse] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    processing: int = Field(..., description="Number of jobs currently processing")
    queued: int = Field(..., description="Number of jobs in queue")


class JobDeleteResponse(BaseModel):
    """Response model for job deletion/cancellation."""

    jobId: str = Field(..., description="The job ID")
    status: str = Field(..., description="New job status after operation")
    message: str = Field(..., description="Operation result message")


class ConfigResponse(BaseModel):
    """Response model for configuration endpoint."""

    model: str = Field(..., description="Default translation model")
    temperature: float = Field(..., description="Default temperature")
    batchSize: int = Field(..., description="Batch size for translation")
    maxConcurrentJobs: int = Field(..., description="Max concurrent translation jobs")
    maxJobs: int = Field(..., description="Max jobs in memory")
    apiKeyConfigured: bool = Field(..., description="Whether API key is configured")
    queueStatus: dict = Field(..., description="Current queue status")


class ConfigUpdateRequest(BaseModel):
    """Request model for updating runtime configuration."""
    
    apiKey: Optional[str] = Field(
        default=None,
        description="OpenRouter API key"
    )
    model: Optional[str] = Field(
        default=None,
        description="Default model for translation"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Default temperature"
    )
    maxConcurrentJobs: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Max concurrent workers"
    )


class ConfigUpdateResponse(BaseModel):
    """Response model for configuration update."""

    status: str = Field(..., description="Update status")
    message: str = Field(default="Configuration updated", description="Status message")


class ServiceStatusResponse(BaseModel):
    """Response model for service status endpoint."""

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    healthy: bool = Field(..., description="Whether service is healthy")
    config: dict = Field(..., description="Current configuration summary")
    queue: dict = Field(..., description="Queue status")