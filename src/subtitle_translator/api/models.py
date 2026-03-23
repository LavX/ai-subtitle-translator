"""Pydantic request/response models for the API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SubtitleLine(BaseModel):
    """A single subtitle line with position and text content."""

    position: int = Field(..., description="Line position/index in the subtitle file")
    line: str = Field(..., description="The subtitle text content")


class ReasoningConfig(BaseModel):
    """Configuration for model reasoning/thinking capabilities."""

    enabled: bool | None = Field(
        default=None,
        description="Enable reasoning (default medium effort). Not all models support this.",
    )
    effort: str | None = Field(
        default=None,
        description="Reasoning effort level: 'xhigh', 'high', 'medium', 'low', 'minimal', 'none'",
    )
    max_tokens: int | None = Field(
        default=None,
        alias="maxTokens",
        ge=100,
        le=32000,
        description="Max tokens for reasoning (alternative to effort)",
    )

    model_config = {"populate_by_name": True}


class ProviderConfig(BaseModel):
    """Configuration for OpenRouter provider routing."""

    order: list[str] | None = Field(
        default=None,
        description="List of provider slugs to try in order (e.g., ['exacto', 'deepinfra'])",
    )
    allow_fallbacks: bool | None = Field(
        default=True,
        alias="allowFallbacks",
        description="Whether to allow fallbacks to other providers",
    )
    sort: str | None = Field(
        default=None, description="Sort providers by: 'price', 'throughput', or 'latency'"
    )
    only: list[str] | None = Field(
        default=None, description="List of provider slugs to allow exclusively"
    )
    ignore: list[str] | None = Field(default=None, description="List of provider slugs to skip")

    model_config = {"populate_by_name": True}


class TranslationConfig(BaseModel):
    """Per-request configuration that can override defaults."""

    api_key: str | None = Field(
        default=None,
        alias="apiKey",
        description="OpenRouter API key (overrides environment variable)",
    )
    model: str | None = Field(default=None, description="Model to use for translation")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)"
    )
    max_concurrent_jobs: int | None = Field(
        default=None,
        alias="maxConcurrentJobs",
        ge=1,
        le=10,
        description="Max concurrent workers (only via PUT /config)",
    )
    reasoning: ReasoningConfig | None = Field(
        default=None, description="Reasoning/thinking configuration (only supported by some models)"
    )
    use_thinking_variant: bool | None = Field(
        default=None,
        alias="useThinkingVariant",
        description="Append :thinking to model ID for extended reasoning (DeepSeek, Qwen)",
    )
    provider: ProviderConfig | None = Field(
        default=None, description="OpenRouter provider routing configuration"
    )
    parallel_batches: int | None = Field(
        default=None,
        alias="parallelBatches",
        ge=1,
        le=10,
        description="Number of batches to process in parallel per job (default: 4)",
    )

    model_config = {"populate_by_name": True}


class TranslateContentRequest(BaseModel):
    """
    Request model for translating subtitle content.

    Compatible with Lingarr API format for seamless Bazarr integration.
    """

    arrMediaId: int | None = Field(
        default=None, description="Media ID from Sonarr/Radarr (optional)"
    )
    title: str | None = Field(
        default=None, description="Title of the media (helps translation context)"
    )
    sourceLanguage: str = Field(..., description="Source language code (e.g., 'en', 'English')")
    targetLanguage: str = Field(..., description="Target language code (e.g., 'es', 'Spanish')")
    mediaType: str | None = Field(default=None, description="Type of media: 'Episode' or 'Movie'")
    lines: list[SubtitleLine] = Field(
        ..., max_length=50_000, description="List of subtitle lines to translate"
    )
    fileName: str | None = Field(default=None, description="Original file name (informational)")
    jobName: str | None = Field(default=None, description="User-provided label for the job")
    model: str | None = Field(
        default=None, description="Override default LLM model for translation"
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Override default temperature (0.0-2.0)"
    )
    config: TranslationConfig | None = Field(
        default=None, description="Per-request configuration overrides"
    )


class TranslateContentResponse(BaseModel):
    """Response model for translated subtitle content."""

    model_config = ConfigDict(populate_by_name=True)

    lines: list[SubtitleLine] = Field(..., description="Translated subtitle lines")
    model_used: str = Field(
        ..., alias="modelUsed", description="The LLM model used for translation"
    )
    tokens_used: int | None = Field(
        default=None, alias="tokensUsed", description="Total tokens consumed (if available)"
    )


class TranslateFileRequest(BaseModel):
    """Request model for translating an entire SRT file."""

    content: str = Field(
        ..., max_length=10_000_000, description="Complete SRT file content as string"
    )
    sourceLanguage: str = Field(..., description="Source language code")
    targetLanguage: str = Field(..., description="Target language code")
    title: str | None = Field(
        default=None, description="Title of the media (helps translation context)"
    )
    mediaType: str | None = Field(default=None, description="Type of media: 'Episode' or 'Movie'")
    fileName: str | None = Field(default=None, description="Original file name (informational)")
    jobName: str | None = Field(default=None, description="User-provided label for the job")
    model: str | None = Field(
        default=None, description="Override default LLM model for translation"
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Override default temperature"
    )
    config: TranslationConfig | None = Field(
        default=None, description="Per-request configuration overrides"
    )


class TranslateFileResponse(BaseModel):
    """Response model for translated SRT file."""

    model_config = ConfigDict(populate_by_name=True)

    content: str = Field(..., description="Translated SRT file content")
    model_used: str = Field(
        ..., alias="modelUsed", description="The LLM model used for translation"
    )
    tokens_used: int | None = Field(
        default=None, alias="tokensUsed", description="Total tokens consumed (if available)"
    )
    subtitle_count: int = Field(
        ..., alias="subtitleCount", description="Number of subtitles translated"
    )


class ModelInfo(BaseModel):
    """Information about an available LLM model."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Model identifier for API calls")
    name: str = Field(..., description="Human-readable model name")
    description: str | None = Field(default=None, description="Model description")
    context_length: int | None = Field(
        default=None, alias="contextLength", description="Maximum context length in tokens"
    )
    pricing: dict | None = Field(
        default=None, description="Pricing information (prompt/completion per token)"
    )
    is_default: bool = Field(
        default=False, alias="isDefault", description="Whether this is the default model"
    )


class ModelsResponse(BaseModel):
    """Response model for listing available models."""

    model_config = ConfigDict(populate_by_name=True)

    models: list[ModelInfo] = Field(..., description="List of available/recommended models")
    default_model: str = Field(..., alias="defaultModel", description="The default model ID")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(..., description="Service status ('healthy' or 'unhealthy')")
    version: str = Field(..., description="Service version")
    openrouter_configured: bool = Field(
        ..., alias="openrouterConfigured", description="Whether OpenRouter API key is configured"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: dict | None = Field(default=None, description="Additional error details")


class TranslationProgress(BaseModel):
    """Progress information for ongoing translation."""

    model_config = ConfigDict(populate_by_name=True)

    total_batches: int = Field(
        ..., alias="totalBatches", description="Total number of batches to process"
    )
    completed_batches: int = Field(
        ..., alias="completedBatches", description="Number of completed batches"
    )
    total_lines: int = Field(
        ..., alias="totalLines", description="Total number of lines to translate"
    )
    completed_lines: int = Field(
        ..., alias="completedLines", description="Number of lines translated"
    )
    percent_complete: float = Field(
        ..., alias="percentComplete", description="Percentage of completion (0-100)"
    )
    status: str = Field(..., description="Current status: 'processing', 'completed', 'failed'")


# Job Queue Models


class JobSubmitResponse(BaseModel):
    """Response model for job submission."""

    jobId: str = Field(..., description="Unique job identifier (UUID)")
    status: str = Field(default="queued", description="Initial job status")
    position: int | None = Field(default=None, description="Position in queue (1-based)")


class JobStatusResponse(BaseModel):
    """Response model for job status."""

    jobId: str = Field(..., description="Unique job identifier (UUID)")
    jobType: str | None = Field(
        default=None, description="Type of job (translate_content, translate_file)"
    )
    status: str = Field(
        ..., description="Job status: queued, processing, completed, failed, cancelled"
    )
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage (0-100)")
    message: str | None = Field(default=None, description="Status message")
    createdAt: datetime = Field(..., description="Job creation timestamp")
    startedAt: datetime | None = Field(default=None, description="Processing start timestamp")
    completedAt: datetime | None = Field(default=None, description="Completion timestamp")
    result: Any | None = Field(default=None, description="Translation result (only when completed)")
    error: str | None = Field(default=None, description="Error message (only when failed)")
    # Input metadata
    jobName: str | None = Field(default=None, description="User-provided job label")
    fileName: str | None = Field(default=None, description="Original file name")
    sourceLanguage: str | None = Field(default=None, description="Source language code")
    targetLanguage: str | None = Field(default=None, description="Target language code")
    title: str | None = Field(default=None, description="Media title")
    mediaType: str | None = Field(default=None, description="Media type (Episode/Movie)")
    model: str | None = Field(default=None, description="Model used for translation")
    totalLines: int | None = Field(default=None, description="Total number of lines to translate")
    # Processing metrics
    totalBatches: int | None = Field(default=None, description="Total number of batches")
    completedBatches: int | None = Field(default=None, description="Number of completed batches")
    completedLines: int | None = Field(
        default=None, description="Number of lines translated so far"
    )
    tokensUsed: int | None = Field(default=None, description="Total tokens consumed so far")
    totalCost: float | None = Field(default=None, description="Total cost in USD (from OpenRouter)")
    elapsedSeconds: float | None = Field(
        default=None, description="Elapsed processing time in seconds"
    )


class JobListResponse(BaseModel):
    """Response model for listing jobs."""

    jobs: list[JobStatusResponse] = Field(..., description="List of jobs")
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
    parallelBatchesPerJob: int = Field(
        ..., description="Number of batches processed in parallel per job"
    )
    maxConcurrentJobs: int = Field(..., description="Max concurrent translation jobs")
    maxJobs: int = Field(..., description="Max jobs in memory")
    apiKeyConfigured: bool = Field(..., description="Whether API key is configured")
    queueStatus: dict = Field(..., description="Current queue status")


class ConfigUpdateRequest(BaseModel):
    """Request model for updating runtime configuration."""

    apiKey: str | None = Field(default=None, description="OpenRouter API key")
    model: str | None = Field(default=None, description="Default model for translation")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Default temperature"
    )
    maxConcurrentJobs: int | None = Field(
        default=None, ge=1, le=10, description="Max concurrent workers"
    )
    parallelBatchesPerJob: int | None = Field(
        default=None, ge=1, le=10, description="Number of batches to process in parallel per job"
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
