"""Pydantic request/response models for the API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from subtitle_translator.core.models import (  # noqa: F401  (re-exported API surface)
    MAX_LINE_CHARS,
    PROVIDER_SORT_VALUES,
    ProviderConfig,
    ReasoningConfig,
    SmartFastPolicy,
    SubtitleLine,
    TranslateContentRequest,
    TranslationConfig,
    _sanitize_text,
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
        ..., max_length=2_000_000, description="Complete SRT file content as string"
    )
    sourceLanguage: str = Field(..., max_length=50, description="Source language code")
    targetLanguage: str = Field(..., max_length=50, description="Target language code")
    title: str | None = Field(
        default=None, max_length=500, description="Title of the media (helps translation context)"
    )
    mediaType: str | None = Field(
        default=None, max_length=50, description="Type of media: 'Episode' or 'Movie'"
    )
    fileName: str | None = Field(
        default=None, max_length=500, description="Original file name (informational)"
    )
    jobName: str | None = Field(
        default=None, max_length=200, description="User-provided label for the job"
    )
    model: str | None = Field(
        default=None, max_length=200, description="Override default LLM model for translation"
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Override default temperature"
    )
    config: TranslationConfig | None = Field(
        default=None, description="Per-request configuration overrides"
    )

    @field_validator(
        "title",
        "sourceLanguage",
        "targetLanguage",
        "mediaType",
        "fileName",
        "jobName",
        mode="before",
    )
    @classmethod
    def sanitize_metadata(cls, v):
        if isinstance(v, str):
            return _sanitize_text(v)
        return v

    @field_validator("content")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        return _sanitize_text(v)


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


class ModelReasoningInfo(BaseModel):
    """Public reasoning choices reported by the model catalog."""

    model_config = ConfigDict(populate_by_name=True)

    mandatory: bool | None = None
    supported_efforts: list[str] = Field(default_factory=list, alias="supportedEfforts")


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
    reasoning: ModelReasoningInfo | None = None


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

    apiKey: str | None = Field(default=None, max_length=500, description="OpenRouter API key")
    model: str | None = Field(
        default=None, max_length=200, description="Default model for translation"
    )
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


class TestConnectionRequest(BaseModel):
    """Request model for testing encryption and API key validity."""

    apiKey: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="API key to test (plaintext or enc: encrypted)",
    )


class ServiceStatusResponse(BaseModel):
    """Response model for service status endpoint."""

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    healthy: bool = Field(..., description="Whether service is healthy")
    config: dict = Field(..., description="Current configuration summary")
    queue: dict = Field(..., description="Queue status")
