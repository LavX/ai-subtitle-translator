"""Request and per-request configuration types shared by the core, the back ends and the server."""

import re
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

# The longest one subtitle line may be, in characters. The server's response model and the
# engine's output guard both read it, so a translation is never accepted by one and
# rejected by the other.
MAX_LINE_CHARS = 2000

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")
_DANGEROUS_CONTROL_CHARS = frozenset(
    chr(c) for c in range(0x00, 0x20) if chr(c) not in ("\n", "\r", "\t")
)


def _sanitize_text(text: str) -> str:
    """Strip null bytes, ANSI escapes, and dangerous control characters."""
    text = _ANSI_RE.sub("", text)
    return "".join(c for c in text if c not in _DANGEROUS_CONTROL_CHARS)


class SubtitleLine(BaseModel):
    """A single subtitle line with position and text content."""

    position: int = Field(
        ..., ge=0, le=100_000, description="Line position/index in the subtitle file"
    )
    line: str = Field(..., max_length=MAX_LINE_CHARS, description="The subtitle text content")

    @field_validator("line")
    @classmethod
    def sanitize_line(cls, v: str) -> str:
        return _sanitize_text(v)


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


PROVIDER_SORT_VALUES = ("default", "throughput", "price", "latency", "nitro", "floor", "smartfast")


class SmartFastPolicy(BaseModel):
    """Opt-in routing controls; prices are USD per million tokens."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    medianPremiumPercent: float = Field(default=50, ge=0, le=1000)
    speedTolerancePercent: float = Field(default=20, ge=0, le=1000)
    sparsePremiumMultiplier: float = Field(default=3, ge=1, le=100)
    maxPromptPrice: float = Field(default=1, ge=0, le=1000)
    maxCompletionPrice: float = Field(default=3, ge=0, le=1000)


class ProviderConfig(BaseModel):
    """Configuration for OpenRouter provider routing."""

    order: list[str] | None = Field(
        default=None,
        max_length=20,
        description="List of provider slugs to try in order (e.g., ['exacto', 'deepinfra'])",
    )
    allow_fallbacks: bool | None = Field(
        default=True,
        alias="allowFallbacks",
        description="Whether to allow fallbacks to other providers",
    )
    sort: str | None = Field(
        default=None,
        description=(
            "Provider routing: 'throughput', 'price' or 'latency' set provider.sort; "
            "'nitro' and 'floor' use the OpenRouter slug shortcuts (which also unlock "
            "the priority/flex tiers); 'smartfast' selects a price-bounded fast provider pool; "
            "'default' keeps OpenRouter's load balancing"
        ),
    )
    only: list[str] | None = Field(
        default=None, max_length=20, description="List of provider slugs to allow exclusively"
    )
    ignore: list[str] | None = Field(
        default=None, max_length=20, description="List of provider slugs to skip"
    )

    data_collection: Literal["allow", "deny"] | None = Field(
        default=None,
        alias="dataCollection",
        description="'deny' uses only providers that do not store or train on request data",
    )

    smart_fast: SmartFastPolicy | None = Field(
        default=None, alias="smartFast", description="SmartFast price and speed policy"
    )

    model_config = {"populate_by_name": True}

    @field_validator("sort", mode="before")
    @classmethod
    def _normalise_sort(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalised = str(value).strip().lower()
        if normalised == "":
            return None
        if normalised not in PROVIDER_SORT_VALUES:
            raise ValueError(
                f"sort must be one of {', '.join(PROVIDER_SORT_VALUES)}, got '{value}'"
            )
        return normalised


class TranslationConfig(BaseModel):
    """Per-request configuration that can override defaults."""

    _smartfast_session_id: str | None = PrivateAttr(default=None)
    # The meter of the attempt this request belongs to; attached by the batch processor.
    _usage_meter: Any = PrivateAttr(default=None)

    def for_operation(self, job_id: str | None = None) -> "TranslationConfig":
        """Copy options and bind an opaque internal identity to one operation."""
        config = self.model_copy()
        config._smartfast_session_id = (
            sha256(f"translation-job:{job_id}".encode()).hexdigest()
            if job_id is not None
            else uuid4().hex
        )
        return config

    api_key: str | None = Field(
        default=None,
        alias="apiKey",
        max_length=500,
        repr=False,
        description="OpenRouter API key (overrides environment variable)",
    )
    model: str | None = Field(default=None, description="Model to use for translation")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)"
    )
    request_timeout: float | None = Field(
        default=None,
        alias="requestTimeout",
        ge=30,
        le=900,
        description="Provider response timeout in seconds for this translation",
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
    service_tier: Literal["default", "flex", "priority"] | None = Field(
        default=None,
        alias="serviceTier",
        description="OpenRouter service tier; default uses standard capacity, null follows routing",
    )
    parallel_batches: int | None = Field(
        default=None,
        alias="parallelBatches",
        ge=1,
        le=10,
        description="Number of batches to process in parallel per job (default: 4)",
    )
    backend: str | None = Field(
        default=None,
        max_length=64,
        description="Use only this configured back-end connection, with no failover",
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
        default=None, max_length=500, description="Title of the media (helps translation context)"
    )
    sourceLanguage: str = Field(
        ..., max_length=50, description="Source language code (e.g., 'en', 'English')"
    )
    targetLanguage: str = Field(
        ..., max_length=50, description="Target language code (e.g., 'es', 'Spanish')"
    )
    mediaType: str | None = Field(
        default=None, max_length=50, description="Type of media: 'Episode' or 'Movie'"
    )
    lines: list[SubtitleLine] = Field(
        ..., max_length=10_000, description="List of subtitle lines to translate"
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
        default=None, ge=0.0, le=2.0, description="Override default temperature (0.0-2.0)"
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
