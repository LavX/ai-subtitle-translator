"""Abstract base class for translation providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

from subtitle_translator.core.context import Language, MediaContext
from subtitle_translator.core.errors import (  # noqa: F401  (re-exported for existing imports)
    AuthenticationError,
    DeclinedError,
    DependencyUnavailableError,
    InvalidConfigError,
    InvalidResponseError,
    ProviderTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
    TranslationProviderError,
)

if TYPE_CHECKING:
    from subtitle_translator.core.models import TranslationConfig


@dataclass
class TranslationResult:
    """Result of a translation operation."""

    translations: list[dict[str, str]]  # List of {"index": "0", "content": "translated text"}
    model_used: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    raw_response: dict | None = None
    note: str | None = None  # why the reply had to be repaired or cut down
    routing_diagnostics: dict | None = None
    # The call answered but reported nothing about what it used, so its hold stays: it may have
    # been billed. Priced back ends set this when the reply carries no usage.
    billing_unknown: bool = False


@dataclass
class TranslationBatch:
    """A batch of subtitle lines to translate."""

    lines: list[dict[str, str]]  # List of {"index": "0", "content": "original text"}
    source_language: str
    target_language: str
    context_title: str | None = None
    context_media_type: str | None = None
    source: "Language | None" = None
    target: "Language | None" = None
    media: "MediaContext | None" = None


class TranslationProvider(ABC):
    """Abstract base class for translation providers."""

    # True when the back end calls reserve_request() itself with each request's exact worst case.
    prices_requests: ClassVar[bool] = False

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    @abstractmethod
    async def translate_batch(
        self,
        batch: TranslationBatch,
        model: str | None = None,
        temperature: float | None = None,
        config_override: Optional["TranslationConfig"] = None,
    ) -> TranslationResult:
        """
        Translate a batch of subtitle lines.

        Args:
            batch: The batch of subtitle lines to translate
            model: Optional model override
            temperature: Optional temperature override
            config_override: Optional per-request configuration override

        Returns:
            TranslationResult containing translated lines and metadata

        Raises:
            TranslationProviderError: On translation failure
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is available and configured correctly.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (close HTTP connections, etc.)."""
        pass

    @abstractmethod
    async def get_available_models(self) -> list[dict]:
        """
        Get list of available models from the provider.

        Returns:
            List of model information dictionaries
        """
        pass

    def get_model_metadata(self, model_id: str) -> dict | None:
        """
        Get metadata for a specific model by ID.

        Args:
            model_id: The model identifier

        Returns:
            Model metadata dict or None if not found
        """
        return None

    def build_system_prompt(
        self,
        target_language: str,
        source_language: str,
        context_title: str | None = None,
        context_media_type: str | None = None,
        *,
        source: "Language | None" = None,
        target: "Language | None" = None,
        media: "MediaContext | None" = None,
    ) -> str:
        """
        Build the system prompt for translation.

        Args:
            target_language: Target language for translation
            source_language: Source language for translation
            context_title: Optional media title for context
            context_media_type: Optional media type (Episode/Movie)
            source: Optional named source language, with its script
            target: Optional named target language, with its script
            media: Optional media context carried as reference data

        Returns:
            System prompt string
        """
        from subtitle_translator.core.prompt import build_system_prompt

        return build_system_prompt(
            target_language,
            source_language,
            context_title,
            context_media_type,
            source=source,
            target=target,
            media=media,
        )

    def format_input_for_translation(self, lines: list[dict[str, str]]) -> str:
        """
        Format subtitle lines for the translation request.

        Args:
            lines: List of {"index": "X", "content": "text"} dictionaries

        Returns:
            JSON string representation of the input
        """
        import json

        return json.dumps(lines, ensure_ascii=False)
