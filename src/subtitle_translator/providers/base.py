"""Abstract base class for translation providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from subtitle_translator.api.models import TranslationConfig


@dataclass
class TranslationResult:
    """Result of a translation operation."""

    translations: list[dict[str, str]]  # List of {"index": "0", "content": "translated text"}
    model_used: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw_response: Optional[dict] = None


@dataclass
class TranslationBatch:
    """A batch of subtitle lines to translate."""

    lines: list[dict[str, str]]  # List of {"index": "0", "content": "original text"}
    source_language: str
    target_language: str
    context_title: Optional[str] = None
    context_media_type: Optional[str] = None


class TranslationProviderError(Exception):
    """Base exception for translation provider errors."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        retryable: bool = False,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class RateLimitError(TranslationProviderError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, provider: str = "unknown", retry_after: Optional[float] = None):
        super().__init__(message, provider, retryable=True, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(TranslationProviderError):
    """Raised when authentication fails."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider, retryable=False, status_code=401)


class InvalidResponseError(TranslationProviderError):
    """Raised when the provider returns an invalid response."""

    def __init__(self, message: str, provider: str = "unknown", raw_response: Optional[str] = None):
        super().__init__(message, provider, retryable=True)
        self.raw_response = raw_response


class TranslationProvider(ABC):
    """Abstract base class for translation providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    @abstractmethod
    async def translate_batch(
        self,
        batch: TranslationBatch,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
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
    async def get_available_models(self) -> list[dict]:
        """
        Get list of available models from the provider.

        Returns:
            List of model information dictionaries
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

    def build_system_prompt(
        self,
        target_language: str,
        context_title: Optional[str] = None,
        context_media_type: Optional[str] = None,
    ) -> str:
        """
        Build the system prompt for translation.

        Args:
            target_language: Target language for translation
            context_title: Optional media title for context
            context_media_type: Optional media type (Episode/Movie)

        Returns:
            System prompt string
        """
        context_info = ""
        if context_title:
            context_info += f"\nMedia title: {context_title}"
        if context_media_type:
            context_info += f"\nMedia type: {context_media_type}"

        return f"""You are a professional subtitle translator specializing in translating subtitles to {target_language}.
{context_info}

Your task is to translate the provided subtitle lines while following these rules:
1. Return ONLY a valid JSON array with the exact format: [{{"index": "0", "content": "translated text"}}, ...]
2. Each object must have "index" (string) and "content" (string) keys
3. Preserve the original meaning and tone of the dialogue
4. Keep translations natural and colloquial for spoken dialogue
5. Maintain approximately 40-50 characters per line when possible for readability
6. Do NOT translate proper nouns (names of people, places, brands) unless they have an official translation
7. Preserve any timing codes, formatting tags (like <i>, <b>), or special markers
8. If the original contains profanity or slang, translate appropriately for the target language
9. Do NOT add explanations, notes, or anything outside the JSON array"""

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