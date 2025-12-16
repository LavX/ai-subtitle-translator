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

    def build_system_prompt(
        self,
        target_language: str,
        source_language: str,
        context_title: Optional[str] = None,
        context_media_type: Optional[str] = None,
    ) -> str:
        """
        Build the system prompt for translation.

        Args:
            target_language: Target language for translation
            source_language: Source language for translation
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

        language_note = f"Follow natural {target_language} grammar, expressions, and use proper diacritics/accents for the language."

        return f"""You are an expert subtitle translator specializing in {source_language} to {target_language} translation.

{context_info}{language_note}

CRITICAL INSTRUCTIONS - READ CAREFULLY:

STEP 1: EXACT FORMAT REQUIREMENT
Return ONLY this exact JSON format - nothing else, no exceptions:
[{{
  "index": "0",
  "content": "TRANSLATED TEXT WITH DIACRITICS"
}},{{
  "index": "1",
  "content": "NEXT TRANSLATED LINE"
}}]

STEP 2: 100% TRANSLATION REQUIREMENT
- Translate EVERY SINGLE LINE from {source_language} to {target_language}
- DO NOT skip any lines - no matter how simple, only if it's reasonable to do so
- DO NOT leave any {source_language} text- no matter how simple, only if it's reasonable to do so
- EVERY "content" field MUST contain {target_language} text- no matter how simple, only if it's reasonable to do so

STEP 3: PRESERVATION REQUIREMENTS
- Keep ALL HTML tags exactly: <i>, <b>, <font color="#FFFFFF">, <font color="#FF0000">
- Keep ALL line breaks (\n) in the same positions
- Keep ALL punctuation exactly as positioned
- Keep ALL capitalization (capital letters start sentences)
- Keep ALL numbers, dates, times

STEP 4: TRANSLATION QUALITY REQUIREMENTS
- Use natural, conversational {target_language} expressions
- For {target_language}, use everyday spoken language that real people actually say
- If the line is casual/slang, make the translation casual/slang in {target_language}
- If the line is formal/polite, make the translation formal/polite in {target_language}
- For profanity: use equivalent intensity {target_language} profanity, not censorship

STEP 5: VERIFICATION CHECKLIST
Before returning, verify every single line:
[ ] Every "content" contains ONLY {target_language} characters/words
[ ] Not a single {source_language} word remains in any "content"
[ ] All HTML tags preserved exactly
[ ] All line breaks preserved exactly
[ ] All diacritics/accents proper for {target_language}
[ ] Natural flow that sounds like native speakers
[ ] Same emotional tone maintained
[ ] Same level of formality maintained

VIOLATION CONSEQUENCES:
- If any line contains {source_language}: FAIL
- If any {target_language} diacritics/accents missing: FAIL
- If different number of lines than input: FAIL
- If HTML tags missing/changed: FAIL

FINAL CHECK: Read through all "content" fields one by one.
Does every single one contain ONLY natural {target_language}? YES or FAIL."""

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