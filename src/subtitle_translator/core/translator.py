"""Main translation orchestration module."""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from subtitle_translator.api.models import SubtitleLine, TranslateContentRequest
from subtitle_translator.config import Settings, get_settings
from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.srt_parser import SRTParser, add_rtl_markers
from subtitle_translator.providers.base import TranslationProvider
from subtitle_translator.providers.openrouter import OpenRouterProvider

if TYPE_CHECKING:
    from subtitle_translator.api.models import TranslationConfig

logger = logging.getLogger(__name__)


@dataclass
class ContentTranslationResult:
    """Result of translating subtitle content."""

    lines: list[SubtitleLine]
    model_used: str
    tokens_used: int | None = None
    success: bool = True
    error: str | None = None


@dataclass
class FileTranslationResult:
    """Result of translating an SRT file."""

    content: str
    model_used: str
    tokens_used: int | None = None
    subtitle_count: int = 0
    success: bool = True
    error: str | None = None


class SubtitleTranslator:
    """Main subtitle translation orchestrator."""

    def __init__(
        self,
        provider: TranslationProvider | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize the subtitle translator.

        Args:
            provider: Optional translation provider. Creates OpenRouter by default.
            settings: Optional settings instance.
        """
        self.settings = settings or get_settings()
        self._provider = provider
        self._srt_parser = SRTParser()

    @property
    def provider(self) -> TranslationProvider:
        """Get or create the translation provider."""
        if self._provider is None:
            self._provider = OpenRouterProvider(self.settings)
        return self._provider

    async def close(self) -> None:
        """Clean up resources."""
        if self._provider is not None:
            await self._provider.close()

    async def translate_content(
        self,
        request: TranslateContentRequest,
        config_override: Optional["TranslationConfig"] = None,
    ) -> ContentTranslationResult:
        """
        Translate subtitle content from a Lingarr-compatible request.

        Args:
            request: Translation request with lines to translate
            config_override: Optional per-request configuration override

        Returns:
            ContentTranslationResult with translated lines
        """
        # Use config from request if not provided explicitly
        effective_config = config_override or request.config

        if not request.lines:
            return ContentTranslationResult(
                lines=[],
                model_used=request.model or self.settings.openrouter_default_model,
                tokens_used=0,
            )

        # Convert request lines to internal format
        lines = [{"index": str(line.position), "content": line.line} for line in request.lines]

        # Create batch processor
        processor = BatchProcessor(self.provider, self.settings)

        try:
            # Process all batches
            result = await processor.process_all_batches(
                lines=lines,
                source_language=request.sourceLanguage,
                target_language=request.targetLanguage,
                context_title=request.title,
                context_media_type=request.mediaType,
                model=request.model,
                temperature=request.temperature,
                config_override=effective_config,
            )

            # Check if translation was successful
            if not result.success:
                failed_batches = [r for r in result.batch_results if not r.success]
                error_msg = "; ".join(r.error or "Unknown error" for r in failed_batches)

                # Return partial results if any
                if result.all_translations:
                    translated_lines = self._map_translations_to_lines(
                        request.lines, result.all_translations, request.targetLanguage
                    )
                    return ContentTranslationResult(
                        lines=translated_lines,
                        model_used=result.model_used,
                        tokens_used=result.total_tokens,
                        success=False,
                        error=f"Partial failure: {error_msg}",
                    )
                else:
                    # Nothing came back, but every attempt was billed.
                    return ContentTranslationResult(
                        lines=[],
                        model_used=result.model_used,
                        tokens_used=result.total_tokens,
                        success=False,
                        error=error_msg,
                    )

            # Map translations back to SubtitleLine format
            translated_lines = self._map_translations_to_lines(
                request.lines, result.all_translations, request.targetLanguage
            )

            return ContentTranslationResult(
                lines=translated_lines,
                model_used=result.model_used,
                tokens_used=result.total_tokens,
            )

        except Exception as e:
            logger.exception(f"Translation failed: {e}")
            return ContentTranslationResult(
                lines=[],
                model_used=request.model or self.settings.openrouter_default_model,
                success=False,
                error=str(e),
            )

    async def translate_file(
        self,
        content: str,
        source_language: str,
        target_language: str,
        title: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        config_override: Optional["TranslationConfig"] = None,
    ) -> FileTranslationResult:
        """
        Translate an entire SRT file.

        Args:
            content: Raw SRT file content
            source_language: Source language code
            target_language: Target language code
            title: Optional media title for context
            model: Optional model override
            temperature: Optional temperature override
            config_override: Optional per-request configuration override

        Returns:
            FileTranslationResult with translated SRT content
        """
        # Determine model to use (config override takes precedence)
        if config_override and config_override.model:
            model_to_use = config_override.model
        else:
            model_to_use = model or self.settings.openrouter_default_model

        try:
            # Parse and validate SRT content
            entries = self._srt_parser.parse(content)
            if not entries:
                return FileTranslationResult(
                    content=content,
                    model_used=model_to_use,
                    subtitle_count=0,
                )

            # Extract lines for translation
            lines = self._srt_parser.extract_lines_for_translation(entries)

            # Create batch processor
            processor = BatchProcessor(self.provider, self.settings)

            # Process all batches
            result = await processor.process_all_batches(
                lines=lines,
                source_language=source_language,
                target_language=target_language,
                context_title=title,
                model=model,
                temperature=temperature,
                config_override=config_override,
            )

            if not result.success:
                failed_batches = [r for r in result.batch_results if not r.success]
                error_msg = "; ".join(r.error or "Unknown error" for r in failed_batches)

                # Return partial results if any
                if result.all_translations:
                    is_rtl = self.settings.is_rtl_language(target_language)
                    translated_entries = self._srt_parser.apply_translations(
                        entries, result.all_translations, is_rtl=is_rtl
                    )
                    translated_content = self._srt_parser.compose(translated_entries)

                    return FileTranslationResult(
                        content=translated_content,
                        model_used=result.model_used,
                        tokens_used=result.total_tokens,
                        subtitle_count=len(entries),
                        success=False,
                        error=f"Partial failure: {error_msg}",
                    )
                else:
                    return FileTranslationResult(
                        content="",
                        model_used=result.model_used,
                        tokens_used=result.total_tokens,
                        success=False,
                        error=error_msg,
                    )

            # Check if target language is RTL
            is_rtl = self.settings.is_rtl_language(target_language)

            # Apply translations back to entries
            translated_entries = self._srt_parser.apply_translations(
                entries, result.all_translations, is_rtl=is_rtl
            )

            # Optionally split long subtitles
            translated_entries = self._srt_parser.split_long_subtitles(translated_entries)

            # Compose back to SRT format
            translated_content = self._srt_parser.compose(translated_entries)

            return FileTranslationResult(
                content=translated_content,
                model_used=result.model_used,
                tokens_used=result.total_tokens,
                subtitle_count=len(entries),
            )

        except Exception as e:
            logger.exception(f"File translation failed: {e}")
            return FileTranslationResult(
                content="",
                model_used=model_to_use,
                success=False,
                error=str(e),
            )

    def _map_translations_to_lines(
        self,
        original_lines: list[SubtitleLine],
        translations: list[dict[str, str]],
        target_language: str,
    ) -> list[SubtitleLine]:
        """Map translated content back to SubtitleLine format."""
        return map_translations_to_lines(
            original_lines, translations, target_language, self.settings
        )

    def _add_rtl_markers(self, text: str) -> str:
        """Add RTL markers to text for proper display."""
        return add_rtl_markers(text)

    async def get_available_models(self) -> list[dict]:
        """Get list of available translation models."""
        return await self.provider.get_available_models()

    async def health_check(self) -> bool:
        """Check if the translator is ready to process requests."""
        return await self.provider.health_check()


def map_translations_to_lines(
    original_lines: list[SubtitleLine],
    translations: list[dict[str, str]],
    target_language: str,
    settings: "Settings",
) -> list[SubtitleLine]:
    """
    Map translated content back to SubtitleLine format.

    Args:
        original_lines: Original subtitle lines
        translations: Translated content
        target_language: Target language for RTL handling
        settings: Settings instance

    Returns:
        List of SubtitleLine with translated content
    """
    translation_map = {t["index"]: t["content"] for t in translations}
    is_rtl = settings.is_rtl_language(target_language)

    result = []
    for line in original_lines:
        translated_text = translation_map.get(str(line.position), line.line)

        if is_rtl:
            translated_text = add_rtl_markers(translated_text)

        result.append(SubtitleLine(position=line.position, line=translated_text))

    return result


# Global translator instance for dependency injection
_translator_instance: SubtitleTranslator | None = None
_translator_lock: asyncio.Lock = asyncio.Lock()


async def get_translator() -> SubtitleTranslator:
    """Get or create the global translator instance."""
    global _translator_instance
    if _translator_instance is None:
        async with _translator_lock:
            if _translator_instance is None:
                _translator_instance = SubtitleTranslator()
    return _translator_instance


async def close_translator() -> None:
    """Close the global translator instance."""
    global _translator_instance
    if _translator_instance is not None:
        await _translator_instance.close()
        _translator_instance = None
