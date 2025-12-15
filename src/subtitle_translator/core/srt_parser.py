"""SRT file parsing and generation utilities."""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import srt

logger = logging.getLogger(__name__)


@dataclass
class SubtitleEntry:
    """A single subtitle entry with timing and content."""

    index: int
    start: timedelta
    end: timedelta
    content: str
    proprietary: str = ""  # For any proprietary data in the subtitle

    def to_srt_subtitle(self) -> srt.Subtitle:
        """Convert to srt library Subtitle object."""
        return srt.Subtitle(
            index=self.index,
            start=self.start,
            end=self.end,
            content=self.content,
            proprietary=self.proprietary,
        )

    @classmethod
    def from_srt_subtitle(cls, sub: srt.Subtitle) -> "SubtitleEntry":
        """Create from srt library Subtitle object."""
        return cls(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            content=sub.content,
            proprietary=sub.proprietary if hasattr(sub, "proprietary") else "",
        )


class SRTParserError(Exception):
    """Exception raised for SRT parsing errors."""

    def __init__(self, message: str, line_number: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.line_number = line_number


class SRTParser:
    """Parser for SRT subtitle files."""

    def __init__(self):
        """Initialize the SRT parser."""
        pass

    def parse(self, content: str) -> list[SubtitleEntry]:
        """
        Parse SRT content into a list of SubtitleEntry objects.

        Args:
            content: Raw SRT file content as string

        Returns:
            List of SubtitleEntry objects

        Raises:
            SRTParserError: If parsing fails
        """
        try:
            # Use the srt library to parse
            subtitles = list(srt.parse(content))
            return [SubtitleEntry.from_srt_subtitle(sub) for sub in subtitles]
        except srt.SRTParseError as e:
            raise SRTParserError(f"Failed to parse SRT content: {str(e)}") from e
        except Exception as e:
            raise SRTParserError(f"Unexpected error parsing SRT: {str(e)}") from e

    def compose(self, entries: list[SubtitleEntry]) -> str:
        """
        Compose SubtitleEntry objects back into SRT format.

        Args:
            entries: List of SubtitleEntry objects

        Returns:
            SRT formatted string
        """
        subtitles = [entry.to_srt_subtitle() for entry in entries]
        return srt.compose(subtitles)

    def extract_lines_for_translation(
        self, entries: list[SubtitleEntry]
    ) -> list[dict[str, str]]:
        """
        Extract subtitle content for translation.

        Converts subtitle entries into the format needed for the translation API.

        Args:
            entries: List of SubtitleEntry objects

        Returns:
            List of {"index": "X", "content": "text"} dictionaries
        """
        return [
            {"index": str(entry.index), "content": entry.content}
            for entry in entries
        ]

    def apply_translations(
        self,
        entries: list[SubtitleEntry],
        translations: list[dict[str, str]],
        is_rtl: bool = False,
    ) -> list[SubtitleEntry]:
        """
        Apply translated content back to subtitle entries.

        Args:
            entries: Original list of SubtitleEntry objects
            translations: List of {"index": "X", "content": "translated"} dictionaries
            is_rtl: Whether to add RTL markers for right-to-left languages

        Returns:
            New list of SubtitleEntry objects with translated content
        """
        # Build a mapping from index to translated content
        translation_map = {t["index"]: t["content"] for t in translations}

        translated_entries = []
        for entry in entries:
            translated_content = translation_map.get(str(entry.index), entry.content)
            
            # Add RTL markers if needed
            if is_rtl:
                translated_content = self._add_rtl_markers(translated_content)

            translated_entries.append(
                SubtitleEntry(
                    index=entry.index,
                    start=entry.start,
                    end=entry.end,
                    content=translated_content,
                    proprietary=entry.proprietary,
                )
            )

        return translated_entries

    def _add_rtl_markers(self, text: str) -> str:
        """
        Add Right-to-Left directional markers to text.

        Args:
            text: Text content

        Returns:
            Text with RTL markers added
        """
        # Unicode RIGHT-TO-LEFT MARK (RLM) and RIGHT-TO-LEFT EMBEDDING (RLE)
        RLM = "\u200F"
        RLE = "\u202B"
        PDF = "\u202C"  # POP DIRECTIONAL FORMATTING
        
        lines = text.split("\n")
        marked_lines = []
        
        for line in lines:
            if line.strip():
                # Wrap each line with RLE...PDF for proper rendering
                marked_lines.append(f"{RLE}{line}{PDF}")
            else:
                marked_lines.append(line)
        
        return "\n".join(marked_lines)

    def validate_srt(self, content: str) -> tuple[bool, Optional[str]]:
        """
        Validate SRT content.

        Args:
            content: Raw SRT file content

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            subtitles = list(srt.parse(content))
            if not subtitles:
                return False, "No subtitles found in content"
            return True, None
        except srt.SRTParseError as e:
            return False, f"Invalid SRT format: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_subtitle_count(self, content: str) -> int:
        """
        Get the number of subtitles in SRT content.

        Args:
            content: Raw SRT file content

        Returns:
            Number of subtitles, or 0 if parsing fails
        """
        try:
            return len(list(srt.parse(content)))
        except Exception:
            return 0

    def merge_multiline_subtitles(self, entries: list[SubtitleEntry]) -> list[SubtitleEntry]:
        """
        Merge subtitles that span multiple lines into single entries.

        This is useful for preserving context during translation.

        Args:
            entries: List of SubtitleEntry objects

        Returns:
            List of SubtitleEntry objects (unchanged in this implementation)
        """
        # SRT already handles multiline content within a single entry
        # This method is here for potential future enhancements
        return entries

    def split_long_subtitles(
        self,
        entries: list[SubtitleEntry],
        max_chars_per_line: int = 42,
        max_lines: int = 2,
    ) -> list[SubtitleEntry]:
        """
        Split subtitles that are too long into multiple lines.

        Args:
            entries: List of SubtitleEntry objects
            max_chars_per_line: Maximum characters per line
            max_lines: Maximum number of lines per subtitle

        Returns:
            List of SubtitleEntry objects with content split if needed
        """
        processed = []
        for entry in entries:
            if len(entry.content) <= max_chars_per_line:
                processed.append(entry)
                continue

            # Split into words and rebuild with line breaks
            words = entry.content.replace("\n", " ").split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                word_len = len(word) + (1 if current_line else 0)
                if current_length + word_len <= max_chars_per_line:
                    current_line.append(word)
                    current_length += word_len
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                lines.append(" ".join(current_line))

            # Limit to max_lines
            lines = lines[:max_lines]
            
            processed.append(
                SubtitleEntry(
                    index=entry.index,
                    start=entry.start,
                    end=entry.end,
                    content="\n".join(lines),
                    proprietary=entry.proprietary,
                )
            )

        return processed


def get_srt_parser() -> SRTParser:
    """Factory function to get an SRT parser instance."""
    return SRTParser()