"""What the caller knows about the languages and the media, handed to the prompt as data."""

from dataclasses import dataclass

SCRIPT_NAMES = {
    "Latn": "Latin",
    "Cyrl": "Cyrillic",
    "Arab": "Arabic",
    "Hebr": "Hebrew",
    "Grek": "Greek",
    "Deva": "Devanagari",
    "Hans": "simplified Han",
    "Hant": "traditional Han",
    "Jpan": "Japanese",
    "Kore": "Korean",
    "Thai": "Thai",
}


@dataclass(frozen=True)
class Language:
    """A BCP-47 language as Bazarr+ normalizes it. code "auto" asks the model to detect it."""

    code: str
    name: str = ""
    script: str | None = None
    rtl: bool = False

    @classmethod
    def auto(cls) -> "Language":
        return cls(code="auto")

    @property
    def is_auto(self) -> bool:
        return self.code.strip().lower() == "auto"

    def label(self) -> str:
        base = (self.name or self.code)[:50]
        if self.script:
            return f"{base}, written in {SCRIPT_NAMES.get(self.script, self.script)[:30]} script"
        return base


@dataclass(frozen=True)
class MediaContext:
    """What the subtitles belong to. Every field is optional."""

    type: str | None = None
    title: str | None = None
    year: int | None = None
    imdb_id: str | None = None
    overview: str | None = None
