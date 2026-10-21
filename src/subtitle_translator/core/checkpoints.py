"""Per-batch checkpoints: what an attempt has already translated, tied to the source text."""

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from subtitle_translator.core.engine import Line


def source_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CheckpointLine:
    source: str
    text: str


@dataclass(frozen=True)
class Checkpoint:
    lines: Mapping[int, CheckpointLine]
    tokens: int
    cost: float
    total_tokens: int
    total_cost: float
    connection_id: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lines": {
                str(p): {"source": line.source, "text": line.text}
                for p, line in self.lines.items()
            },
            "tokens": self.tokens,
            "cost": self.cost,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "connection_id": self.connection_id,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Checkpoint":
        if not isinstance(data, Mapping) or not isinstance(data.get("lines"), Mapping):
            raise ValueError("a checkpoint needs a lines mapping")
        lines = {}
        for position, line in data["lines"].items():
            if (
                not isinstance(line, Mapping)
                or not isinstance(line.get("source"), str)
                or not isinstance(line.get("text"), str)
            ):
                raise ValueError(f"checkpoint line {position!r} is malformed")
            lines[int(position)] = CheckpointLine(line["source"], line["text"])
        numbers = {}
        for key in ("tokens", "total_tokens"):
            value = data.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"checkpoint {key} is malformed")
            numbers[key] = value
        for key in ("cost", "total_cost"):
            value = data.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"checkpoint {key} is malformed")
            numbers[key] = float(value)
        return cls(
            lines=lines,
            connection_id=str(data.get("connection_id", "")),
            model=str(data.get("model", "")),
            **numbers,
        )


def usable_translations(
    lines: Sequence["Line"], checkpoints: Iterable[Checkpoint]
) -> dict[int, str]:
    """Earlier translations whose source text still matches, later checkpoints winning."""
    digests = {line.position: source_digest(line.text) for line in lines}
    usable: dict[int, str] = {}
    for checkpoint in checkpoints:
        for position, line in checkpoint.lines.items():
            if digests.get(position) == line.source:
                usable[position] = line.text
    return usable
