"""Which back ends one plugin worker serves: the built-in ones, or one extension's."""

from dataclasses import dataclass
from typing import Literal

from subtitle_translator.core.backends import BackendScope, normalize_distribution


@dataclass(frozen=True)
class PluginScope:
    kind: Literal["builtin", "extension"]
    distribution: str | None = None

    @classmethod
    def builtin(cls) -> "PluginScope":
        return cls("builtin")

    @classmethod
    def extension(cls, distribution: str) -> "PluginScope":
        return cls("extension", normalize_distribution(distribution))

    def backend_scope(self) -> BackendScope:
        if self.kind == "builtin":
            return BackendScope.builtin()
        return BackendScope.extension(self.distribution or "")
