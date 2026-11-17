"""The bazarr.translator.v1 operations on plain dictionaries. No SDK import here."""

import importlib.metadata
from collections.abc import Callable, Iterable
from typing import Any

from subtitle_translator import __version__
from subtitle_translator.core.backends import BackendRegistry
from subtitle_translator.core.models import MAX_LINE_CHARS
from subtitle_translator.plugin.scope import PluginScope

KIND_API = "bazarr.translator.v1"
SETTINGS_VERSION = 1
MAX_LINES = 10_000


class TranslatorContract:
    def __init__(
        self,
        scope: PluginScope,
        entry_points: Callable[..., Iterable[Any]] = importlib.metadata.entry_points,
    ) -> None:
        self.scope = scope
        self._entry_points = entry_points
        self._registry: BackendRegistry | None = None

    @property
    def registry(self) -> BackendRegistry:
        if self._registry is None:
            self._registry = BackendRegistry.discover(self.scope.backend_scope(), self._entry_points)
        return self._registry

    def describe(self) -> dict[str, Any]:
        served, unavailable = [], []
        for entry in sorted(self.registry.entries(), key=lambda e: e.type_id):
            if entry.backend is not None:
                served.append({"type": entry.type_id, "name": entry.backend.display_name})
            elif entry.error:
                unavailable.append({"type": entry.type_id, "error": entry.error})
        return {
            "kind_api": KIND_API,
            "plugin_version": __version__,
            "settings_version": SETTINGS_VERSION,
            "backend_types": served,
            "unavailable": unavailable,
            "languages": None,
            "detects_source": True,
            "limits": {"max_lines": MAX_LINES, "max_line_chars": MAX_LINE_CHARS},
        }
