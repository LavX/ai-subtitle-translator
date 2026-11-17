"""The bazarr.translator.v1 operations on plain dictionaries. No SDK import here."""

import importlib.metadata
from collections.abc import Callable, Iterable
from typing import Any

from subtitle_translator import __version__
from subtitle_translator.core.backends import BackendRegistry
from subtitle_translator.core.errors import (
    DependencyUnavailableError,
    InvalidConfigError,
    TranslationProviderError,
)
from subtitle_translator.core.models import MAX_LINE_CHARS
from subtitle_translator.plugin.scope import PluginScope
from subtitle_translator.plugin.wire import (
    MAX_LINES,
    WireError,
    read_connection,
    read_engine_settings,
)

KIND_API = "bazarr.translator.v1"
SETTINGS_VERSION = 1
SECRET_FIELDS_V1 = {"openrouter": {"api_key"}, "gemini": {"api_keys"}, "openai_compatible": {"api_key"}}


class ContractError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        field_errors: dict[str, str] | None = None,
        retry_after: float | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field_errors = dict(field_errors or {})
        self.retry_after = retry_after
        self.reason = reason


def secret_fields(contract: "TranslatorContract") -> dict[str, set[str]]:
    result = {}
    for type_id in contract.registry.types():
        backend = contract.registry.get(type_id)
        result[type_id] = {spec.key for spec in backend.fields() if spec.secret}
    return result


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

    def _backend(self, type_id: str):
        """The back end of a type this worker serves. A type outside its scope, or one refused
        because another package owns the id, is invalid_config naming the type. A type in scope
        whose back end could not be loaded is dependency_unavailable, so Bazarr+ moves on to its
        next connection."""
        try:
            return self.registry.get(type_id)
        except InvalidConfigError:
            raise ContractError(
                "invalid_config",
                f"This plugin does not serve {type_id}",
                field_errors={"connection.type": f"this plugin does not serve {type_id}"},
            ) from None
        except DependencyUnavailableError as error:
            raise ContractError("dependency_unavailable", error.message) from None

    def validate(self, config: dict) -> dict:
        try:
            connection = read_connection(config, {})
            read_engine_settings(config)
        except WireError as error:
            raise ContractError(
                "invalid_config", error.message, field_errors={error.field: error.message}
            ) from None
        backend = self._backend(connection.backend_type)
        secret_keys = {spec.key for spec in backend.fields() if spec.secret}
        outcome = backend.validate(connection.settings, {})
        errors = {
            f"connection.settings.{k}": v for k, v in outcome.errors.items() if k not in secret_keys
        }
        if errors:
            raise ContractError("invalid_config", "Some settings are not valid", field_errors=errors)
        return {
            "settings": dict(config.get("settings") or {}),
            "connection": {
                "id": connection.id,
                "type": connection.backend_type,
                "enabled": connection.enabled,
                "settings": {k: v for k, v in outcome.patch.items() if k not in secret_keys},
            },
        }

    def migrate(self, from_version: int, settings: dict, connections: list[dict]) -> dict:
        if from_version > SETTINGS_VERSION or from_version < 1:
            raise ContractError("invalid_config", f"Cannot migrate settings from version {from_version}")
        migrated = []
        for index, connection in enumerate(connections):
            backend = self._backend(str(connection.get("type")))
            try:
                new_settings = backend.migrate(from_version, dict(connection.get("settings") or {}))
            except TranslationProviderError as error:
                raise ContractError(
                    "invalid_config", error.message, field_errors={f"connections[{index}]": error.message}
                ) from None
            migrated.append({**connection, "settings": new_settings})
        return {"settings": dict(settings), "connections": migrated}
