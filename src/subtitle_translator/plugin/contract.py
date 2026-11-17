"""The bazarr.translator.v1 operations on plain dictionaries. No SDK import here."""

import asyncio
import importlib.metadata
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, replace
from typing import Any, Protocol

import httpx

from subtitle_translator import __version__
from subtitle_translator.core.backends import (
    BackendContext,
    BackendRegistry,
    Connection,
    FieldType,
)
from subtitle_translator.core.engine import EngineError, EngineEvents, TranslationEngine
from subtitle_translator.core.errors import (
    DependencyUnavailableError,
    InvalidConfigError,
    TranslationProviderError,
    error_kind,
)
from subtitle_translator.core.models import MAX_LINE_CHARS
from subtitle_translator.core.usage import UsageMeter
from subtitle_translator.plugin.scope import PluginScope
from subtitle_translator.plugin.wire import (
    MAX_LINES,
    WireError,
    read_budget,
    read_connection,
    read_engine_settings,
    read_job,
    write_checkpoint,
    write_error,
    write_outcome,
    write_usage,
)

KIND_API = "bazarr.translator.v1"
SETTINGS_VERSION = 1
# Used when the host names no heartbeat interval; the SDK binding passes the host's value.
HEARTBEAT_SECONDS = 10.0
# How many engine-settings snapshots a worker keeps state for.
MAX_ENGINES = 8
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


class Sink(Protocol):
    def progress(self, done: int, total: int, message: str) -> None: ...
    def heartbeat(self) -> None: ...
    async def reserve(self, amount: float, model: str) -> str | None: ...
    async def usage(self, report: dict) -> None: ...
    async def checkpoint(self, data: dict) -> None: ...


def _meter_sinks(sink: "Sink | None"):
    """The usage and reservation callbacks of a meter that reports to Bazarr+ through the sink."""
    if sink is None:
        return None, None

    async def usage(report) -> None:
        await sink.usage(write_usage(report))

    async def reserve(amount: float, model: str) -> str | None:
        return await sink.reserve(amount, model)

    return usage, reserve


def _settings_fields(field_errors: dict[str, str] | None) -> dict[str, str]:
    """A back end names its own settings; Bazarr+ addresses them as connection.settings.<key>."""
    return {
        key if key.startswith("connection.") else f"connection.settings.{key}": message
        for key, message in (field_errors or {}).items()
    }


def key_list(value: str) -> list[str]:
    """The keys of a key-list secret as the host delivers it. The wire carries a secret as one
    text value, so the host sends a key list as a JSON list in text; one key per line is read
    too. Blank entries are dropped."""
    text = value.strip()
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except ValueError:
            decoded = None
        if isinstance(decoded, list):
            return [item.strip() for item in decoded if isinstance(item, str) and item.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def decode_secrets(backend, connection: Connection) -> Connection:
    """The connection with its key-list secrets as lists, whatever form they arrived in."""
    lists = {spec.key for spec in backend.fields() if spec.type == FieldType.KEY_LIST}
    secrets = {
        key: key_list(value) if key in lists and isinstance(value, str) else value
        for key, value in connection.secrets.items()
    }
    return replace(connection, secrets=secrets)


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
        # One engine per distinct engine-settings snapshot, kept for the worker's life: it holds
        # no event loop, so its key health and learned batch sizes serve every call.
        self._engines: OrderedDict[str, TranslationEngine] = OrderedDict()
        self._engines_lock = threading.Lock()

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

    async def options(
        self, field: str, config: dict, secrets: dict, context: BackendContext | None = None
    ) -> list[dict]:
        try:
            connection = read_connection(config, secrets)
        except WireError as error:
            raise ContractError(
                "invalid_config", error.message, field_errors={error.field: error.message}
            ) from None
        # The host names the field by its bare key; the prefixed form is accepted too.
        prefix = "connection.settings."
        key = field[len(prefix) :] if field.startswith(prefix) else field
        if not key or "." in key:
            raise ContractError(
                "invalid_config", f"No live choices for {field}", field_errors={field: "unknown"}
            )
        backend = self._backend(connection.backend_type)
        connection = decode_secrets(backend, connection)
        try:
            choices = await backend.options(
                key, connection.settings, connection.secrets, context or BackendContext()
            )
        except TranslationProviderError as error:
            raise ContractError(
                str(error_kind(error)), error.message, retry_after=error.retry_after
            ) from None
        except (httpx.HTTPError, OSError) as error:
            # A back end that lets a transport failure through, such as an unreachable server.
            raise ContractError(
                "dependency_unavailable", f"The server could not be reached: {type(error).__name__}"
            ) from None
        # A live choice fills nothing else, so the preset values are not part of the answer.
        return [{k: v for k, v in asdict(choice).items() if k != "fills"} for choice in choices]

    async def test(
        self,
        config: dict,
        secrets: dict,
        context: BackendContext | None = None,
        sink: "Sink | None" = None,
        budget: float | None = None,
    ) -> dict:
        try:
            connection = read_connection(config, secrets)
            base = read_engine_settings(config)
            budget = read_budget(budget)
        except WireError as error:
            raise ContractError(
                "invalid_config", error.message, field_errors={error.field: error.message}
            ) from None
        backend = self._backend(connection.backend_type)
        connection = decode_secrets(backend, connection)
        context = context or BackendContext()
        if backend.uses_sign_in:
            status = await backend.sign_in_status(connection, context)
            if status is not None and not status.signed_in:
                return {
                    "ok": False,
                    "code": "auth_required",
                    "reason": None,
                    "message": status.detail or "Signed out",
                    "model": None,
                    "signed_in": False,
                    "tokens": 0,
                    "cost": 0.0,
                }
        # A paid test is held and reported like a translation call.
        meter = UsageMeter(budget, *_meter_sinks(sink))
        try:
            outcome = await backend.test(connection, base, context, meter)
        except TranslationProviderError as error:
            # The back end refused the connection before any call, so there is no result to
            # report; like settings it cannot read, this names the fields at fault.
            raise ContractError(
                str(error_kind(error)),
                error.message,
                field_errors=_settings_fields(getattr(error, "field_errors", None)),
                retry_after=error.retry_after,
            ) from None
        return {
            "ok": outcome.ok,
            "code": str(outcome.kind) if outcome.kind else None,
            "reason": outcome.reason,
            "message": outcome.message,
            "model": outcome.model,
            "signed_in": True if backend.uses_sign_in else None,
            "tokens": meter.tokens,
            "cost": meter.cost,
        }

    async def close_clients(self) -> None:
        """Close the HTTP clients every opened back end holds, keeping the back ends. A host that
        runs each call on an event loop of its own calls this after each call, because a client
        belongs to the loop that first used it; key health, rotation and learned batch sizes
        stay, and a provider opens a new client on its next use."""
        with self._engines_lock:
            engines = list(self._engines.values())
        for engine in engines:
            await engine.close_clients()

    def _engine(self, base, context) -> TranslationEngine:
        key = json.dumps(base.model_dump(), sort_keys=True)
        with self._engines_lock:
            engine = self._engines.get(key)
            if engine is None:
                engine = self._engines[key] = TranslationEngine(
                    self.registry, base, lambda type_id: context
                )
            self._engines.move_to_end(key)
            while len(self._engines) > MAX_ENGINES:
                self._engines.popitem(last=False)
            return engine

    async def translate(
        self,
        request: dict,
        sink: Sink,
        *,
        heartbeat_interval: float = HEARTBEAT_SECONDS,
        context: BackendContext | None = None,
        cancelled: asyncio.Event | None = None,
    ) -> dict:
        try:
            job, connection, base = read_job(request)
        except WireError as error:
            raise ContractError(
                "invalid_config", error.message, field_errors={error.field: error.message}
            ) from None
        connection = decode_secrets(self._backend(connection.backend_type), connection)
        if cancelled is not None and cancelled.is_set():
            raise ContractError("cancelled", "Bazarr+ cancelled this job before it started")
        engine = self._engine(base, context or BackendContext())

        async def beat() -> None:
            while True:
                sink.heartbeat()
                await asyncio.sleep(heartbeat_interval)

        usage, reserve = _meter_sinks(sink)

        async def checkpoint(data) -> None:
            await sink.checkpoint(write_checkpoint(data))

        events = EngineEvents(
            on_progress=lambda p: sink.progress(p.done_lines, p.total_lines, p.message),
            on_usage=usage,
            on_checkpoint=checkpoint,
            on_reserve=reserve,
        )
        # The attempt runs as its own task, so either way Bazarr+ cancels a job, by cancelling
        # this coroutine or by setting the SDK's cancel event, it stops at once and reports
        # nothing more.
        heartbeat = asyncio.create_task(beat())
        stop = asyncio.create_task(cancelled.wait()) if cancelled is not None else None
        work = asyncio.create_task(engine.attempt(job, connection, events))
        try:
            await asyncio.wait({work, stop} - {None}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (work, stop, heartbeat):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(t for t in (work, stop, heartbeat) if t is not None), return_exceptions=True
            )
        if work.cancelled():
            raise ContractError("cancelled", "Bazarr+ cancelled this job")
        error = work.exception()
        if isinstance(error, EngineError):
            shaped = write_error(error)
            raise ContractError(
                shaped["code"],
                shaped["message"],
                _settings_fields(shaped["field_errors"]),
                shaped["retry_after"],
                shaped["reason"],
            ) from None
        if error is not None:
            raise error
        return write_outcome(work.result())
