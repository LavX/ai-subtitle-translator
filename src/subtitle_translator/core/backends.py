"""The back-end plugin interface, and the registry that finds back ends through entry points.

A back end is one way of reaching a model: OpenRouter, Gemini, or a harness such as a vendor
CLI. Each is a BackendType registered under the ``subtitle_translator.backends`` entry-point
group; the entry point's name is the type id. A scope decides which distributions' back ends
a process may load, and it is applied before anything is imported, so a Bazarr+ worker for
the built-in back ends never runs extension code and an extension's worker never offers the
built-ins.
"""

import hashlib
import importlib.metadata
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from subtitle_translator.core.errors import (
    DependencyUnavailableError,
    ErrorKind,
    InvalidConfigError,
    TranslationProviderError,
    error_kind,
)
from subtitle_translator.core.models import TranslationConfig
from subtitle_translator.core.provider import TranslationBatch, TranslationProvider
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.usage import UsageMeter, call_scope, reserve_request

ENTRY_POINT_GROUP = "subtitle_translator.backends"
BUILTIN_DISTRIBUTION = "lavx-ai-subtitle-translator"


def normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class FieldType(StrEnum):
    TEXT = "text"
    SECRET = "secret"
    NUMBER = "number"
    INTEGER = "integer"
    TOGGLE = "toggle"
    CHOICE = "choice"
    LIST = "list"
    KEY_LIST = "key_list"
    ORDERABLE_LIST = "orderable_list"
    LIVE_CHOICE = "live_choice"


_SECRET_TYPES = frozenset({FieldType.SECRET, FieldType.KEY_LIST})
_FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class Choice:
    """One option of a choice field. fills makes it a preset: values this option puts into
    other fields of the same form, which stay editable."""

    value: str
    label: str
    description: str | None = None
    prompt_price: float | None = None
    completion_price: float | None = None
    context_length: int | None = None
    fills: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class SettingField:
    """One per-connection setting. In Bazarr+ the signed manifest carries these; the standalone
    reads them directly. The plugin package generates the manifest's settings block from them."""

    key: str
    type: FieldType
    label: str
    default: Any = None
    required: bool = False
    help: str | None = None
    help_url: str | None = None
    choices: tuple[Choice, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    group: str | None = None
    show_when: tuple[str, Any] | None = None
    destination: bool = False

    @property
    def secret(self) -> bool:
        return self.type in _SECRET_TYPES


@dataclass(frozen=True)
class Validation:
    patch: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class SignInStatus:
    signed_in: bool
    detail: str | None = None


@dataclass(frozen=True)
class TestOutcome:
    ok: bool
    kind: ErrorKind | None = None
    message: str = ""
    model: str | None = None
    sign_in: SignInStatus | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Connection:
    """One entry of the engine's ordered back-end list."""

    id: str
    backend_type: str
    enabled: bool = True
    settings: Mapping[str, Any] = field(default_factory=dict)
    secrets: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def fingerprint(self) -> str:
        payload = json.dumps(
            [self.backend_type, dict(self.settings), dict(self.secrets)],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class BackendContext:
    data_dir: Path | None = None
    artifacts_dir: Path | None = None
    temp_dir: Path | None = None

    def sign_in_dir(self, connection_id: str) -> Path | None:
        """The connection's own sign-in folder inside the data folder: HOME and the XDG folders
        of its vendor binary, so two connections of one type keep two accounts.
        Bazarr+ creates it with the connection and deletes it with it, and the login command
        uses the same path. None when there is no data folder."""
        if self.data_dir is None:
            return None
        if not _FOLDER_NAME.fullmatch(connection_id):
            raise InvalidConfigError(
                f"Connection id {connection_id!r} cannot name a folder",
                field_errors={"connection.id": "letters, digits, '.', '_' and '-' only"},
            )
        return self.data_dir / "sign-in" / connection_id


@dataclass
class OpenedBackend:
    provider: TranslationProvider
    settings: CoreSettings
    config: TranslationConfig
    model: str
    batch_size: int | None = None


_PROBE = TranslationBatch(
    lines=[{"index": "0", "content": "Hello, how are you?"}],
    source_language="English",
    target_language="Spanish",
)


class BackendType(ABC):
    type_id: ClassVar[str]
    display_name: ClassVar[str]
    settings_version: ClassVar[int] = 1
    uses_sign_in: ClassVar[bool] = False
    login_argv: ClassVar[tuple[str, ...]] = ()
    endpoints: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def fields(self) -> list[SettingField]: ...

    @abstractmethod
    def validate(self, settings: Mapping[str, Any], secrets: Mapping[str, Any]) -> Validation:
        """Normalize and check without a network call."""

    @abstractmethod
    def open(
        self, connection: Connection, base: CoreSettings, context: BackendContext
    ) -> OpenedBackend:
        """Build the provider and per-request config for one connection."""

    async def options(
        self,
        field: str,
        settings: Mapping[str, Any],
        secrets: Mapping[str, Any],
        context: BackendContext,
    ) -> list[Choice]:
        return []

    async def test(
        self,
        connection: Connection,
        base: CoreSettings,
        context: BackendContext,
        meter: UsageMeter | None = None,
    ) -> TestOutcome:
        """A real one-line call through this connection. It is held and settled on the meter like
        any translation call, so a paid test counts against the caps and is refused, without
        being sent, when its worst case does not fit; a call that may have been billed without
        reporting keeps its hold. The call's failures are in the outcome; a connection that
        open() refuses raises open()'s error before anything is sent."""
        opened = self.open(connection, base, context)
        meter = meter or UsageMeter()
        config = opened.config.for_operation()
        config._usage_meter = meter
        try:
            with call_scope() as holds:
                try:
                    if not getattr(opened.provider, "prices_requests", False):
                        await reserve_request(
                            config, None, opened.provider.provider_name, opened.model
                        )
                    result = await opened.provider.translate_batch(_PROBE, config_override=config)
                except TranslationProviderError as error:
                    settled = () if error.billing_unknown else holds
                    await meter.record(
                        error.tokens_used, error.cost, opened.model, reservations=settled
                    )
                    return TestOutcome(
                        ok=False,
                        kind=error_kind(error),
                        message=error.message,
                        model=opened.model,
                        reason=getattr(error, "reason", None),
                    )
                await meter.record(
                    result.total_tokens or 0,
                    result.cost or 0.0,
                    result.model_used or opened.model,
                    reservations=holds,
                )
        finally:
            await opened.provider.close()
        ok = bool(result.translations) and bool(result.translations[0].get("content", "").strip())
        return TestOutcome(
            ok=ok,
            kind=None if ok else ErrorKind.PARTIAL,
            message="" if ok else "The model answered without a translation",
            model=result.model_used or opened.model,
        )

    async def sign_in_status(
        self, connection: Connection, context: BackendContext
    ) -> SignInStatus | None:
        return None

    def migrate(self, from_version: int, settings: Mapping[str, Any]) -> dict[str, Any]:
        if from_version == self.settings_version:
            return dict(settings)
        raise InvalidConfigError(
            f"{self.display_name} cannot read settings version {from_version}",
            provider=self.type_id,
        )

    def required_errors(
        self, settings: Mapping[str, Any], secrets: Mapping[str, Any]
    ) -> dict[str, str]:
        errors = {}
        for spec in self.fields():
            if not spec.required:
                continue
            value = (secrets if spec.secret else settings).get(spec.key)
            if value is None or value == "" or value == [] or value == ():
                errors[spec.key] = f"{spec.label} is required"
        return errors


@dataclass(frozen=True)
class BackendScope:
    distributions: frozenset[str] | None

    @classmethod
    def builtin(cls) -> "BackendScope":
        return cls(frozenset({BUILTIN_DISTRIBUTION}))

    @classmethod
    def extension(cls, distribution: str) -> "BackendScope":
        return cls(frozenset({normalize_distribution(distribution)}))

    @classmethod
    def everything(cls) -> "BackendScope":
        return cls(None)

    def admits(self, distribution: str | None) -> bool:
        if self.distributions is None:
            return True
        return distribution is not None and normalize_distribution(distribution) in self.distributions


@dataclass(frozen=True)
class RegisteredBackend:
    type_id: str
    backend: "BackendType | None"
    distribution: str | None
    version: str | None
    error: str | None
    # Another package's entry under a type id it may not serve: one reserved for the built-in
    # back end, or one another package already provides. It never serves that type.
    refused: bool = False

    @property
    def builtin(self) -> bool:
        return _is_builtin(self.distribution)


def _is_builtin(distribution: str | None) -> bool:
    return distribution is not None and normalize_distribution(distribution) == BUILTIN_DISTRIBUTION


def _distribution_of(point: Any) -> str | None:
    dist = getattr(point, "dist", None)
    return getattr(dist, "name", None) if dist is not None else None


class BackendRegistry:
    def __init__(self, entries: Iterable[RegisteredBackend] = ()) -> None:
        self._active: dict[str, RegisteredBackend] = {}
        self._all: list[RegisteredBackend] = []
        for entry in entries:
            self._add(entry)

    def _add(self, entry: RegisteredBackend) -> None:
        current = self._active.get(entry.type_id)
        if current is None:
            self._active[entry.type_id] = entry
            self._all.append(entry)
            return
        if entry.builtin != current.builtin:
            # A built-in owns its type id whether or not it loaded. When it is broken the type is
            # unavailable; it is never served by another package, which would then receive the
            # built-in connections and their keys.
            winner, loser = (entry, current) if entry.builtin else (current, entry)
        elif current.backend is None and entry.backend is not None:
            # Between two packages of the same standing, one that loads replaces one that did not.
            self._active[entry.type_id] = entry
            self._all.append(entry)
            return
        else:
            winner, loser = current, entry
        if winner is entry:
            self._active[entry.type_id] = entry
            self._all = [e for e in self._all if e is not current] + [entry]
        reason = (
            f"type id {loser.type_id!r} is reserved for the built-in back end"
            if winner.builtin
            else f"type id {loser.type_id!r} already provided by {winner.distribution}"
        )
        self._all.append(
            RegisteredBackend(
                loser.type_id, None, loser.distribution, loser.version, reason, refused=True
            )
        )

    @classmethod
    def discover(
        cls,
        scope: BackendScope,
        entry_points: Callable[..., Iterable[Any]] = importlib.metadata.entry_points,
    ) -> "BackendRegistry":
        points = list(entry_points(group=ENTRY_POINT_GROUP))
        # The built-in type ids, read from metadata without importing anything, so no scope
        # ever loads another package's back end under one of those ids.
        reserved = {point.name for point in points if _is_builtin(_distribution_of(point))}
        entries = []
        for point in points:
            distribution = _distribution_of(point)
            dist = getattr(point, "dist", None)
            version = getattr(dist, "version", None) if dist is not None else None
            if not scope.admits(distribution):
                continue
            if point.name in reserved and not _is_builtin(distribution):
                entries.append(
                    RegisteredBackend(
                        point.name,
                        None,
                        distribution,
                        version,
                        f"type id {point.name!r} is reserved for the built-in back end",
                        refused=True,
                    )
                )
                continue
            try:
                target = point.load()
                backend = target() if isinstance(target, type) else target
                if not isinstance(backend, BackendType):
                    raise TypeError(f"{point.value} is not a BackendType")
                if backend.type_id != point.name:
                    raise ValueError(f"entry point {point.name!r} names type {backend.type_id!r}")
                entries.append(RegisteredBackend(point.name, backend, distribution, version, None))
            except Exception as exc:  # noqa: BLE001 - one broken back end must not hide the others
                entries.append(
                    RegisteredBackend(
                        point.name, None, distribution, version, f"{type(exc).__name__}: {exc}"
                    )
                )
        return cls(entries)

    def get(self, type_id: str) -> BackendType:
        entry = self._active.get(type_id)
        if entry is None or entry.refused:
            raise InvalidConfigError(
                f"No back end of type {type_id!r} is installed",
                field_errors={"backend_type": "unknown"},
            )
        if entry.backend is None:
            raise DependencyUnavailableError(f"The {type_id} back end could not be loaded: {entry.error}")
        return entry.backend

    def entries(self) -> list[RegisteredBackend]:
        return list(self._all)

    def types(self) -> list[str]:
        return sorted(t for t, e in self._active.items() if e.backend is not None)
