# src/subtitle_translator/plugin/wire.py
"""The translator kind's dictionaries, read into core types and written back.

The key names are this package's proposal until the Bazarr+ plugin SDK and host fix them;
this module is the only place that knows them.
"""

import math
from collections.abc import Mapping
from typing import Any

from subtitle_translator.core.backends import Connection
from subtitle_translator.core.checkpoints import Checkpoint
from subtitle_translator.core.context import Language, MediaContext
from subtitle_translator.core.engine import EngineError, Line, TranslationJob, TranslationOutcome
from subtitle_translator.core.models import MAX_LINE_CHARS
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.usage import Usage

# The most lines one request may carry; describe() reports it.
MAX_LINES = 10_000
# The engine settings a request may carry, with the bounds the manifest declares.
ENGINE_LIMITS = {"request_timeout": (30, 900), "max_retries": (0, 6)}


class WireError(ValueError):
    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WireError(field, "must be an object")
    return value


def _text(value: Any, field: str, *, required: bool = True, limit: int | None = None) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise WireError(field, "must be text")
    if limit is not None and len(value) > limit:
        raise WireError(field, f"is longer than {limit} characters")
    return value


def read_language(data: Any, field: str) -> Language:
    data = _mapping(data, field)
    code = _text(data.get("code"), f"{field}.code", limit=35)
    if code.strip().lower() == "auto":
        return Language.auto()
    return Language(
        code=code,
        name=_text(data.get("name"), f"{field}.name", required=False, limit=100) or "",
        script=_text(data.get("script"), f"{field}.script", required=False, limit=8),
        rtl=bool(data.get("rtl", False)),
    )


def read_media(data: Any) -> MediaContext:
    if data is None:
        return MediaContext()
    data = _mapping(data, "media")
    year = data.get("year")
    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        raise WireError("media.year", "must be a whole number")
    return MediaContext(
        type=_text(data.get("type"), "media.type", required=False, limit=50),
        title=_text(data.get("title"), "media.title", required=False, limit=500),
        year=year,
        imdb_id=_text(data.get("imdb_id"), "media.imdb_id", required=False, limit=20),
        overview=_text(data.get("overview"), "media.overview", required=False, limit=5000),
    )


def read_lines(data: Any) -> list[Line]:
    if not isinstance(data, list):
        raise WireError("lines", "must be a list")
    if len(data) > MAX_LINES:
        raise WireError("lines", f"has more than {MAX_LINES} entries")
    lines = []
    for index, item in enumerate(data):
        item = _mapping(item, f"lines[{index}]")
        position = item.get("position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise WireError(f"lines[{index}].position", "must be a whole number from 0")
        lines.append(Line(position, _text(item.get("text"), f"lines[{index}].text", limit=MAX_LINE_CHARS)))
    return lines


def read_connection(config: Any, secrets: Any) -> Connection:
    connection = _mapping(_mapping(config, "config").get("connection"), "config.connection")
    return Connection(
        id=_text(connection.get("id"), "config.connection.id", limit=64),
        backend_type=_text(connection.get("type"), "config.connection.type", limit=64),
        enabled=bool(connection.get("enabled", True)),
        settings=dict(_mapping(connection.get("settings") or {}, "config.connection.settings")),
        secrets=dict(_mapping(secrets or {}, "secrets")),
    )


def read_engine_settings(config: Any) -> CoreSettings:
    settings = _mapping(_mapping(config, "config").get("settings") or {}, "config.settings")
    values = {}
    for key, (low, high) in ENGINE_LIMITS.items():
        if key in settings:
            value = settings[key]
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise WireError(f"config.settings.{key}", f"must be a whole number from {low} to {high}")
            values[key] = value
    return CoreSettings(**values)


def read_budget(value: Any, field: str = "budget") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise WireError(field, "must be a number from 0, or null")
    return float(value)


def read_job(request: Any) -> tuple[TranslationJob, Connection, CoreSettings]:
    request = _mapping(request, "request")
    lines = read_lines(request.get("lines"))
    checkpoints = []
    held = request.get("checkpoints") or []
    if not isinstance(held, list):
        raise WireError("checkpoints", "must be a list")
    for index, item in enumerate(held):
        try:
            checkpoints.append(Checkpoint.from_dict(item))
        except ValueError as error:
            raise WireError(f"checkpoints[{index}]", str(error)) from None
    budget = read_budget(request.get("budget"))
    job = TranslationJob(
        source=read_language(request.get("source"), "source"),
        target=read_language(request.get("target"), "target"),
        lines=lines,
        media=read_media(request.get("media")),
        checkpoints=checkpoints,
        budget=budget,
        add_rtl_marks=False,
        session_id=_text(request.get("job_id"), "job_id", required=False, limit=128),
    )
    config = request.get("config")
    return job, read_connection(config, request.get("secrets")), read_engine_settings(config)


def write_outcome(outcome: TranslationOutcome) -> dict[str, Any]:
    return {
        "lines": [{"position": line.position, "text": line.text} for line in outcome.lines],
        "status": outcome.status,
        "missing": list(outcome.missing),
        "backend": outcome.backend,
        "connection": outcome.connection_id,
        "model": outcome.model,
        "tokens": outcome.tokens,
        "cost": outcome.cost,
        "batches": [{"positions": list(batch.positions), "model": batch.model} for batch in outcome.batches],
    }


def write_error(error: EngineError) -> dict[str, Any]:
    return {
        "code": str(error.kind),
        "message": error.message,
        "retry_after": error.retry_after,
        "field_errors": dict(error.field_errors),
        "reason": error.reason,
    }


def write_usage(usage: Usage) -> dict[str, Any]:
    return {
        "tokens": usage.tokens,
        "cost": usage.cost,
        "model": usage.model,
        "total_tokens": usage.total_tokens,
        "total_cost": usage.total_cost,
        "reservations": list(usage.reservations),
    }


def write_checkpoint(checkpoint: Checkpoint) -> dict[str, Any]:
    return checkpoint.to_dict()
