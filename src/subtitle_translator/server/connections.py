"""The standalone's ordered back-end list, read from the environment.

OPENROUTER_* and GEMINI_* configure those two built-in back ends. Any other back end named in
BACKEND_ORDER, the built-in OpenAI-compatible one or an installed back-end package, reads each
of its settings from <TYPE>_<FIELD>, for example OPENAI_COMPATIBLE_BASE_URL or VENDOR_CLI_MODEL,
and is turned on when its settings validate. They are read from the process environment and
from the same .env file as every other setting; the environment wins, as it does for them.
"""

import logging
import os
from collections.abc import Mapping

from dotenv import dotenv_values

from subtitle_translator.core.backends import BackendRegistry, Connection, FieldType, SettingField
from subtitle_translator.core.errors import TranslationProviderError
from subtitle_translator.server.config import Settings

logger = logging.getLogger(__name__)
_LISTS = {FieldType.LIST, FieldType.KEY_LIST, FieldType.ORDERABLE_LIST}


def env_name(type_id: str, key: str) -> str:
    return f"{type_id}_{key}".upper().replace("-", "_")


def _split(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _parse(spec: SettingField, raw: str):
    if spec.type in _LISTS:
        return _split(raw)
    if spec.type is FieldType.TOGGLE:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if spec.type is FieldType.INTEGER:
        return int(raw)
    if spec.type is FieldType.NUMBER:
        return float(raw)
    return raw


def environment() -> dict[str, str]:
    """The process environment over the .env file the other settings are read from."""
    values: dict[str, str] = {}
    env_file = Settings.model_config.get("env_file")
    for name in (env_file,) if isinstance(env_file, (str, os.PathLike)) else tuple(env_file or ()):
        try:
            values.update({k: v for k, v in dotenv_values(name).items() if v is not None})
        except OSError:
            continue
    values.update(os.environ)
    return values


def _configured(name: str, backend, environ: Mapping[str, str]) -> Connection:
    values, secrets = {}, {}
    for spec in backend.fields():
        raw = environ.get(env_name(name, spec.key))
        if raw is None or not raw.strip():
            # Docker Compose hands a variable the .env file lacks through as an empty string.
            continue
        try:
            (secrets if spec.secret else values)[spec.key] = _parse(spec, raw)
        except ValueError:
            logger.warning("%s is not a valid %s", env_name(name, spec.key), spec.type)
    validation = backend.validate(values, secrets)
    if not validation.ok:
        logger.warning("%s is off: %s", name, ", ".join(f"{k}: {v}" for k, v in validation.errors.items()))
    return Connection(name, name, enabled=validation.ok, settings=validation.patch or values, secrets=secrets)


def build_connections(
    settings: Settings, registry: BackendRegistry, environ: Mapping[str, str] | None = None
) -> list[Connection]:
    if environ is None:
        environ = environment()
    connections: list[Connection] = []
    for name in _split(settings.backend_order):
        if any(c.id == name for c in connections):
            continue
        if name == "openrouter":
            key = settings.openrouter_api_key
            connections.append(
                Connection(
                    "openrouter",
                    "openrouter",
                    enabled=bool(key),
                    settings={
                        "model": settings.openrouter_default_model,
                        "temperature": settings.openrouter_temperature,
                        "max_tokens": settings.openrouter_max_tokens,
                        "deny_data_collection": settings.openrouter_deny_data_collection,
                    },
                    secrets={"api_key": key},
                )
            )
            continue
        if name == "gemini":
            keys = _split(settings.gemini_api_keys)
            connections.append(
                Connection(
                    "gemini",
                    "gemini",
                    enabled=bool(keys),
                    settings={
                        "model": settings.gemini_model,
                        "batch_size": settings.gemini_batch_size,
                        "reasoning": settings.gemini_reasoning,
                        "price_input": settings.gemini_price_input,
                        "price_output": settings.gemini_price_output,
                        "key_projects": _split(settings.gemini_key_projects),
                    },
                    secrets={"api_keys": keys},
                )
            )
            continue
        try:
            backend = registry.get(name)
        except TranslationProviderError as error:
            logger.warning("BACKEND_ORDER names %s, which is not available: %s", name, error.message)
            continue
        try:
            connections.append(_configured(name, backend, environ))
        except Exception as error:  # noqa: BLE001 - a broken extension must not stop the service
            logger.warning("%s is off: its setup failed (%s: %s)", name, type(error).__name__, error)
    return connections
