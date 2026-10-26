"""The standalone's ordered back-end list, read from the environment.

OPENROUTER_* and GEMINI_* configure those two built-in back ends. Any other back end named in
BACKEND_ORDER, the built-in OpenAI-compatible one or an installed back-end package, reads each
of its settings from <TYPE>_<FIELD>, for example OPENAI_COMPATIBLE_BASE_URL or VENDOR_CLI_MODEL,
and is turned on when its settings validate.
"""

import logging
import os
from collections.abc import Mapping

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


def build_connections(
    settings: Settings, registry: BackendRegistry, environ: Mapping[str, str] = os.environ
) -> list[Connection]:
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
        values, secrets = {}, {}
        for spec in backend.fields():
            raw = environ.get(env_name(name, spec.key))
            if raw is None:
                continue
            try:
                (secrets if spec.secret else values)[spec.key] = _parse(spec, raw)
            except ValueError:
                logger.warning("%s is not a valid %s", env_name(name, spec.key), spec.type)
        validation = backend.validate(values, secrets)
        if not validation.ok:
            logger.warning("%s is off: %s", name, ", ".join(f"{k}: {v}" for k, v in validation.errors.items()))
        connections.append(
            Connection(name, name, enabled=validation.ok, settings=validation.patch or values, secrets=secrets)
        )
    return connections
