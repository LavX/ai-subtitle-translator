"""The OpenAI-compatible back end: the OpenAI API, LM Studio, Ollama or any other server that
speaks chat completions.

One back-end type. A connection picks a preset, which fills its server address (the address
stays editable) and carries what that server expects: the name of the output limit, a
temperature that is dropped when a model refuses one, and how JSON mode is asked for. The
presets live here, so nothing they change reaches the Gemini back end on the shared client.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from subtitle_translator.backends.openai_compat import OpenAICompatProvider
from subtitle_translator.core.backends import (
    BackendContext,
    BackendType,
    Choice,
    Connection,
    FieldType,
    OpenedBackend,
    SettingField,
    Validation,
    request_timeout_of,
)
from subtitle_translator.core.errors import (
    AuthenticationError,
    InvalidConfigError,
    QuotaExhaustedError,
    TranslationProviderError,
)
from subtitle_translator.core.models import TranslationConfig
from subtitle_translator.core.settings import CoreSettings

# LM Studio documents structured output only through a JSON schema. The schema is the shape the
# prompt asks for; a server that refuses it has response_format dropped for that model.
TRANSLATIONS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "translations",
        "schema": {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"index": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["index", "content"],
                    },
                }
            },
            "required": ["translations"],
        },
    },
}


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    base_url: str
    output_limit: str = "max_tokens"
    relaxable: tuple[str, ...] = ()
    response_format: Mapping[str, Any] | None = field(default=None, hash=False)
    parallel_batches: int = 1


PRESETS = {
    # OpenAI wants max_completion_tokens, and its reasoning models accept only the default
    # temperature.
    "openai": Preset(
        "openai",
        "OpenAI",
        "https://api.openai.com/v1",
        output_limit="max_completion_tokens",
        relaxable=("temperature",),
        parallel_batches=4,
    ),
    "lmstudio": Preset(
        "lmstudio", "LM Studio", "http://localhost:1234/v1", response_format=TRANSLATIONS_SCHEMA
    ),
    "ollama": Preset("ollama", "Ollama", "http://localhost:11434/v1"),
    "custom": Preset("custom", "OpenAI-compatible server", ""),
}
DEFAULT_PRESET = "custom"
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_TOKENS = 8000


def _base_url(value: Any) -> tuple[str, str | None]:
    """The address normalized, and what is wrong with it, if anything."""
    text = str(value or "").strip().rstrip("/")
    if not text:
        return "", "Enter the server's address, such as http://192.168.1.10:1234/v1"
    try:
        parts = urlsplit(text)
        parts.port  # noqa: B018 - raises ValueError for a port that is not a number
    except ValueError:
        return text, "This is not a valid address"
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return text, "Use an http:// or https:// address"
    if parts.username or parts.password or parts.query or parts.fragment:
        return text, "The address takes no user name, password or query; put a key in the API key field"
    return text, None


def _insufficient_quota(response: httpx.Response) -> bool:
    try:
        body = response.json()
    except ValueError:
        return False
    error = body.get("error") if isinstance(body, dict) else None
    return isinstance(error, dict) and "insufficient_quota" in (
        str(error.get("code")),
        str(error.get("type")),
    )


class OpenAICompatibleServerProvider(OpenAICompatProvider):
    """The client with one preset's request details."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        preset: Preset,
        model: str,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
        price_input: float = 0.0,
        price_output: float = 0.0,
        settings: CoreSettings | None = None,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(
            base_url,
            [api_key] if api_key else [],
            default_model=model,
            settings=settings,
            client=client,
            name="openai_compatible",
            clock=clock,
            price_input=price_input,
            price_output=price_output,
        )
        self.preset = preset
        self.error_label = preset.label
        self.OPTIONAL_FIELDS = {
            **OpenAICompatProvider.OPTIONAL_FIELDS,
            **{name: (name,) for name in preset.relaxable},
        }
        self.max_output_tokens = max_output_tokens
        self._rejected: dict[str, set[str]] = {}

    def _extra_payload(self, model: str, config) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.preset.response_format is not None:
            extra["response_format"] = dict(self.preset.response_format)
        if self.max_output_tokens:
            extra[self.preset.output_limit] = self.max_output_tokens
        return extra

    def _payload(self, batch, model, temperature, config) -> dict[str, Any]:
        payload = super()._payload(batch, model, temperature, config)
        for name in self._rejected.get(model, ()):
            payload.pop(name, None)
        return payload

    def _note_rejected(self, model: str, field: str) -> None:
        # Remembered per model, so a server that refuses a field costs one extra request per
        # model, not one per batch.
        self._rejected.setdefault(model, set()).add(field)

    def classify(self, response: httpx.Response, model: str) -> TranslationProviderError | None:
        code = response.status_code
        if code == 403:
            return AuthenticationError(
                f"{self.error_label} refused the request (403)", provider=self._name
            )
        if code == 429 and _insufficient_quota(response):
            return QuotaExhaustedError(
                f"{self.error_label} says the account has no quota left",
                provider=self._name,
                status_code=429,
            )
        return None


class OpenAICompatibleBackendType(BackendType):
    type_id = "openai_compatible"
    display_name = "OpenAI-compatible server"
    # No fixed host: the address is a destination the user sets, not an endpoint of the package.
    endpoints = ()

    def __init__(
        self, transport: httpx.AsyncBaseTransport | None = None, clock: Callable[[], float] = time.time
    ):
        self._transport = transport
        self._clock = clock

    def fields(self) -> list[SettingField]:
        return [
            SettingField(
                "preset",
                FieldType.CHOICE,
                "Server",
                default=DEFAULT_PRESET,
                choices=tuple(
                    Choice(
                        p.id,
                        p.label if p.id != "custom" else "Custom",
                        fills={"base_url": p.base_url, "parallel_batches": p.parallel_batches},
                    )
                    for p in PRESETS.values()
                ),
                help="Fills in the address and the request details that server expects.",
            ),
            SettingField(
                "base_url",
                FieldType.TEXT,
                "Server address",
                required=True,
                destination=True,
                help="Inside a container, localhost is the container itself: for a server on your "
                "machine use its LAN address or host.docker.internal.",
            ),
            SettingField(
                "api_key",
                FieldType.SECRET,
                "API key",
                help="Only if the server asks for one. LM Studio and Ollama need none.",
            ),
            SettingField(
                "model",
                FieldType.LIVE_CHOICE,
                "Model",
                required=True,
                help="Pick one the server has loaded, or type its id.",
            ),
            SettingField(
                "temperature", FieldType.NUMBER, "Temperature", default=0.3, minimum=0, maximum=2
            ),
            SettingField(
                "batch_size",
                FieldType.INTEGER,
                "Lines per request",
                default=DEFAULT_BATCH_SIZE,
                minimum=10,
                maximum=500,
                help="Local models often have a small context window: lower this if replies come back cut short.",
            ),
            SettingField(
                "parallel_batches",
                FieldType.INTEGER,
                "Requests in parallel",
                default=1,
                minimum=1,
                maximum=10,
                help="A local server usually answers one request at a time.",
            ),
            SettingField(
                "max_tokens",
                FieldType.INTEGER,
                "Output budget per request",
                default=DEFAULT_MAX_TOKENS,
                minimum=0,
                maximum=200000,
                help="0 sends no budget.",
            ),
            SettingField(
                "price_input",
                FieldType.NUMBER,
                "Input price (USD per 1M tokens)",
                default=0.0,
                minimum=0,
                maximum=100000,
                help="0 for a free or local server. A set price makes the budget hold each request's worst case.",
            ),
            SettingField(
                "price_output",
                FieldType.NUMBER,
                "Output price (USD per 1M tokens)",
                default=0.0,
                minimum=0,
                maximum=100000,
            ),
        ]

    def validate(self, settings: Mapping[str, Any], secrets: Mapping[str, Any]) -> Validation:
        errors: dict[str, str] = {}
        preset = PRESETS.get(str(settings.get("preset") or DEFAULT_PRESET))
        if preset is None:
            errors["preset"] = f"Choose one of {', '.join(PRESETS)}"
            preset = PRESETS[DEFAULT_PRESET]
        base_url, problem = _base_url(settings.get("base_url") or preset.base_url)
        if problem:
            errors["base_url"] = problem
        model = str(settings.get("model") or "").strip()
        if not model:
            errors["model"] = "Pick or type a model"
        patch: dict[str, Any] = {"preset": preset.id, "base_url": base_url, "model": model}
        try:
            temperature = float(settings.get("temperature", 0.3))
            if not 0 <= temperature <= 2:
                raise ValueError
            patch["temperature"] = temperature
        except (TypeError, ValueError):
            errors["temperature"] = "Temperature is a number from 0 to 2"
        for key, label, default, low, high in (
            ("batch_size", "Lines per request", DEFAULT_BATCH_SIZE, 10, 500),
            ("parallel_batches", "Requests in parallel", preset.parallel_batches, 1, 10),
            ("max_tokens", "Output budget", DEFAULT_MAX_TOKENS, 0, 200000),
        ):
            value = settings.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                errors[key] = f"{label} is a whole number from {low} to {high}"
            else:
                patch[key] = value
        for key, label in (("price_input", "Input price"), ("price_output", "Output price")):
            value = settings.get(key, 0.0)
            try:
                price = float(value)
                if isinstance(value, bool) or not 0 <= price <= 100000:
                    raise ValueError
                patch[key] = price
            except (TypeError, ValueError):
                errors[key] = f"{label} is a number from 0 to 100000"
        if preset.id == "openai" and not (patch.get("price_input") or patch.get("price_output")):
            errors["price_input"] = "The OpenAI preset needs its prices so the budget can hold each call"
        return Validation(patch=patch, errors=errors)

    def _client(self, base: CoreSettings, base_url: str) -> httpx.AsyncClient | None:
        if self._transport is None:
            return None
        return httpx.AsyncClient(
            transport=self._transport,
            base_url=base_url + "/",
            timeout=httpx.Timeout(base.request_timeout),
        )

    def _provider(
        self, settings: Mapping[str, Any], api_key: str | None, base: CoreSettings
    ) -> OpenAICompatibleServerProvider:
        return OpenAICompatibleServerProvider(
            settings["base_url"],
            api_key,
            preset=PRESETS[settings["preset"]],
            model=settings.get("model", ""),
            max_output_tokens=settings.get("max_tokens", DEFAULT_MAX_TOKENS),
            price_input=float(settings.get("price_input", 0.0)),
            price_output=float(settings.get("price_output", 0.0)),
            settings=base,
            client=self._client(base, settings["base_url"]),
            clock=self._clock,
        )

    def open(
        self, connection: Connection, base: CoreSettings, context: BackendContext
    ) -> OpenedBackend:
        checked = self.validate(connection.settings, connection.secrets)
        if not checked.ok:
            raise InvalidConfigError(
                "The OpenAI-compatible connection is not complete",
                provider=self.type_id,
                field_errors=checked.errors,
            )
        settings = checked.patch
        timeout = request_timeout_of(connection.settings)
        if timeout is not None:
            # The request's own timeout bounds the HTTP call and the batch alike.
            base = base.model_copy(update={"request_timeout": timeout})
        api_key = str(connection.secrets.get("api_key") or "").strip() or None
        config = TranslationConfig(
            model=settings["model"],
            temperature=settings["temperature"],
            parallelBatches=settings["parallel_batches"],
            requestTimeout=timeout,
        )
        return OpenedBackend(
            self._provider(settings, api_key, base),
            base,
            config,
            settings["model"],
            settings["batch_size"],
        )

    async def options(self, field, settings, secrets, context) -> list[Choice]:
        """The models the server lists. None when it lists nothing, cannot be reached or wants a
        key it was not given; the form then takes the model as free text."""
        if field != "model":
            return []
        preset = PRESETS.get(str(settings.get("preset") or DEFAULT_PRESET), PRESETS[DEFAULT_PRESET])
        base_url, problem = _base_url(settings.get("base_url") or preset.base_url)
        if problem:
            return []
        api_key = str(secrets.get("api_key") or "").strip() or None
        base = CoreSettings(request_timeout=15.0)
        provider = self._provider(
            {"preset": preset.id, "base_url": base_url}, api_key, base
        )
        try:
            models = await provider.get_available_models()
        except (TranslationProviderError, httpx.HTTPError, ValueError):
            return []
        finally:
            await provider.close()
        return [Choice(item["id"], item["name"]) for item in models]
