"""Google Gemini through its OpenAI-compatible endpoint.

Fixes over the engine Bazarr+ 2.7 carried: the key travels in the Authorization header, never
the URL; the retry time comes from the RetryInfo in the response body; a per-day quota waits
for midnight Pacific; keys of one project share its limits; one batch goes out at a time;
thinking is never "turned off" on models that cannot turn it off; retired models are mapped.
"""

import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as clock_time
from typing import Any
from zoneinfo import ZoneInfo

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
    TestOutcome,
    Validation,
)
from subtitle_translator.core.errors import (
    AuthenticationError,
    DependencyUnavailableError,
    InvalidConfigError,
    QuotaExhaustedError,
    RateLimitError,
    TranslationProviderError,
)
from subtitle_translator.core.models import TranslationConfig
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.usage import UsageMeter

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
GEMINI_DEFAULT_BATCH_SIZE = 300
PACIFIC = ZoneInfo("America/Los_Angeles")

# Google's deprecations page, read 2026-09-24: shut-down model -> listed replacement.
RETIRED_GEMINI_MODELS = {
    "gemini-2.0-flash": GEMINI_DEFAULT_MODEL,
    "gemini-2.0-flash-001": GEMINI_DEFAULT_MODEL,
    "gemini-2.0-flash-lite": "gemini-3.1-flash-lite",
    "gemini-2.0-flash-lite-001": "gemini-3.1-flash-lite",
    "gemini-2.5-flash-preview-05-20": GEMINI_DEFAULT_MODEL,
    "gemini-2.5-flash-preview-09-2025": GEMINI_DEFAULT_MODEL,
    "gemini-2.5-flash-lite-preview-09-2025": "gemini-3.1-flash-lite",
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
}
REASONING_CHOICES = ("disabled", "low", "medium", "high")
_THINKING_ALWAYS_ON = re.compile(r"^gemini-(3|2\.5-pro)")
_NOT_FOR_TEXT = ("embedding", "tts", "image", "live", "aqa", "veo", "imagen", "lyria")

# Published output-token ceilings, so a priced request can be held at its worst case. A model
# this map does not know has no known bound, and a priced request to it is refused only when
# the attempt carries a budget.
GEMINI_OUTPUT_LIMITS = {
    "gemini-2.0-flash": 8192,
    "gemini-2.5-flash": 65536,
    "gemini-2.5-pro": 65536,
    "gemini-3.6-flash": 65536,
}


def next_pacific_midnight(now: float) -> float:
    local = datetime.fromtimestamp(now, PACIFIC)
    return datetime.combine(
        (local + timedelta(days=1)).date(), clock_time(0, 0), tzinfo=PACIFIC
    ).timestamp()


def resolve_model(model: str) -> str:
    bare = (model or "").strip().removeprefix("models/")
    return RETIRED_GEMINI_MODELS.get(bare, bare)


def reasoning_effort(model: str, setting: str) -> str | None:
    if setting in ("low", "medium", "high"):
        return setting
    return "low" if _THINKING_ALWAYS_ON.match(model.removeprefix("models/")) else "none"


@dataclass(frozen=True)
class GoogleError:
    status: str | None
    message: str
    retry_delay: float | None
    quota_ids: tuple[str, ...]
    reason: str | None


def _seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value.endswith("s"):
        return None
    try:
        seconds = float(value[:-1])
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def parse_google_error(body: Any) -> GoogleError | None:
    """Google's error envelope; the OpenAI-compatible surface sometimes wraps it in a list."""
    if isinstance(body, list):
        body = body[0] if body else None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None
    delay, quota_ids, reason = None, [], None
    for detail in error.get("details") if isinstance(error.get("details"), list) else []:
        if not isinstance(detail, dict):
            continue
        kind = str(detail.get("@type", ""))
        if kind.endswith("google.rpc.RetryInfo"):
            delay = _seconds(detail.get("retryDelay"))
        elif kind.endswith("google.rpc.QuotaFailure"):
            for violation in detail.get("violations") or []:
                if isinstance(violation, dict) and isinstance(violation.get("quotaId"), str):
                    quota_ids.append(violation["quotaId"])
        elif kind.endswith("google.rpc.ErrorInfo") and isinstance(detail.get("reason"), str):
            reason = detail["reason"]
    status = error.get("status")
    return GoogleError(
        status if isinstance(status, str) else None,
        str(error.get("message") or "")[:500],
        delay,
        tuple(quota_ids),
        reason,
    )


class GeminiProvider(OpenAICompatProvider):
    error_label = "Gemini"
    OPTIONAL_FIELDS = {
        **OpenAICompatProvider.OPTIONAL_FIELDS,
        "reasoning_effort": ("reasoning", "thinking"),
    }

    def __init__(
        self,
        api_keys,
        *,
        key_projects=(),
        model: str,
        reasoning: str = "disabled",
        price_input: float = 0.0,
        price_output: float = 0.0,
        settings: CoreSettings | None = None,
        client: httpx.AsyncClient | None = None,
        base_url: str = GEMINI_BASE_URL,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(
            base_url,
            api_keys,
            key_groups=key_projects,
            default_model=model,
            settings=settings,
            client=client,
            name="gemini",
            clock=clock,
            price_input=price_input,
            price_output=price_output,
        )
        self.reasoning = reasoning

    def _output_ceiling(self, model_used: str) -> int | None:
        return GEMINI_OUTPUT_LIMITS.get(model_used.removeprefix("models/"))

    def _extra_payload(self, model, config):
        effort = reasoning_effort(model, self.reasoning)
        return {"reasoning_effort": effort} if effort else {}

    def classify(self, response: httpx.Response, model: str) -> TranslationProviderError | None:
        code = response.status_code
        if code < 400:
            return None
        try:
            parsed = parse_google_error(response.json())
        except ValueError:
            parsed = None
        message = parsed.message if parsed and parsed.message else f"HTTP {code}"
        key_rejected = parsed is not None and (
            parsed.reason in ("API_KEY_INVALID", "API_KEY_SERVICE_BLOCKED")
            or "api key not valid" in message.lower()
        )
        if code in (401, 403) or key_rejected:
            return AuthenticationError(f"Gemini rejected the key: {message}", provider="gemini")
        if code == 429:
            daily = parsed is not None and any(
                "perday" in q.lower() or "daily" in q.lower() for q in parsed.quota_ids
            )
            if daily:
                now = self._clock()
                return QuotaExhaustedError(
                    f"Gemini's daily quota for {model} is used up until midnight Pacific",
                    provider="gemini",
                    retry_after=next_pacific_midnight(now) - now,
                    status_code=429,
                )
            delay = parsed.retry_delay if parsed and parsed.retry_delay is not None else 60.0
            return RateLimitError(
                f"Gemini rate limit: {message}", provider="gemini", retry_after=delay
            )
        if code == 404:
            return InvalidConfigError(
                f"Gemini has no model {model!r}",
                provider="gemini",
                field_errors={"model": "not found"},
                status_code=404,
            )
        if code >= 500:
            return DependencyUnavailableError(
                f"Gemini server error {code}", provider="gemini", status_code=code
            )
        return None


def _labels(settings: Mapping[str, Any]) -> list[str]:
    value = settings.get("key_projects") or []
    return [str(v).strip() for v in value] if isinstance(value, (list, tuple)) else []


def _keys(secrets: Mapping[str, Any]) -> list[str]:
    value = secrets.get("api_keys") or []
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in value if str(v).strip()]


class GeminiBackendType(BackendType):
    type_id = "gemini"
    display_name = "Google Gemini"
    endpoints = ("generativelanguage.googleapis.com",)

    def __init__(
        self, transport: httpx.AsyncBaseTransport | None = None, clock: Callable[[], float] = time.time
    ):
        self._transport = transport
        self._clock = clock

    def fields(self) -> list[SettingField]:
        return [
            SettingField(
                "api_keys",
                FieldType.KEY_LIST,
                "API keys",
                required=True,
                help="Several keys rotate when one is limited. Keys from one Google Cloud project share its limits.",
                help_url="https://aistudio.google.com/apikey",
            ),
            SettingField(
                "key_projects",
                FieldType.LIST,
                "Project of each key",
                help="Optional. Give keys from the same project the same name, in the order of the keys.",
            ),
            SettingField("model", FieldType.LIVE_CHOICE, "Model", default=GEMINI_DEFAULT_MODEL),
            SettingField(
                "reasoning",
                FieldType.CHOICE,
                "Thinking",
                default="disabled",
                choices=tuple(Choice(v, v.capitalize()) for v in REASONING_CHOICES),
                help="Gemini 3 and 2.5 Pro always think; Disabled uses their lowest level.",
            ),
            SettingField(
                "temperature", FieldType.NUMBER, "Temperature", default=0.3, minimum=0, maximum=2
            ),
            SettingField(
                "batch_size",
                FieldType.INTEGER,
                "Lines per request",
                default=GEMINI_DEFAULT_BATCH_SIZE,
                minimum=10,
                maximum=500,
            ),
            SettingField(
                "price_input",
                FieldType.NUMBER,
                "Input price (USD per 1M tokens)",
                default=0.0,
                minimum=0,
                maximum=100000,
                help="0 for the free tier. A set price makes the budget hold each request's worst case.",
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

    def validate(self, settings, secrets) -> Validation:
        errors = {}
        if not _keys(secrets):
            errors["api_keys"] = "API keys is required"
        patch: dict[str, Any] = {
            "model": resolve_model(str(settings.get("model") or GEMINI_DEFAULT_MODEL))
        }
        reasoning = settings.get("reasoning") or "disabled"
        if reasoning not in REASONING_CHOICES:
            errors["reasoning"] = f"Choose one of {', '.join(REASONING_CHOICES)}"
        patch["reasoning"] = reasoning
        try:
            temperature = float(settings.get("temperature", 0.3))
            if not 0 <= temperature <= 2:
                raise ValueError
            patch["temperature"] = temperature
        except (TypeError, ValueError):
            errors["temperature"] = "Temperature is a number from 0 to 2"
        size = settings.get("batch_size", GEMINI_DEFAULT_BATCH_SIZE)
        if isinstance(size, bool) or not isinstance(size, int) or not 10 <= size <= 500:
            errors["batch_size"] = "Lines per request is a whole number from 10 to 500"
        else:
            patch["batch_size"] = size
        for key, label in (("price_input", "Input price"), ("price_output", "Output price")):
            value = settings.get(key, 0.0)
            try:
                price = float(value)
                if isinstance(value, bool) or not 0 <= price <= 100000:
                    raise ValueError
                patch[key] = price
            except (TypeError, ValueError):
                errors[key] = f"{label} is a number from 0 to 100000"
        labels = _labels(settings)
        # Bazarr+ validates the settings without the keys, which are secrets, so the names can
        # only be counted against keys that were given; test() counts them when it has the keys.
        keys = _keys(secrets)
        if keys and len(labels) > len(keys):
            errors["key_projects"] = "There are more project names than keys"
        patch["key_projects"] = labels
        return Validation(patch=patch, errors=errors)

    def _client(self, base: CoreSettings) -> httpx.AsyncClient | None:
        if self._transport is None:
            return None
        return httpx.AsyncClient(
            transport=self._transport,
            base_url=GEMINI_BASE_URL + "/",
            timeout=httpx.Timeout(base.request_timeout),
        )

    def open(
        self, connection: Connection, base: CoreSettings, context: BackendContext
    ) -> OpenedBackend:
        settings = connection.settings
        prices: dict[str, float] = {}
        for key, label in (("price_input", "Input price"), ("price_output", "Output price")):
            value = settings.get(key, 0.0)
            try:
                price = float(value)
                if isinstance(value, bool) or not 0 <= price <= 100000:
                    raise ValueError
                prices[key] = price
            except (TypeError, ValueError):
                raise InvalidConfigError(
                    "The Gemini connection is not complete",
                    provider=self.type_id,
                    field_errors={key: f"{label} is a number from 0 to 100000"},
                ) from None
        model = resolve_model(str(settings.get("model") or GEMINI_DEFAULT_MODEL))
        provider = GeminiProvider(
            _keys(connection.secrets),
            key_projects=_labels(settings),
            model=model,
            reasoning=str(settings.get("reasoning") or "disabled"),
            price_input=prices["price_input"],
            price_output=prices["price_output"],
            settings=base,
            client=self._client(base),
            clock=self._clock,
        )
        config = TranslationConfig(
            model=model,
            temperature=float(settings.get("temperature", 0.3)),
            parallelBatches=1,
        )
        size = settings.get("batch_size", GEMINI_DEFAULT_BATCH_SIZE)
        return OpenedBackend(
            provider,
            base,
            config,
            model,
            size if isinstance(size, int) else GEMINI_DEFAULT_BATCH_SIZE,
        )

    async def test(
        self,
        connection: Connection,
        base: CoreSettings,
        context: BackendContext,
        meter: UsageMeter | None = None,
    ) -> TestOutcome:
        """The one place with both the keys and their project names, so it counts them before
        the call. open() does not refuse: the key ring leaves names past the last key unused."""
        keys = _keys(connection.secrets)
        if keys and len(_labels(connection.settings)) > len(keys):
            raise InvalidConfigError(
                "There are more project names than keys",
                provider=self.type_id,
                field_errors={"key_projects": "There are more project names than keys"},
            )
        return await super().test(connection, base, context, meter)

    async def options(self, field, settings, secrets, context) -> list[Choice]:
        if field != "model":
            return []
        keys = _keys(secrets)
        if not keys:
            raise AuthenticationError(
                "Add a Gemini API key to list the models", provider="gemini"
            )
        provider = GeminiProvider(
            [keys[0]],
            model=GEMINI_DEFAULT_MODEL,
            settings=CoreSettings(),
            client=self._client(CoreSettings()),
            clock=self._clock,
        )
        try:
            models = await provider.get_available_models()
        finally:
            await provider.close()
        choices = []
        for item in models:
            model_id = item["id"].removeprefix("models/")
            if model_id.startswith("gemini-") and not any(
                word in model_id for word in _NOT_FOR_TEXT
            ):
                choices.append(Choice(model_id, item.get("name", model_id).removeprefix("models/")))
        return choices
