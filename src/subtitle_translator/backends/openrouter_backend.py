"""OpenRouter as a back-end type: connection settings, validation, live model choices, tests."""

from collections.abc import Callable, Mapping
from typing import Any

import httpx

from subtitle_translator.backends.openrouter import (
    ROUTING_SUFFIXES,
    OpenRouterProvider,
    split_routing_suffix,
)
from subtitle_translator.core.backends import (
    BackendContext,
    BackendType,
    Choice,
    Connection,
    FieldType,
    OpenedBackend,
    SettingField,
    Validation,
)
from subtitle_translator.core.errors import InvalidConfigError
from subtitle_translator.core.models import (
    PROVIDER_SORT_VALUES,
    ProviderConfig,
    ReasoningConfig,
    SmartFastPolicy,
    TranslationConfig,
)
from subtitle_translator.core.settings import CoreSettings

OPENROUTER_DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
# Carried over from Bazarr+ (bazarr/app/config.py RETIRED_OPENROUTER_MODELS), plus the retired
# preview that the translator's own .env.example still names.
RETIRED_OPENROUTER_MODELS = {
    "google/gemini-2.5-flash-preview-05-20": "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-preview-09-2025": "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite-preview-06-17": "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-lite-preview-09-2025": "google/gemini-2.5-flash-lite",
    "google/gemini-3-pro-preview": "google/gemini-3.1-pro-preview",
}
REASONING_CHOICES = ("disabled", "minimal", "low", "medium", "high", "xhigh")
_PROVIDER_KEYS = (
    "routing",
    "provider_order",
    "allow_fallbacks",
    "only",
    "ignore",
    "smart_fast",
    "deny_data_collection",
    "data_collection",
)
SERVICE_TIERS = ("default", "flex", "priority")


def resolve_model(model: str) -> str:
    slug, suffix = split_routing_suffix((model or "").strip())
    slug = RETIRED_OPENROUTER_MODELS.get(slug, slug)
    return f"{slug}:{suffix}" if suffix else slug


def _reasoning(value: Any) -> ReasoningConfig | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return ReasoningConfig(**value)
    return ReasoningConfig(effort="none" if value == "disabled" else str(value))


def translation_config(
    settings: Mapping[str, Any], secrets: Mapping[str, Any]
) -> TranslationConfig:
    routing = settings.get("routing") or None
    policy = settings.get("smart_fast")
    provider_fields = {
        "order": settings.get("provider_order") or None,
        "allowFallbacks": settings.get("allow_fallbacks", True),
        "sort": routing,
        # An empty list is the unset value of a list field, not "no provider allowed".
        "only": settings.get("only") or None,
        "ignore": settings.get("ignore") or None,
        "smartFast": SmartFastPolicy(**policy) if isinstance(policy, dict) else None,
        # A request's explicit choice ("allow" or "deny") wins; the toggle can only deny.
        "dataCollection": settings.get("data_collection")
        or ("deny" if settings.get("deny_data_collection") else None),
    }
    has_provider = any(key in settings for key in _PROVIDER_KEYS)
    return TranslationConfig(
        apiKey=secrets.get("api_key") or None,
        model=resolve_model(settings["model"]) if settings.get("model") else None,
        temperature=settings.get("temperature"),
        requestTimeout=settings.get("request_timeout"),
        reasoning=_reasoning(settings.get("reasoning")),
        useThinkingVariant=settings.get("use_thinking_variant"),
        provider=ProviderConfig(**provider_fields) if has_provider else None,
        serviceTier=settings.get("service_tier"),
        parallelBatches=settings.get("parallel_batches"),
    )


def settings_from_config(config: TranslationConfig) -> dict[str, Any]:
    """The connection settings of a request that brought its own configuration."""
    settings: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "request_timeout": config.request_timeout,
        "reasoning": config.reasoning.model_dump(by_alias=True) if config.reasoning else None,
        "use_thinking_variant": config.use_thinking_variant,
        "service_tier": config.service_tier,
        "parallel_batches": config.parallel_batches,
    }
    if config.provider is not None:
        p = config.provider
        settings.update(
            routing=p.sort,
            provider_order=p.order,
            allow_fallbacks=p.allow_fallbacks,
            only=p.only,
            ignore=p.ignore,
            smart_fast=p.smart_fast.model_dump() if p.smart_fast else None,
            deny_data_collection=True if p.data_collection == "deny" else None,
            data_collection=p.data_collection,
        )
    return settings


class OpenRouterBackendType(BackendType):
    type_id = "openrouter"
    display_name = "OpenRouter"
    endpoints = ("openrouter.ai",)

    def __init__(
        self,
        provider_factory: Callable[[CoreSettings], OpenRouterProvider] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._factory = provider_factory or OpenRouterProvider
        self._transport = transport

    def fields(self) -> list[SettingField]:
        def choice(values):
            return tuple(Choice(v, v) for v in values)

        return [
            SettingField(
                "api_key",
                FieldType.SECRET,
                "API key",
                required=True,
                help_url="https://openrouter.ai/keys",
            ),
            SettingField("model", FieldType.LIVE_CHOICE, "Model", default=OPENROUTER_DEFAULT_MODEL),
            SettingField(
                "temperature", FieldType.NUMBER, "Temperature", default=0.3, minimum=0, maximum=2
            ),
            SettingField(
                "reasoning",
                FieldType.CHOICE,
                "Reasoning",
                default="disabled",
                choices=choice(REASONING_CHOICES),
            ),
            SettingField(
                "routing",
                FieldType.CHOICE,
                "Provider routing",
                default="throughput",
                choices=choice(PROVIDER_SORT_VALUES),
            ),
            SettingField(
                "smart_fast",
                FieldType.TEXT,
                "SmartFast policy",
                group="SmartFast",
                show_when=("routing", "smartfast"),
            ),
            SettingField(
                "provider_order", FieldType.ORDERABLE_LIST, "Provider order", group="Providers"
            ),
            SettingField("only", FieldType.LIST, "Only these providers", group="Providers"),
            SettingField("ignore", FieldType.LIST, "Never these providers", group="Providers"),
            SettingField(
                "allow_fallbacks",
                FieldType.TOGGLE,
                "Allow fallback providers",
                default=True,
                group="Providers",
            ),
            SettingField(
                "deny_data_collection",
                FieldType.TOGGLE,
                "Avoid providers that store or train on your data",
                default=False,
                group="Providers",
                help="Many free models retain data, so this can leave no provider for them.",
            ),
            SettingField(
                "service_tier",
                FieldType.CHOICE,
                "Service tier",
                choices=choice(SERVICE_TIERS),
            ),
            SettingField(
                "parallel_batches",
                FieldType.INTEGER,
                "Requests in parallel",
                default=4,
                minimum=1,
                maximum=10,
            ),
            SettingField(
                "max_tokens",
                FieldType.INTEGER,
                "Output budget per request",
                default=8000,
                minimum=0,
                maximum=200000,
                help="0 sends no budget.",
            ),
        ]

    def validate(self, settings, secrets) -> Validation:
        errors = self.required_errors(settings, secrets)
        patch = dict(settings)
        patch["model"] = resolve_model(str(settings.get("model") or OPENROUTER_DEFAULT_MODEL))
        patch["deny_data_collection"] = bool(settings.get("deny_data_collection", False))
        routing = settings.get("routing") or "throughput"
        if routing not in PROVIDER_SORT_VALUES:
            errors["routing"] = f"Choose one of {', '.join(PROVIDER_SORT_VALUES)}"
        elif routing == "smartfast" and settings.get("provider_order"):
            errors["routing"] = "SmartFast picks its own providers; clear the provider order"
        elif routing == "smartfast" and split_routing_suffix(patch["model"])[1] in ROUTING_SUFFIXES:
            errors["routing"] = "The model already carries a routing shortcut"
        patch["routing"] = routing
        parallel = settings.get("parallel_batches", 4)
        if isinstance(parallel, bool) or not isinstance(parallel, int) or not 1 <= parallel <= 10:
            errors["parallel_batches"] = "Requests in parallel is a whole number from 1 to 10"
        budget = settings.get("max_tokens", 8000)
        if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 200000:
            errors["max_tokens"] = "Output budget is a whole number from 0 to 200000"
        reasoning = settings.get("reasoning", "disabled")
        if not isinstance(reasoning, dict) and reasoning not in REASONING_CHOICES:
            errors["reasoning"] = f"Choose one of {', '.join(REASONING_CHOICES)}"
        if settings.get("service_tier") not in (None, "", *SERVICE_TIERS):
            errors["service_tier"] = f"Choose one of {', '.join(SERVICE_TIERS)}"
        # Bazarr+ validates without the key, so a missing key alone must not skip this check:
        # a policy that fails here would fail every job at open().
        secret_keys = {spec.key for spec in self.fields() if spec.secret}
        if not set(errors) - secret_keys:
            try:
                translation_config(patch, secrets)
            except ValueError as error:
                errors["smart_fast"] = str(error).splitlines()[0]
        return Validation(patch=patch, errors=errors)

    def open(
        self, connection: Connection, base: CoreSettings, context: BackendContext
    ) -> OpenedBackend:
        settings = dict(connection.settings)
        settings.setdefault("model", OPENROUTER_DEFAULT_MODEL)
        try:
            config = translation_config(settings, connection.secrets)
        except ValueError as error:
            # A settings error the engine and the plugin report by field, like any other.
            raise InvalidConfigError(
                "The OpenRouter connection settings are not valid",
                provider=self.type_id,
                field_errors={"smart_fast": str(error).splitlines()[0]},
            ) from None
        budget = settings.get("max_tokens")
        core = base.model_copy(
            update={
                "openrouter_api_key": config.api_key or "",
                "openrouter_max_tokens": budget
                if isinstance(budget, int)
                else base.openrouter_max_tokens,
            }
        )
        return OpenedBackend(
            self._factory(core), core, config, config.model or OPENROUTER_DEFAULT_MODEL, None
        )

    async def options(self, field, settings, secrets, context) -> list[Choice]:
        if field != "model":
            return []
        base = CoreSettings()
        async with httpx.AsyncClient(
            transport=self._transport,
            base_url=base.openrouter_api_base,
            timeout=15.0,
            follow_redirects=False,
        ) as client:
            response = await client.get("/models")
        records = response.json().get("data") if response.status_code == 200 else None
        choices = []
        for model in records if isinstance(records, list) else []:
            if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                continue
            pricing = model.get("pricing") if isinstance(model.get("pricing"), dict) else {}

            def per_million(key, pricing=pricing):
                try:
                    return float(pricing.get(key)) * 1_000_000
                except (TypeError, ValueError):
                    return None

            context_length = model.get("context_length")
            choices.append(
                Choice(
                    model["id"],
                    str(model.get("name") or model["id"]),
                    prompt_price=per_million("prompt"),
                    completion_price=per_million("completion"),
                    context_length=context_length if isinstance(context_length, int) else None,
                )
            )
        return choices
