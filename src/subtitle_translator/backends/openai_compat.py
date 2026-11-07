"""Any endpoint that speaks OpenAI chat completions. The Gemini back end (on Gemini's OpenAI
surface) and the OpenAI-compatible back end for OpenAI, LM Studio, Ollama and any other server
are both built on it; neither one's request details reach the other."""

import logging
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Optional

import httpx

from subtitle_translator.backends.chat_completions import ChatCompletionsProvider
from subtitle_translator.backends.keyring import KeyRing, KeyState
from subtitle_translator.core.errors import (
    AuthenticationError,
    DependencyUnavailableError,
    InvalidConfigError,
    ProviderTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
    TranslationProviderError,
)
from subtitle_translator.core.provider import TranslationBatch, TranslationResult
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.usage import release_request, reserve_request

if TYPE_CHECKING:
    from subtitle_translator.core.models import TranslationConfig

logger = logging.getLogger(__name__)


class OpenAICompatProvider(ChatCompletionsProvider):
    error_label = "Endpoint"
    debug_logger = logging.getLogger(f"{__name__}.debug")
    OPTIONAL_FIELDS: dict[str, tuple[str, ...]] = {"response_format": ("response_format", "json")}

    def __init__(
        self,
        base_url: str,
        api_keys: Sequence[str] = (),
        *,
        key_groups: Sequence[str | None] = (),
        default_model: str | None = None,
        settings: CoreSettings | None = None,
        key_cooldown_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
        name: str = "openai_compat",
        clock: Callable[[], float] = time.time,
        price_input: float = 0.0,
        price_output: float = 0.0,
    ) -> None:
        self.settings = settings or CoreSettings()
        self.base_url = base_url.rstrip("/")
        self.keys = KeyRing(api_keys, key_groups, clock=clock)
        self.default_model = default_model
        self._cooldown = key_cooldown_seconds
        self._client = client
        self._name = name
        self._clock = clock
        self.price_input = price_input
        self.price_output = price_output
        self.prices_requests = price_input > 0 or price_output > 0

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url + "/",
                timeout=httpx.Timeout(self.settings.request_timeout),
                follow_redirects=False,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _model(self, model: str | None, config: Optional["TranslationConfig"]) -> str:
        chosen = (config.model if config and config.model else None) or model or self.default_model
        if not chosen:
            raise InvalidConfigError(
                "No model is configured", provider=self._name, field_errors={"model": "required"}
            )
        return chosen

    def _extra_payload(self, model: str, config: Optional["TranslationConfig"]) -> dict[str, Any]:
        return {}

    def classify(self, response: httpx.Response, model: str) -> TranslationProviderError | None:
        return None

    def _note_rejected(self, model: str, field: str) -> None:
        """A server refused an optional field for this model. The client drops it for the rest of
        this request only; a subclass may remember it for later requests of the model."""

    def _payload(self, batch, model, temperature, config) -> dict[str, Any]:
        system = self.build_system_prompt(
            batch.target_language,
            batch.source_language,
            batch.context_title,
            batch.context_media_type,
            source=batch.source,
            target=batch.target,
            media=batch.media,
        )
        temp = (
            config.temperature
            if config is not None and config.temperature is not None
            else temperature
            if temperature is not None
            else self.settings.openrouter_temperature
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self.format_input_for_translation(batch.lines)},
            ],
            "temperature": temp,
            "response_format": {"type": "json_object"},
        }
        payload.update(self._extra_payload(model, config))
        return payload

    def _output_limit(self, model: str, payload: dict[str, Any]) -> int | None:
        """The request's own output limit, or the model's ceiling when it carries none."""
        for key in ("max_tokens", "max_completion_tokens"):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return self._output_ceiling(model)

    def _worst_case(
        self, model: str, payload: dict[str, Any], config: Optional["TranslationConfig"]
    ) -> float:
        """What this request may cost at most, from the configured prices.

        An output price with no output limit leaves the bound unknown. A request with a budget
        is refused rather than sent unbounded; without a budget it goes out as before.
        """
        limit = self._output_limit(model, payload)
        meter = getattr(config, "_usage_meter", None)
        if limit is None and self.price_output > 0 and meter is not None and meter.budget is not None:
            raise InvalidConfigError(
                "This connection prices output tokens but the request carries no output limit",
                provider=self._name,
                field_errors={
                    "max_tokens": "Set an output budget, or pick a model with a known output ceiling"
                },
            )
        messages = payload.get("messages") or []
        input_tokens = sum(
            len(str(message.get("content", "")).encode("utf-8"))
            for message in messages
            if isinstance(message, dict)
        )
        worst_case = (input_tokens + 16) * self.price_input / 1e6
        if limit is not None:
            worst_case += limit * self.price_output / 1e6
        return worst_case

    def _priced_cost(
        self, prompt: int | None, completion: int | None, total: int | None
    ) -> float | None:
        """The call's cost from the configured prices, never from the server's own figure."""
        if (
            isinstance(prompt, int)
            and not isinstance(prompt, bool)
            and isinstance(completion, int)
            and not isinstance(completion, bool)
        ):
            return prompt * self.price_input / 1e6 + completion * self.price_output / 1e6
        if isinstance(total, int) and not isinstance(total, bool) and total > 0:
            return total * max(self.price_input, self.price_output) / 1e6
        return None

    def _price_error(self, error: TranslationProviderError) -> None:
        if self.prices_requests and error.tokens_used:
            error.cost = float(error.tokens_used) * max(self.price_input, self.price_output) / 1e6

    def _no_key_left(self, model: str) -> TranslationProviderError:
        if self.keys.all_invalid():
            return AuthenticationError(
                f"Every {self.error_label} key was rejected", provider=self._name
            )
        wait = self.keys.wait(model)
        if wait is not None and wait[0] is KeyState.EXHAUSTED:
            return QuotaExhaustedError(
                f"Every {self.error_label} key has used its quota for {model}",
                provider=self._name,
                retry_after=max(0.0, wait[1]),
            )
        return RateLimitError(
            f"Every {self.error_label} key is rate limited for {model}",
            provider=self._name,
            retry_after=max(0.0, wait[1]) if wait is not None else self._cooldown,
        )

    @staticmethod
    async def _release_unbilled(config, reservation, error: TranslationProviderError) -> None:
        """A rejection that reports no usage cost nothing, so the next attempt may reuse its hold."""
        if not (error.tokens_used or error.cost or error.billing_unknown):
            await release_request(config, reservation)

    async def translate_batch(
        self,
        batch: TranslationBatch,
        model: str | None = None,
        temperature: float | None = None,
        config_override: Optional["TranslationConfig"] = None,
    ) -> TranslationResult:
        model_id = self._model(model, config_override)
        payload = self._payload(batch, model_id, temperature, config_override)
        pinned_key = (
            config_override.api_key if config_override and config_override.api_key else None
        )
        dropped: set[str] = set()
        for _ in range(len(self.keys) + len(self.OPTIONAL_FIELDS) + 1):
            key = pinned_key or (self.keys.acquire(model_id) if len(self.keys) else None)
            if key is None and len(self.keys):
                raise self._no_key_left(model_id)
            reservation = None
            if self.prices_requests:
                reservation = await reserve_request(
                    config_override,
                    self._worst_case(model_id, payload, config_override),
                    self._name,
                    model_id,
                )
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            try:
                response = await self.client.post(
                    "chat/completions", json=payload, headers=headers
                )
            except httpx.TimeoutException:
                raise ProviderTimeoutError(
                    f"{self.error_label} did not answer in time", provider=self._name
                ) from None
            except httpx.RequestError as error:
                # Only a failed connect proves the request never left; after it, the server may
                # have answered and billed it.
                raise DependencyUnavailableError(
                    f"{self.error_label} could not be reached ({type(error).__name__})",
                    provider=self._name,
                    billing_unknown=not isinstance(error, httpx.ConnectError),
                ) from None
            if response.status_code == 400:
                text = response.text.lower()
                relaxed = next(
                    (
                        field
                        for field, words in self.OPTIONAL_FIELDS.items()
                        if field in payload and field not in dropped and any(w in text for w in words)
                    ),
                    None,
                )
                if relaxed is not None:
                    # The server turned the request down before doing any work.
                    await release_request(config_override, reservation)
                    payload.pop(relaxed)
                    dropped.add(relaxed)
                    self._note_rejected(model_id, relaxed)
                    logger.info(f"{self.error_label} rejected {relaxed}; retrying without it")
                    continue
            try:
                special = self.classify(response, model_id)
                if special is not None:
                    raise special
                result = await self._process_response(response, model_id)
            except AuthenticationError as error:
                self._price_error(error)
                if pinned_key or not key:
                    raise
                await self._release_unbilled(config_override, reservation, error)
                self.keys.mark_invalid(key)
                continue
            except QuotaExhaustedError as error:
                self._price_error(error)
                if pinned_key or not key:
                    raise
                await self._release_unbilled(config_override, reservation, error)
                self.keys.mark_exhausted(
                    key, self._clock() + (error.retry_after or 3600.0), model_id
                )
                continue
            except RateLimitError as error:
                self._price_error(error)
                if pinned_key or not key:
                    raise
                await self._release_unbilled(config_override, reservation, error)
                self.keys.mark_rate_limited(key, error.retry_after or self._cooldown, model_id)
                continue
            if key and not pinned_key:
                # An override key is not in the ring, and the ring never sees it.
                self.keys.mark_success(key, model_id)
            if self.prices_requests:
                cost = self._priced_cost(
                    result.prompt_tokens, result.completion_tokens, result.total_tokens
                )
                if cost is None:
                    # Nothing says what the call used, so its hold stays: it may have been billed.
                    result.billing_unknown = True
                else:
                    result.cost = cost
            self._validate_and_warn_unchanged(batch.lines, result.translations)
            return result
        raise self._no_key_left(model_id)

    async def health_check(self) -> bool:
        try:
            return (await self._get_models()).status_code // 100 == 2
        except httpx.HTTPError:
            return False

    async def _get_models(self) -> httpx.Response:
        key = self.keys.acquire("models") if len(self.keys) else None
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return await self.client.get("models", headers=headers)

    async def get_available_models(self) -> list[dict]:
        response = await self._get_models()
        special = self.classify(response, "")
        if special is not None:
            raise special
        data = response.json().get("data") if response.status_code == 200 else None
        models = []
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append({"id": item["id"], "name": item.get("name") or item["id"]})
        return models
