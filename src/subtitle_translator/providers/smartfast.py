"""Price-bounded endpoint selection for OpenRouter translation requests.

Endpoint catalog prices are USD/token. Policy and outbound ``max_price`` values
are USD/million tokens. This module deliberately has no API-model dependency.
"""

import asyncio
import hashlib
import json
import math
import re
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from statistics import median
from time import monotonic
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

from subtitle_translator.config import Settings
from subtitle_translator.providers.base import TranslationProviderError

CATALOG_TTL = 300
STATE_TTL = 1800
QUARANTINE_SECONDS = 30
MAX_CATALOGS = 128
MAX_SESSIONS = 4096
MAX_OBSERVATIONS = 4096
MAX_ACTIVE = 8192
_PRICE_KEYS = frozenset(
    {
        "prompt",
        "completion",
        "request",
        "input_cache_read",
        "input_cache_write",
        "input_cache_write_1h",
        "internal_reasoning",
        "image",
        "audio",
        "audio_output",
        "web_search",
    }
)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


class SmartFastPolicy(BaseModel):
    """Opt-in routing controls; prices are USD per million tokens."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    medianPremiumPercent: float = Field(default=50, ge=0, le=1000)
    speedTolerancePercent: float = Field(default=20, ge=0, le=1000)
    sparsePremiumMultiplier: float = Field(default=3, ge=1, le=100)
    maxPromptPrice: float = Field(default=1, ge=0, le=1000)
    maxCompletionPrice: float = Field(default=3, ge=0, le=1000)


class SmartFastRoutingError(TranslationProviderError):
    """No safe endpoint selection can be made before the completion call."""

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(
            message,
            provider="openrouter",
            retryable=retry_after is not None,
            retry_after=retry_after,
        )


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _within(value: float, limit: float) -> bool:
    return value <= limit or math.isclose(value, limit, rel_tol=1e-12, abs_tol=0)


def _scope(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _tokens(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number.is_integer() and number <= 100_000_000 else None


def _metric(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    return _number(value.get("p50")) if isinstance(value, Mapping) else None


def _tier(tag: str) -> str:
    suffix = tag.rsplit("/", 1)[-1]
    return (
        "priority" if suffix in {"fast", "priority"} else "flex" if suffix == "flex" else "default"
    )


def _matches(tag: str, restriction: str) -> bool:
    tag = tag.removesuffix("/priority") + ("/fast" if tag.endswith("/priority") else "")
    restriction = restriction.removesuffix("/priority") + (
        "/fast" if restriction.endswith("/priority") else ""
    )
    return tag == restriction or ("/" not in restriction and tag.startswith(restriction + "/"))


@dataclass(frozen=True)
class _PriceOverride:
    prices: tuple[tuple[str, float], ...]
    min_prompt_tokens: float | None
    utc_start: int | None
    utc_end: int | None
    utc_days: tuple[str, ...] | None

    def matches(self, input_tokens: int, now: datetime) -> bool:
        if self.min_prompt_tokens is not None and input_tokens <= self.min_prompt_tokens:
            return False
        if self.utc_days is not None and _WEEKDAYS[now.weekday()] not in self.utc_days:
            return False
        if self.utc_start is None:
            return True
        minute = now.hour * 60 + now.minute
        if self.utc_end > self.utc_start:
            return self.utc_start <= minute < self.utc_end
        return minute >= self.utc_start or minute < self.utc_end


def _parse_overrides(pricing: Mapping[str, Any]) -> tuple[_PriceOverride, ...] | None:
    records = pricing.get("overrides", [])
    if not isinstance(records, list) or len(records) > 64:
        return None
    result = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) - _PRICE_KEYS - {
            "min_prompt_tokens",
            "utc_start",
            "utc_end",
            "utc_days",
        }:
            return None
        minimum = _tokens(record.get("min_prompt_tokens"))
        if "min_prompt_tokens" in record and minimum is None:
            return None
        start, end = record.get("utc_start"), record.get("utc_end")
        if "utc_start" in record or "utc_end" in record:
            if any(
                type(value) is not int or value < 0 or value // 100 >= 24 or value % 100 >= 60
                for value in (start, end)
            ):
                return None
            start, end = start // 100 * 60 + start % 100, end // 100 * 60 + end % 100
        days = record.get("utc_days")
        if "utc_days" in record and (
            not isinstance(days, list) or not days or any(day not in _WEEKDAYS for day in days)
        ):
            return None
        prices = []
        for key in _PRICE_KEYS & record.keys():
            value = _number(record[key])
            if value is None or not math.isfinite(value * 1_000_000):
                return None
            prices.append((key, value if key == "request" else value * 1_000_000))
        result.append(
            _PriceOverride(
                tuple(sorted(prices)),
                minimum,
                start,
                end,
                tuple(days) if days is not None else None,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class Endpoint:
    tag: str
    provider_name: str
    prompt_price: float
    completion_price: float
    request_price: float
    latency: float | None
    throughput: float | None
    supported_parameters: frozenset[str]
    context_length: float | None
    max_prompt_tokens: float | None
    max_completion_tokens: float | None
    cache_read_price: float | None
    cache_write_price: float | None
    reasoning_price: float | None
    overrides: tuple[_PriceOverride, ...]

    def for_request(self, input_tokens: int, now: datetime) -> "Endpoint":
        values = {}
        names = {
            "prompt": "prompt_price",
            "completion": "completion_price",
            "request": "request_price",
            "input_cache_read": "cache_read_price",
            "input_cache_write": "cache_write_price",
            "internal_reasoning": "reasoning_price",
        }
        for override in self.overrides:
            if override.matches(input_tokens, now):
                values.update((names[key], value) for key, value in override.prices if key in names)
        return replace(self, **values) if values else self

    def cold_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            max(self.prompt_price, self.cache_write_price or 0) * input_tokens
            + max(self.completion_price, self.reasoning_price or 0) * output_tokens
        ) / 1_000_000 + self.request_price

    def estimated_seconds(self, output_tokens: int) -> float | None:
        if self.latency is None or self.throughput is None or self.throughput <= 0:
            return None
        estimate = self.latency + output_tokens / self.throughput
        return estimate if math.isfinite(estimate) else None


def _parse_endpoint(record: Mapping[str, Any]) -> Endpoint | None:
    tag = record.get("tag")
    if (
        not isinstance(tag, str)
        or len(tag) > 160
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]*", tag)
    ):
        return None
    if type(record.get("status")) is not int or record["status"] != 0:
        return None
    pricing = record.get("pricing")
    if not isinstance(pricing, Mapping):
        return None
    prompt, completion = (_number(pricing.get(key)) for key in ("prompt", "completion"))
    # Some catalogs omit the optional fee. Admission then relies on the
    # mandatory upstream max_price.request=0, not a claimed metadata guarantee.
    request = _number(pricing.get("request", 0))
    if prompt is None or completion is None or request is None:
        return None
    prompt, completion = prompt * 1_000_000, completion * 1_000_000
    if not math.isfinite(prompt) or not math.isfinite(completion):
        return None
    cache_prices = {}
    for key in ("input_cache_read", "input_cache_write", "internal_reasoning"):
        value = _number(pricing[key]) if key in pricing else None
        if key in pricing and (value is None or not math.isfinite(value * 1_000_000)):
            return None
        cache_prices[key] = value * 1_000_000 if value is not None else None
    overrides = _parse_overrides(pricing)
    if overrides is None:
        return None
    parameters = record.get("supported_parameters")
    latency_ms = _metric(record, "latency_last_30m")
    return Endpoint(
        tag=tag,
        provider_name=record.get("provider_name")
        if isinstance(record.get("provider_name"), str)
        else "",
        prompt_price=prompt,
        completion_price=completion,
        request_price=request,
        # Catalog latency is milliseconds; estimates and observations use seconds.
        latency=latency_ms / 1000 if latency_ms is not None else None,
        throughput=_metric(record, "throughput_last_30m"),
        supported_parameters=frozenset(p for p in parameters if isinstance(p, str))
        if isinstance(parameters, list)
        else frozenset(),
        context_length=_number(record.get("context_length")),
        max_prompt_tokens=_number(record.get("max_prompt_tokens")),
        max_completion_tokens=_number(record.get("max_completion_tokens")),
        cache_read_price=cache_prices["input_cache_read"],
        cache_write_price=cache_prices["input_cache_write"],
        reasoning_price=cache_prices["internal_reasoning"],
        overrides=overrides,
    )


@dataclass(frozen=True)
class _Catalog:
    endpoints: tuple[Endpoint, ...]
    tags: tuple[str, ...]
    excluded: dict[str, str]
    fetched_at: float


@dataclass
class _Session:
    scope: str
    endpoint: str | None
    touched_at: float
    cached_ratio: float | None = None
    degradation: int = 0


@dataclass
class _Observation:
    seconds_per_token: deque[float]
    touched_at: float


@dataclass(frozen=True)
class RoutingDecision:
    provider: dict[str, Any]
    session_id: str
    endpoint: str | None
    diagnostics: dict[str, Any] = field(repr=False)
    identity: str = field(repr=False)
    _scope: str = field(repr=False)
    _session_key: str = field(repr=False)
    _endpoints: tuple[Endpoint, ...] = field(repr=False)
    _output_tokens: int = field(repr=False)
    _speed_limit: float | None = field(repr=False)
    _invalidated_endpoints: set[str] = field(default_factory=set, repr=False)

    def supports_parameter(self, parameter: str) -> bool:
        """Whether every endpoint allowed for this request supports a parameter."""
        return all(parameter in endpoint.supported_parameters for endpoint in self._endpoints)


class SmartFastRouter:
    """Choose a safe provider pool using the caller's authenticated HTTP client."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._catalogs: OrderedDict[str, _Catalog] = OrderedDict()
        self._refreshes: dict[str, asyncio.Task[_Catalog]] = {}
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._observations: OrderedDict[tuple[str, str, bool], _Observation] = OrderedDict()
        self._quarantines: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._active: dict[str, RoutingDecision] = {}
        self._observed: set[str] = set()

    def _prune(self) -> None:
        now = monotonic()
        active_sessions = {decision._session_key for decision in self._active.values()}
        for key, catalog in list(self._catalogs.items()):
            if now - catalog.fetched_at >= CATALOG_TTL:
                del self._catalogs[key]
        for key, session in list(self._sessions.items()):
            if key not in active_sessions and now - session.touched_at >= STATE_TTL:
                del self._sessions[key]
        for key, observation in list(self._observations.items()):
            if now - observation.touched_at >= STATE_TTL:
                del self._observations[key]
        for key, expires_at in list(self._quarantines.items()):
            if now >= expires_at:
                del self._quarantines[key]

    async def _catalog(
        self,
        client: httpx.AsyncClient,
        model: str,
        account: str,
        headers: Mapping[str, str] | None = None,
    ) -> _Catalog:
        key = _scope(account, model)
        cached = self._catalogs.get(key)
        if cached is not None and monotonic() - cached.fetched_at < CATALOG_TTL:
            self._catalogs.move_to_end(key)
            return cached
        task = self._refreshes.get(key)
        if task is None:
            if len(self._refreshes) >= MAX_CATALOGS:
                raise SmartFastRoutingError("SmartFast endpoint refresh capacity is busy")
            task = asyncio.create_task(self._refresh(client, model, key, headers))
            self._refreshes[key] = task
            # A cancelled waiter must neither cancel a shared refresh nor leave
            # an unhandled exception if it was the last waiter.
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(task)

    async def _refresh(
        self,
        client: httpx.AsyncClient,
        model: str,
        key: str,
        headers: Mapping[str, str] | None = None,
    ) -> _Catalog:
        try:
            catalog = await self._fetch(client, model, headers)
            self._catalogs[key] = catalog
            self._catalogs.move_to_end(key)
            while len(self._catalogs) > MAX_CATALOGS:
                self._catalogs.popitem(last=False)
            return catalog
        finally:
            self._refreshes.pop(key, None)

    async def _fetch(
        self, client: httpx.AsyncClient, model: str, headers: Mapping[str, str] | None = None
    ) -> _Catalog:
        try:
            configured_timeout = _number(self.settings.request_timeout)
            timeout = min(15.0, configured_timeout) if configured_timeout else 15.0
            # A shared refresh outlives a cancelled waiter, but never its own
            # deadline, including providers that send continuous keepalive bytes.
            async with asyncio.timeout(timeout):
                response = await client.get(
                    f"/models/{quote(model, safe='/:')}/endpoints", headers=headers
                )
            response.raise_for_status()
            data = response.json()
            records = data.get("data", {}).get("endpoints")
            if not isinstance(records, list) or len(records) > 1024:
                raise ValueError("Missing endpoint list")
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
            raise SmartFastRoutingError("SmartFast endpoint catalog is unavailable") from exc
        unique: dict[str, Endpoint | None] = {}
        excluded: dict[str, str] = {}
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("tag"), str):
                continue
            tag = record["tag"]
            if len(tag) > 160 or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]*", tag):
                continue
            parsed = _parse_endpoint(record)
            # Conflicting duplicates are not independently selectable. Exclude
            # the tag instead of inventing a price or trusting a healthy copy.
            if tag in unique and unique[tag] != parsed:
                unique[tag] = None
                excluded[tag] = "conflicting_duplicate"
            elif tag not in unique:
                unique[tag] = parsed
                if parsed is None:
                    excluded[tag] = "invalid_or_unhealthy"
        return _Catalog(
            tuple(e for e in unique.values() if e is not None), tuple(unique), excluded, monotonic()
        )

    def _estimate(
        self, endpoint: Endpoint, scope: str, output_tokens: int, warm: bool
    ) -> float | None:
        observation = self._observations.get((scope, endpoint.tag, warm))
        if observation is not None and monotonic() - observation.touched_at < STATE_TTL:
            estimate = median(observation.seconds_per_token) * output_tokens
            if math.isfinite(estimate):
                return estimate
        return endpoint.estimated_seconds(output_tokens)

    def _load(self, scope: str, tag: str) -> int:
        return sum(d._scope == scope and tag in d.provider["only"] for d in self._active.values())

    def _admit_session(self, key: str, scope: str, endpoint: str | None) -> _Session:
        session = self._sessions.get(key)
        if session is None:
            active_keys = {decision._session_key for decision in self._active.values()}
            while len(self._sessions) >= MAX_SESSIONS:
                evict = next((k for k in self._sessions if k not in active_keys), None)
                if evict is None:
                    raise SmartFastRoutingError("SmartFast session capacity is busy")
                del self._sessions[evict]
            session = _Session(scope, endpoint, monotonic())
            self._sessions[key] = session
        if session.endpoint != endpoint:
            session.endpoint = endpoint
            session.cached_ratio = None
            session.degradation = 0
        session.touched_at = monotonic()
        self._sessions.move_to_end(key)
        return session

    async def select(
        self,
        client: httpx.AsyncClient,
        *,
        model: str,
        session_id: str,
        account_scope: str,
        input_tokens: int,
        output_tokens: int,
        policy: SmartFastPolicy | None = None,
        only: Sequence[str] | None = None,
        ignore: Sequence[str] | None = None,
        service_tier: str | None = None,
        context_key: str = "",
        required_parameters: Sequence[str] = (),
        headers: Mapping[str, str] | None = None,
    ) -> RoutingDecision:
        """Select before a completion; ``release`` must run in the caller's finally."""
        if any(
            type(value) is not int or not 0 <= value <= 100_000_000
            for value in (input_tokens, output_tokens)
        ):
            raise SmartFastRoutingError(
                "SmartFast requires finite nonnegative integer token estimates"
            )
        if not session_id or not account_scope or not model or "/" not in model:
            raise SmartFastRoutingError(
                "SmartFast requires a model and opaque account/session identifiers"
            )
        if service_tier not in (None, "default", "priority", "fast", "flex"):
            raise SmartFastRoutingError("SmartFast does not recognize the requested service tier")
        self._prune()
        policy = policy or SmartFastPolicy()
        catalog = await self._catalog(client, model, account_scope, headers)
        if len(self._active) >= MAX_ACTIVE:
            raise SmartFastRoutingError("SmartFast request capacity is busy")
        scope = _scope(account_scope, model, context_key)
        session_key = _scope(scope, session_id)
        session = self._sessions.get(session_key)
        now = _utc_now()
        endpoints = [endpoint.for_request(input_tokens, now) for endpoint in catalog.endpoints]
        free = "free" in model.rsplit("/", 1)[-1].split(":")[1:]
        prompt_cap = 0 if free else policy.maxPromptPrice
        completion_cap = 0 if free else policy.maxCompletionPrice
        requested_tier = "priority" if service_tier == "fast" else service_tier or "default"
        excluded = dict(catalog.excluded)
        eligible: list[Endpoint] = []
        for endpoint in endpoints:
            reason = None
            explicit_tier = only and any(
                _matches(endpoint.tag, item) and _tier(item) != "default" for item in only
            )
            if (only is not None and not any(_matches(endpoint.tag, item) for item in only)) or any(
                _matches(endpoint.tag, item) for item in ignore or ()
            ):
                reason = "restriction"
            elif _tier(endpoint.tag) != requested_tier and not (
                service_tier is None and explicit_tier
            ):
                reason = "service_tier"
            elif not set(required_parameters).issubset(endpoint.supported_parameters):
                reason = "capability"
            elif any(
                limit is not None and amount > limit
                for limit, amount in (
                    (endpoint.context_length, input_tokens + output_tokens),
                    (endpoint.max_prompt_tokens, input_tokens),
                    (endpoint.max_completion_tokens, output_tokens),
                )
            ):
                reason = "context_limit"
            elif (
                not _within(endpoint.prompt_price, prompt_cap)
                or not _within(endpoint.completion_price, completion_cap)
                or (
                    endpoint.reasoning_price is not None
                    and not _within(endpoint.reasoning_price, completion_cap)
                )
                or (
                    endpoint.cache_read_price is not None
                    and not _within(endpoint.cache_read_price, prompt_cap)
                )
                or (
                    endpoint.cache_write_price is not None
                    and not _within(endpoint.cache_write_price, prompt_cap)
                )
                or endpoint.request_price != 0
            ):
                reason = "price_cap"
            if reason:
                excluded[endpoint.tag] = reason
            else:
                eligible.append(endpoint)
        if not eligible:
            raise SmartFastRoutingError(
                "SmartFast found no healthy endpoint within the configured limits"
            )
        costs = {e.tag: e.cold_cost(input_tokens, output_tokens) for e in eligible}
        cutoff = (
            median(costs.values()) * (1 + policy.medianPremiumPercent / 100)
            if len(eligible) >= 5
            else min(costs.values()) * policy.sparsePremiumMultiplier
        )
        # Cooldowns cannot enlarge the price pool or make an otherwise ineligible
        # endpoint look retryable. First apply every lasting eligibility constraint.
        pool = []
        cooldowns = []
        now_monotonic = monotonic()
        for endpoint in eligible:
            expires_at = self._quarantines.get((scope, endpoint.tag), 0)
            if not _within(costs[endpoint.tag], cutoff):
                excluded[endpoint.tag] = "price_outlier"
            elif expires_at > now_monotonic:
                excluded[endpoint.tag] = "quarantine"
                cooldowns.append(expires_at)
            else:
                pool.append(endpoint)
        if not pool:
            raise SmartFastRoutingError(
                "SmartFast eligible endpoints are temporarily cooling down",
                retry_after=min(cooldowns) - now_monotonic,
            )
        estimates = {
            e.tag: self._estimate(
                e,
                scope,
                output_tokens,
                bool(session and session.endpoint == e.tag and session.cached_ratio),
            )
            for e in pool
        }
        known = [value for value in estimates.values() if value is not None]
        bootstrap = not known
        speed_limit = min(known) * (1 + policy.speedTolerancePercent / 100) if known else None
        fast = [
            e
            for e in pool
            if speed_limit is None
            or (estimates[e.tag] is not None and _within(estimates[e.tag], speed_limit))
        ]
        effective_cost = dict(costs)
        for endpoint in pool:
            if (
                session
                and session.endpoint == endpoint.tag
                and session.cached_ratio
                and endpoint.cache_read_price is not None
            ):
                # Cache savings require actual usage for this exact session and
                # a quoted read price. Cold prices still determine the pool.
                cold_price = max(endpoint.prompt_price, endpoint.cache_write_price or 0)
                read_price = min(cold_price, endpoint.cache_read_price)
                effective_cost[endpoint.tag] -= (
                    input_tokens * session.cached_ratio * (cold_price - read_price) / 1_000_000
                )
        pinned = next(
            (e for e in pool if session and session.endpoint == e.tag and session.degradation < 3),
            None,
        )
        cheapest = min(effective_cost[e.tag] for e in fast)
        balanced = [e for e in fast if _within(effective_cost[e.tag], cheapest * 1.05)]
        selected = pinned or (
            None
            if bootstrap
            else min(
                balanced, key=lambda e: (self._load(scope, e.tag), effective_cost[e.tag], e.tag)
            )
        )
        self._admit_session(session_key, scope, selected.tag if selected else None)
        allowed = [selected.tag] if selected else [e.tag for e in pool]
        provider = {
            "only": allowed,
            "max_price": {"prompt": prompt_cap, "completion": completion_cap, "request": 0},
            "allow_fallbacks": True,
            "require_parameters": True,
        }
        if bootstrap:
            provider["sort"] = "throughput"
        blocked = set(ignore or ()) | {
            tag
            for tag in catalog.tags
            if tag not in allowed and any(_matches(tag, item) for item in allowed)
        }
        if blocked:
            provider["ignore"] = sorted(blocked)
        decision = RoutingDecision(
            provider=provider,
            session_id=session_id,
            endpoint=selected.tag if selected else None,
            diagnostics={
                "bootstrap": bootstrap,
                "selected_endpoint": selected.tag if selected else None,
                "price_pool": [e.tag for e in pool],
                "fast_pool": [e.tag for e in fast],
                "estimated_seconds": estimates,
                "cold_cost": costs,
                "effective_cost": effective_cost,
                "excluded": excluded,
            },
            identity=uuid4().hex,
            _scope=scope,
            _session_key=session_key,
            _endpoints=tuple(e for e in pool if e.tag in allowed),
            _output_tokens=output_tokens,
            _speed_limit=speed_limit,
        )
        self._active[decision.identity] = decision
        return decision

    def _attribution(self, decision: RoutingDecision, result: Mapping[str, Any]) -> str | None:
        exact = result.get("endpoint") or result.get("provider_slug")
        provider = result.get("provider") or result.get("provider_name")
        if isinstance(exact, str):
            return next((e.tag for e in decision._endpoints if e.tag == exact), None)
        if isinstance(provider, str):
            matches = [
                e.tag
                for e in decision._endpoints
                if e.tag == provider or e.provider_name == provider
            ]
            return matches[0] if len(matches) == 1 else None
        return decision.endpoint

    def observe(
        self,
        decision: RoutingDecision,
        elapsed: float,
        result: Mapping[str, Any] | None,
        success: bool,
        *,
        healthy: bool | None = None,
    ) -> None:
        """Observe raw completion JSON once, without retaining its text or errors.

        Call before ``release``. Missing or ambiguous provider attribution does
        not train exact endpoint measurements. Cancellation needs only release.
        Healthy but incomplete or repaired output preserves affinity and actual
        cache usage without training successful speed or degradation measurements.
        """
        if (
            self._active.get(decision.identity) is not decision
            or decision.identity in self._observed
        ):
            return
        self._observed.add(decision.identity)
        healthy = success if healthy is None else healthy
        result = result if isinstance(result, Mapping) else {}
        elapsed_value = _number(elapsed)
        tag = self._attribution(decision, result)
        usage = result.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, Mapping) else {}
        cached = _tokens(details.get("cached_tokens"))
        written = _tokens(details.get("cache_write_tokens"))
        prompt = _tokens(usage.get("prompt_tokens"))
        output = _tokens(usage.get("completion_tokens"))
        cost = _number(usage.get("cost"))
        if cached is not None and (prompt is None or cached > prompt):
            cached = None
        decision.diagnostics.update(
            success=bool(success),
            healthy=bool(healthy),
            elapsed_seconds=elapsed_value,
            observed_endpoint=tag,
            cache_read_tokens=cached,
            cache_write_tokens=written,
            reported_cost=cost,
        )
        # A response admitted before a health failure may arrive after recovery.
        # Its billing remains observable, but it cannot overwrite the newer state.
        if (tag or decision.endpoint) in decision._invalidated_endpoints:
            return
        if not healthy:
            error = result.get("error")
            status = _tokens(error.get("code")) if isinstance(error, Mapping) else None
            decision.diagnostics["failure_status"] = (
                int(status) if status is not None and 100 <= status <= 599 else None
            )
            failed = tag or decision.endpoint
            if failed:
                for active in self._active.values():
                    if active._scope == decision._scope and failed in active.provider["only"]:
                        active._invalidated_endpoints.add(failed)
                key = (decision._scope, failed)
                self._quarantines[key] = monotonic() + QUARANTINE_SECONDS
                self._quarantines.move_to_end(key)
                while len(self._quarantines) > MAX_OBSERVATIONS:
                    self._quarantines.popitem(last=False)
                for key, session in list(self._sessions.items()):
                    if session.scope == decision._scope and session.endpoint == failed:
                        del self._sessions[key]
            return
        if tag is None:
            return
        session = self._sessions.get(decision._session_key)
        if session and (session.endpoint is None or session.endpoint == tag):
            session.endpoint = tag
            session.touched_at = monotonic()
            if cached is not None and prompt:
                ratio = min(0.9, cached / prompt)
                session.cached_ratio = (
                    min(session.cached_ratio, ratio) if session.cached_ratio else ratio
                )
            if (
                success
                and elapsed_value is not None
                and output
                and decision._speed_limit is not None
            ):
                normalized = elapsed_value * decision._output_tokens / output
                session.degradation = (
                    session.degradation + 1 if not _within(normalized, decision._speed_limit) else 0
                )
        if not success or elapsed_value is None or elapsed_value <= 0 or not output:
            return
        key = (decision._scope, tag, bool(cached))
        observation = self._observations.get(key)
        if observation is None:
            observation = _Observation(deque(maxlen=8), monotonic())
            self._observations[key] = observation
        observation.seconds_per_token.append(elapsed_value / output)
        observation.touched_at = monotonic()
        self._observations.move_to_end(key)
        while len(self._observations) > MAX_OBSERVATIONS:
            self._observations.popitem(last=False)

    def release(self, decision: RoutingDecision) -> None:
        """Idempotently release admission after success, exception, or cancellation."""
        if self._active.get(decision.identity) is decision:
            del self._active[decision.identity]
            self._observed.discard(decision.identity)
