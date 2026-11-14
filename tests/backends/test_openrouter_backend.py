"""The OpenRouter back-end type: settings round trip, data retention, worst-case charges, options, discovery."""

import json

import httpx
import pytest

from subtitle_translator.backends.openrouter import OpenRouterProvider
from subtitle_translator.backends.openrouter_backend import (
    OPENROUTER_DEFAULT_MODEL,
    OpenRouterBackendType,
    resolve_model,
    settings_from_config,
    translation_config,
)
from subtitle_translator.core.backends import (
    BackendContext,
    BackendRegistry,
    BackendScope,
    Connection,
)
from subtitle_translator.core.errors import InvalidConfigError
from subtitle_translator.core.models import (
    ProviderConfig,
    ReasoningConfig,
    SmartFastPolicy,
    TranslationConfig,
)
from subtitle_translator.core.provider import TranslationBatch
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.core.usage import BudgetExhaustedError, UsageMeter

BATCH = TranslationBatch(lines=[{"index": "0", "content": "Hello"}], source_language="en", target_language="hu")


def test_legacy_request_config_round_trips_through_a_pinned_connection():
    legacy = TranslationConfig(
        apiKey="enc-decrypted-key",
        model="deepseek/deepseek-v4-flash:smartfast",
        temperature=0.2,
        requestTimeout=300,
        reasoning=ReasoningConfig(effort="low", maxTokens=2000),
        useThinkingVariant=False,
        provider=ProviderConfig(
            order=["deepinfra", "novita"], allowFallbacks=False, sort="smartfast",
            only=["deepinfra"], ignore=["x"], smartFast=SmartFastPolicy(maxPromptPrice=0.5),
        ),
        serviceTier="default",
        parallelBatches=3,
    )
    rebuilt = translation_config(settings_from_config(legacy), {"api_key": legacy.api_key})
    assert rebuilt.model_dump() == legacy.model_dump()


def test_retired_models_keep_their_routing_suffix():
    assert resolve_model("google/gemini-3-pro-preview:floor") == "google/gemini-3.1-pro-preview:floor"
    assert resolve_model("google/gemini-2.5-flash-preview-09-2025") == "google/gemini-2.5-flash"
    assert resolve_model("openai/gpt-5.6-luna") == "openai/gpt-5.6-luna"


def test_validate_normalizes_and_reports():
    backend = OpenRouterBackendType()
    ok = backend.validate({"model": "google/gemini-2.5-flash-lite-preview-06-17", "routing": "floor"}, {"api_key": "k"})
    assert ok.ok and ok.patch["model"] == "google/gemini-2.5-flash-lite" and ok.patch["deny_data_collection"] is False
    bad = backend.validate({"routing": "smartfast", "provider_order": ["a"], "parallel_batches": 99}, {})
    assert set(bad.errors) == {"api_key", "routing", "parallel_batches"}


def test_a_smartfast_policy_is_checked_without_the_key():
    backend = OpenRouterBackendType()
    policy = {"routing": "smartfast", "smart_fast": {"maxPromptPrice": -1}}
    assert set(backend.validate(policy, {}).errors) == {"api_key", "smart_fast"}
    assert set(backend.validate(policy, {"api_key": "k"}).errors) == {"smart_fast"}
    fine = {"routing": "smartfast", "smart_fast": {"maxPromptPrice": 0.5}}
    assert backend.validate(fine, {}).errors == {"api_key": "API key is required"}


def test_open_refuses_a_policy_it_cannot_read():
    with pytest.raises(InvalidConfigError) as caught:
        OpenRouterBackendType().open(
            Connection("or", "openrouter", settings={"model": "x/y", "smart_fast": {"warp": 9}}, secrets={"api_key": "k"}),
            CoreSettings(),
            BackendContext(),
        )
    assert set(caught.value.field_errors) == {"smart_fast"}


def test_deny_data_collection_goes_into_the_provider_payload():
    config = translation_config(
        {"model": "x/y", "routing": "throughput", "deny_data_collection": True}, {"api_key": "k"}
    )
    _, params = OpenRouterProvider(CoreSettings())._build_provider_payload(config, "x/y")
    assert params["provider"]["data_collection"] == "deny"


def test_data_collection_is_off_by_default():
    config = translation_config({"model": "x/y"}, {"api_key": "k"})
    assert config.provider is None or config.provider.data_collection is None


def test_worst_case_charge_counts_every_input_byte_and_the_output_limit():
    provider = OpenRouterProvider(CoreSettings())
    provider._model_price_cache["x/y"] = (0.000001, 0.000002)
    payload = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1000}
    size = len(json.dumps(payload["messages"], ensure_ascii=False).encode()) + 16
    assert provider._worst_case_charge(payload, "x/y") == pytest.approx(size * 0.000001 + 1000 * 0.000002)
    assert provider._worst_case_charge(payload, "x/y:free") == 0.0
    assert provider._worst_case_charge({"messages": []}, "x/y") is None  # no output bound known
    assert provider._worst_case_charge(payload, "unknown/model") is None
    capped = {**payload, "provider": {"max_price": {"prompt": 1, "completion": 3}}}
    assert provider._worst_case_charge(capped, "x/y") == pytest.approx(size * 1e-6 + 1000 * 3e-6)


@pytest.mark.asyncio
async def test_a_request_whose_worst_case_does_not_fit_is_never_sent():
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(500)

    provider = OpenRouterProvider(CoreSettings(openrouter_api_key="k", openrouter_max_tokens=8000))
    provider._model_params_fetched = True
    provider._model_price_cache["x/y"] = (0.000001, 0.000002)
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://openrouter.ai/api/v1")
    config = TranslationConfig(model="x/y").for_operation("job")
    config._usage_meter = UsageMeter(budget=0.001)  # 8000 output tokens alone could cost 0.016
    with pytest.raises(BudgetExhaustedError):
        await provider.translate_batch(BATCH, config_override=config)
    assert sent == []


@pytest.mark.asyncio
async def test_options_list_models_with_prices_and_context():
    def handler(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"id": "x/y", "name": "X Y", "context_length": 131072,
                                                    "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}}]})

    choices = await OpenRouterBackendType(transport=httpx.MockTransport(handler)).options("model", {}, {}, BackendContext())
    assert choices[0].value == "x/y" and choices[0].prompt_price == pytest.approx(0.1)
    assert choices[0].completion_price == pytest.approx(0.4) and choices[0].context_length == 131072


@pytest.mark.asyncio
async def test_options_return_nothing_when_the_model_list_cannot_be_fetched(caplog):
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    backend = OpenRouterBackendType(transport=httpx.MockTransport(handler))
    secrets = {"api_key": "sk-or-synthetic-key"}
    with caplog.at_level("WARNING"):
        assert await backend.options("model", {}, secrets, BackendContext()) == []
    assert "ConnectError" in caplog.text
    assert "sk-or-synthetic-key" not in caplog.text


@pytest.mark.asyncio
async def test_options_return_nothing_for_a_malformed_api_base(caplog):
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "x/y"}]})

    backend = OpenRouterBackendType(transport=httpx.MockTransport(handler), api_base=lambda: "http://[::1/v1")
    with caplog.at_level("WARNING"):
        assert await backend.options("model", {}, {}, BackendContext()) == []
    assert "InvalidURL" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"<html>maintenance</html>", b"[1, 2]", b"null"])
async def test_options_return_nothing_for_a_body_that_is_not_a_model_list(body):
    def handler(request):
        return httpx.Response(200, content=body)

    backend = OpenRouterBackendType(transport=httpx.MockTransport(handler))
    assert await backend.options("model", {}, {}, BackendContext()) == []


@pytest.mark.asyncio
async def test_options_read_the_configured_api_base():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "x/y"}]})

    backend = OpenRouterBackendType(
        transport=httpx.MockTransport(handler), api_base=lambda: "https://gateway.example/api/v1"
    )
    choices = await backend.options("model", {}, {}, BackendContext())
    assert seen == ["https://gateway.example/api/v1/models"]
    assert [choice.value for choice in choices] == ["x/y"]


@pytest.mark.asyncio
async def test_options_use_the_public_base_when_the_host_sets_none():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": []})

    backend = OpenRouterBackendType(transport=httpx.MockTransport(handler))
    await backend.options("model", {}, {}, BackendContext())
    assert seen == ["https://openrouter.ai/api/v1/models"]


@pytest.mark.asyncio
async def test_options_leave_out_models_that_do_not_produce_text():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "a/chat", "architecture": {"output_modalities": ["text"]}},
                    {"id": "b/both", "architecture": {"output_modalities": ["text", "image"]}},
                    {"id": "c/image", "architecture": {"output_modalities": ["image"]}},
                    {"id": "d/silent"},
                    {"id": "e/odd", "architecture": {"output_modalities": "text"}},
                ]
            },
        )

    backend = OpenRouterBackendType(transport=httpx.MockTransport(handler))
    choices = await backend.options("model", {}, {}, BackendContext())
    assert [choice.value for choice in choices] == ["a/chat", "b/both", "d/silent", "e/odd"]


def test_open_uses_the_connection_budget_and_adaptive_batches():
    opened = OpenRouterBackendType().open(
        Connection("or", "openrouter", settings={"model": "x/y", "max_tokens": 0}, secrets={"api_key": "k"}),
        CoreSettings(openrouter_max_tokens=8000),
        BackendContext(),
    )
    assert opened.settings.openrouter_max_tokens == 0 and opened.batch_size is None
    assert opened.config.api_key == "k" and opened.model == "x/y"


def test_the_builtin_back_ends_are_discovered_from_the_installed_package():
    registry = BackendRegistry.discover(BackendScope.builtin())
    assert registry.types() == ["gemini", "openai_compatible", "openrouter"]
    assert all(entry.builtin for entry in registry.entries())
    assert OPENROUTER_DEFAULT_MODEL


@pytest.mark.parametrize("value", ["allow", "deny"])
def test_a_request_data_collection_choice_survives_the_pinned_connection(value):
    config = TranslationConfig(
        apiKey="k", model="x/y", provider=ProviderConfig(sort="throughput", dataCollection=value)
    )
    rebuilt = translation_config(
        {k: v for k, v in settings_from_config(config).items() if v is not None}, {"api_key": "k"}
    )
    assert rebuilt.provider.data_collection == value
    _, params = OpenRouterProvider(CoreSettings())._build_provider_payload(rebuilt, "x/y")
    assert params["provider"]["data_collection"] == value


def test_a_request_that_says_nothing_about_data_collection_sends_nothing():
    config = TranslationConfig(apiKey="k", model="x/y", provider=ProviderConfig(sort="throughput"))
    rebuilt = translation_config(
        {k: v for k, v in settings_from_config(config).items() if v is not None}, {"api_key": "k"}
    )
    assert rebuilt.provider.data_collection is None


def test_an_empty_only_list_restricts_nothing_through_a_connection_and_in_the_payload():
    # An empty list is the unset value of a list field; it must not become "no provider allowed".
    config = TranslationConfig(
        apiKey="k", model="x/y", provider=ProviderConfig(sort="throughput", only=[], ignore=[])
    )
    rebuilt = translation_config(
        {k: v for k, v in settings_from_config(config).items() if v is not None}, {"api_key": "k"}
    )
    assert rebuilt.provider.only is None and rebuilt.provider.ignore is None
    _, params = OpenRouterProvider(CoreSettings())._build_provider_payload(rebuilt, "x/y")
    assert "only" not in params["provider"] and "ignore" not in params["provider"]


def _priced_provider(reply):
    provider = OpenRouterProvider(CoreSettings(openrouter_api_key="k", openrouter_max_tokens=1000))
    provider._model_params_fetched = True
    provider._model_price_cache["x/y"] = (0.000001, 0.000002)
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply)),
        base_url="https://openrouter.ai/api/v1",
    )
    return provider


def _translated(usage):
    content = json.dumps({"translations": [{"index": "0", "content": "Szia"}]})
    return {"choices": [{"message": {"content": content}}], "usage": usage}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage,charged",
    [({"total_tokens": 30}, False), ({"total_tokens": 30, "cost": 0.0004}, True)],
)
async def test_a_reply_that_states_no_charge_keeps_its_hold(usage, charged):
    from subtitle_translator.core.batch_processor import BatchProcessor

    provider = _priced_provider(_translated(usage))
    meter = UsageMeter(budget=1.0)
    result = await BatchProcessor(provider, CoreSettings(), meter=meter).process_all_batches(
        BATCH.lines, "en", "hu", batch_size=10, model="x/y"
    )
    assert result.success
    if charged:
        assert meter.reserved == pytest.approx(0.0) and meter.cost == pytest.approx(0.0004)
    else:
        # Nothing says what the call cost, so the hold stays and later requests plan around it.
        assert meter.reserved > 0 and meter.cost == 0.0


@pytest.mark.asyncio
async def test_a_free_model_reply_without_a_charge_costs_nothing():
    from subtitle_translator.core.batch_processor import BatchProcessor

    provider = _priced_provider(_translated({"total_tokens": 30}))
    meter = UsageMeter(budget=1.0)
    result = await BatchProcessor(provider, CoreSettings(), meter=meter).process_all_batches(
        BATCH.lines, "en", "hu", batch_size=10, model="x/y:free"
    )
    assert result.success and meter.reserved == 0.0 and meter.cost == 0.0


def _reply_with(content=None, *, choices=True, usage=None):
    message = {"content": content} if content is not None else {}
    body = {"choices": [{"message": message}] if choices else []}
    if usage is not None:
        body["usage"] = usage
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        _reply_with(choices=False, usage={"total_tokens": 30}),
        _reply_with(content=None, usage={"total_tokens": 30}),
        _reply_with(content="this is not json", usage={"total_tokens": 30}),
        _reply_with(content=json.dumps({"translations": "nope"}), usage={"total_tokens": 30}),
        _reply_with(content="this is not json"),
    ],
    ids=["no-choices", "no-content", "not-json", "bad-translations", "no-usage-at-all"],
)
async def test_a_reply_that_fails_processing_without_a_stated_charge_keeps_its_hold(reply):
    from subtitle_translator.core.batch_processor import BatchProcessor

    provider = _priced_provider(reply)
    meter = UsageMeter(budget=1.0)
    result = await BatchProcessor(
        provider, CoreSettings(max_retries=1, retry_delay=0.0), meter=meter
    ).process_all_batches(BATCH.lines, "en", "hu", batch_size=10, model="x/y")
    assert not result.success
    # The call may have been billed, so its hold stays and a retry cannot spend it again.
    assert meter.reserved > 0 and meter.cost == 0.0


@pytest.mark.asyncio
async def test_a_reply_that_fails_processing_after_stating_a_zero_charge_releases_its_hold():
    from subtitle_translator.core.batch_processor import BatchProcessor

    reply = _reply_with(content="this is not json", usage={"total_tokens": 30, "cost": 0})
    provider = _priced_provider(reply)
    meter = UsageMeter(budget=1.0)
    result = await BatchProcessor(
        provider, CoreSettings(max_retries=1, retry_delay=0.0), meter=meter
    ).process_all_batches(BATCH.lines, "en", "hu", batch_size=10, model="x/y")
    assert not result.success and meter.reserved == 0.0 and meter.cost == 0.0


@pytest.mark.asyncio
async def test_a_reply_that_fails_processing_after_stating_a_charge_settles_that_charge():
    from subtitle_translator.core.batch_processor import BatchProcessor

    reply = _reply_with(content="this is not json", usage={"total_tokens": 30, "cost": 0.0004})
    provider = _priced_provider(reply)
    meter = UsageMeter(budget=1.0)
    await BatchProcessor(
        provider, CoreSettings(max_retries=1, retry_delay=0.0), meter=meter
    ).process_all_batches(BATCH.lines, "en", "hu", batch_size=10, model="x/y")
    assert meter.reserved == 0.0 and meter.cost > 0.0


def _capturing_provider(seen, **settings):
    reply = _translated({"total_tokens": 30, "cost": 0.0004})

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    provider = OpenRouterProvider(
        CoreSettings(openrouter_api_key="k", openrouter_max_tokens=1000, **settings)
    )
    provider._model_params_fetched = True
    provider._model_price_cache["x/y"] = (0.000001, 0.000002)
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://openrouter.ai/api/v1"
    )
    return provider


@pytest.mark.asyncio
async def test_a_budgeted_request_caps_endpoint_prices_at_the_reserved_prices():
    seen = []
    provider = _capturing_provider(seen)
    config = TranslationConfig()
    config._usage_meter = UsageMeter(budget=1.0)
    await provider.translate_batch(BATCH, model="x/y", config_override=config)
    # USD per million tokens, the catalog's 1e-6 and 2e-6 per token.
    # The SDK sends prices as decimal strings, as it does for SmartFast.
    sent = {key: float(value) for key, value in seen[0]["provider"]["max_price"].items()}
    # A fixed fee per request is capped at nothing, because the reservation prices tokens only.
    assert sent == {"prompt": 1.0, "completion": 2.0, "request": 0.0}


@pytest.mark.asyncio
async def test_a_request_without_a_budget_sends_no_price_cap():
    seen = []
    provider = _capturing_provider(seen)
    config = TranslationConfig()
    config._usage_meter = UsageMeter()
    await provider.translate_batch(BATCH, model="x/y", config_override=config)
    assert "max_price" not in (seen[0].get("provider") or {})


@pytest.mark.asyncio
async def test_a_cap_the_request_already_carries_stays_where_it_is_lower():
    seen = []
    provider = _capturing_provider(seen)
    payload = {"provider": {"max_price": {"prompt": 0.5, "completion": 9, "request": 0}}, "messages": []}
    config = TranslationConfig()
    config._usage_meter = UsageMeter(budget=1.0)
    provider._cap_prices_to_the_reservation(payload, "x/y", config)
    assert payload["provider"]["max_price"] == {"prompt": 0.5, "completion": 2.0, "request": 0}
    payload = {"provider": {"max_price": {"request": 0.01}}, "messages": []}
    provider._cap_prices_to_the_reservation(payload, "x/y", config)
    assert payload["provider"]["max_price"]["request"] == 0


def test_the_reservation_never_assumes_a_price_above_the_cap():
    provider = _capturing_provider([])
    payload = {
        "messages": [{"role": "user", "content": "x" * 100}],
        "max_tokens": 1000,
        "provider": {"max_price": {"completion": 0.5}},  # prompt left open, completion capped
    }
    worst = provider._worst_case_charge(payload, "x/y")
    input_bytes = len(json.dumps(payload["messages"]).encode()) + 16
    # The open prompt price is the catalog's; the completion price is the cap, below the catalog's.
    assert worst == pytest.approx(input_bytes * 1e-6 + 1000 * 0.5e-6)
    free = {"messages": [], "max_tokens": 1000, "provider": {"max_price": {"prompt": 9, "completion": 9}}}
    assert provider._worst_case_charge(free, "x/y") == pytest.approx(18 * 9e-6 + 1000 * 9e-6)
