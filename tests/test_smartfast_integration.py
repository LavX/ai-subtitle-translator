"""SmartFast integration through real provider HTTP and operation boundaries."""

import asyncio
import json
import logging
from functools import partial
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from subtitle_translator.api.models import ProviderConfig, TranslationConfig
from subtitle_translator.config import Settings
from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.translator import SubtitleTranslator
from subtitle_translator.providers.base import (
    InvalidResponseError,
    ProviderTimeoutError,
    RateLimitError,
    TranslationBatch,
    TranslationProviderError,
)
from subtitle_translator.providers.openrouter import OpenRouterProvider
from subtitle_translator.queue.job_manager import JobManager, JobStatus, JobType
from subtitle_translator.queue.worker import (
    process_content_translation_job,
    process_file_translation_job,
)


def endpoint(tag, price=0.1, *, throughput=100):
    return {
        "tag": tag,
        "provider_name": tag.title(),
        "status": 0,
        "pricing": {
            "prompt": str(price / 1_000_000),
            "completion": str(price / 1_000_000),
            "request": "0",
        },
        "latency_last_30m": {"p50": 100},  # Milliseconds, equivalent to 0.1 seconds.
        "throughput_last_30m": {"p50": throughput},
        "context_length": 131072,
        "max_completion_tokens": 32768,
        "supported_parameters": ["temperature", "response_format", "reasoning"],
    }


def quoted_prices(body):
    """Read the SDK's decimal-string quotes as unit rates for policy assertions."""
    return {key: float(value) for key, value in body["provider"]["max_price"].items()}


class OpenRouterWire:
    def __init__(self):
        self.endpoints = [endpoint("a"), endpoint("b")]
        self.models = [
            {
                "id": "fixture/model",
                "supported_parameters": ["temperature", "response_format", "reasoning"],
            }
        ]
        self.requests = []
        self.completions = []
        self.reply_modes = []
        self.completion_gate = None
        self.catalog_gate = None
        self.catalog_stream = None
        self.started = asyncio.Event()
        self.disconnected = asyncio.Event()

    async def __call__(self, request):
        self.requests.append(request)
        if request.method == "GET" and request.url.path.endswith("/endpoints"):
            if self.catalog_stream is not None:
                return httpx.Response(200, stream=self.catalog_stream)
            if self.catalog_gate is not None:
                await self.catalog_gate.wait()
            return httpx.Response(200, json={"data": {"endpoints": self.endpoints}})
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": self.models})
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        self.completions.append(body)
        self.started.set()
        if self.completion_gate is not None:
            try:
                await self.completion_gate.wait()
            except asyncio.CancelledError:
                self.disconnected.set()
                raise
        source = json.loads(body["messages"][-1]["content"])
        translations = [{"index": line["index"], "content": "Forditas"} for line in source]
        mode = self.reply_modes.pop(0) if self.reply_modes else "complete"
        content = json.dumps({"translations": translations})
        if mode == "partial":
            content = json.dumps({"translations": translations[:1]})
        elif mode == "blank":
            translations[0]["content"] = " \t\n"
            content = json.dumps({"translations": translations})
        elif mode == "repaired":
            content = content[:-2]
        elif mode == "trailing_comma":
            content = content[:-2] + ",]}"
        elif mode == "invalid_escape":
            content = content.replace("Forditas", r"Fordit\'as")
        elif mode == "outer_trailing_comma":
            content = content[:-1] + ",}"
        elif mode == "fenced_invalid_escape":
            content = "```json\n" + content.replace("Forditas", r"Fordit\'as") + "\n```"
        elif mode == "prose_invalid_escape":
            content = "Here is the translation: " + content.replace("Forditas", r"Fordit\'as")
        elif mode == "duplicate_keys":
            content = '{"index":"0","content":"Forditas","index":"1","content":"Forditas"}'
        elif mode == "invalid":
            content = "bad response"
        elif mode == "empty":
            content = json.dumps({"translations": [{"index": source[0]["index"], "content": ""}]})
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "cost": 0.001,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        data = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": usage,
            "provider": body.get("provider", {}).get("only", ["a"])[0].title(),
        }
        if mode == "error":
            data = {"error": {"code": 429, "message": "synthetic failure"}, "usage": usage}
        if mode == "no_choices":
            data["choices"] = []
        elif mode == "bad_choices":
            data["choices"] = ["invalid choice"]
        elif mode == "bad_message":
            data["choices"][0]["message"] = []
        elif mode == "bad_content":
            data["choices"][0]["message"]["content"] = {"unexpected": "object"}
        elif mode == "http429":
            return httpx.Response(
                429, json={"error": {"code": 429}, "usage": usage}, headers={"Retry-After": "0"}
            )
        elif mode == "http503":
            return httpx.Response(
                503, json={"error": {"code": 503}, "usage": usage}, headers={"Retry-After": "0"}
            )
        elif mode == "transport":
            raise httpx.ConnectError("synthetic connection loss", request=request)
        return httpx.Response(200, json=data)


@pytest.fixture
async def environment(monkeypatch):
    wire = OpenRouterWire()
    # HTTP is the sole replacement boundary, including public model discovery.
    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(wire))
    )
    settings = Settings(
        _env_file=None,
        openrouter_api_key="synthetic-default-key",
        openrouter_api_base="https://fixture.invalid/api/v1",
        openrouter_default_model="fixture/model",
        retry_delay=0,
        batch_size=2,
        parallel_batches_per_job=2,
    )
    provider = OpenRouterProvider(settings)
    yield wire, provider, settings
    await provider.close()


def batch(count=2):
    return TranslationBatch(
        lines=[{"index": str(i), "content": f"Original {i}"} for i in range(count)],
        source_language="English",
        target_language="Hungarian",
        context_title="private-media-title",
    )


@pytest.mark.parametrize("model,price", [("fixture/model", 0.1), ("fixture/model:free", 0)])
@pytest.mark.parametrize("catalog_has_json", [False, True])
async def test_json_mode_is_optional_for_sole_paid_and_free_endpoints(
    environment, model, price, catalog_has_json
):
    wire, provider, _ = environment
    wire.endpoints = [endpoint("a", price)]
    wire.endpoints[0]["supported_parameters"].remove("response_format")
    if not catalog_has_json:
        wire.models[0]["supported_parameters"].remove("response_format")
    result = await provider.translate_batch(
        batch(),
        model=model + ":smartfast",
        config_override=TranslationConfig(reasoning={"effort": "none"}),
    )
    assert result.translations == [
        {"index": "0", "content": "Forditas"},
        {"index": "1", "content": "Forditas"},
    ]
    assert len(wire.completions) == 1
    body = wire.completions[0]
    assert "response_format" not in body
    assert body["reasoning"] == {"effort": "none"}
    assert body["temperature"] == 0.3
    assert body["provider"]["only"] == ["a"]
    assert body["provider"]["require_parameters"] is True
    assert quoted_prices(body) == (
        {"prompt": 0, "completion": 0, "request": 0}
        if price == 0
        else {"prompt": 1, "completion": 3, "request": 0}
    )
    assert "translations" in body["messages"][0]["content"]


@pytest.mark.parametrize(
    "record",
    json.loads((Path(__file__).parent / "fixtures/smartfast-json-capabilities.json").read_text()),
    ids=lambda record: record["model"]["id"],
)
async def test_json_mode_absent_from_real_ling_and_nemotron_metadata_still_translates(
    environment, record
):
    wire, provider, _ = environment
    wire.models = [record["model"]]
    wire.endpoints = record["catalog"]["data"]["endpoints"]
    result = await provider.translate_batch(
        batch(),
        model=record["model"]["id"] + ":smartfast",
        config_override=TranslationConfig(reasoning={"effort": "none"}),
    )
    assert len(result.translations) == 2
    assert len(wire.completions) == 1
    body = wire.completions[0]
    assert body["model"] == record["model"]["id"]
    assert "response_format" not in body
    assert body["reasoning"] == {"effort": "none"}
    assert body["provider"]["only"] == [wire.endpoints[0]["tag"]]
    assert quoted_prices(body) == {"prompt": 0, "completion": 0, "request": 0}
    assert result.routing_diagnostics["bootstrap"] is True


@pytest.mark.parametrize("first_has_json", [False, True])
async def test_json_mode_matches_mixed_bootstrap_then_pinned_route(environment, first_has_json):
    wire, provider, _ = environment
    wire.endpoints = [endpoint("a", throughput=None), endpoint("b", throughput=None)]
    wire.endpoints[0 if not first_has_json else 1]["supported_parameters"].remove("response_format")
    config = TranslationConfig()
    config._smartfast_session_id = "same-operation"
    first = await provider.translate_batch(
        batch(), model="fixture/model:smartfast", config_override=config
    )
    second = await provider.translate_batch(
        batch(), model="fixture/model:smartfast", config_override=config
    )
    assert len(first.translations) == len(second.translations) == 2
    bootstrap, pinned = wire.completions
    assert bootstrap["provider"]["only"] == ["a", "b"]
    assert bootstrap["provider"]["sort"] == "throughput"
    assert "response_format" not in bootstrap
    assert pinned["provider"]["only"] == ["a"]
    assert ("response_format" in pinned) is first_has_json
    if first_has_json:
        assert pinned["response_format"] == {"type": "json_object"}
    assert bootstrap["session_id"] == pinned["session_id"]
    assert first.routing_diagnostics["price_pool"] == ["a", "b"]
    assert second.routing_diagnostics["price_pool"] == ["a", "b"]


@pytest.mark.parametrize("selection", ["price", "speed"])
async def test_json_mode_does_not_exclude_best_price_or_speed_endpoint(environment, selection):
    wire, provider, _ = environment
    wire.endpoints = [
        endpoint("a"),
        endpoint("b", 0.4 if selection == "price" else 0.1, throughput=20),
    ]
    wire.endpoints[0]["supported_parameters"].remove("response_format")
    result = await provider.translate_batch(batch(), model="fixture/model:smartfast")
    assert len(result.translations) == 2
    assert wire.completions[0]["provider"]["only"] == ["a"]
    assert "response_format" not in wire.completions[0]
    assert result.routing_diagnostics["excluded"].get("b") == (
        "price_outlier" if selection == "price" else None
    )


@pytest.mark.parametrize("first_has_json", [False, True])
async def test_json_mode_is_recomputed_after_provider_failover(environment, first_has_json):
    wire, provider, settings = environment
    wire.endpoints[0 if not first_has_json else 1]["supported_parameters"].remove("response_format")
    wire.reply_modes = ["http503"]
    result = await BatchProcessor(provider, settings).process_batch(
        batch(),
        0,
        model="fixture/model:smartfast",
        _deadline=asyncio.get_running_loop().time() + 1,
    )
    assert result.success
    assert [body["provider"]["only"] for body in wire.completions] == [["a"], ["b"]]
    assert [("response_format" in body) for body in wire.completions] == [
        first_has_json,
        not first_has_json,
    ]
    assert len({body["session_id"] for body in wire.completions}) == 1


async def test_json_mode_retained_for_supported_route_despite_unsupported_pool_member(environment):
    wire, provider, _ = environment
    wire.endpoints[1]["supported_parameters"].remove("response_format")
    result = await provider.translate_batch(batch(), model="fixture/model:smartfast")
    assert result.routing_diagnostics["price_pool"] == ["a", "b"]
    assert wire.completions[0]["provider"]["only"] == ["a"]
    assert wire.completions[0]["response_format"] == {"type": "json_object"}


async def test_json_mode_omission_preserves_explicit_provider_restrictions(environment):
    wire, provider, _ = environment
    wire.endpoints[1]["supported_parameters"].remove("response_format")
    result = await provider.translate_batch(
        batch(),
        model="fixture/model:smartfast",
        config_override=TranslationConfig(
            provider={"only": ["b"], "ignore": ["a"], "allowFallbacks": False}
        ),
    )
    assert len(result.translations) == 2
    body = wire.completions[0]
    assert "response_format" not in body
    assert body["provider"]["only"] == ["b"]
    assert body["provider"]["ignore"] == ["a"]
    assert body["provider"]["allow_fallbacks"] is False
    assert quoted_prices(body) == {"prompt": 1, "completion": 3, "request": 0}


@pytest.mark.parametrize("missing", ["reasoning", "temperature"])
async def test_json_mode_fallback_keeps_actual_required_parameters_strict(environment, missing):
    wire, provider, _ = environment
    wire.endpoints = [endpoint("a")]
    wire.endpoints[0]["supported_parameters"] = [
        key for key in ["reasoning", "temperature"] if key != missing
    ]
    with pytest.raises(TranslationProviderError, match="no healthy endpoint"):
        await provider.translate_batch(
            batch(),
            model="fixture/model:smartfast",
            config_override=TranslationConfig(reasoning={"effort": "none"}),
        )
    assert wire.completions == []


async def test_json_mode_fallback_cannot_disable_model_mandatory_reasoning(environment):
    wire, provider, _ = environment
    wire.models[0]["reasoning"] = {"mandatory": True}
    wire.endpoints[0]["supported_parameters"].remove("response_format")
    with pytest.raises(TranslationProviderError, match="Reasoning is mandatory"):
        await provider.translate_batch(
            batch(),
            model="fixture/model:smartfast",
            config_override=TranslationConfig(reasoning={"effort": "none"}),
        )
    assert wire.completions == []


async def test_json_mode_fallback_does_not_enable_format_alongside_active_reasoning(environment):
    wire, provider, _ = environment
    wire.models[0]["reasoning"] = {"mandatory": True, "supported_efforts": ["low"]}
    await provider.translate_batch(
        batch(),
        model="fixture/model:smartfast",
        config_override=TranslationConfig(reasoning={"effort": "low"}),
    )
    assert wire.completions[0]["reasoning"] == {"effort": "low"}
    assert "response_format" not in wire.completions[0]


async def test_json_mode_fallback_invalid_output_retries_remain_bounded(environment):
    wire, provider, settings = environment
    wire.endpoints = [endpoint("a", 0)]
    wire.endpoints[0]["supported_parameters"].remove("response_format")
    wire.reply_modes = ["invalid"] * 20
    settings.max_retries = 3
    async with asyncio.timeout(1):
        result = await BatchProcessor(provider, settings).process_batch(
            batch(),
            0,
            model="fixture/model:free:smartfast",
        )
    assert not result.success
    assert result.translations == []
    assert len(wire.completions) == 4
    assert result.retries == 3
    assert len({body["session_id"] for body in wire.completions}) == 1
    assert all("response_format" not in body for body in wire.completions)
    assert all(body["provider"]["only"] == ["a"] for body in wire.completions)
    assert all(
        quoted_prices(body) == {"prompt": 0, "completion": 0, "request": 0}
        for body in wire.completions
    )
    assert result.tokens_used == 4200
    assert result.cost == pytest.approx(0.004)


@pytest.mark.parametrize(
    "model",
    ["fixture/model:smartfast", "fixture/model:free:smartfast", "fixture/model:thinking:smartfast"],
)
async def test_local_suffix_preserves_underlying_variant_and_emits_price_caps(environment, model):
    wire, provider, _ = environment
    if ":free:" in model:
        wire.endpoints = [endpoint("a", 0)]
    await provider.translate_batch(batch(), model=model)
    body = wire.completions[0]
    assert body["model"] == model.removesuffix(":smartfast")
    assert body["session_id"]
    assert quoted_prices(body) == (
        {"prompt": 0, "completion": 0, "request": 0}
        if ":free:" in model
        else {"prompt": 1, "completion": 3, "request": 0}
    )
    assert "order" not in body["provider"]
    assert "service_tier" not in body
    assert not any("smartfast" in str(request.url) for request in wire.requests)


async def test_config_policy_and_exact_restrictions_reach_catalog_and_completion(environment):
    wire, provider, _ = environment
    wire.endpoints = [endpoint("a"), endpoint("a/priority"), endpoint("b")]
    config = TranslationConfig(
        apiKey="synthetic-job-key",
        provider={
            "sort": "smartfast",
            "only": ["a", "b"],
            "ignore": ["b"],
            "allowFallbacks": False,
            "smartFast": {"maxPromptPrice": 0.2, "maxCompletionPrice": 0.4},
        },
    )
    result = await provider.translate_batch(batch(), config_override=config)
    body = wire.completions[0]
    assert body["provider"]["only"] == ["a"]
    assert set(body["provider"]["ignore"]) == {"b", "a/priority"}
    assert body["provider"]["allow_fallbacks"] is False
    assert quoted_prices(body) == {"prompt": 0.2, "completion": 0.4, "request": 0}
    authenticated = [
        request for request in wire.requests if not request.url.path.endswith("/models")
    ]
    assert all(
        request.headers["Authorization"] == "Bearer synthetic-job-key" for request in authenticated
    )
    assert provider.client.headers["Authorization"] == "Bearer synthetic-default-key"
    assert result.routing_diagnostics["bootstrap"] is False
    assert result.routing_diagnostics["reported_cost"] == 0.001


@pytest.mark.parametrize(
    "field,value",
    [
        ("maxPromptPrice", -1),
        ("maxCompletionPrice", float("inf")),
        ("medianPremiumPercent", float("nan")),
        ("speedTolerancePercent", 1001),
        ("sparsePremiumMultiplier", 0.5),
        ("unexpected", 1),
    ],
)
def test_policy_validation_rejects_unsafe_limits(field, value):
    with pytest.raises(ValidationError):
        ProviderConfig(smartFast={field: value})


@pytest.mark.parametrize(
    "model,config",
    [
        ("fixture/model:smartfast", {"provider": {"order": ["a"]}}),
        ("fixture/model:smartfast", {"provider": {"sort": "nitro"}}),
        ("fixture/model:nitro", {"provider": {"sort": "smartfast"}}),
        ("fixture/model:nitro:smartfast", {}),
        ("fixture/model:smartfast:floor", {}),
        ("fixture/model:smartfast:smartfast", {}),
    ],
)
async def test_conflicting_routing_fails_before_completion(environment, model, config):
    wire, provider, _ = environment
    with pytest.raises((TranslationProviderError, ValidationError)):
        await provider.translate_batch(
            batch(), model=model, config_override=TranslationConfig(**config)
        )
    assert wire.completions == []


async def test_empty_only_filter_cannot_expand_to_paid_pool(environment):
    wire, provider, _ = environment
    with pytest.raises(TranslationProviderError):
        await provider.translate_batch(
            batch(),
            model="fixture/model:smartfast",
            config_override=TranslationConfig(provider={"only": []}),
        )
    assert wire.completions == []


async def test_legacy_routing_has_no_new_wire_fields(environment):
    wire, provider, _ = environment
    await provider.translate_batch(batch())
    body = wire.completions[0]
    assert body["model"] == "fixture/model"
    assert body["provider"] == {"sort": "throughput"}
    assert "session_id" not in body
    assert not any(request.url.path.endswith("/endpoints") for request in wire.requests)


@pytest.mark.parametrize(
    "mode",
    [
        "partial",
        "invalid",
        "empty",
        "blank",
        "repaired",
        "trailing_comma",
        "invalid_escape",
        "outer_trailing_comma",
        "fenced_invalid_escape",
        "prose_invalid_escape",
        "duplicate_keys",
        "no_choices",
        "bad_choices",
        "bad_message",
        "bad_content",
    ],
)
async def test_unusable_output_keeps_endpoint_and_preserves_usage_without_speed_training(
    environment, mode
):
    wire, provider, _ = environment
    wire.reply_modes = [mode]
    try:
        result = await provider.translate_batch(batch(), model="fixture/model:smartfast")
    except InvalidResponseError as error:
        assert error.tokens_used == 1050
        assert error.cost == 0.001
    else:
        assert result.total_tokens == 1050
        assert result.cost == 0.001
        assert result.routing_diagnostics["success"] is False
    next_result = await provider.translate_batch(batch(), model="fixture/model:smartfast")
    assert wire.completions[0]["provider"]["only"] == ["a"]
    assert wire.completions[1]["provider"]["only"] == ["a"]
    assert (
        next_result.routing_diagnostics["estimated_seconds"]["a"]
        == next_result.routing_diagnostics["estimated_seconds"]["b"]
    )


async def test_cancellation_releases_local_reservation_and_disconnects(environment):
    wire, provider, _ = environment
    wire.completion_gate = asyncio.Event()
    task = asyncio.create_task(provider.translate_batch(batch(), model="fixture/model:smartfast"))
    await wire.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wire.disconnected.is_set()
    wire.completion_gate.set()
    await provider.translate_batch(batch(), model="fixture/model:smartfast")
    # A leaked reservation on a would send a new session to b.
    assert wire.completions[1]["provider"]["only"] == ["a"]


async def test_metadata_discovery_uses_translation_wall_clock_budget(environment):
    wire, provider, _ = environment
    wire.catalog_gate = asyncio.Event()
    provider.settings.request_timeout = 0.03
    try:
        with pytest.raises(ProviderTimeoutError):
            async with asyncio.timeout(0.5):
                await provider.translate_batch(batch(), model="fixture/model:smartfast")
    finally:
        wire.catalog_gate.set()
        await asyncio.sleep(0)
    assert wire.completions == []


async def test_sanitized_routing_diagnostics_do_not_contain_ownership_or_context(
    environment, caplog
):
    wire, provider, _ = environment
    with caplog.at_level(logging.DEBUG):
        result = await provider.translate_batch(
            batch(),
            model="fixture/model:smartfast",
            config_override=TranslationConfig(apiKey="synthetic-private-key"),
        )
    diagnostics = json.dumps(result.routing_diagnostics)
    for forbidden in [
        "synthetic-private-key",
        "private-media-title",
        "Original 0",
        wire.completions[0]["session_id"],
    ]:
        assert forbidden not in diagnostics
    assert "synthetic-private-key" not in caplog.text
    assert wire.completions[0]["session_id"] not in caplog.text


@pytest.mark.parametrize("origin", ["config", "top-level", "default"])
async def test_sync_operation_retains_one_identity_without_mutating_reused_options(
    environment, origin
):
    from subtitle_translator.api.models import TranslateContentRequest

    wire, provider, settings = environment
    model = "fixture/model:smartfast"
    options = TranslationConfig(**({"model": model} if origin == "config" else {}))
    if origin == "default":
        settings.openrouter_default_model = model
    request = TranslateContentRequest(
        sourceLanguage="English",
        targetLanguage="Hungarian",
        lines=[{"position": i, "line": f"Original {i}"} for i in range(operation_line_count())],
        config=options,
        model=model if origin == "top-level" else None,
    )
    translator = SubtitleTranslator(provider, settings)
    first = await translator.translate_content(request)
    second = await translator.translate_content(request)
    assert first.success and second.success
    sessions = [body["session_id"] for body in wire.completions]
    assert sessions[0] == sessions[1]
    assert sessions[2] == sessions[3]
    assert sessions[0] != sessions[2]
    assert options._smartfast_session_id is None
    assert "_smartfast_session_id" not in options.model_dump()


async def test_streaming_operations_do_not_share_session_with_reused_config(environment):
    wire, provider, settings = environment
    processor = BatchProcessor(provider, settings)
    options = TranslationConfig(
        model="fixture/model:smartfast", _smartfast_session_id="untrusted-owner"
    )
    for _ in range(2):
        outcomes = [
            result
            async for result, progress in processor.process_batches_stream(
                batch(4).lines,
                "English",
                "Hungarian",
                batch_size=2,
                config_override=options,
            )
        ]
        assert all(result.success for result in outcomes)
    sessions = [body["session_id"] for body in wire.completions]
    assert sessions[0] == sessions[1]
    assert sessions[2] == sessions[3]
    assert sessions[0] != sessions[2]
    assert "untrusted-owner" not in sessions


@pytest.mark.parametrize("mode", ["partial", "invalid", "empty", "blank"])
@pytest.mark.parametrize(
    "model,price",
    [
        ("fixture/model:smartfast", 0.1),
        ("fixture/model:smartfast", 0),
        ("fixture/model:free:smartfast", 0),
    ],
)
async def test_single_endpoint_adaptive_recovery_retains_identity_and_billed_usage(
    environment, mode, model, price
):
    from subtitle_translator.core.batch_sizing import get_batch_size_resolver

    wire, provider, settings = environment
    get_batch_size_resolver().reset()
    settings.batch_size = 12
    wire.endpoints = [endpoint("a", price)]
    wire.reply_modes = [mode]
    result = await BatchProcessor(provider, settings).process_all_batches(
        batch(12).lines,
        "English",
        "Hungarian",
        model=model,
        batch_size=12,
    )
    assert result.success
    assert len(result.all_translations) == 12
    assert len(wire.completions) >= 2
    assert len({body["session_id"] for body in wire.completions}) == 1
    assert wire.completions[0]["provider"]["only"] == ["a"]
    assert wire.completions[1]["provider"]["only"] == ["a"]
    assert all(line["content"].strip() for line in result.all_translations)
    request_lines = [json.loads(body["messages"][-1]["content"]) for body in wire.completions]
    assert len(request_lines[0]) == 12
    assert all(len(lines) < 12 for lines in request_lines[1:])
    if mode == "partial":
        assert {line["index"] for lines in request_lines[1:] for line in lines} == {
            str(i) for i in range(1, 12)
        }
    if mode == "blank":
        assert request_lines[1] == [{"index": "0", "content": "Original 0"}]
    assert result.total_tokens == 1050 * len(wire.completions)
    assert result.progress.total_cost == pytest.approx(0.001 * len(wire.completions))
    assert all(
        quoted_prices(body)
        == (
            {"prompt": 0, "completion": 0, "request": 0}
            if ":free:" in model
            else {"prompt": 1, "completion": 3, "request": 0}
        )
        for body in wire.completions
    )
    get_batch_size_resolver().reset()


def operation_line_count():
    from subtitle_translator.core.batch_sizing import get_batch_size_resolver

    return 2 * get_batch_size_resolver().resolve("fixture/model:smartfast")


def job_data(job_type):
    shared = {
        "sourceLanguage": "English",
        "targetLanguage": "Hungarian",
        "model": "fixture/model:smartfast",
    }
    if job_type == JobType.TRANSLATE_CONTENT:
        return {
            **shared,
            "lines": [
                {"position": i, "line": f"Original {i}"} for i in range(operation_line_count())
            ],
        }
    return {
        **shared,
        "content": "".join(
            f"{i + 1}\n00:00:00,000 --> 00:00:01,000\nOriginal {i}\n\n"
            for i in range(operation_line_count())
        ),
    }


async def test_concurrent_content_and_file_jobs_balance_new_sessions_and_pin_batches(environment):
    wire, provider, settings = environment
    wire.completion_gate = asyncio.Event()
    translator = SubtitleTranslator(provider, settings)
    manager = JobManager()
    types = [JobType.TRANSLATE_CONTENT, JobType.TRANSLATE_FILE]
    ids = [await manager.submit_job(job_data(kind), kind) for kind in types]
    workers = [process_content_translation_job, process_file_translation_job]
    tasks = [
        asyncio.create_task(worker(manager, job_id, translator))
        for worker, job_id in zip(workers, ids, strict=True)
    ]
    try:
        async with asyncio.timeout(2):
            while len(wire.completions) < 4:
                await asyncio.sleep(0.01)
        wire.completion_gate.set()
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert all(manager.get_job(job_id).status == JobStatus.COMPLETED for job_id in ids)
    by_session = {}
    for body in wire.completions:
        by_session.setdefault(body["session_id"], []).append(body["provider"]["only"])
    assert sorted(by_session.values()) == [[["a"], ["a"]], [["b"], ["b"]]]


@pytest.mark.parametrize(
    "job_type,worker",
    [
        (JobType.TRANSLATE_CONTENT, process_content_translation_job),
        (JobType.TRANSLATE_FILE, process_file_translation_job),
    ],
)
async def test_recovered_job_keeps_identity_and_ignores_supplied_session_scope(
    environment, tmp_path, job_type, worker, caplog
):
    from subtitle_translator.queue.job_store import JobStore

    wire, provider, settings = environment
    wire.completion_gate = asyncio.Event()
    store = JobStore(str(tmp_path / "jobs.db"))
    manager = JobManager()
    manager.set_store(store)
    data = job_data(job_type)
    data["config"] = {
        "_smartfast_session_id": "untrusted-owner",
        "model": "fixture/model:smartfast",
    }
    job_id = await manager.submit_job(data, job_type)
    manager.set_job_processing(job_id)
    with caplog.at_level(logging.INFO):
        task = asyncio.create_task(worker(manager, job_id, SubtitleTranslator(provider, settings)))
        await wire.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    first_session = wire.completions[0]["session_id"]
    assert first_session != "untrusted-owner"
    assert "untrusted-owner" not in caplog.text
    wire.completion_gate.set()
    store.close()
    restored_store = JobStore(str(tmp_path / "jobs.db"))
    restored = JobManager()
    restored.set_store(restored_store)
    assert await restored.recover_jobs() == 1
    restored_provider = OpenRouterProvider(settings)
    try:
        await worker(restored, job_id, SubtitleTranslator(restored_provider, settings))
        assert restored.get_job(job_id).status == JobStatus.COMPLETED
        assert {body["session_id"] for body in wire.completions} == {first_session}
    finally:
        await restored_provider.close()
        restored_store.close()


async def test_request_deadline_is_failed_feedback_and_keeps_sanitized_diagnostics(environment):
    wire, provider, settings = environment
    settings.request_timeout = 0.2
    wire.completion_gate = asyncio.Event()
    with pytest.raises(ProviderTimeoutError) as raised:
        await provider.translate_batch(batch(), model="fixture/model:smartfast")
    assert raised.value.routing_diagnostics["success"] is False
    assert wire.disconnected.is_set()
    wire.completion_gate.set()
    await provider.translate_batch(batch(), model="fixture/model:smartfast")
    assert wire.completions[1]["provider"]["only"] == ["b"]


@pytest.mark.parametrize("change", ["account", "context"])
async def test_failed_observation_cannot_contaminate_another_account_or_context(
    environment, change
):
    wire, provider, _ = environment
    wire.reply_modes = ["http429"]
    with pytest.raises(RateLimitError):
        await provider.translate_batch(batch(), model="fixture/model:smartfast")
    next_batch = batch()
    if change == "context":
        next_batch.target_language = "French"
    config = TranslationConfig(apiKey="synthetic-other-account") if change == "account" else None
    await provider.translate_batch(
        next_batch, model="fixture/model:smartfast", config_override=config
    )
    assert wire.completions[1]["provider"]["only"] == ["a"]


async def test_persisted_invalid_smartfast_config_fails_closed_in_file_worker(environment):
    wire, provider, settings = environment
    manager = JobManager()
    data = job_data(JobType.TRANSLATE_FILE)
    data["config"] = {"provider": {"sort": "smartfast", "smartFast": {"maxCompletionPrice": -1}}}
    job_id = await manager.submit_job(data, JobType.TRANSLATE_FILE)
    await process_file_translation_job(manager, job_id, SubtitleTranslator(provider, settings))
    assert manager.get_job(job_id).status == JobStatus.FAILED
    assert wire.completions == []


async def test_cancelled_metadata_waiter_does_not_leave_continuous_refresh_running(environment):
    wire, provider, settings = environment

    class ContinuousMetadata(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = asyncio.Event()
            self.chunks = 0
            self.stop = False

        async def __aiter__(self):
            while not self.stop:
                self.chunks += 1
                yield b" "
                await asyncio.sleep(0.005)

        async def aclose(self):
            self.closed.set()

    stream = wire.catalog_stream = ContinuousMetadata()
    settings.request_timeout = 0.03
    try:
        with pytest.raises(TranslationProviderError):
            await provider.translate_batch(batch(), model="fixture/model:smartfast")
        assert stream.chunks > 1
        async with asyncio.timeout(0.3):
            await stream.closed.wait()
        assert wire.completions == []
    finally:
        stream.stop = True
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("origin", ["config", "top-level", "default"])
async def test_invalid_stored_options_cannot_drop_smartfast_model_or_restrictions(
    environment, origin
):
    wire, provider, settings = environment
    manager = JobManager()
    data = job_data(JobType.TRANSLATE_FILE)
    data.pop("model")
    data["config"] = {
        "temperature": 9,
        "provider": {"only": ["b"], "allowFallbacks": False},
    }
    if origin == "config":
        data["config"]["model"] = "fixture/model:smartfast"
    elif origin == "top-level":
        data["model"] = "fixture/model:smartfast"
    else:
        settings.openrouter_default_model = "fixture/model:smartfast"

    job_id = await manager.submit_job(data, JobType.TRANSLATE_FILE)
    await process_file_translation_job(manager, job_id, SubtitleTranslator(provider, settings))

    assert manager.get_job(job_id).status == JobStatus.FAILED
    assert wire.completions == []


@pytest.mark.parametrize("origin", ["top-level", "default"])
async def test_non_object_stored_smartfast_config_is_rejected(environment, origin):
    wire, provider, settings = environment
    manager = JobManager()
    data = job_data(JobType.TRANSLATE_FILE)
    if origin == "default":
        data.pop("model")
        settings.openrouter_default_model = "fixture/model:smartfast"
    data["config"] = "invalid stored options"
    job_id = await manager.submit_job(data, JobType.TRANSLATE_FILE)
    await process_file_translation_job(manager, job_id, SubtitleTranslator(provider, settings))
    assert manager.get_job(job_id).status == JobStatus.FAILED
    assert wire.completions == []


async def test_invalid_legacy_stored_options_retain_existing_default_fallback(environment):
    wire, provider, settings = environment
    manager = JobManager()
    data = job_data(JobType.TRANSLATE_FILE)
    data["model"] = "fixture/model"
    data["config"] = {"temperature": 9}
    job_id = await manager.submit_job(data, JobType.TRANSLATE_FILE)
    await process_file_translation_job(manager, job_id, SubtitleTranslator(provider, settings))
    assert manager.get_job(job_id).status == JobStatus.COMPLETED
    assert wire.completions
    assert all(body["provider"] == {"sort": "throughput"} for body in wire.completions)


@pytest.mark.parametrize("mode", ["http429", "http503", "error", "transport"])
async def test_real_failures_quarantine_endpoint_and_allow_bounded_failover(environment, mode):
    wire, provider, settings = environment
    wire.reply_modes = [mode]
    if mode == "error":
        with pytest.raises(RateLimitError) as raised:
            await provider.translate_batch(batch(), model="fixture/model:smartfast")
        assert raised.value.routing_diagnostics["failure_status"] == 429
    result = await BatchProcessor(provider, settings).process_batch(
        batch(),
        0,
        model="fixture/model:smartfast",
        _deadline=asyncio.get_running_loop().time() + 0.5,
    )
    assert result.success
    assert [body["provider"]["only"] for body in wire.completions] == [["a"], ["b"]]
    assert all(
        quoted_prices(body) == {"prompt": 1, "completion": 3, "request": 0}
        for body in wire.completions
    )


@pytest.mark.parametrize(
    "model,price",
    [
        ("fixture/model:smartfast", 0.1),
        ("fixture/model:smartfast", 0),
        ("fixture/model:free:smartfast", 0),
    ],
)
async def test_single_endpoint_cooldown_waits_then_retries_with_same_session(
    environment, monkeypatch, model, price
):
    from subtitle_translator.providers import smartfast

    wire, provider, settings = environment
    wire.endpoints = [endpoint("a", price)]
    wire.reply_modes = ["http429"]
    monkeypatch.setattr(smartfast, "QUARANTINE_SECONDS", 0.02)
    started = asyncio.get_running_loop().time()
    result = await BatchProcessor(provider, settings).process_batch(
        batch(),
        0,
        model=model,
        _deadline=started + 0.5,
    )
    assert result.success
    assert asyncio.get_running_loop().time() - started >= 0.02
    assert len(wire.completions) == 2
    assert len({body["session_id"] for body in wire.completions}) == 1
    assert result.tokens_used == 2100
    assert result.cost == pytest.approx(0.002)

    assert all(
        quoted_prices(body)
        == (
            {"prompt": 0, "completion": 0, "request": 0}
            if ":free:" in model
            else {"prompt": 1, "completion": 3, "request": 0}
        )
        for body in wire.completions
    )


async def test_quarantine_wait_stops_at_existing_batch_deadline(environment, monkeypatch):
    from subtitle_translator.providers import smartfast

    wire, provider, settings = environment
    wire.endpoints = [endpoint("a")]
    wire.reply_modes = ["http429"]
    monkeypatch.setattr(smartfast, "QUARANTINE_SECONDS", 30)
    started = asyncio.get_running_loop().time()
    async with asyncio.timeout(0.3):
        result = await BatchProcessor(provider, settings).process_batch(
            batch(),
            0,
            model="fixture/model:smartfast",
            _deadline=started + 0.2,
        )
    assert not result.success
    assert "timeout budget" in result.error.lower()
    assert len(wire.completions) == 1
    assert result.tokens_used == 1050
    assert result.cost == 0.001


async def test_permanent_routing_rejection_reports_cause_without_retry_exhaustion(environment):
    wire, provider, settings = environment
    result = await BatchProcessor(provider, settings).process_batch(
        batch(),
        0,
        model="fixture/model:smartfast",
        config_override=TranslationConfig(provider={"only": ["absent"]}),
    )
    assert not result.success
    assert result.retries == 0
    assert "no healthy endpoint" in result.error
    assert "Max retries exceeded" not in result.error
    assert wire.completions == []


async def test_free_endpoint_failure_never_falls_through_to_paid_endpoint(environment, monkeypatch):
    from subtitle_translator.providers import smartfast

    wire, provider, settings = environment
    wire.endpoints = [endpoint("a", 0), endpoint("paid", 0.1)]
    wire.reply_modes = ["http429"]
    monkeypatch.setattr(smartfast, "QUARANTINE_SECONDS", 0.02)
    result = await BatchProcessor(provider, settings).process_batch(
        batch(),
        0,
        model="fixture/model:free:smartfast",
        _deadline=asyncio.get_running_loop().time() + 0.5,
    )
    assert result.success
    assert [body["provider"]["only"] for body in wire.completions] == [["a"], ["a"]]
    assert all(
        quoted_prices(body) == {"prompt": 0, "completion": 0, "request": 0}
        for body in wire.completions
    )
