"""Outgoing request regressions for runtime defaults and explicit reasoning."""

import asyncio
import json

import httpx
import pytest

from subtitle_translator import config
from subtitle_translator.api import routes
from subtitle_translator.api.models import ReasoningConfig, TranslationConfig
from subtitle_translator.core import translator as translator_module
from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.main import app
from subtitle_translator.providers.base import TranslationBatch
from subtitle_translator.providers.openrouter import OpenRouterProvider
from subtitle_translator.queue.job_manager import JobManager, JobStatus
from subtitle_translator.queue.worker import job_worker_handler
from tests.test_batch_reliability import response, settings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["openai/gpt-5", "anthropic/claude-sonnet-4.5", "future/model", "plain/model"]
)
@pytest.mark.parametrize("reasoning", [{"effort": "none"}, {"enabled": False}])
async def test_explicit_disable_reaches_http_and_keeps_json(monkeypatch, model, reasoning):
    requests = []
    client_class = httpx.AsyncClient

    async def send(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": model,
                            "supported_parameters": ["reasoning"],
                            "reasoning": {"mandatory": False, "supported_efforts": ["high", "low"]},
                        }
                    ]
                },
            )
        requests.append(json.loads(request.content))
        return response([{"index": "1", "content": "translated"}])

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(send)),
    )
    provider = OpenRouterProvider(settings())
    try:
        await provider.translate_batch(
            TranslationBatch([{"index": "1", "content": "source"}], "en", "hu"),
            model=model,
            config_override=TranslationConfig(reasoning=ReasoningConfig(**reasoning)),
        )
        assert requests[0]["reasoning"] == {"effort": "none"}
        assert requests[0]["response_format"] == {"type": "json_object"}
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["openai/gpt-5", "future/mandatory"])
async def test_mandatory_reasoning_fails_before_paid_request(monkeypatch, model):
    paid_requests = []
    client_class = httpx.AsyncClient

    async def send(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": model,
                            "supported_parameters": ["reasoning"],
                            "reasoning": {"mandatory": True},
                        }
                    ]
                },
            )
        paid_requests.append(request)
        return response([{"index": "1", "content": "translated"}])

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(send)),
    )
    provider = OpenRouterProvider(settings())
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": "1", "content": "source"}],
            "en",
            "hu",
            model=model,
            config_override=TranslationConfig(reasoning=ReasoningConfig(enabled=False)),
        )
        assert not result.success
        assert "mandatory" in result.batch_results[0].error.lower()
        assert not paid_requests
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["high", "medium"])
async def test_catalog_effort_selection_is_not_silently_ignored(monkeypatch, effort):
    sent = []
    client_class = httpx.AsyncClient

    async def send(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "anthropic/claude-sonnet-4.5",
                            "supported_parameters": ["reasoning"],
                            "reasoning": {"supported_efforts": ["high", "low"]},
                        }
                    ]
                },
            )
        sent.append(json.loads(request.content))
        return response([{"index": "1", "content": "translated"}])

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(send)),
    )
    provider = OpenRouterProvider(settings())
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": "1", "content": "source"}],
            "en",
            "hu",
            model="anthropic/claude-sonnet-4.5",
            config_override=TranslationConfig(reasoning=ReasoningConfig(effort=effort)),
        )
        if effort == "high":
            assert result.success
            assert sent[0].get("reasoning") == {"effort": "high"}
        else:
            assert not result.success
            assert "not supported" in result.batch_results[0].error
            assert not sent
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "use_thinking", "expected_model", "expected_reasoning"),
    [
        ("deepseek/deepseek-r1", True, "deepseek/deepseek-r1:thinking", None),
        ("deepseek/deepseek-r1:thinking", True, "deepseek/deepseek-r1:thinking", None),
        ("deepseek/deepseek-r1", False, "deepseek/deepseek-r1", {"effort": "high"}),
    ],
)
async def test_catalog_effort_preserves_explicit_thinking_variant_at_http(
    monkeypatch, model, use_thinking, expected_model, expected_reasoning
):
    sent = []
    client_class = httpx.AsyncClient

    async def send(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "deepseek/deepseek-r1",
                            "supported_parameters": ["reasoning", "temperature"],
                            "reasoning": {"supported_efforts": ["high", "low"]},
                        }
                    ]
                },
            )
        sent.append(json.loads(request.content))
        return response([{"index": "1", "content": "translated"}])

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(send)),
    )
    provider = OpenRouterProvider(settings())
    try:
        result = await provider.translate_batch(
            TranslationBatch([{"index": "1", "content": "source"}], "en", "hu"),
            config_override=TranslationConfig(
                model=model,
                useThinkingVariant=use_thinking,
                reasoning=ReasoningConfig(effort="high"),
            ),
        )
        assert result.translations == [{"index": "1", "content": "translated"}]
        assert len(sent) == 1
        assert sent[0]["model"] == expected_model
        assert sent[0].get("reasoning") == expected_reasoning
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_http_runtime_defaults_snapshot_running_and_queued_jobs(monkeypatch):
    config.reset_settings()
    initial = settings(openrouter_default_model="old/model", openrouter_temperature=0.2)
    monkeypatch.setattr(config, "_settings", initial)
    monkeypatch.setattr(translator_module, "_translator_instance", None)
    monkeypatch.setattr(translator_module, "_translator_lock", asyncio.Lock())
    manager = JobManager(max_concurrent=1)
    manager.set_worker_handler(job_worker_handler)
    monkeypatch.setattr(routes, "job_manager", manager)
    monkeypatch.setattr(routes, "_auth_token", None)
    clients = []
    sent = []
    old_started = asyncio.Event()
    old_release = asyncio.Event()
    new_pair = asyncio.Event()
    new_release = asyncio.Event()
    active = {"old": 0, "queued": 0, "explicit": 0}
    maxima = dict(active)
    client_class = httpx.AsyncClient
    key_names = {
        initial.openrouter_api_key: "old",
        "synthetic-rotated": "new",
        "synthetic-override": "explicit",
    }

    async def send(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        payload = json.loads(request.content)
        lines = json.loads(payload["messages"][-1]["content"])
        name = lines[0]["content"].split()[0]
        key_name = key_names.get(
            request.headers["authorization"].removeprefix("Bearer "), "unexpected"
        )
        sent.append((name, payload["model"], payload["temperature"], key_name))
        active[name] += 1
        maxima[name] = max(maxima[name], active[name])
        try:
            if name == "old":
                old_started.set()
                await old_release.wait()
            if name == "queued":
                if active[name] == 2:
                    new_pair.set()
                await new_release.wait()
            return response([{**x, "content": "translated"} for x in lines])
        finally:
            active[name] -= 1

    def make_client(**kwargs):
        client = client_class(**kwargs, transport=httpx.MockTransport(send))
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    api = client_class(transport=httpx.ASGITransport(app=app), base_url="http://test")
    try:
        assert (await api.get("/health")).status_code == 200
        ids = []
        for name in active:
            data = {
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [{"position": i, "line": f"{name} source"} for i in range(200)],
            }
            if name == "explicit":
                data["config"] = {
                    "model": "explicit/model",
                    "temperature": 0,
                    "parallelBatches": 1,
                    "apiKey": "synthetic-override",
                }
            result = await api.post("/api/v1/jobs/translate/content", json=data)
            assert result.status_code == 200
            ids.append(result.json()["jobId"])
        await manager.start_workers()
        await asyncio.wait_for(old_started.wait(), 2)
        result = await api.put(
            "/api/v1/config",
            json={
                "model": "new/model",
                "temperature": 0.8,
                "parallelBatchesPerJob": 2,
                "apiKey": "synthetic-rotated",
            },
        )
        assert result.status_code == 200
        current = (await api.get("/api/v1/config")).json()
        assert (current["model"], current["temperature"], current["parallelBatchesPerJob"]) == (
            "new/model",
            0.8,
            2,
        )
        assert all(not client.is_closed for client in clients)
        old_release.set()
        await asyncio.wait_for(new_pair.wait(), 2)
        new_release.set()
        await asyncio.wait_for(manager.queue.join(), 3)
        assert (
            sent
            == [("old", "old/model", 0.2, "old")] * 2
            + [("queued", "new/model", 0.8, "new")] * 2
            + [("explicit", "explicit/model", 0, "explicit")] * 2
        )
        assert maxima == {"old": 1, "queued": 2, "explicit": 1}
        assert all(manager.get_job(i).status == JobStatus.COMPLETED for i in ids)
    finally:
        old_release.set()
        new_release.set()
        await manager.stop_workers()
        await translator_module.close_translator()
        await api.aclose()
        config.reset_settings()
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["requestTimeout", "request_timeout"])
async def test_timeout_override_survives_file_config_and_reaches_http(field):
    from subtitle_translator.queue.worker import _extract_config_override_from_dict

    observed = []

    async def send(request):
        observed.append(request.extensions["timeout"])
        return response([{"index": "1", "content": "translated"}])

    provider = OpenRouterProvider(settings())
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    try:
        override = _extract_config_override_from_dict({field: 45})
        assert override.request_timeout == 45
        await provider.translate_batch(
            TranslationBatch([{"index": "1", "content": "source"}], "en", "hu"),
            config_override=override,
        )
        assert observed == [{"connect": 45, "read": 45, "write": 45, "pool": 45}]
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("file_job", [False, True])
@pytest.mark.parametrize("override", [None, "request", "config"])
async def test_job_start_persists_effective_model_before_provider_request(
    tmp_path, monkeypatch, file_job, override
):
    from subtitle_translator.queue.job_store import JobStore

    config.reset_settings()
    monkeypatch.setattr(config, "_settings", settings(openrouter_default_model="old/model"))
    monkeypatch.setattr(translator_module, "_translator_instance", None)
    monkeypatch.setattr(translator_module, "_translator_lock", asyncio.Lock())
    manager = JobManager(max_concurrent=1)
    store = JobStore(str(tmp_path / "models.db"))
    manager.set_store(store)
    manager.set_worker_handler(job_worker_handler)
    monkeypatch.setattr(routes, "job_manager", manager)
    monkeypatch.setattr(routes, "_auth_token", None)
    entered = asyncio.Event()
    release = asyncio.Event()
    outgoing = []
    initial_progress_models = []
    original_progress = manager.update_progress

    def progress(*args, **kwargs):
        initial_progress_models.append(store.load_job(args[0]).model)
        original_progress(*args, **kwargs)

    monkeypatch.setattr(manager, "update_progress", progress)
    client_class = httpx.AsyncClient

    async def send(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        payload = json.loads(request.content)
        outgoing.append(payload["model"])
        entered.set()
        await release.wait()
        lines = json.loads(payload["messages"][-1]["content"])
        return response([{**x, "content": "translated"} for x in lines])

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(send)),
    )
    api = client_class(transport=httpx.ASGITransport(app=app), base_url="http://test")
    data = {"sourceLanguage": "en", "targetLanguage": "hu"}
    if file_job:
        data["content"] = "1\n00:00:01,000 --> 00:00:02,000\nsource\n"
    else:
        data["lines"] = [{"position": 1, "line": "source"}]
    if override:
        data["model"] = "request/model"
    if override == "config":
        data["config"] = {"model": "config/model"}
    expected_model = f"{override}/model" if override else "new/model"
    try:
        route = "/api/v1/jobs/translate/file" if file_job else "/api/v1/jobs/translate/content"
        submitted = await api.post(route, json=data)
        assert submitted.status_code == 200
        job_id = submitted.json()["jobId"]
        updated = await api.put("/api/v1/config", json={"model": "new/model"})
        assert updated.status_code == 200
        await manager.start_workers()
        await asyncio.wait_for(entered.wait(), 2)
        active = (await api.get(f"/api/v1/jobs/{job_id}")).json()
        assert active["model"] == expected_model
        assert store.load_job(job_id).model == expected_model
        assert initial_progress_models[0] == expected_model
        assert outgoing == [expected_model]
        release.set()
        await asyncio.wait_for(manager.queue.join(), 2)
        terminal = (await api.get(f"/api/v1/jobs/{job_id}")).json()
        persisted = store.load_job(job_id)
        assert terminal["model"] == expected_model
        assert persisted.model == persisted.result["model_used"] == expected_model
    finally:
        release.set()
        await manager.stop_workers()
        await translator_module.close_translator()
        await api.aclose()
        store.close()
        config.reset_settings()
