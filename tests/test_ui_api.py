"""Standalone UI authentication, job ownership and restart boundaries."""

import json
from collections import OrderedDict
from unittest.mock import AsyncMock

import httpx
import pytest
from httpx import ASGITransport, AsyncClient, MockTransport, Response

from subtitle_translator import config, main, ui_api
from subtitle_translator.api import routes
from subtitle_translator.api.models import TranslateFileRequest
from subtitle_translator.config import Settings
from subtitle_translator.queue import worker
from subtitle_translator.queue.job_manager import JobManager, JobStatus, JobType
from subtitle_translator.queue.job_store import JobStore

FILE_REQUEST = {
    "content": "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
    "sourceLanguage": "en",
    "targetLanguage": "hu",
    "fileName": "example.srt",
}
KEY_A = "test-openrouter-key-a"
KEY_B = "test-openrouter-key-b"


def headers(key=KEY_A):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def ui_environment(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None,
        ui_enabled=True,
        debug=False,
        encryption_enabled=True,
        encryption_strict=True,
        openrouter_api_key="test-server-key",
    )
    monkeypatch.setattr(config, "_settings", settings)
    monkeypatch.setattr(config, "_runtime_overrides", {})
    monkeypatch.setattr(config, "_overridden_settings", None)
    monkeypatch.setattr(routes, "_auth_token", "test-legacy-auth")
    manager = JobManager()
    store = JobStore(db_path=str(tmp_path / "jobs.db"), crypto_key=b"a" * 32)
    manager.set_store(store)
    monkeypatch.setattr(routes, "job_manager", manager)
    network = {"calls": [], "status": 200, "malformed": False, "unreachable": False, "now": 0}

    def respond(request):
        if str(request.url) == "https://openrouter.ai/api/v1/models":
            assert "Authorization" not in request.headers
            return Response(
                network.get("catalog_status", 200),
                json={
                    "data": [
                        {
                            "id": "outside/curated-list",
                            "name": "External model",
                            "context_length": 12345,
                            "reasoning": network.get("reasoning"),
                        },
                        {"id": "openai/gpt-5.6-luna", "name": "OpenAI: GPT-5.6 Luna"},
                        {"id": "outside/curated-list", "name": "Duplicate"},
                    ]
                },
            )
        assert str(request.url) == "https://openrouter.ai/api/v1/key"
        assert request.method == "GET"
        network["calls"].append(request.headers["Authorization"])
        if network["unreachable"]:
            raise httpx.ConnectError(f"Unavailable: {KEY_A}", request=request)
        payload = (
            []
            if network["malformed"]
            else {"data": {"label": KEY_A, "is_free_tier": False}, "error": KEY_A}
        )
        return Response(network["status"], json=payload)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: AsyncClient(transport=MockTransport(respond))
    )
    monkeypatch.setattr(ui_api, "job_manager", manager)
    monkeypatch.setattr(ui_api, "monotonic", lambda: network["now"])
    monkeypatch.setattr(ui_api, "_validated_keys", OrderedDict())
    monkeypatch.setattr(ui_api, "_catalog_cache", None, raising=False)
    yield main.create_app(), manager, store, settings, network
    store.close()


@pytest.fixture
async def client(ui_environment):
    app, *_ = ui_environment
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


async def submit(client, key=KEY_A, **overrides):
    response = await client.post(
        "/ui/api/jobs/translate/file", headers=headers(key), json={**FILE_REQUEST, **overrides}
    )
    assert response.status_code == 200, response.text
    return response.json()["jobId"]


async def test_connect_validates_key_and_returns_no_provider_secrets(client, ui_environment):
    response = await client.post("/ui/api/connect", headers=headers())
    assert response.status_code == 200
    assert response.json()["connected"] is True
    assert len(response.json()["ownerScope"]) == 64
    assert KEY_A not in response.text
    assert "test-server-key" not in response.text
    assert "test-legacy-auth" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers
    assert ui_environment[4]["calls"] == [f"Bearer {KEY_A}"]


@pytest.mark.parametrize("authorization", [None, "Basic wrong", "Bearer "])
async def test_missing_bearer_cannot_use_server_key(client, ui_environment, authorization):
    request_headers = {"Authorization": authorization} if authorization is not None else {}
    response = await client.post(
        "/ui/api/jobs/translate/file", headers=request_headers, json=FILE_REQUEST
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["cache-control"] == "no-store"
    assert not ui_environment[1].jobs
    assert not ui_environment[4]["calls"]


@pytest.mark.parametrize(
    "provider_status, expected", [(401, 401), (403, 401), (429, 429), (500, 503)]
)
async def test_failed_validation_never_submits_or_echoes_key(
    client, ui_environment, provider_status, expected
):
    ui_environment[4]["status"] = provider_status
    for _ in range(2):
        response = await client.post(
            "/ui/api/jobs/translate/file", headers=headers(), json=FILE_REQUEST
        )
        assert response.status_code == expected
        assert KEY_A not in response.text
        assert response.headers["cache-control"] == "no-store"
    assert not ui_environment[1].jobs
    assert len(ui_environment[4]["calls"]) == 2


@pytest.mark.parametrize("failure", ["malformed", "unreachable"])
async def test_unusable_validation_response_fails_closed(client, ui_environment, failure):
    ui_environment[4][failure] = True
    response = await client.post("/ui/api/connect", headers=headers())
    assert response.status_code == 503
    assert KEY_A not in response.text


async def test_models_use_full_catalog_and_default_to_luna(client):
    assert (await client.get("/ui/api/models")).status_code == 401
    response = await client.get("/ui/api/models", headers=headers())
    assert response.status_code == 200
    assert response.json()["models"]
    assert [model["id"] for model in response.json()["models"]] == [
        "openai/gpt-5.6-luna",
        "outside/curated-list",
    ]
    assert response.json()["defaultModel"] == "openai/gpt-5.6-luna:floor"
    assert response.json()["models"][0]["isDefault"] is True
    assert all(model["id"] and model["name"] for model in response.json()["models"])
    assert response.headers["cache-control"] == "no-store"


async def test_catalog_failure_is_sanitized_without_losing_auth(client, ui_environment):
    ui_environment[4]["catalog_status"] = 503
    response = await client.get("/ui/api/models", headers=headers())
    assert response.status_code == 503
    assert KEY_A not in response.text
    assert (await client.post("/ui/api/connect", headers=headers())).status_code == 200


@pytest.mark.parametrize("mandatory", [True, False, None])
async def test_catalog_exposes_only_supported_reasoning_choices(client, ui_environment, mandatory):
    ui_environment[4]["reasoning"] = {
        "mandatory": mandatory,
        "supported_efforts": ["high", "low", "high", "unknown", {"private": KEY_A}],
        "private": KEY_A,
    }
    response = await client.get("/ui/api/models", headers=headers())
    assert response.status_code == 200
    model = response.json()["models"][1]
    assert model.get("reasoning") == {"mandatory": mandatory, "supportedEfforts": ["high", "low"]}
    assert KEY_A not in response.text
    assert "unknown" not in response.text


async def test_positive_validation_cache_is_hash_only_bounded_and_expires(client, ui_environment):
    network = ui_environment[4]
    assert (await client.post("/ui/api/connect", headers=headers())).status_code == 200
    assert (await client.get("/ui/api/models", headers=headers())).status_code == 200
    assert len(network["calls"]) == 1
    assert KEY_A not in repr(ui_api._validated_keys)
    assert len(next(iter(ui_api._validated_keys))) == 64
    network["now"] = 301
    assert (await client.post("/ui/api/connect", headers=headers())).status_code == 200
    assert len(network["calls"]) == 2
    for i in range(128):
        assert (
            await client.post("/ui/api/connect", headers=headers(f"test-distinct-key-{i}"))
        ).status_code == 200
    assert len(ui_api._validated_keys) == 128
    assert (await client.post("/ui/api/connect", headers=headers())).status_code == 200
    assert len(network["calls"]) == 131


async def test_submission_uses_only_bearer_and_persists_only_encrypted_key(client, ui_environment):
    job_id = await submit(
        client,
        config={"model": "test/model", "provider": {"sort": "floor"}},
        _ui_owner="forged-owner",
    )
    _, manager, store, _, _ = ui_environment
    job = manager.get_job(job_id)
    assert job.api_key_override == KEY_A
    dumped = json.dumps(job.request_data)
    assert KEY_A not in dumped
    assert "test-body-key" not in dumped
    assert "test-server-key" not in dumped
    assert job.request_data["_ui_owner"] != "forged-owner"
    assert len(job.request_data["_ui_owner"]) == 64
    assert job.request_data["config"]["provider"]["sort"] == "floor"
    assert store.load_job(job_id).api_key_override == KEY_A
    stored_key = store._conn.execute(
        "SELECT api_key_override FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()[0]
    assert stored_key.startswith("enc:")
    assert KEY_A not in stored_key
    assert "_ui_owner" not in TranslateFileRequest(**job.request_data).model_dump()


async def test_foreign_keys_and_legacy_jobs_are_not_accessible(client, ui_environment):
    job_id = await submit(client)
    manager = ui_environment[1]
    legacy_id = await manager.submit_job(
        job_type=JobType.TRANSLATE_FILE, request_data=FILE_REQUEST, api_key_override=KEY_A
    )
    for path, auth in [(job_id, headers(KEY_B)), (legacy_id, headers())]:
        for method in (client.get, client.delete):
            response = await method(f"/ui/api/jobs/{path}", headers=auth)
            assert response.status_code == 404
            assert response.headers["cache-control"] == "no-store"
    assert manager.get_job(job_id).status == JobStatus.QUEUED
    assert manager.get_job(legacy_id).status == JobStatus.QUEUED
    own = await client.get(f"/ui/api/jobs/{job_id}", headers=headers())
    assert own.status_code == 200
    assert KEY_A not in own.text
    assert "_ui_owner" not in own.text


@pytest.mark.parametrize("terminal", [False, True])
async def test_ui_cancellation_never_deletes_finished_results(client, ui_environment, terminal):
    job_id = await submit(client)
    manager = ui_environment[1]
    if terminal:
        manager.set_job_completed(job_id, {"content": "Translated subtitle"})
    response = await client.delete(f"/ui/api/jobs/{job_id}?onlyQueued=true", headers=headers())
    assert response.status_code == 200
    assert response.json()["status"] == ("completed" if terminal else "cancelled")
    assert manager.get_job(job_id) is not None


async def test_openrouter_bearer_does_not_bypass_legacy_auth(client):
    response = await client.get("/api/v1/jobs", headers=headers())
    assert response.status_code == 401


async def test_body_provider_key_is_rejected_without_echo(client, ui_environment):
    response = await client.post(
        "/ui/api/jobs/translate/file",
        headers=headers(),
        json={**FILE_REQUEST, "config": {"apiKey": KEY_B}},
    )
    assert response.status_code == 422
    assert KEY_B not in response.text
    assert not ui_environment[1].jobs


async def test_ui_job_list_filters_ownership_before_limit_and_counts(client, ui_environment):
    manager = ui_environment[1]
    first = await submit(client)
    manager.set_job_completed(first, {"content": "Translated subtitle"})
    second = await submit(client)
    await submit(client, KEY_B)
    await manager.submit_job(job_type=JobType.TRANSLATE_FILE, request_data=FILE_REQUEST)
    response = await client.get("/ui/api/jobs?limit=1", headers=headers())
    assert response.status_code == 200
    data = response.json()
    assert [job["jobId"] for job in data["jobs"]] == [second]
    assert (data["total"], data["queued"], data["processing"]) == (2, 1, 0)


async def test_public_legacy_endpoints_cannot_read_or_delete_ui_jobs(
    client, ui_environment, monkeypatch
):
    monkeypatch.setattr(routes, "_auth_token", None)
    manager = ui_environment[1]
    legacy_id = await manager.submit_job(job_type=JobType.TRANSLATE_FILE, request_data=FILE_REQUEST)
    job_id = await submit(client)
    manager.set_job_completed(job_id, {"content": "Private translated subtitle"})
    for method in (client.get, client.delete):
        assert (await method(f"/api/v1/jobs/{job_id}")).status_code == 404
    listed = (await client.get("/api/v1/jobs?limit=1")).json()
    assert [job["jobId"] for job in listed["jobs"]] == [legacy_id]
    assert (listed["total"], listed["queued"], listed["processing"]) == (1, 1, 0)
    assert (await client.get(f"/api/v1/jobs/{legacy_id}")).status_code == 200
    assert (await client.delete(f"/api/v1/jobs/{legacy_id}")).json()["status"] == "cancelled"
    assert manager.get_job(job_id).result == {"content": "Private translated subtitle"}


async def test_owned_job_visibility_survives_restart(client, ui_environment):
    job_id = await submit(client)
    manager = ui_environment[1]
    manager.set_job_completed(job_id, {"content": "Retained translated subtitle"})
    manager.jobs.clear()
    await manager.recover_jobs()
    own = await client.get(f"/ui/api/jobs/{job_id}", headers=headers())
    assert own.status_code == 200
    assert own.json()["result"]["content"] == "Retained translated subtitle"
    foreign = await client.get(f"/ui/api/jobs/{job_id}", headers=headers(KEY_B))
    assert foreign.status_code == 404


async def test_private_ui_validation_errors_do_not_echo_submitted_secrets(client):
    response = await client.post(
        "/ui/api/jobs/translate/file",
        headers=headers(),
        json={**FILE_REQUEST, "config": {"apiKey": KEY_A, "temperature": KEY_A}},
    )
    assert response.status_code == 422
    assert KEY_A not in response.text
    assert response.headers["cache-control"] == "no-store"


async def test_disabled_ui_has_no_api_routes(ui_environment):
    ui_environment[3].ui_enabled = False
    async with AsyncClient(
        transport=ASGITransport(app=main.create_app()), base_url="http://testserver"
    ) as client:
        for method, path in [
            (client.post, "/ui/api/connect"),
            (client.get, "/ui/api/models"),
            (client.post, "/ui/api/jobs/translate/file"),
            (client.get, "/ui/api/jobs/unknown"),
        ]:
            assert (await method(path, headers=headers())).status_code == 404


async def test_ui_api_works_under_proxy_root_path(ui_environment):
    async with AsyncClient(
        transport=ASGITransport(app=ui_environment[0], root_path="/translator"),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/translator/ui/api/connect", headers=headers())
    assert response.status_code == 200


@pytest.mark.parametrize("stored_key", [None, "enc:unavailable-key-material"])
async def test_recovered_ui_job_without_usable_key_never_gets_server_translator(
    ui_environment, monkeypatch, stored_key
):
    manager = ui_environment[1]
    job_id = await manager.submit_job(
        job_type=JobType.TRANSLATE_FILE,
        request_data={**FILE_REQUEST, "_ui_owner": "test-owner-fingerprint"},
        api_key_override=stored_key,
    )
    translator = AsyncMock(side_effect=AssertionError("Server translator must not be used"))
    monkeypatch.setattr(worker, "get_translator", translator)
    await worker.job_worker_handler(manager, job_id, JobType.TRANSLATE_FILE)
    job = manager.get_job(job_id)
    assert job.status == JobStatus.FAILED
    assert "OpenRouter key" in job.error
    assert not translator.called
