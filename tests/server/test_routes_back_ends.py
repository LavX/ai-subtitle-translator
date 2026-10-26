"""The translate routes accept a request when the server has any back end turned on, or when the
request brings its own OpenRouter key, and answer 401 otherwise."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from subtitle_translator.core.models import SubtitleLine
from subtitle_translator.core.translator import ContentTranslationResult, FileTranslationResult
from subtitle_translator.server import runtime as runtime_module
from subtitle_translator.server.api import routes
from subtitle_translator.server.config import Settings
from subtitle_translator.server.main import app
from subtitle_translator.server.queue.job_manager import job_manager

SRT = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
CONTENT = {"sourceLanguage": "en", "targetLanguage": "hu", "lines": [{"position": 1, "line": "Hello"}]}
FILE = {"content": SRT, "sourceLanguage": "en", "targetLanguage": "hu"}
ROUTES = [
    ("/api/v1/translate/content", CONTENT),
    ("/api/v1/translate/file", FILE),
    ("/api/v1/jobs/translate/content", CONTENT),
    ("/api/v1/jobs/translate/file", FILE),
]
LM_STUDIO = {
    "OPENAI_COMPATIBLE_PRESET": "lmstudio",
    "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:1234/v1",
    "OPENAI_COMPATIBLE_MODEL": "qwen3-8b",
}


@pytest.fixture
def server(monkeypatch):
    """A server with these settings and environment, checked by the real runtime. The
    translation is stubbed and the queued jobs never run: only the check is under test."""
    monkeypatch.setattr(routes, "_auth_token", None)
    monkeypatch.setattr(routes, "_crypto_key", None)
    translator = AsyncMock()
    translator.translate_content.return_value = ContentTranslationResult(
        lines=[SubtitleLine(position=1, line="Szia")], model_used="qwen3-8b", tokens_used=3
    )
    translator.translate_file.return_value = FileTranslationResult(
        content=SRT.replace("Hello", "Szia"), model_used="qwen3-8b", tokens_used=3, subtitle_count=1
    )
    monkeypatch.setattr(routes, "get_translator", AsyncMock(return_value=translator))

    def configure(environ: dict[str, str], **values) -> TestClient:
        for name in LM_STUDIO:
            monkeypatch.delenv(name, raising=False)
        for name, value in environ.items():
            monkeypatch.setenv(name, value)
        settings = Settings(_env_file=None, openrouter_api_key="", gemini_api_keys="", **values)
        monkeypatch.setattr(runtime_module, "get_settings", lambda: settings)
        monkeypatch.setattr(routes, "get_settings", lambda: settings)
        monkeypatch.setattr(runtime_module, "_runtime", None)
        monkeypatch.setattr(routes, "get_runtime", runtime_module.get_runtime)
        return TestClient(app)

    yield configure
    job_manager.jobs.clear()
    while not job_manager.queue.empty():
        job_manager.queue.get_nowait()


@pytest.mark.parametrize(("path", "body"), ROUTES)
def test_a_server_with_only_a_keyless_local_server_accepts_requests(server, path, body):
    client = server(LM_STUDIO, backend_order="openai_compatible")
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(("path", "body"), ROUTES)
def test_a_server_with_no_back_end_turned_on_answers_401(server, path, body):
    client = server({}, backend_order="openrouter,gemini,openai_compatible")
    response = client.post(path, json=body)
    assert response.status_code == 401
    assert response.json()["detail"] == {"error": "configuration_error", "message": routes.NO_BACK_END}


@pytest.mark.parametrize(("path", "body"), ROUTES)
def test_a_request_with_its_own_key_needs_no_server_back_end(server, path, body):
    client = server({}, backend_order="openrouter")
    response = client.post(path, json={**body, "config": {"apiKey": "sk-or-request"}})
    assert response.status_code == 200, response.text
