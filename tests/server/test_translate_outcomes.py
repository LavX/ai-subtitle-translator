"""What the synchronous routes and run_lines make of an engine outcome: blank cues are not
translations, and a line the engine cannot hold is kept as source instead of failing the job."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from subtitle_translator.core.errors import InvalidResponseError
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.server.api import routes
from subtitle_translator.server.config import Settings
from subtitle_translator.server.main import app
from subtitle_translator.server.runtime import EngineTranslator, single_provider_runtime
from subtitle_translator.core.engine import Line
from subtitle_translator.server.translate import run_lines
from tests.core.engine_fakes import ScriptedProvider

CORE = CoreSettings(openrouter_api_key="server", max_retries=1, retry_delay=0.0, batch_size=2,
                    parallel_batches_per_job=1, request_timeout=5.0)
BAD = InvalidResponseError("garbled")
# one blank cue and two that the model keeps garbling: nothing is translated
BLANK_AND_TWO_BAD = [Line(0, "Hello"), Line(1, "   "), Line(2, "Bye")]


def runtime(provider):
    return single_provider_runtime(provider, CORE)


@pytest.mark.asyncio
async def test_a_blank_cue_does_not_count_as_a_translation():
    result = await run_lines(runtime(ScriptedProvider([BAD] * 20)), BLANK_AND_TWO_BAD, source="en", target="hu")
    assert result.status == "failed" and result.done == set()


@pytest.mark.asyncio
async def test_a_file_of_only_blank_cues_has_nothing_to_translate_and_completes():
    result = await run_lines(runtime(ScriptedProvider()), [Line(0, " "), Line(1, "")], source="en", target="hu")
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_the_sync_content_route_answers_500_when_only_blank_cues_came_back(monkeypatch):
    monkeypatch.setattr(routes, "_auth_token", None)
    monkeypatch.setattr(routes, "_crypto_key", None)
    translator = EngineTranslator(runtime(ScriptedProvider([BAD] * 20)))
    monkeypatch.setattr(routes, "get_translator", AsyncMock(return_value=translator))
    settings = Settings(_env_file=None, openrouter_api_key="server")
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "get_runtime", AsyncMock(return_value=translator.runtime))
    body = {
        "sourceLanguage": "en",
        "targetLanguage": "hu",
        "lines": [{"position": 0, "line": "Hello"}, {"position": 1, "line": "  "}, {"position": 2, "line": "Bye"}],
    }
    response = TestClient(app).post("/api/v1/translate/content", json=body)
    assert response.status_code == 500, response.text
    assert response.json()["detail"]["error"] == "translation_failed"
