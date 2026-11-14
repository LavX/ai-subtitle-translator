"""What the synchronous routes and run_lines make of an engine outcome: blank cues are not
translations, and a line the engine cannot hold is kept as source instead of failing the job."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from subtitle_translator.core.engine import Line
from subtitle_translator.core.errors import InvalidResponseError
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.server.api import routes
from subtitle_translator.server.config import Settings
from subtitle_translator.server.main import app
from subtitle_translator.server.runtime import EngineTranslator, single_provider_runtime
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


def _long_first(batch):
    from subtitle_translator.core.models import MAX_LINE_CHARS
    from subtitle_translator.core.provider import TranslationResult

    return TranslationResult(
        translations=[
            {"index": line["index"], "content": "x" * (MAX_LINE_CHARS - 1) if line["index"] == "0" else "ok"}
            for line in batch.lines
        ],
        model_used="scripted-model",
        total_tokens=10,
        cost=0.01,
    )


@pytest.mark.asyncio
async def test_a_checkpointed_line_the_rtl_marks_push_past_the_cap_keeps_its_source_after_a_stop():
    from subtitle_translator.core.errors import QuotaExhaustedError

    lines = [Line(i, f"l{i}") for i in range(4)]
    provider = ScriptedProvider([_long_first, QuotaExhaustedError("spent")])
    result = await run_lines(runtime(provider), lines, source="en", target="ar")
    assert result.status == "partial"
    assert result.lines[0].text == "l0" and 0 not in result.done
    assert result.lines[1].text.startswith("‫") and 1 in result.done


@pytest.mark.asyncio
async def test_the_sync_content_translator_returns_the_source_for_a_line_the_marks_overflow():
    from subtitle_translator.core.models import SubtitleLine, TranslateContentRequest

    translator = EngineTranslator(runtime(ScriptedProvider([_long_first])))
    request = TranslateContentRequest(
        sourceLanguage="en",
        targetLanguage="ar",
        lines=[SubtitleLine(position=i, line=f"l{i}") for i in range(2)],
    )
    result = await translator.translate_content(request)
    assert [item.line for item in result.lines][0] == "l0"
