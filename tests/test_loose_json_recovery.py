"""One malformed line from a loose model must not cost the lines around it.

A small model that answers with an unescaped quote in one cue used to fail
the whole batch, and once the batch was split, the first failed child ended
the split and left every sibling untranslated. Now the parser keeps the cues
before the break, the siblings are still attempted, and the result says which
cues stayed in the source language and why.
"""

import json

import httpx
import pytest

from subtitle_translator.api.models import SubtitleLine, TranslateContentRequest
from subtitle_translator.config import Settings
from subtitle_translator.core.batch_sizing import get_batch_size_resolver
from subtitle_translator.core.translator import SubtitleTranslator
from subtitle_translator.providers.openrouter import OpenRouterProvider

BAD_LINE = "7"


@pytest.fixture(autouse=True)
def reset_sizing():
    get_batch_size_resolver().reset()
    yield
    get_batch_size_resolver().reset()


def loose_reply(lines):
    """A model that puts an unescaped quote in one cue's translation."""
    parts = []
    for line in lines:
        if line["index"] == BAD_LINE:
            parts.append('{"index": "7", "content": "say "hi" now"}')
        else:
            parts.append(json.dumps({"index": line["index"], "content": f"T{line['index']}"}))
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": '{"translations": [' + ", ".join(parts) + "]}"}}],
            "usage": {"total_tokens": len(lines), "cost": len(lines) / 1000},
        },
    )


@pytest.mark.asyncio
async def test_a_bad_line_costs_only_the_child_batch_holding_it():
    calls = []

    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append((lines[0]["index"], len(lines)))
        return loose_reply(lines)

    provider = OpenRouterProvider(
        Settings(
            _env_file=None,
            openrouter_api_key="synthetic-test-only",
            max_retries=1,
            retry_delay=0,
            parallel_batches_per_job=1,
        )
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    provider._model_params_fetched = True
    translator = SubtitleTranslator(provider, provider.settings)
    try:
        result = await translator.translate_content(
            TranslateContentRequest(
                sourceLanguage="en",
                targetLanguage="hu",
                lines=[SubtitleLine(position=i, line=f"source {i}") for i in range(1, 21)],
            )
        )
    finally:
        await translator.close()

    # The 20-line reply keeps cues 1-6 before the break, so only 7-20 are sent
    # again as children of 10: the child holding cue 7 fails after its retry and
    # teaches the floor, and its sibling (17-20) is still sent.
    assert calls == [("1", 20), ("7", 10), ("7", 10), ("17", 4)]
    assert not result.success
    translated = {str(line.position) for line in result.lines if line.line.startswith("T")}
    assert translated == {str(i) for i in range(1, 7)} | {str(i) for i in range(17, 21)}
    assert len(result.lines) == 20
    assert "cues 7-16 at size 10" in result.error
    assert "Failed to parse JSON" in result.error
    assert '"hi"' in result.error
