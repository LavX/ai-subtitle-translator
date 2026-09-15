"""A cue with no text must never reach the provider.

Subtitle files carry blank cues: a numbered entry with timings and no dialogue.
There is nothing in one to translate, and a reply cannot contain a translation
for it, so the batch carrying it is judged short, split, retried down to a
single line and finally failed. The file then reports partial with a coverage
figure short by exactly the number of blank cues, even though every position
came back intact. The Matrix reference file has two of them.
"""

import json

import httpx
import pytest

from subtitle_translator.config import Settings
from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.batch_sizing import get_batch_size_resolver
from subtitle_translator.providers.openrouter import OpenRouterProvider


@pytest.fixture(autouse=True)
def reset_sizing():
    get_batch_size_resolver().reset()
    yield
    get_batch_size_resolver().reset()


def response(lines, tokens=10):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"translations": [{**line, "content": "translated"} for line in lines]}
                        )
                    }
                }
            ],
            "usage": {"total_tokens": tokens, "cost": 0.01},
        },
    )


@pytest.fixture
def provider():
    provider = OpenRouterProvider(
        Settings(
            _env_file=None,
            openrouter_api_key="synthetic-test-only",
            max_retries=1,
            retry_delay=0,
            parallel_batches_per_job=1,
        )
    )
    provider._model_params_fetched = True
    yield provider


def with_transport(provider, sent):
    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        sent.append(lines)
        # A blank cue cannot come back translated, so the stub answers only what a
        # real model could answer, which is every line that had text in it.
        return response([line for line in lines if line["content"].strip()])

    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    return provider


MIXED = [
    {"index": "0", "content": "Is everything in place?"},
    {"index": "1", "content": ""},
    {"index": "2", "content": "I know, but I felt like taking a shift."},
    {"index": "3", "content": "   "},
    {"index": "4", "content": "You like him, don't you?"},
]


@pytest.mark.asyncio
async def test_blank_cues_are_never_sent_to_the_provider(provider):
    sent = []
    with_transport(provider, sent)
    try:
        await BatchProcessor(provider, provider.settings).process_all_batches(
            MIXED, "en", "hu", model="test/blank", batch_size=100
        )
    finally:
        await provider.close()

    assert sent, "expected at least one request"
    for batch in sent:
        assert all(line["content"].strip() for line in batch), (
            f"a blank cue was sent to the provider: {batch}"
        )


@pytest.mark.asyncio
async def test_a_file_with_blank_cues_still_completes(provider):
    sent = []
    with_transport(provider, sent)
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            MIXED, "en", "hu", model="test/blank", batch_size=100
        )
    finally:
        await provider.close()

    assert result.success, [r.error for r in result.batch_results]
    assert len(sent) == 1, f"blank cues caused extra requests: {len(sent)} sent"


@pytest.mark.asyncio
async def test_blank_cues_come_back_at_their_own_positions(provider):
    sent = []
    with_transport(provider, sent)
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            MIXED, "en", "hu", model="test/blank", batch_size=100
        )
    finally:
        await provider.close()

    by_index = {t["index"]: t["content"] for t in result.all_translations}
    assert set(by_index) == {"0", "1", "2", "3", "4"}, "every position must be accounted for"
    assert by_index["1"] == ""
    assert by_index["3"] == "   ", "a blank cue is passed through exactly as it arrived"
    assert by_index["0"] == "translated"


@pytest.mark.asyncio
async def test_a_file_of_nothing_but_blank_cues_needs_no_request(provider):
    sent = []
    with_transport(provider, sent)
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            [{"index": "0", "content": ""}, {"index": "1", "content": "\n"}],
            "en",
            "hu",
            model="test/blank",
            batch_size=100,
        )
    finally:
        await provider.close()

    assert sent == [], "there was nothing to translate"
    assert result.success
    assert {t["index"] for t in result.all_translations} == {"0", "1"}
