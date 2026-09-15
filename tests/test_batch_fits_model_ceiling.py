"""The first batch must fit what the model will actually write.

OPENROUTER_MAX_TOKENS sets the room the reply gets, but the provider trims that
to the model's own output ceiling before sending. Planning from the configured
budget alone hands a low-ceiling model a batch it can never answer: the first
reply is cut short, billed and discarded, and only then does the adaptive sizer
begin halving.

tencent/hy-mt2-1.8b publishes a 4096-token ceiling. At the default 8000-token
budget its first batch was planned at 80 and the job fell 80 -> 40 -> 20 -> 10
-> 5 -> 1, spending 51 single-cue requests to finish one film.
"""

import json

import httpx
import pytest

from subtitle_translator.config import Settings
from subtitle_translator.core.batch_processor import BatchProcessor
from subtitle_translator.core.batch_sizing import get_batch_size_resolver
from subtitle_translator.providers.openrouter import OpenRouterProvider


@pytest.fixture(autouse=True)
def reset_sizing(monkeypatch):
    resolver = get_batch_size_resolver()
    resolver.reset()
    # The resolver holds the global settings, not the provider's, so the budget
    # has to be pinned there or whatever an earlier test left behind decides it.
    monkeypatch.setattr(resolver._settings, "openrouter_max_tokens", 8000)
    monkeypatch.setattr(resolver._settings, "batch_size", 100)
    yield
    resolver.reset()


def response(lines):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"translations": [{**line, "content": "leforditva"} for line in lines]}
                        )
                    }
                }
            ],
            "usage": {"total_tokens": 10, "cost": 0.01},
        },
    )


def provider_with(ceiling, sizes):
    async def send(request):
        lines = json.loads(json.loads(request.content)["messages"][-1]["content"])
        sizes.append(len(lines))
        return response(lines)

    provider = OpenRouterProvider(
        Settings(
            _env_file=None,
            openrouter_api_key="synthetic-test-only",
            max_retries=1,
            retry_delay=0,
            parallel_batches_per_job=1,
            batch_size=100,
            openrouter_max_tokens=8000,
        )
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(send), base_url="https://fake"
    )
    provider._model_params_fetched = True
    if ceiling is not None:
        provider._model_max_output_cache["tencent/hy-mt2-1.8b"] = ceiling
    return provider


LINES = [{"index": str(i), "content": "source"} for i in range(160)]


@pytest.mark.asyncio
async def test_a_low_ceiling_model_gets_a_batch_it_can_answer():
    sizes = []
    provider = provider_with(4096, sizes)
    try:
        result = await BatchProcessor(provider, provider.settings).process_all_batches(
            LINES, "en", "hu", model="tencent/hy-mt2-1.8b"
        )
    finally:
        await provider.close()

    assert result.success
    assert sizes == [40, 40, 40, 40], sizes


@pytest.mark.asyncio
async def test_a_high_ceiling_model_still_plans_from_the_budget():
    sizes = []
    provider = provider_with(131072, sizes)
    try:
        await BatchProcessor(provider, provider.settings).process_all_batches(
            LINES, "en", "hu", model="tencent/hy-mt2-1.8b"
        )
    finally:
        await provider.close()

    assert sizes == [80, 80], sizes


@pytest.mark.asyncio
async def test_an_unknown_ceiling_leaves_planning_unchanged():
    sizes = []
    provider = provider_with(None, sizes)
    try:
        await BatchProcessor(provider, provider.settings).process_all_batches(
            LINES, "en", "hu", model="tencent/hy-mt2-1.8b"
        )
    finally:
        await provider.close()

    assert sizes == [80, 80], sizes
