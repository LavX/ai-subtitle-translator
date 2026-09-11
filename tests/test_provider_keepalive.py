"""Keepalive bytes cannot extend a translation request's wall-clock deadline."""

import asyncio
import json
from functools import partial

import httpx
import pytest

from subtitle_translator.config import Settings
from subtitle_translator.providers.base import ProviderTimeoutError, TranslationBatch
from subtitle_translator.providers.openrouter import OpenRouterProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [b"", b'{"choices":', b'{"choices": []}'])
async def test_continuous_body_is_cut_off_and_shared_client_remains_usable(prefix, monkeypatch):
    class ContinuousBody(httpx.AsyncByteStream):
        def __init__(self):
            self.close_count = 0
            self.active = False
            self.chunks = 0

        async def __aiter__(self):
            self.active = True
            try:
                yield prefix
                while True:
                    self.chunks += 1
                    yield b"                  "
                    await asyncio.sleep(0.005)
            finally:
                self.active = False

        async def aclose(self):
            self.close_count += 1

    body = ContinuousBody()
    posts = 0

    async def respond(request):
        nonlocal posts
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        posts += 1
        if posts == 1:
            return httpx.Response(200, stream=body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"translations": [{"index": "1", "content": "Szia."}]}
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    provider = OpenRouterProvider(
        Settings(
            _env_file=None,
            openrouter_api_key="synthetic-test-only",
            openrouter_default_model="test/model",
            request_timeout=0.2,
        )
    )
    client = provider._client = httpx.AsyncClient(
        base_url="https://openrouter.example/api/v1",
    )
    batch = TranslationBatch([{"index": "1", "content": "Hello."}], "en", "hu")
    try:
        # MockTransport keeps delivering bytes and supplies no inactivity timeout.
        # The outer guard makes a missing application deadline fail promptly.
        with pytest.raises(ProviderTimeoutError):
            async with asyncio.timeout(0.5):
                await provider.translate_batch(batch)
        assert body.chunks > 1
        assert body.close_count == 1
        assert not body.active
        assert not client.is_closed

        result = await provider.translate_batch(batch)
        assert result.translations == [{"index": "1", "content": "Szia."}]
        assert posts == 2
    finally:
        await provider.close()
