"""Service capacity selection survives request validation and queued config loading."""

import json

import httpx
import pytest
from pydantic import ValidationError

from subtitle_translator.api.models import TranslationConfig
from subtitle_translator.providers.base import TranslationBatch
from subtitle_translator.providers.openrouter import OpenRouterProvider
from subtitle_translator.queue.worker import _extract_config_override_from_dict
from tests.test_batch_reliability import response, settings


@pytest.mark.parametrize("field", ["serviceTier", "service_tier"])
def test_tier_validation_and_persisted_round_trip(field):
    config = TranslationConfig(**{field: "default"})
    assert config.service_tier == "default"
    restored = _extract_config_override_from_dict(config.model_dump(exclude_none=True))
    assert restored.service_tier == "default"
    assert _extract_config_override_from_dict({field: "default"}).service_tier == "default"
    with pytest.raises(ValidationError):
        TranslationConfig(**{field: "automatic"})


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", [None, "default", "flex", "priority"])
@pytest.mark.parametrize("route", ["floor", "nitro"])
async def test_tier_reaches_provider_without_changing_route(monkeypatch, tier, route):
    sent = []
    client_class = httpx.AsyncClient

    async def send(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        sent.append(json.loads(request.content))
        return response([{"index": "1", "content": "translated"}])

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(send)),
    )
    provider = OpenRouterProvider(settings())
    try:
        config = TranslationConfig(serviceTier=tier, provider={"sort": route})
        restored = _extract_config_override_from_dict(config.model_dump(exclude_none=True))
        await provider.translate_batch(
            TranslationBatch([{"index": "1", "content": "source"}], "en", "hu"),
            model="fixture/model",
            config_override=restored,
        )
        assert sent[0]["model"] == f"fixture/model:{route}"
        if tier is None:
            assert "service_tier" not in sent[0]
        else:
            assert sent[0]["service_tier"] == tier
    finally:
        await provider.close()
