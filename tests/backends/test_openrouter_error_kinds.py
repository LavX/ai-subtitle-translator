"""OpenRouter's failures land on the right kinds; SmartFast conflicts are configuration; a
transport failure after the connection opened may have been billed."""

import httpx
import pytest

from subtitle_translator.backends.openrouter import OpenRouterProvider, _route_refused_the_request
from subtitle_translator.backends.smartfast import SmartFastRoutingError
from subtitle_translator.core.errors import ErrorKind, TranslationProviderError, error_kind
from subtitle_translator.core.provider import TranslationBatch
from subtitle_translator.core.settings import CoreSettings


def test_smartfast_conflict_is_invalid_config_and_no_route_now_is_dependency():
    assert error_kind(SmartFastRoutingError("conflicts with provider order")) is ErrorKind.INVALID_CONFIG
    assert error_kind(SmartFastRoutingError("no endpoint", retry_after=30)) is ErrorKind.DEPENDENCY_UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,kind",
    [(402, ErrorKind.QUOTA_EXHAUSTED), (404, ErrorKind.INVALID_CONFIG), (503, ErrorKind.DEPENDENCY_UNAVAILABLE)],
)
async def test_envelope_codes_map_to_kinds(code, kind):
    provider = OpenRouterProvider(CoreSettings(openrouter_api_key="k"))
    with pytest.raises(TranslationProviderError) as caught:
        await provider._process_response(
            httpx.Response(code, json={"error": {"code": code, "message": "x"}}), "m"
        )
    assert error_kind(caught.value) is kind


def test_a_data_policy_404_from_a_pooled_route_is_a_route_refusal():
    error = TranslationProviderError(
        "OpenRouter API error 404: No endpoints found matching your data policy", status_code=404
    )
    error.routing_diagnostics = {"selected_endpoint": "a", "fast_pool": ["a", "b"]}
    assert _route_refused_the_request(error) is True
    plain = TranslationProviderError("OpenRouter API error 404: model not found", status_code=404)
    plain.routing_diagnostics = {"selected_endpoint": "a", "fast_pool": ["a", "b"]}
    assert _route_refused_the_request(plain) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,unknown",
    [
        (httpx.ConnectError, False),
        (httpx.ReadError, True),
        (httpx.RemoteProtocolError, True),
        (httpx.ReadTimeout, True),
    ],
)
async def test_a_transport_failure_after_connecting_may_have_been_billed(failure, unknown):
    def explode(request):
        raise failure(failure.__name__, request=request)

    provider = OpenRouterProvider(CoreSettings(openrouter_api_key="synthetic-test-only"))
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(explode), base_url="https://fake")
    provider._model_params_fetched = True
    batch = TranslationBatch(
        lines=[{"index": "0", "content": "Cue"}], source_language="en", target_language="hu"
    )
    try:
        with pytest.raises(TranslationProviderError) as caught:
            await provider.translate_batch(batch, model="test/model")
    finally:
        await provider._client.aclose()
    assert caught.value.billing_unknown is unknown
