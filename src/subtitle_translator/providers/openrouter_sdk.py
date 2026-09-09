"""Official async SDK dispatch with the translator's raw-response semantics."""

import json
from collections.abc import Mapping
from typing import Any

import httpx
from openrouter import OpenRouter, components
from openrouter.errors import OpenRouterError
from openrouter.utils.logger import NoOpLogger


class _AsyncOnly:
    """Resource-free SDK sync-client boundary for this exclusively async adapter."""

    def build_request(self, *args: Any, **kwargs: Any) -> httpx.Request:
        raise RuntimeError("Synchronous OpenRouter requests are not supported")

    def send(self, *args: Any, **kwargs: Any) -> httpx.Response:
        raise RuntimeError("Synchronous OpenRouter requests are not supported")

    def close(self) -> None:
        pass


class _ResponseClient:
    """SDK custom HTTP client that borrows a pool and retains its raw response."""

    def __init__(self, client: httpx.AsyncClient, reasoning_extensions: dict[str, Any]):
        self.client = client
        self.reasoning_extensions = reasoning_extensions
        self.response: httpx.Response | None = None

    def build_request(self, *args: Any, **kwargs: Any) -> httpx.Request:
        request = self.client.build_request(*args, **kwargs)
        if not self.reasoning_extensions:
            return request
        # SDK 1.1.133 omits these supported OpenRouter reasoning fields. Restore
        # only those fields, preserving every other SDK-serialized request value.
        body = json.loads(request.content)
        body.setdefault("reasoning", {}).update(self.reasoning_extensions)
        headers = dict(request.headers)
        headers.pop("content-length", None)
        return httpx.Request(
            request.method,
            request.url,
            headers=headers,
            content=json.dumps(body, ensure_ascii=False).encode(),
            extensions=request.extensions,
        )

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        self.response = await self.client.send(request, **kwargs)
        return self.response

    async def aclose(self) -> None:
        # The provider owns the shared pool, including across concurrent jobs.
        pass


async def send_completion(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    server_url: str,
    payload: dict[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> httpx.Response:
    """Let the SDK send once; retain raw error envelopes and usage for parsing."""
    arguments = dict(payload)
    if isinstance(arguments.get("provider"), dict):
        provider = dict(arguments["provider"])
        if "max_price" in provider:
            # SDK prices are decimal strings. Explicit validation prevents its
            # optional union from silently treating invalid routing as unset.
            provider["max_price"] = {
                key: str(value) for key, value in provider["max_price"].items()
            }
        arguments["provider"] = components.ProviderPreferences(**provider)
    reasoning = arguments.get("reasoning") or {}
    captured = _ResponseClient(
        client, {key: reasoning[key] for key in ("max_tokens", "enabled") if key in reasoning}
    )
    # Prevent unused synchronous TLS-pool setup from blocking the event loop.
    # Both supplied clients remain caller-owned and SDK retries stay disabled.
    with OpenRouter(
        api_key=api_key,
        server_url=server_url,
        client=_AsyncOnly(),
        async_client=captured,
        retry_config=None,
        debug_logger=NoOpLogger(),
    ) as sdk:
        try:
            await sdk.chat.send_async(
                **arguments,
                retries=None,
                timeout_ms=max(1, round(timeout_seconds * 1000)),
                http_headers=headers,
            )
        except OpenRouterError as error:
            # Strict SDK response validation must not discard partial/error usage
            # or reinterpret embedded HTTP 200 errors before our existing parser.
            return error.raw_response
    if captured.response is None:
        raise RuntimeError("OpenRouter SDK returned no HTTP response")
    return captured.response
