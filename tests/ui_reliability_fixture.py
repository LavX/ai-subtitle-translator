"""Local HTTP transport for browser checks through the real OpenRouter provider."""

import asyncio
import json

import httpx

from subtitle_translator.providers.openrouter import OpenRouterProvider


def install_provider(settings, evidence):
    if evidence:
        evidence.write_text("")
    client_class = httpx.AsyncClient
    attempts = {}
    models = [
        {
            "id": "openai/gpt-5.6-luna",
            "name": "GPT-5.6 Luna",
            "reasoning": {
                "mandatory": False,
                "supported_efforts": ["low", "high"],
            },
        },
        {
            "id": "fixture/mandatory",
            "name": "Mandatory reasoning",
            "reasoning": {
                "mandatory": True,
                "supported_efforts": ["low", "high"],
            },
        },
        {
            "id": "fixture/low-only",
            "name": "Low effort only",
            "reasoning": {
                "supported_efforts": ["low"],
            },
        },
        {
            "id": "anthropic/claude-sonnet-4.5",
            "name": "Catalog effort model",
            "reasoning": {
                "mandatory": False,
                "supported_efforts": ["low", "high"],
            },
        },
    ]
    for model in models:
        model["supported_parameters"] = ["reasoning"]

    def record(payload, lines, outcome):
        if evidence:
            with evidence.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "model": payload["model"],
                            "lineCount": len(lines),
                            "reasoning": payload.get("reasoning"),
                            "outcome": outcome,
                            "temperatureSent": "temperature" in payload,
                            "serviceTier": payload.get("service_tier"),
                            "provider": payload.get("provider", {}),
                        }
                    )
                    + "\n"
                )

    async def send(request):
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": models})
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            raise AssertionError("Unexpected fixture HTTP request")
        if request.headers.get("authorization") not in {
            "Bearer sk-or-v1-demo-not-a-real-key",
            "Bearer sk-or-v1-demo-second-key",
        }:
            raise AssertionError("Submitted UI key did not reach provider")
        payload = json.loads(request.content)
        lines = json.loads(payload["messages"][-1]["content"])
        key = tuple(line["content"] for line in lines)
        attempt = attempts[key] = attempts.get(key, 0) + 1
        if any("RATE_LIMIT" in line["content"] for line in lines) and attempt == 1:
            record(payload, lines, "rate-limited")
            return httpx.Response(
                200,
                headers={"Retry-After": "2"},
                json={
                    "error": {"code": 429, "message": "temporarily rate-limited upstream"},
                    "usage": {"total_tokens": 7, "cost": 0.0001},
                },
            )
        if any("RECOVER" in line["content"] for line in lines):
            if len(lines) > 5:
                if attempt == 1:
                    await asyncio.sleep(3)
                    record(payload, lines, "retryable-error")
                    return httpx.Response(
                        503, json={"error": {"message": "Temporary fixture failure"}}
                    )
                await asyncio.sleep(3)
                record(payload, lines, "timeout")
                raise httpx.ReadTimeout("Synthetic response timeout", request=request)
            await asyncio.sleep(5)
        elif any("TIMEOUT" in line["content"] for line in lines):
            await asyncio.sleep(3)
            record(payload, lines, "timeout")
            raise httpx.ReadTimeout("Synthetic response timeout", request=request)
        record(payload, lines, "success")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                [
                                    {"index": line["index"], "content": "HU: " + line["content"]}
                                    for line in lines
                                ]
                            )
                        }
                    }
                ],
                "model": payload["model"],
                "usage": {"total_tokens": 10, "cost": 0.001},
            },
        )

    # Every HTTP client in this disposable process uses the local transport.
    # Real request construction, response parsing, worker and API remain active.
    httpx.AsyncClient = lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(send))
    return OpenRouterProvider(settings)
