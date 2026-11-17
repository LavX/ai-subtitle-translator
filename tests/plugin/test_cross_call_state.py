"""State that outlives a call, with a real HTTP client on a different event loop each call."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

pytest.importorskip("bazarr_plugin_sdk")

from bazarr_plugin_sdk import PluginError  # noqa: E402

from subtitle_translator.plugin.sdk_binding import BuiltinTranslatorPlugin  # noqa: E402

OK = {
    "choices": [{"message": {"content": json.dumps({"translations": [{"index": "0", "content": "Szia"}]})}}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
}


class Server:
    """A loopback chat-completions server that answers from a script and records what it saw."""

    def __init__(self):
        self.seen = []
        self.script = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length") or 0))
                owner.seen.append(self.headers.get("authorization"))
                status, headers, body = owner.script.pop(0) if owner.script else (200, {}, OK)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    running = Server()
    yield running
    running.close()


def request(url):
    return {
        "job_id": "job-1",
        "source": {"code": "en", "name": "English"},
        "target": {"code": "hu", "name": "Hungarian"},
        "lines": [{"position": 0, "text": "Hello"}],
        "config": {
            "settings": {"max_retries": 1},
            "connection": {
                "id": "local",
                "type": "openai_compatible",
                "enabled": True,
                "settings": {"preset": "custom", "base_url": url, "model": "m"},
            },
        },
        "secrets": {"api_key": "sk-local"},
    }


def context():
    async def stored(*args, **kwargs):
        return None

    return SimpleNamespace(
        progress=lambda *args: None,
        heartbeat=lambda: None,
        reserve=stored,
        usage=stored,
        checkpoint=stored,
        cancelled=asyncio.Event(),
        heartbeat_interval=0.05,
        budget=None,
        data_dir=None,
        temp_dir=None,
        artifacts_dir=None,
    )


def call(plugin, url):
    """One translate call on an event loop of its own, as the SDK runs it."""
    return asyncio.to_thread(lambda: asyncio.run(plugin.translate(request(url), context())))


@pytest.mark.asyncio
async def test_a_real_http_client_is_reopened_on_every_new_event_loop(server):
    plugin = BuiltinTranslatorPlugin()
    for _ in range(3):
        result = await call(plugin, server.url)
        assert result["status"] == "complete" and result["lines"][0]["text"] == "Szia"
    assert server.seen == ["Bearer sk-local"] * 3


@pytest.mark.asyncio
async def test_a_rate_limited_key_stays_cooling_for_the_next_call(server):
    plugin = BuiltinTranslatorPlugin()
    await call(plugin, server.url)
    server.script.append((429, {"Retry-After": "3600"}, {"error": {"message": "slow down"}}))
    with pytest.raises(PluginError) as first:
        await call(plugin, server.url)
    assert first.value.code == "rate_limited"
    sent = len(server.seen)
    with pytest.raises(PluginError) as second:
        await call(plugin, server.url)
    # The key's cooldown outlived the call and its event loop: nothing was sent.
    assert second.value.code == "rate_limited" and len(server.seen) == sent


@pytest.mark.asyncio
async def test_a_learned_batch_size_outlives_the_call(server):
    plugin = BuiltinTranslatorPlugin()
    await call(plugin, server.url)
    (engine,) = plugin._contract._engines.values()
    (opened,) = engine._opened.values()
    resolver = opened.resolver
    resolver.record_failure("m", 100)
    learned = resolver.limit_planned_size("m", 100)
    assert learned < 100
    await call(plugin, server.url)
    (opened_again,) = engine._opened.values()
    assert opened_again.resolver is resolver and resolver.limit_planned_size("m", 100) == learned
