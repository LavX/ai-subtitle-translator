"""The route suites give the server its OpenRouter key, or take it away, by patching
routes.get_settings. The translate routes decide from the runtime's connections whether the server
has a back end, so here the runtime they read follows that patched key. A test of the check itself
puts the real get_runtime back."""

from types import SimpleNamespace

import pytest

from subtitle_translator.core.backends import Connection


@pytest.fixture(autouse=True)
def route_runtime_follows_the_patched_key(monkeypatch):
    from subtitle_translator.server.api import routes

    async def runtime():
        key = getattr(routes.get_settings(), "openrouter_api_key", "")
        return SimpleNamespace(connections=[Connection("openrouter", "openrouter", enabled=bool(key))])

    monkeypatch.setattr(routes, "get_runtime", runtime)
