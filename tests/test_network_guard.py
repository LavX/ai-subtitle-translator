"""The guard that keeps the suite off the internet has to work, or it protects nothing."""

import socket

import pytest


def test_an_outbound_lookup_is_blocked_and_recorded(no_outbound_network):
    with pytest.raises(OSError):
        socket.getaddrinfo("openrouter.ai", 443)
    assert no_outbound_network == ["openrouter.ai:443"]
    # Handled here, so this test does not fail itself at teardown.
    no_outbound_network.clear()


def test_an_outbound_connection_is_blocked_and_recorded(no_outbound_network):
    with pytest.raises(OSError):
        socket.create_connection(("openrouter.ai", 443))
    assert no_outbound_network == ["('openrouter.ai', 443)"]
    no_outbound_network.clear()


def test_loopback_is_left_alone(no_outbound_network):
    """A test's own stub server is not the internet."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
    socket.getaddrinfo("localhost", port)
    assert no_outbound_network == []


def test_a_provider_that_stands_its_catalog_down_reaches_nothing(no_outbound_network):
    """The line every provider-building test needs, and what it buys."""
    from unittest.mock import MagicMock

    from subtitle_translator.providers.openrouter import OpenRouterProvider

    settings = MagicMock()
    settings.openrouter_api_base = "https://openrouter.ai/api/v1"
    provider = OpenRouterProvider(settings=settings)
    provider._model_params_fetched = True

    import asyncio

    asyncio.run(provider._ensure_model_params_cache())
    assert no_outbound_network == []
