"""Shared test setup.

The provider loads the OpenRouter model catalog on its own, so a test that builds
one without standing that fetch down reaches the real internet: it passes while the
network is healthy, then turns slow or red when it is not, and the failure surfaces
somewhere unrelated to the cause. The guard below makes that impossible to do by
accident, and names the host that escaped.

Raising at the call site is not enough on its own. The catalog fetch catches every
exception so a provider outage cannot break translation, which would swallow the
complaint and leave the test green. The attempt is recorded instead, and the test
fails when it ends.
"""

import socket

import pytest

# Loopback is a test's own stub server, not the internet.
_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


def _is_local(host: object) -> bool:
    if not isinstance(host, str):
        return False
    return host in _ALLOWED_HOSTS or host.startswith("127.")


@pytest.fixture(autouse=True)
def no_outbound_network(monkeypatch, request):
    """Fail any test that opens a connection off this machine.

    Mark a test ``@pytest.mark.allow_network`` to opt out; nothing in the suite
    needs it today, and a new one should be argued for rather than added quietly.
    """
    if request.node.get_closest_marker("allow_network"):
        yield []
        return

    escaped: list[str] = []
    real_getaddrinfo = socket.getaddrinfo
    real_create_connection = socket.create_connection

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        if not _is_local(host):
            escaped.append(f"{host}:{port}")
            raise OSError(f"blocked outbound connection to {host}:{port}")
        return real_getaddrinfo(host, port, *args, **kwargs)

    def guarded_create_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if not _is_local(host):
            escaped.append(str(address))
            raise OSError(f"blocked outbound connection to {address}")
        return real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)

    # Yielded so the guard's own test can inspect what it caught and clear it.
    yield escaped

    if escaped:
        pytest.fail(
            "test reached for the real internet: "
            + ", ".join(sorted(set(escaped)))
            + ". Stand the call down (a provider usually needs "
            "`_model_params_fetched = True`, or patch its client), or mark the test "
            "with @pytest.mark.allow_network.",
            pytrace=False,
        )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_network: this test is allowed to reach the real internet"
    )
