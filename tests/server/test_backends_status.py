"""The back-end panel data: what is installed, what is on, and never a key."""

import pytest

from subtitle_translator.core.backends import BackendRegistry, BackendScope
from subtitle_translator.core.engine import TranslationEngine
from subtitle_translator.server.backends_status import backend_status
from subtitle_translator.server.config import Settings
from subtitle_translator.server.runtime import single_provider_runtime
from tests.core import registry_fakes as fakes
from tests.core.engine_fakes import ScriptedProvider


@pytest.mark.asyncio
async def test_status_lists_installed_back_ends_with_errors_and_sign_in():
    run = single_provider_runtime(ScriptedProvider(), Settings(_env_file=None, openrouter_api_key="sk-or-hidden"))
    run.registry = BackendRegistry.discover(
        BackendScope.everything(),
        fakes.entry_points_from(
            fakes.FakeEntryPoint("signin", fakes.SignInBackend, dist="lavx-translator-harness", version="0.1.0"),
            fakes.FakeEntryPoint("broken", fakes.EchoBackend, dist="other", explode=True),
        ),
    )
    status = await backend_status(run)
    by_type = {b["type"]: b for b in status["backends"]}
    assert by_type["signin"]["usesSignIn"] is True
    # The image ships no sign-in wrapper, so no command is advertised for the user to run.
    assert all("loginCommand" not in b for b in status["backends"])
    assert "translator-login" not in repr(status)
    assert by_type["signin"]["source"] == "lavx-translator-harness 0.1.0"
    assert by_type["broken"]["available"] is False and "ImportError" in by_type["broken"]["error"]
    assert "sk-or-hidden" not in repr(status)


@pytest.mark.asyncio
async def test_gemini_keys_are_reported_by_position_and_state():
    from subtitle_translator.server.connections import build_connections

    settings = Settings(_env_file=None, gemini_api_keys="AIza-one,AIza-two", backend_order="gemini")
    run = single_provider_runtime(ScriptedProvider(), settings)
    run.registry = BackendRegistry.discover(BackendScope.builtin())
    run.connections = build_connections(settings, run.registry, environ={})
    run.engine = TranslationEngine(run.registry, run.core)
    status = await backend_status(run)
    keys = status["connections"][0]["keys"]
    assert [(k["index"], k["state"]) for k in keys] == [(1, "ok"), (2, "ok")]
    assert "AIza" not in repr(status)


@pytest.mark.asyncio
async def test_sign_in_state_is_read_from_the_connection_sign_in_folder(tmp_path):
    from subtitle_translator.core.backends import Connection
    from subtitle_translator.server.runtime import _context_for

    settings = Settings(_env_file=None, backends_data_dir=str(tmp_path))
    run = single_provider_runtime(ScriptedProvider(), settings)
    run.registry = BackendRegistry.discover(
        BackendScope.everything(),
        fakes.entry_points_from(fakes.FakeEntryPoint("signin", fakes.SignInBackend, dist="lavx-translator-harness")),
    )
    run.engine = TranslationEngine(run.registry, run.core, _context_for(settings))
    run.connections = [Connection("signin", "signin", enabled=False)]
    assert (await backend_status(run))["connections"][0]["signedIn"] is False
    folder = tmp_path / "signin" / "sign-in" / "signin"
    folder.mkdir(parents=True)
    (folder / "signed-in").touch()
    assert (await backend_status(run))["connections"][0]["signedIn"] is True


def test_the_api_route_lists_back_ends_without_secrets(monkeypatch):
    from fastapi.testclient import TestClient

    from subtitle_translator.server import runtime as runtime_module
    from subtitle_translator.server.api import routes
    from subtitle_translator.server.main import app

    settings = Settings(_env_file=None, openrouter_api_key="sk-or-hidden", gemini_api_keys="AIza-one")
    monkeypatch.setattr(routes, "_auth_token", None)
    monkeypatch.setattr(runtime_module, "get_settings", lambda: settings)
    monkeypatch.setattr(runtime_module, "_runtime", None)
    response = TestClient(app).get("/api/v1/backends")
    assert response.status_code == 200
    body = response.json()
    assert {b["type"] for b in body["backends"]} >= {"openrouter", "gemini", "openai_compatible"}
    assert [c["id"] for c in body["connections"]] == ["openrouter", "gemini"]
    assert "sk-or-hidden" not in response.text and "AIza" not in response.text


@pytest.mark.asyncio
async def test_one_broken_connection_does_not_hide_the_others():
    from subtitle_translator.core.backends import Connection

    run = single_provider_runtime(ScriptedProvider(), Settings(_env_file=None, openrouter_api_key="k"))

    def broken(connection):
        raise RuntimeError("boom")

    run.engine.provider_for = broken
    run.connections = [Connection("openrouter", "openrouter", enabled=True), Connection("x", "openrouter", enabled=False)]
    status = await backend_status(run)
    assert [c["id"] for c in status["connections"]] == ["openrouter", "x"]
