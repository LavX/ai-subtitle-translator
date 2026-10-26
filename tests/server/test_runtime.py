"""The standalone's connection list, request pinning and the translator surface the routes use."""

import pytest

from subtitle_translator.core.backends import BackendRegistry, BackendScope
from subtitle_translator.core.context import Language
from subtitle_translator.core.engine import EngineError, Line, TranslationJob
from subtitle_translator.core.errors import ErrorKind, RateLimitError
from subtitle_translator.core.models import ProviderConfig, SmartFastPolicy, TranslationConfig
from subtitle_translator.server import runtime as runtime_module
from subtitle_translator.server.config import Settings
from subtitle_translator.server.connections import build_connections, env_name
from subtitle_translator.server.runtime import REQUEST_CONNECTION, single_provider_runtime
from tests.core import registry_fakes as fakes
from tests.core.engine_fakes import ScriptedProvider


def settings(**kw):
    # Settings reads the process environment; these tests assert on the values they pass, so
    # an exported OPENROUTER_API_KEY or BACKEND_ORDER must not leak in.
    defaults = {
        "openrouter_api_key": "",
        "gemini_api_keys": "",
        "gemini_key_projects": "",
        "backend_order": "openrouter,gemini",
    }
    return Settings(_env_file=None, **{**defaults, **kw})


def test_env_names():
    assert env_name("vendor-cli", "model") == "VENDOR_CLI_MODEL"


def test_connections_follow_backend_order_and_are_on_only_when_configured():
    registry = BackendRegistry.discover(BackendScope.builtin())
    connections = build_connections(
        settings(openrouter_api_key="or-key", gemini_api_keys="g1, g2", gemini_key_projects="p,p",
                 backend_order="gemini,openrouter,nonsense"),
        registry,
        environ={},
    )
    assert [(c.id, c.enabled) for c in connections] == [("gemini", True), ("openrouter", True)]
    gemini = connections[0]
    assert gemini.secrets["api_keys"] == ["g1", "g2"] and gemini.settings["key_projects"] == ["p", "p"]
    assert "g1" not in repr(gemini)
    off = build_connections(settings(), registry, environ={})
    assert [(c.id, c.enabled) for c in off] == [("openrouter", False), ("gemini", False)]


def test_an_installed_back_end_is_configured_from_its_environment_variables():
    registry = BackendRegistry.discover(
        BackendScope.everything(),
        fakes.entry_points_from(fakes.FakeEntryPoint("signin", fakes.SignInBackend, dist="lavx-translator-harness")),
    )
    connections = build_connections(
        settings(backend_order="signin"), registry, environ={"SIGNIN_MODEL": "harness-2"}
    )
    assert connections[0].enabled and connections[0].settings["model"] == "harness-2"


def test_a_request_with_its_own_key_is_pinned_with_every_option():
    run = single_provider_runtime(ScriptedProvider(), settings(openrouter_api_key="server", gemini_api_keys="g"))
    legacy = TranslationConfig(
        apiKey="user-key",
        model="x/y:smartfast",
        provider=ProviderConfig(sort="smartfast", smartFast=SmartFastPolicy(maxPromptPrice=0.5), order=None),
        serviceTier="default",
    )
    connections, pinned = run.plan(legacy)
    assert pinned == REQUEST_CONNECTION and [c.id for c in connections] == [REQUEST_CONNECTION]
    assert connections[0].secrets == {"api_key": "user-key"}
    assert connections[0].settings["model"] == "x/y:smartfast"
    assert connections[0].settings["smart_fast"]["maxPromptPrice"] == 0.5


@pytest.mark.asyncio
async def test_a_legacy_request_is_pinned_to_its_own_key_and_never_fails_over():
    provider = ScriptedProvider([RateLimitError("slow", retry_after=120)])
    run = single_provider_runtime(provider, settings(openrouter_api_key="server"))
    connections, pinned = run.plan(TranslationConfig(apiKey="user-key", model="x/y"))
    job = TranslationJob(Language("en", "en"), Language("hu", "hu"), [Line(0, "Hello")])
    with pytest.raises(EngineError) as caught:
        await run.engine.translate(job, connections, pinned=pinned)
    assert caught.value.kind is ErrorKind.RATE_LIMITED and caught.value.failovers == []


def test_a_legacy_model_applies_to_openrouter_connections_only():
    registry = BackendRegistry.discover(BackendScope.builtin())
    run = single_provider_runtime(ScriptedProvider(), settings(openrouter_api_key="k"))
    run.connections = build_connections(settings(openrouter_api_key="k", gemini_api_keys="g"), registry, environ={})
    connections, pinned = run.plan(None, model="google/gemini-2.5-flash", temperature=0.1)
    by_id = {c.id: c for c in connections}
    assert pinned is None
    assert by_id["openrouter"].settings["model"] == "google/gemini-2.5-flash"
    assert by_id["gemini"].settings["model"] != "google/gemini-2.5-flash"


def test_backend_pins_a_configured_connection():
    run = single_provider_runtime(ScriptedProvider(), settings(openrouter_api_key="k"))
    assert run.plan(TranslationConfig(backend="openrouter")) == (run.connections, "openrouter")


@pytest.mark.asyncio
async def test_get_runtime_rebuilds_connections_but_keeps_the_engine(monkeypatch):
    first = settings(openrouter_api_key="a")
    second = settings(openrouter_api_key="b")
    current = {"value": first}
    monkeypatch.setattr(runtime_module, "get_settings", lambda: current["value"])
    monkeypatch.setattr(runtime_module, "_runtime", None)
    one = await runtime_module.get_runtime()
    opened = one.engine.provider_for(one.connections[0])
    current["value"] = second
    two = await runtime_module.get_runtime()
    assert two.engine is one.engine
    assert two.connections[0].secrets["api_key"] == "b"
    assert two.engine.provider_for(two.connections[0]) is opened    # one OpenRouter pool for every key
    await runtime_module.close_runtime()


@pytest.mark.asyncio
async def test_a_changed_timeout_reaches_every_connection(monkeypatch):
    from subtitle_translator.backends.openrouter import OpenRouterProvider

    probed = []

    async def probe(self):
        probed.append(self.client.timeout.read)
        return True

    monkeypatch.setattr(OpenRouterProvider, "health_check", probe)
    current = {"value": settings(openrouter_api_key="a", request_timeout=120)}
    monkeypatch.setattr(runtime_module, "get_settings", lambda: current["value"])
    monkeypatch.setattr(runtime_module, "_runtime", None)
    one = await runtime_module.get_runtime()
    before = one.engine.provider_for(one.connections[0]).settings.request_timeout
    current["value"] = settings(openrouter_api_key="a", request_timeout=300)
    two = await runtime_module.get_runtime()
    after = two.engine.provider_for(two.connections[0]).settings.request_timeout
    assert (before, after) == (120, 300) and two.engine is one.engine
    assert await runtime_module.EngineTranslator(two).health_check() is True
    assert probed == [300]                                  # the probe's client has it too
    await runtime_module.close_runtime()


@pytest.mark.asyncio
async def test_a_new_openrouter_key_leaves_the_other_connections_open(monkeypatch):
    current = {"value": settings(openrouter_api_key="a", gemini_api_keys="g")}
    monkeypatch.setattr(runtime_module, "get_settings", lambda: current["value"])
    monkeypatch.setattr(runtime_module, "_runtime", None)
    one = await runtime_module.get_runtime()
    gemini = one.engine.provider_for(next(c for c in one.connections if c.id == "gemini"))
    current["value"] = settings(openrouter_api_key="b", gemini_api_keys="g")
    two = await runtime_module.get_runtime()
    assert two.engine.provider_for(next(c for c in two.connections if c.id == "gemini")) is gemini
    assert next(c for c in two.connections if c.id == "openrouter").secrets["api_key"] == "b"
    await runtime_module.close_runtime()


@pytest.mark.asyncio
async def test_a_new_temperature_or_output_budget_reaches_requests_with_their_own_key(monkeypatch):
    current = {"value": settings(openrouter_api_key="a", gemini_api_keys="g",
                                 openrouter_temperature=0.3, openrouter_max_tokens=8000)}
    monkeypatch.setattr(runtime_module, "get_settings", lambda: current["value"])
    monkeypatch.setattr(runtime_module, "_runtime", None)
    one = await runtime_module.get_runtime()
    gemini = one.engine.provider_for(next(c for c in one.connections if c.id == "gemini"))
    current["value"] = settings(openrouter_api_key="a", gemini_api_keys="g",
                                openrouter_temperature=0.9, openrouter_max_tokens=2000)
    two = await runtime_module.get_runtime()
    assert two.engine.provider_for(next(c for c in two.connections if c.id == "gemini")) is gemini
    (request,), _ = two.plan(TranslationConfig(apiKey="user-key", model="x/y"))
    assert (request.settings["temperature"], request.settings["max_tokens"]) == (0.9, 2000)
    assert two.engine.provider_for(request).settings.openrouter_max_tokens == 2000
    # The engine was not reconfigured, so its base still has 0.3 and 8000; the connection's
    # own values are the ones sent.
    opened = two.registry.get("openrouter").open(request, one.core, two.engine.context_for("openrouter"))
    assert opened.config.temperature == 0.9 and opened.settings.openrouter_max_tokens == 2000
    (own,), _ = two.plan(TranslationConfig(apiKey="user-key", model="x/y", temperature=0.5), temperature=0.7)
    (routed,), _ = two.plan(TranslationConfig(apiKey="user-key", model="x/y"), temperature=0.7)
    assert (own.settings["temperature"], routed.settings["temperature"]) == (0.5, 0.7)
    await runtime_module.close_runtime()


@pytest.mark.asyncio
async def test_the_health_probe_uses_the_server_key(monkeypatch):
    from subtitle_translator.backends.openrouter import OpenRouterProvider

    probed = []

    async def probe(self):
        probed.append(self.settings.openrouter_api_key)
        return True

    monkeypatch.setattr(OpenRouterProvider, "health_check", probe)
    monkeypatch.setattr(runtime_module, "get_settings", lambda: settings(openrouter_api_key="server-key"))
    monkeypatch.setattr(runtime_module, "_runtime", None)
    translator = await runtime_module.get_translator()
    assert await translator.health_check() is True and probed == ["server-key"]
    await runtime_module.close_runtime()


def test_an_openai_compatible_server_is_configured_from_its_environment():
    registry = BackendRegistry.discover(BackendScope.builtin())
    connections = build_connections(
        settings(openrouter_api_key="k", backend_order="openai_compatible,openrouter"),
        registry,
        environ={
            "OPENAI_COMPATIBLE_PRESET": "lmstudio",
            "OPENAI_COMPATIBLE_BASE_URL": "http://192.168.1.10:1234/v1/",
            "OPENAI_COMPATIBLE_MODEL": "qwen3-8b",
        },
    )
    assert [(c.id, c.enabled) for c in connections] == [("openai_compatible", True), ("openrouter", True)]
    local = connections[0]
    assert local.settings["base_url"] == "http://192.168.1.10:1234/v1" and local.secrets == {}


def test_the_gemini_connection_carries_its_prices():
    registry = BackendRegistry.discover(BackendScope.builtin())
    priced = build_connections(
        settings(gemini_api_keys="k", gemini_price_input=1.5, gemini_price_output=3.0), registry, environ={}
    )
    gemini = next(c for c in priced if c.id == "gemini")
    assert (gemini.settings["price_input"], gemini.settings["price_output"]) == (1.5, 3.0)
    default = next(c for c in build_connections(settings(gemini_api_keys="k"), registry, environ={}) if c.id == "gemini")
    assert (default.settings["price_input"], default.settings["price_output"]) == (0.0, 0.0)
