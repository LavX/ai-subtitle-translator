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


def test_a_keyless_request_with_routing_leaves_the_servers_data_retention_option_alone():
    registry = BackendRegistry.discover(BackendScope.builtin())
    run = single_provider_runtime(ScriptedProvider(), settings(openrouter_api_key="k"))
    run.connections = build_connections(
        settings(openrouter_api_key="k", openrouter_deny_data_collection=True), registry, environ={}
    )
    config = TranslationConfig(provider=ProviderConfig(order=["x"]))
    connections, _ = run.plan(config)
    server = next(c for c in connections if c.id == "openrouter")
    assert server.settings["deny_data_collection"] is True


def test_the_runtime_and_settings_reprs_hide_keys():
    run = single_provider_runtime(
        ScriptedProvider(), settings(openrouter_api_key="SECRET-or", gemini_api_keys="SECRET-g")
    )
    assert "SECRET" not in repr(run) and "SECRET" not in repr(run.settings)


def test_a_back_end_whose_setup_raises_is_skipped_not_fatal():
    class Exploding(fakes.EchoBackend):
        type_id = "exploding"

        def fields(self):
            raise RuntimeError("boom")

    registry = BackendRegistry.discover(
        BackendScope.everything(),
        fakes.entry_points_from(fakes.FakeEntryPoint("exploding", Exploding, dist="other")),
    )
    connections = build_connections(
        settings(openrouter_api_key="k", backend_order="exploding,openrouter"), registry, environ={}
    )
    assert [c.id for c in connections] == ["openrouter"]


def test_a_legacy_request_never_reaches_a_server_with_gemini_enabled():
    registry = BackendRegistry.discover(BackendScope.builtin())
    run = single_provider_runtime(ScriptedProvider(), settings(openrouter_api_key="server"))
    run.connections = build_connections(
        settings(openrouter_api_key="server", gemini_api_keys="g"), registry, environ={}
    )
    connections, pinned = run.plan(TranslationConfig(apiKey="user-key", model="x/y"))
    assert pinned == REQUEST_CONNECTION
    assert [c.id for c in connections] == [REQUEST_CONNECTION] and connections[0].backend_type == "openrouter"


@pytest.mark.asyncio
async def test_the_next_keyless_job_is_sent_after_the_only_connection_timed_out():
    # Default BACKEND_ORDER with Gemini off leaves one enabled connection. A read timeout
    # puts it into its default cooldown, and the next job must still be sent, as in 2.1.1.
    from subtitle_translator.core.errors import ProviderTimeoutError
    from subtitle_translator.server.translate import run_lines

    provider = ScriptedProvider([ProviderTimeoutError("read timed out")])
    run = single_provider_runtime(provider, settings(openrouter_api_key="server"))
    first = await run_lines(run, [Line(0, "Hello")], source="en", target="hu")
    assert first.status == "failed"
    sent = len(provider.batches)
    second = await run_lines(run, [Line(0, "Hello")], source="en", target="hu")
    assert second.status == "completed" and len(provider.batches) > sent


@pytest.mark.asyncio
async def test_the_next_keyless_job_is_still_attempted_after_a_402():
    from subtitle_translator.core.errors import QuotaExhaustedError
    from subtitle_translator.server.translate import run_lines

    provider = ScriptedProvider([QuotaExhaustedError("payment required")])
    run = single_provider_runtime(provider, settings(openrouter_api_key="server"))
    assert (await run_lines(run, [Line(0, "Hello")], source="en", target="hu")).status == "failed"
    sent = len(provider.batches)
    again = await run_lines(run, [Line(0, "Hello")], source="en", target="hu")
    assert again.status == "completed" and len(provider.batches) > sent


def test_settings_and_request_config_reprs_hide_every_secret():
    s = settings(
        openrouter_api_key="SECRET-or",
        gemini_api_keys="SECRET-g",
        encryption_key="SECRET-enc",
        admin_api_key="SECRET-admin",
    )
    for rendered in (repr(s), str(s)):
        assert "SECRET" not in rendered
    # hiding them from the repr must not change what is read or written
    assert s.encryption_key == "SECRET-enc" and s.admin_api_key == "SECRET-admin"
    assert s.model_dump()["admin_api_key"] == "SECRET-admin"
    config = TranslationConfig(apiKey="SECRET-user", model="x/y")
    assert "SECRET" not in repr(config) and "SECRET" not in str(config)
    assert config.api_key == "SECRET-user" and config.model_dump(by_alias=True)["apiKey"] == "SECRET-user"


def test_an_opened_back_end_does_not_show_a_request_key_in_its_repr():
    from subtitle_translator.core.backends import Connection

    run = single_provider_runtime(ScriptedProvider(), settings(openrouter_api_key="server"))
    connections, _ = run.plan(TranslationConfig(apiKey="SECRET-user", model="x/y"))
    entry = run.engine._open(connections[0])
    assert "SECRET" not in repr(entry) and "SECRET" not in repr(entry.opened) and "SECRET" not in repr(run.engine._opened)
    assert isinstance(connections[0], Connection)


LM_STUDIO_ENV = (
    "OPENAI_COMPATIBLE_PRESET=lmstudio\n"
    "OPENAI_COMPATIBLE_BASE_URL=http://192.168.1.10:1234/v1\n"
    "OPENAI_COMPATIBLE_MODEL=from-dotenv\n"
)


def test_openai_compatible_settings_in_the_dotenv_file_turn_the_back_end_on(tmp_path, monkeypatch):
    for name in ("PRESET", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"OPENAI_COMPATIBLE_{name}", raising=False)
    (tmp_path / ".env").write_text(LM_STUDIO_ENV)
    monkeypatch.chdir(tmp_path)
    registry = BackendRegistry.discover(BackendScope.builtin())
    connections = build_connections(settings(backend_order="openai_compatible"), registry)
    assert [(c.id, c.enabled) for c in connections] == [("openai_compatible", True)]
    assert connections[0].settings["model"] == "from-dotenv"


def test_the_process_environment_wins_over_the_dotenv_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(LM_STUDIO_ENV)
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "from-environment")
    monkeypatch.chdir(tmp_path)
    registry = BackendRegistry.discover(BackendScope.builtin())
    connections = build_connections(settings(backend_order="openai_compatible"), registry)
    assert connections[0].settings["model"] == "from-environment"


def test_a_blank_value_from_compose_counts_as_not_set(tmp_path, monkeypatch):
    # docker-compose passes ${VAR:-} as an empty string when the .env file has no such line.
    for name, value in {"PRESET": "lmstudio", "BASE_URL": "http://127.0.0.1:1234/v1", "MODEL": "m",
                        "PRICE_INPUT": "", "BATCH_SIZE": "", "API_KEY": ""}.items():
        monkeypatch.setenv(f"OPENAI_COMPATIBLE_{name}", value)
    monkeypatch.chdir(tmp_path)
    registry = BackendRegistry.discover(BackendScope.builtin())
    connection = build_connections(settings(backend_order="openai_compatible"), registry)[0]
    assert connection.enabled and connection.secrets == {}
    assert connection.settings["batch_size"] == 100 and connection.settings["price_input"] == 0.0


def test_no_dotenv_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("PRESET", "BASE_URL", "MODEL"):
        monkeypatch.delenv(f"OPENAI_COMPATIBLE_{name}", raising=False)
    registry = BackendRegistry.discover(BackendScope.builtin())
    connection = build_connections(settings(backend_order="openai_compatible"), registry)[0]
    assert not connection.enabled


@pytest.mark.asyncio
async def test_the_standalone_lists_models_from_its_configured_api_base(monkeypatch):
    import httpx

    from subtitle_translator.core.backends import BackendContext

    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(
        runtime_module, "get_settings", lambda: settings(openrouter_api_base="https://gateway.example/v1")
    )
    registry = runtime_module._registry(lambda _core: None)
    backend = registry.get("openrouter")
    backend._transport = httpx.MockTransport(handler)
    await backend.options("model", {}, {}, BackendContext())
    assert seen == ["https://gateway.example/v1/models"]


def _runtime_with_every_backend():
    registry = BackendRegistry.discover(BackendScope.builtin())
    run = single_provider_runtime(ScriptedProvider(), settings(openrouter_api_key="k"))
    run.connections = build_connections(
        settings(openrouter_api_key="k", gemini_api_keys="g"), registry, environ={}
    )
    return run


def _request_options(collection="deny", **extra):
    return TranslationConfig(
        model="x/y",
        temperature=0.9,
        requestTimeout=77.0,
        provider=ProviderConfig(only=["safe-host"], ignore=["bad-host"], dataCollection=collection),
        **extra,
    )


def test_a_backend_pin_to_openrouter_keeps_every_request_option():
    run = _runtime_with_every_backend()
    connections, pinned = run.plan(_request_options(backend="openrouter"))
    chosen = next(c for c in connections if c.id == pinned).settings
    assert chosen["only"] == ["safe-host"] and chosen["ignore"] == ["bad-host"]
    assert chosen["data_collection"] == "deny" and chosen["deny_data_collection"] is True
    assert chosen["model"] == "x/y" and chosen["temperature"] == 0.9
    assert chosen["request_timeout"] == 77.0


def test_a_backend_pin_to_another_backend_keeps_the_options_it_can_take():
    run = _runtime_with_every_backend()
    connections, pinned = run.plan(_request_options("allow", backend="gemini"))
    chosen = next(c for c in connections if c.id == pinned).settings
    assert pinned == "gemini" and chosen["model"] == "x/y" and chosen["temperature"] == 0.9
    # OpenRouter's routing vocabulary never reaches a back end that has no such thing.
    assert not {"only", "ignore", "data_collection", "request_timeout"} & set(chosen)


def test_a_backend_pin_to_an_openai_compatible_connection_keeps_its_model_and_parallelism():
    from subtitle_translator.core.backends import Connection

    run = _runtime_with_every_backend()
    run.connections = [*run.connections, Connection("local", "openai_compatible", settings={"model": "m0"})]
    connections, pinned = run.plan(_request_options("allow", backend="local", parallelBatches=3))
    chosen = next(c for c in connections if c.id == pinned).settings
    assert chosen["model"] == "x/y" and chosen["temperature"] == 0.9 and chosen["parallel_batches"] == 3
    assert not {"only", "ignore", "data_collection", "request_timeout"} & set(chosen)


@pytest.mark.parametrize("pin,kind", [("gemini", "gemini"), ("local", "openai_compatible")])
def test_a_pin_to_a_back_end_that_cannot_promise_no_collection_refuses_deny(pin, kind):
    from subtitle_translator.core.backends import Connection
    from subtitle_translator.core.errors import InvalidConfigError

    run = _runtime_with_every_backend()
    run.connections = [*run.connections, Connection("local", "openai_compatible", settings={"model": "m0"})]
    config = TranslationConfig(backend=pin, provider=ProviderConfig(dataCollection="deny"))
    with pytest.raises(InvalidConfigError) as caught:
        run.plan(config)
    assert kind in str(caught.value) and "dataCollection" in str(caught.value)
    assert "provider.dataCollection" in caught.value.field_errors


@pytest.mark.parametrize("pin", ["gemini", "local"])
def test_a_pin_to_another_back_end_still_accepts_allow_and_ignores_routing_lists(pin, caplog):
    from subtitle_translator.core.backends import Connection

    run = _runtime_with_every_backend()
    run.connections = [*run.connections, Connection("local", "openai_compatible", settings={"model": "m0"})]
    config = TranslationConfig(
        backend=pin,
        provider=ProviderConfig(only=["safe"], ignore=["bad"], dataCollection="allow"),
    )
    with caplog.at_level("WARNING"):
        connections, pinned = run.plan(config)
    chosen = next(c for c in connections if c.id == pinned).settings
    assert pinned == pin and "only" not in chosen and "ignore" not in chosen
    assert "OpenRouter only" in caplog.text


def test_a_pin_to_openrouter_still_takes_deny():
    run = _runtime_with_every_backend()
    config = TranslationConfig(backend="openrouter", provider=ProviderConfig(dataCollection="deny"))
    connections, pinned = run.plan(config)
    assert next(c for c in connections if c.id == pinned).settings["data_collection"] == "deny"
