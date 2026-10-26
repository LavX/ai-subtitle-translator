"""Back ends are found through entry points, per scope, without importing out-of-scope code."""

import pytest

from subtitle_translator.core.backends import (
    BUILTIN_DISTRIBUTION,
    BackendRegistry,
    BackendScope,
    Connection,
    FieldType,
    RegisteredBackend,
    SettingField,
    normalize_distribution,
)
from subtitle_translator.core.errors import DependencyUnavailableError, InvalidConfigError
from tests.core import registry_fakes as fakes


@pytest.fixture(autouse=True)
def _reset():
    fakes.LOADED.clear()


def test_builtin_scope_never_loads_another_distribution():
    registry = BackendRegistry.discover(
        BackendScope.builtin(),
        fakes.entry_points_from(
            fakes.FakeEntryPoint("echo", fakes.EchoBackend, dist="lavx_ai_subtitle_translator"),
            fakes.FakeEntryPoint("signin", fakes.SignInBackend, dist="lavx-translator-harness"),
        ),
    )
    assert registry.types() == ["echo"]
    assert fakes.LOADED == ["echo"]  # the extension's module was never imported


def test_extension_scope_serves_only_that_extension():
    registry = BackendRegistry.discover(
        BackendScope.extension("lavx-translator-harness"),
        fakes.entry_points_from(
            fakes.FakeEntryPoint("echo", fakes.EchoBackend, dist=BUILTIN_DISTRIBUTION),
            fakes.FakeEntryPoint("signin", fakes.SignInBackend, dist="Lavx.Translator_Harness"),
        ),
    )
    assert registry.types() == ["signin"] and fakes.LOADED == ["signin"]
    assert registry.get("signin").uses_sign_in is True


def test_everything_scope_and_a_broken_back_end_does_not_hide_the_others():
    registry = BackendRegistry.discover(
        BackendScope.everything(),
        fakes.entry_points_from(
            fakes.FakeEntryPoint("echo", fakes.EchoBackend, dist=BUILTIN_DISTRIBUTION),
            fakes.FakeEntryPoint("broken", fakes.EchoBackend, dist="other", explode=True),
            fakes.FakeEntryPoint("misnamed", fakes.Misnamed, dist="other"),
        ),
    )
    entries = {entry.type_id: entry for entry in registry.entries()}
    assert entries["echo"].backend is not None and entries["echo"].builtin
    assert entries["broken"].backend is None and "ImportError" in entries["broken"].error
    assert not entries["broken"].refused  # a load failure is not a refusal
    assert "names type" in entries["misnamed"].error
    with pytest.raises(DependencyUnavailableError):
        registry.get("broken")
    with pytest.raises(InvalidConfigError):
        registry.get("nothing-like-this")


def test_a_builtin_type_id_cannot_be_taken_over():
    registry = BackendRegistry.discover(
        BackendScope.everything(),
        fakes.entry_points_from(
            fakes.FakeEntryPoint("echo", fakes.EchoBackend, dist="squatter"),
            fakes.FakeEntryPoint("echo", fakes.EchoBackend, dist=BUILTIN_DISTRIBUTION),
        ),
    )
    winner = [e for e in registry.entries() if e.type_id == "echo" and e.backend is not None]
    assert len(winner) == 1 and winner[0].distribution == BUILTIN_DISTRIBUTION
    assert any(e.refused and "reserved for the built-in" in e.error for e in registry.entries())
    assert fakes.LOADED == ["echo"]  # the squatter's module was never imported


@pytest.mark.parametrize("squatter_first", [True, False])
def test_a_broken_builtin_still_owns_its_type_id(squatter_first):
    points = [
        fakes.FakeEntryPoint("echo", fakes.EchoBackend, dist="squatter"),
        fakes.FakeEntryPoint("echo", fakes.EchoBackend, dist=BUILTIN_DISTRIBUTION, explode=True),
    ]
    registry = BackendRegistry.discover(
        BackendScope.everything(), fakes.entry_points_from(*(points if squatter_first else points[::-1]))
    )
    with pytest.raises(DependencyUnavailableError):
        registry.get("echo")  # unavailable, never the squatter's
    assert registry.types() == [] and fakes.LOADED == ["echo"]


def test_a_built_in_entry_owns_its_type_id_in_either_order():
    broken = RegisteredBackend("echo", None, BUILTIN_DISTRIBUTION, "1", "ImportError: gone")
    squatter = RegisteredBackend("echo", fakes.EchoBackend(), "squatter", "1", None)
    for entries in ([broken, squatter], [squatter, broken]):
        with pytest.raises(DependencyUnavailableError):
            BackendRegistry(entries).get("echo")


def test_an_extension_scope_never_loads_a_built_in_type_id():
    registry = BackendRegistry.discover(
        BackendScope.extension("squatter"),
        fakes.entry_points_from(
            fakes.FakeEntryPoint("echo", fakes.EchoBackend, dist=BUILTIN_DISTRIBUTION),
            fakes.FakeEntryPoint("echo", fakes.EchoBackend, dist="squatter"),
        ),
    )
    assert registry.types() == [] and fakes.LOADED == []
    assert [entry.refused for entry in registry.entries()] == [True]
    with pytest.raises(InvalidConfigError):
        registry.get("echo")  # not served here, not a broken back end


def test_connection_repr_hides_secrets():
    connection = Connection(
        "c1", "echo", settings={"model": "m"}, secrets={"api_key": "sk-live-123456789"}
    )
    assert "sk-live" not in repr(connection)
    assert (
        connection.fingerprint()
        != Connection(
            "c1", "echo", settings={"model": "m"}, secrets={"api_key": "other"}
        ).fingerprint()
    )


def test_required_errors_and_secret_fields():
    backend = fakes.EchoBackend()
    key = SettingField("api_key", FieldType.SECRET, "API key", required=True)
    assert key.secret and not SettingField("model", FieldType.TEXT, "Model").secret
    backend.fields = lambda: [key]
    assert backend.required_errors({}, {}) == {"api_key": "API key is required"}
    assert backend.required_errors({}, {"api_key": "x"}) == {}


def test_migrate_is_identity_for_the_current_version_and_refuses_the_unknown():
    backend = fakes.EchoBackend()
    assert backend.migrate(1, {"model": "m"}) == {"model": "m"}
    with pytest.raises(InvalidConfigError):
        backend.migrate(2, {"model": "m"})


def test_distribution_names_are_normalized():
    assert normalize_distribution("Lavx.Translator_Harness") == "lavx-translator-harness"


def test_each_connection_has_its_own_sign_in_folder(tmp_path):
    from subtitle_translator.core.backends import BackendContext

    context = BackendContext(data_dir=tmp_path)
    assert context.sign_in_dir("claude-1") == tmp_path / "sign-in" / "claude-1"
    assert context.sign_in_dir("claude-2") != context.sign_in_dir("claude-1")
    assert BackendContext().sign_in_dir("claude-1") is None
    for bad in ("../escape", "", ".hidden", "a/b"):
        with pytest.raises(InvalidConfigError):
            context.sign_in_dir(bad)


@pytest.mark.asyncio
async def test_a_paid_test_call_is_held_and_settled_like_a_translation():
    from subtitle_translator.core.backends import BackendContext
    from subtitle_translator.core.errors import ErrorKind
    from subtitle_translator.core.settings import CoreSettings
    from subtitle_translator.core.usage import UsageMeter
    from tests.core.engine_fakes import ScriptedBackend, ScriptedProvider

    holds, settled = [], []

    async def reserve(amount, model):
        holds.append(amount)
        return f"hold-{len(holds)}"

    async def usage(report):
        settled.extend(report.reservations)

    provider = ScriptedProvider(price=0.02)
    backend = ScriptedBackend({"c1": provider})
    outcome = await backend.test(
        Connection("c1", "scripted"), CoreSettings(), BackendContext(), UsageMeter(1.0, usage, reserve)
    )
    assert outcome.ok and holds == [0.02] and settled == ["hold-1"]
    spent = ScriptedProvider(price=0.02)
    refused = await ScriptedBackend({"c1": spent}).test(
        Connection("c1", "scripted"), CoreSettings(), BackendContext(), UsageMeter(budget=0.0)
    )
    assert refused.ok is False and refused.kind is ErrorKind.CANCELLED and refused.reason == "budget_exhausted"
    assert spent.batches == []


@pytest.mark.asyncio
async def test_a_paid_test_call_that_timed_out_keeps_its_hold():
    from subtitle_translator.core.backends import BackendContext
    from subtitle_translator.core.errors import ErrorKind, ProviderTimeoutError
    from subtitle_translator.core.settings import CoreSettings
    from subtitle_translator.core.usage import UsageMeter
    from tests.core.engine_fakes import ScriptedBackend, ScriptedProvider

    settled = []

    async def reserve(amount, model):
        return "hold-1"

    async def usage(report):
        settled.extend(report.reservations)

    meter = UsageMeter(1.0, usage, reserve)
    provider = ScriptedProvider([ProviderTimeoutError("late")], price=0.02)
    outcome = await ScriptedBackend({"c1": provider}).test(
        Connection("c1", "scripted"), CoreSettings(), BackendContext(), meter
    )
    assert outcome.ok is False and outcome.kind is ErrorKind.DEPENDENCY_UNAVAILABLE
    assert meter.reserved == pytest.approx(0.02) and settled == []


def test_override_serves_a_configured_instance():
    registry = BackendRegistry.discover(
        BackendScope.everything(),
        fakes.entry_points_from(fakes.FakeEntryPoint("echo", fakes.EchoBackend, dist=BUILTIN_DISTRIBUTION)),
    )
    configured = fakes.EchoBackend()
    registry.override(configured)
    assert registry.get("echo") is configured
    assert [e.distribution for e in registry.entries() if e.type_id == "echo"] == [BUILTIN_DISTRIBUTION]
