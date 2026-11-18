"""The SDK classes delegate to the contract and map its errors onto the SDK's."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("bazarr_plugin_sdk")

from bazarr_plugin_sdk import PluginError  # noqa: E402
from bazarr_plugin_sdk.testing import run_conformance  # noqa: E402

from subtitle_translator.core.backends import BackendRegistry, RegisteredBackend  # noqa: E402
from subtitle_translator.plugin.sdk_binding import (  # noqa: E402
    BuiltinTranslatorPlugin,
    ExtensionTranslatorPlugin,
)
from tests.core.engine_fakes import ScriptedBackend, ScriptedProvider  # noqa: E402
from tests.plugin.test_translate import request  # noqa: E402


def test_builtin_plugin_describes_the_builtin_scope():
    facts = BuiltinTranslatorPlugin().describe()
    assert [t["type"] for t in facts["backend_types"]] == [
        "gemini",
        "openai_compatible",
        "openrouter",
    ]


def test_an_extension_subclass_names_its_distribution():
    class Harness(ExtensionTranslatorPlugin):
        distribution = "lavx-translator-harness"

    assert Harness().describe()["backend_types"] == []  # not installed here


def test_an_extension_plugin_without_a_distribution_is_refused():
    with pytest.raises(TypeError):
        ExtensionTranslatorPlugin()


def test_contract_errors_become_plugin_errors():
    plugin = BuiltinTranslatorPlugin()
    with pytest.raises(PluginError) as caught:
        plugin.validate(
            {"settings": {}, "connection": {"id": "c", "type": "vendor-cli", "settings": {}}}
        )
    assert caught.value.code == "invalid_config"
    assert caught.value.field_errors == {"connection.type": "this plugin does not serve vendor-cli"}


def scripted_plugin(provider):
    plugin = BuiltinTranslatorPlugin()
    plugin._contract._registry = BackendRegistry(
        [
            RegisteredBackend(
                "scripted",
                ScriptedBackend({"c1": provider}),
                "lavx-ai-subtitle-translator",
                "0",
                None,
            )
        ]
    )
    return plugin


def fake_context(**extra):
    calls = []

    async def reserve(amount, model):
        calls.append(("reserve", amount, model))
        return f"hold-{len(calls)}"

    async def usage(tokens, cost, model, reservations=()):
        calls.append(("usage", tokens, cost, model, tuple(reservations)))

    async def checkpoint(data):
        calls.append(("checkpoint",))

    ctx = SimpleNamespace(
        progress=lambda *args: None,
        heartbeat=lambda: None,
        reserve=reserve,
        usage=usage,
        checkpoint=checkpoint,
        cancelled=asyncio.Event(),
        heartbeat_interval=0.01,
        budget=None,
        data_dir=None,
        temp_dir=None,
        artifacts_dir=None,
        **extra,
    )
    return ctx, calls


@pytest.mark.asyncio
async def test_a_translation_reports_holds_and_usage_through_the_sdk_context():
    plugin = scripted_plugin(ScriptedProvider(price=0.02))
    ctx, calls = fake_context()
    result = await plugin.translate({**request(5), "budget": 1.0}, ctx)
    assert result["status"] == "complete"
    assert {c[0] for c in calls} >= {"reserve", "usage", "checkpoint"}
    usage = next(c for c in calls if c[0] == "usage")
    assert usage[4] == ("hold-1",) and usage[1] == 10


@pytest.mark.asyncio
async def test_a_call_closes_its_clients_and_keeps_the_back_end_for_the_next_call():
    provider = ScriptedProvider()
    plugin = scripted_plugin(provider)
    backend = plugin._contract.registry.get("scripted")

    def one_call_on_its_own_loop():
        ctx, _ = fake_context()
        return asyncio.run(plugin.translate(request(5), ctx))

    await asyncio.to_thread(one_call_on_its_own_loop)
    assert provider.closed and len(plugin._contract._engines) == 1
    provider.closed = False
    await asyncio.to_thread(one_call_on_its_own_loop)
    # The second call, on another event loop, reused what the first opened.
    assert backend.opened == ["c1"] and provider.closed


@pytest.mark.asyncio
async def test_a_failing_close_does_not_replace_a_finished_result():
    class BrokenClose(ScriptedProvider):
        async def close(self):
            raise RuntimeError("socket gone")

    plugin = scripted_plugin(BrokenClose())
    ctx, _ = fake_context()
    result = await plugin.translate(request(5), ctx)
    assert result["status"] == "complete"


def refusing_context(error):
    ctx, calls = fake_context()

    async def reserve(amount, model):
        calls.append(("reserve", amount))
        raise error

    ctx.reserve = reserve
    return ctx, calls


@pytest.mark.asyncio
async def test_a_hold_the_host_refuses_ends_the_attempt_as_a_budget_stop():
    provider = ScriptedProvider(price=0.02)
    ctx, calls = refusing_context(
        PluginError("cancelled", "the cost budget does not cover this call (month_spent)")
    )
    with pytest.raises(PluginError) as caught:
        await scripted_plugin(provider).translate(request(), ctx)
    assert caught.value.code == "cancelled" and caught.value.reason == "budget_exhausted"
    assert provider.batches == []
    # A refused hold is final: the remaining batches do not ask again.
    assert len([c for c in calls if c[0] == "reserve"]) == 1


@pytest.mark.asyncio
async def test_a_hold_that_fails_for_another_reason_keeps_its_own_kind():
    ctx, _ = refusing_context(PluginError("dependency_unavailable", "the host is busy"))
    with pytest.raises(PluginError) as caught:
        await scripted_plugin(ScriptedProvider(price=0.02)).translate(request(), ctx)
    assert caught.value.code == "dependency_unavailable"


@pytest.mark.asyncio
async def test_the_sdk_cancel_event_stops_a_translation():
    started = asyncio.Event()

    class Hanging(ScriptedProvider):
        async def translate_batch(self, batch, model=None, temperature=None, config_override=None):
            started.set()
            await asyncio.Event().wait()

    plugin = scripted_plugin(Hanging())
    ctx, _ = fake_context()
    task = asyncio.create_task(plugin.translate(request(), ctx))
    await started.wait()
    ctx.cancelled.set()
    with pytest.raises(PluginError) as caught:
        await asyncio.wait_for(task, 1)
    assert caught.value.code == "cancelled"


def test_conformance():
    run_conformance(
        BuiltinTranslatorPlugin,
        kind_api="bazarr.translator.v1",
        samples={
            "validate": {
                "args": {
                    "config": {
                        "settings": {},
                        "connection": {"id": "c", "type": "vendor-cli", "settings": {}},
                    }
                },
                "expect_error": "invalid_config",
            }
        },
    )
