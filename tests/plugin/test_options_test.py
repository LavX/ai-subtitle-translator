# tests/plugin/test_options_test.py
"""options and test receive secrets, use one connection, and report sign-in state without credentials."""

import httpx
import pytest

from subtitle_translator.backends.gemini import GeminiBackendType
from subtitle_translator.core.backends import BackendRegistry, RegisteredBackend
from subtitle_translator.plugin.contract import ContractError, TranslatorContract
from subtitle_translator.plugin.scope import PluginScope
from tests.plugin.fake_dist import HARNESS, points

CONFIG = {"settings": {}, "connection": {"id": "g", "type": "gemini", "enabled": True, "settings": {}}}


def contract_with(backend):
    contract = TranslatorContract(PluginScope.builtin())
    contract._registry = BackendRegistry([RegisteredBackend(backend.type_id, backend, "lavx-ai-subtitle-translator", "0", None)])
    return contract


@pytest.mark.asyncio
async def test_options_use_the_secrets_they_are_given():
    def handler(request):
        assert request.headers["authorization"] == "Bearer from-secrets"
        return httpx.Response(200, json={"data": [{"id": "models/gemini-2.5-flash"}]})

    contract = contract_with(GeminiBackendType(transport=httpx.MockTransport(handler)))
    choices = await contract.options("connection.settings.model", CONFIG, {"api_keys": ["from-secrets"]})
    assert choices == [{"value": "gemini-2.5-flash", "label": "gemini-2.5-flash", "description": None,
                        "prompt_price": None, "completion_price": None, "context_length": None}]


@pytest.mark.asyncio
async def test_options_without_a_key_is_auth_required():
    contract = contract_with(GeminiBackendType(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    with pytest.raises(ContractError) as caught:
        await contract.options("connection.settings.model", CONFIG, {})
    assert caught.value.code == "auth_required"


@pytest.mark.asyncio
async def test_test_reports_a_rejected_key_as_a_result_not_a_crash():
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "API key not valid."}})

    contract = contract_with(GeminiBackendType(transport=httpx.MockTransport(handler)))
    result = await contract.test(CONFIG, {"api_keys": ["bad"]})
    assert result["ok"] is False and result["code"] == "auth_required" and "bad" not in repr(result)


@pytest.mark.asyncio
async def test_a_sign_in_back_end_reports_signed_in_or_out():
    contract = TranslatorContract(PluginScope.extension(HARNESS), points())
    backend = contract.registry.get("signin")

    async def signed_out(connection, context):
        from subtitle_translator.core.backends import SignInStatus

        return SignInStatus(False, "Run the login command")

    backend.sign_in_status = signed_out
    config = {"settings": {}, "connection": {"id": "s", "type": "signin", "enabled": True, "settings": {}}}
    result = await contract.test(config, {})
    assert result["signed_in"] is False and result["ok"] is False


@pytest.mark.asyncio
async def test_a_connection_the_back_end_cannot_open_is_invalid_config_naming_the_field():
    builtin = TranslatorContract(PluginScope.builtin())
    local = {"settings": {}, "connection": {"id": "oc", "type": "openai_compatible", "enabled": True,
                                              "settings": {"preset": "lmstudio"}}}
    with pytest.raises(ContractError) as caught:
        await builtin.test(local, {})
    assert caught.value.code == "invalid_config"
    assert set(caught.value.field_errors) == {"connection.settings.model"}
    projects = {**CONFIG, "connection": {**CONFIG["connection"], "settings": {"key_projects": ["a", "b"]}}}
    with pytest.raises(ContractError) as caught:
        await builtin.test(projects, {"api_keys": ["k"]})
    assert set(caught.value.field_errors) == {"connection.settings.key_projects"}


class Holds:
    """A sink that records holds and usage; test calls report nothing else."""

    def __init__(self):
        self.events = []

    def progress(self, done, total, message):
        pass

    def heartbeat(self):
        pass

    async def reserve(self, amount, model):
        self.events.append(("reserve", amount))
        return f"hold-{len(self.events)}"

    async def usage(self, report):
        self.events.append(("usage", report["tokens"], tuple(report["reservations"])))

    async def checkpoint(self, data):
        pass


@pytest.mark.asyncio
async def test_a_paid_test_is_held_and_reported_like_a_translation():
    from tests.core.engine_fakes import ScriptedBackend, ScriptedProvider

    scripted = {"settings": {}, "connection": {"id": "c1", "type": "scripted", "enabled": True, "settings": {}}}
    sink = Holds()
    result = await contract_with(ScriptedBackend({"c1": ScriptedProvider(price=0.02)})).test(scripted, {}, sink=sink, budget=1.0)
    assert result["ok"] is True and result["tokens"] == 10
    assert sink.events == [("reserve", 0.02), ("usage", 10, ("hold-1",))]
    spent = ScriptedProvider(price=0.02)
    refused = await contract_with(ScriptedBackend({"c1": spent})).test(scripted, {}, sink=Holds(), budget=0.0)
    assert refused["ok"] is False and refused["code"] == "cancelled" and refused["reason"] == "budget_exhausted"
    assert spent.batches == []
