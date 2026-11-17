# tests/plugin/test_options_test.py
"""options and test receive secrets, use one connection, and report sign-in state without credentials."""

import json

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


@pytest.mark.asyncio
async def test_a_key_list_arrives_as_json_text_and_is_read_as_a_list():
    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json={"data": [{"id": "models/gemini-2.5-flash"}]})

    contract = contract_with(GeminiBackendType(transport=httpx.MockTransport(handler)))
    await contract.options("connection.settings.model", CONFIG, {"api_keys": '["first","second"]'})
    assert seen == ["Bearer first"]


def test_key_list_text_forms():
    from subtitle_translator.plugin.contract import key_list

    assert key_list('["a", " b ", "", 3]') == ["a", "b"]
    assert key_list("a\nb\r\n\n c ") == ["a", "b", "c"]
    assert key_list("single-key") == ["single-key"]
    assert key_list("123456") == ["123456"]  # a number is a key, not a JSON value
    assert key_list("[not json") == ["[not json"]


@pytest.mark.asyncio
async def test_a_translate_request_decodes_its_key_list_for_the_engine():
    from subtitle_translator.core.backends import Connection
    from subtitle_translator.plugin.contract import decode_secrets

    backend = GeminiBackendType()
    connection = Connection("g", "gemini", secrets={"api_keys": '["k1","k2"]', "other": "x"})
    decoded = decode_secrets(backend, connection)
    assert decoded.secrets == {"api_keys": ["k1", "k2"], "other": "x"}
    already = Connection("g", "gemini", secrets={"api_keys": ["k1"]})
    assert decode_secrets(backend, already).secrets == {"api_keys": ["k1"]}


@pytest.mark.asyncio
async def test_an_unreachable_server_while_listing_options_is_dependency_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    contract = contract_with(GeminiBackendType(transport=httpx.MockTransport(handler)))
    with pytest.raises(ContractError) as caught:
        await contract.options("connection.settings.model", CONFIG, {"api_keys": ["k"]})
    assert caught.value.code == "dependency_unavailable"


@pytest.mark.asyncio
async def test_options_take_the_bare_field_key_the_host_sends():
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "models/gemini-2.5-flash"}]})

    contract = contract_with(GeminiBackendType(transport=httpx.MockTransport(handler)))
    bare = await contract.options("model", CONFIG, {"api_keys": ["k"]})
    prefixed = await contract.options("connection.settings.model", CONFIG, {"api_keys": ["k"]})
    assert bare == prefixed and bare[0]["value"] == "gemini-2.5-flash"
    with pytest.raises(ContractError) as caught:
        await contract.options("", CONFIG, {"api_keys": ["k"]})
    assert caught.value.code == "invalid_config"


@pytest.mark.asyncio
async def test_a_translation_sends_the_first_key_of_a_json_text_key_list():
    from tests.plugin.test_translate import Sink, request

    seen = []

    def handler(req):
        seen.append(req.headers["authorization"])
        content = json.dumps({"translations": [{"index": "0", "content": "Szia"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 5}})

    contract = contract_with(GeminiBackendType(transport=httpx.MockTransport(handler)))
    job = request(1)
    job["config"]["connection"] = {"id": "g", "type": "gemini", "enabled": True, "settings": {}}
    job["secrets"] = {"api_keys": '["first-key","second-key"]'}
    result = await contract.translate(job, Sink())
    assert result["status"] == "complete" and seen == ["Bearer first-key"]
