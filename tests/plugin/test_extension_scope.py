# tests/plugin/test_extension_scope.py
"""Built-in and extension workers each serve only their own back ends."""

import pytest

from subtitle_translator.core.backends import BUILTIN_DISTRIBUTION
from subtitle_translator.plugin.contract import ContractError, TranslatorContract
from subtitle_translator.plugin.scope import PluginScope
from tests.core import registry_fakes
from tests.plugin.fake_dist import HARNESS, points
from tests.plugin.test_translate import Sink, request


@pytest.mark.asyncio
async def test_builtin_worker_refuses_an_extension_type():
    registry_fakes.LOADED.clear()
    builtin = TranslatorContract(PluginScope.builtin(), points())
    extension_request = request()
    extension_request["config"]["connection"]["type"] = "signin"
    with pytest.raises(ContractError) as caught:
        await builtin.translate(extension_request, Sink())
    assert caught.value.code == "invalid_config" and "signin" in caught.value.message
    assert "signin" not in registry_fakes.LOADED


@pytest.mark.asyncio
async def test_extension_worker_refuses_a_builtin_type():
    extension = TranslatorContract(PluginScope.extension(HARNESS), points())
    builtin_request = request()
    builtin_request["config"]["connection"]["type"] = "echo"
    with pytest.raises(ContractError):
        await extension.translate(builtin_request, Sink())
    with pytest.raises(ContractError):
        await extension.options("connection.settings.model", builtin_request["config"], {})


@pytest.mark.asyncio
async def test_a_served_type_that_failed_to_load_fails_over_instead_of_being_refused():
    broken = TranslatorContract(PluginScope.extension(HARNESS), points(explode_harness=True))
    signin = request()
    signin["config"]["connection"]["type"] = "signin"
    with pytest.raises(ContractError) as caught:
        await broken.translate(signin, Sink())
    assert caught.value.code == "dependency_unavailable" and caught.value.field_errors == {}
    with pytest.raises(ContractError) as caught:
        await broken.test(signin["config"], {})
    assert caught.value.code == "dependency_unavailable" and caught.value.field_errors == {}
    assert [entry["type"] for entry in broken.describe()["unavailable"]] == ["signin"]


@pytest.mark.asyncio
async def test_an_extension_entry_under_a_built_in_type_id_stays_invalid_config():
    squatter = TranslatorContract(
        PluginScope.extension("squatter"),
        registry_fakes.entry_points_from(
            registry_fakes.FakeEntryPoint("echo", registry_fakes.EchoBackend, dist=BUILTIN_DISTRIBUTION),
            registry_fakes.FakeEntryPoint("echo", registry_fakes.EchoBackend, dist="squatter"),
        ),
    )
    echo = request()
    echo["config"]["connection"]["type"] = "echo"
    with pytest.raises(ContractError) as caught:
        await squatter.translate(echo, Sink())
    assert caught.value.code == "invalid_config"
    assert caught.value.field_errors == {"connection.type": "this plugin does not serve echo"}
