# tests/plugin/test_validate_migrate.py
"""validate normalizes without secrets or network; migrate is pure and keeps secret fields."""

import copy

import pytest

from subtitle_translator.core.backends import BUILTIN_DISTRIBUTION
from subtitle_translator.plugin.contract import (
    SECRET_FIELDS_V1,
    ContractError,
    TranslatorContract,
    secret_fields,
)
from subtitle_translator.plugin.scope import PluginScope
from tests.core.registry_fakes import EchoBackend, FakeEntryPoint, entry_points_from
from tests.plugin.fake_dist import HARNESS, points

BUILTIN = TranslatorContract(PluginScope.builtin())


def config(type_, settings):
    return {"settings": {"request_timeout": 120}, "connection": {"id": "c1", "type": type_, "enabled": True, "settings": settings}}


def test_validate_normalizes_a_retired_model_and_needs_no_secret(no_outbound_network):
    result = BUILTIN.validate(config("gemini", {"model": "gemini-2.0-flash"}))
    assert result["connection"]["settings"]["model"] == "gemini-3.6-flash"
    assert result["connection"]["settings"]["batch_size"] == 300
    assert no_outbound_network == []


def test_validate_names_the_fields_at_fault():
    with pytest.raises(ContractError) as caught:
        BUILTIN.validate(config("openrouter", {"routing": "warp", "parallel_batches": 0}))
    assert caught.value.code == "invalid_config"
    assert set(caught.value.field_errors) == {"connection.settings.routing", "connection.settings.parallel_batches"}


def test_validate_keeps_project_names_it_has_no_keys_to_count():
    result = BUILTIN.validate(config("gemini", {"key_projects": ["a", "b"]}))
    assert result["connection"]["settings"]["key_projects"] == ["a", "b"]


def test_validate_checks_a_smartfast_policy_without_the_key():
    with pytest.raises(ContractError) as caught:
        BUILTIN.validate(config("openrouter", {"routing": "smartfast", "smart_fast": {"maxPromptPrice": -1}}))
    assert set(caught.value.field_errors) == {"connection.settings.smart_fast"}


def test_validate_refuses_a_type_outside_the_scope():
    with pytest.raises(ContractError) as caught:
        BUILTIN.validate(config("vendor-cli", {}))
    assert caught.value.field_errors == {"connection.type": "this plugin does not serve vendor-cli"}


def test_a_served_type_that_failed_to_load_is_dependency_unavailable():
    broken = TranslatorContract(PluginScope.extension(HARNESS), points(explode_harness=True))
    with pytest.raises(ContractError) as caught:
        broken.validate(config("signin", {}))
    assert caught.value.code == "dependency_unavailable" and caught.value.field_errors == {}
    assert "ImportError" in caught.value.message


def test_an_extension_entry_under_a_built_in_type_id_is_not_served():
    squatter = TranslatorContract(
        PluginScope.extension("squatter"),
        entry_points_from(
            FakeEntryPoint("echo", EchoBackend, dist=BUILTIN_DISTRIBUTION),
            FakeEntryPoint("echo", EchoBackend, dist="squatter"),
        ),
    )
    with pytest.raises(ContractError) as caught:
        squatter.validate(config("echo", {}))
    assert caught.value.code == "invalid_config"
    assert caught.value.field_errors == {"connection.type": "this plugin does not serve echo"}


def test_migrate_is_pure_and_identity_for_version_1():
    settings = {"request_timeout": 120}
    connections = [{"id": "c1", "type": "gemini", "settings": {"model": "gemini-2.5-flash"}}]
    before = copy.deepcopy((settings, connections))
    result = BUILTIN.migrate(1, settings, connections)
    assert result == {"settings": settings, "connections": connections}
    assert result["settings"] is not settings and result["connections"][0] is not connections[0]
    assert (settings, connections) == before


def test_migrate_refuses_an_unknown_version():
    with pytest.raises(ContractError) as caught:
        BUILTIN.migrate(7, {}, [])
    assert caught.value.code == "invalid_config"


def test_secret_fields_of_version_1_never_change():
    assert secret_fields(BUILTIN) == SECRET_FIELDS_V1


def test_validate_fills_an_openai_compatible_address_from_its_preset():
    result = BUILTIN.validate(config("openai_compatible", {"preset": "ollama", "model": "llama3.3", "api_key": "x"}))
    settings = result["connection"]["settings"]
    assert settings["base_url"] == "http://localhost:11434/v1" and settings["model"] == "llama3.3"
    assert "api_key" not in settings                      # secrets never travel in settings
