# tests/plugin/test_manifest.py
"""The signed manifest's settings come from the code, and the checked-in copy never drifts."""

import json
from pathlib import Path

from subtitle_translator.plugin.contract import SETTINGS_VERSION, TranslatorContract
from subtitle_translator.plugin.manifest import settings_block
from subtitle_translator.plugin.scope import PluginScope

CHECKED_IN = Path(__file__).resolve().parents[2] / "catalog" / "ai_translator.settings.json"


def test_block_covers_every_served_type_and_marks_secrets():
    contract = TranslatorContract(PluginScope.builtin())
    block = settings_block(contract)
    assert block["settings_version"] == SETTINGS_VERSION
    assert set(block["backend_types"]) == {t["type"] for t in contract.describe()["backend_types"]}
    gemini = {f["key"]: f for f in block["backend_types"]["gemini"]["connection"]}
    assert gemini["api_keys"]["secret"] is True and gemini["api_keys"]["type"] == "key_list"
    assert gemini["batch_size"]["default"] == 300
    assert {f["key"] for f in block["engine"]} == {"request_timeout", "max_retries"}
    assert block["endpoints"] == ["generativelanguage.googleapis.com", "openrouter.ai"]


def test_every_fixed_host_the_builtins_contact_is_declared():
    from urllib.parse import urlparse

    from subtitle_translator.backends.gemini import GEMINI_BASE_URL
    from subtitle_translator.core.settings import CoreSettings

    declared = set(settings_block(TranslatorContract(PluginScope.builtin()))["endpoints"])
    for url in (GEMINI_BASE_URL, CoreSettings().openrouter_api_base):
        assert urlparse(url).hostname in declared, url


def test_the_openai_compatible_address_is_a_destination_and_its_presets_fill_it():
    block = settings_block(TranslatorContract(PluginScope.builtin()))
    fields = {f["key"]: f for f in block["backend_types"]["openai_compatible"]["connection"]}
    assert fields["base_url"]["destination"] is True and fields["base_url"]["required"] is True
    assert fields["api_key"]["secret"] is True and fields["api_key"]["required"] is False
    fills = {c["value"]: c["fills"]["base_url"] for c in fields["preset"]["choices"]}
    assert fills["lmstudio"] == "http://localhost:1234/v1" and fills["custom"] == ""
    assert not any("openai" in host or "localhost" in host for host in block["endpoints"])


def test_the_engine_bounds_are_the_ones_a_request_is_held_to():
    from subtitle_translator.plugin.wire import ENGINE_LIMITS

    block = settings_block(TranslatorContract(PluginScope.builtin()))
    assert {f["key"]: (f["min"], f["max"]) for f in block["engine"]} == ENGINE_LIMITS


def test_the_checked_in_copy_matches_the_code():
    generated = settings_block(TranslatorContract(PluginScope.builtin()))
    assert json.loads(CHECKED_IN.read_text()) == json.loads(json.dumps(generated))


def test_the_price_fields_carry_the_names_the_host_uses():
    block = settings_block(TranslatorContract(PluginScope.builtin()))
    for type_id in ("gemini", "openai_compatible"):
        keys = {f["key"] for f in block["backend_types"][type_id]["connection"]}
        assert {"price_input_per_million", "price_output_per_million"} <= keys
        assert not keys & {"price_input", "price_output"}


def test_the_smartfast_policy_is_declared_as_text_and_read_as_text():
    block = settings_block(TranslatorContract(PluginScope.builtin()))
    fields = {f["key"]: f for f in block["backend_types"]["openrouter"]["connection"]}
    assert fields["smart_fast"]["type"] == "text" and "maxPromptPrice" in fields["smart_fast"]["help"]
