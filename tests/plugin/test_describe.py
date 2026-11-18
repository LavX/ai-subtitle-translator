"""describe reports runtime facts for its own scope, and never a settings field."""

from subtitle_translator import __version__
from subtitle_translator.plugin.contract import KIND_API, SETTINGS_VERSION, TranslatorContract
from subtitle_translator.plugin.scope import PluginScope
from tests.core import registry_fakes
from tests.plugin.fake_dist import HARNESS, points


def test_builtin_scope_describes_only_builtins_and_imports_nothing_else():
    registry_fakes.LOADED.clear()
    facts = TranslatorContract(PluginScope.builtin(), points()).describe()
    assert facts["kind_api"] == KIND_API and facts["plugin_version"] == __version__
    assert facts["settings_version"] == SETTINGS_VERSION
    assert facts["backend_types"] == [{"type": "echo", "name": "Echo"}]
    assert registry_fakes.LOADED == ["echo"]
    assert facts["detects_source"] is True and facts["languages"] is None
    assert facts["limits"] == {"max_lines": 10_000, "max_line_chars": 2000}


def test_extension_scope_describes_only_that_extension():
    facts = TranslatorContract(PluginScope.extension(HARNESS), points()).describe()
    assert facts["backend_types"] == [{"type": "signin", "name": "Sign-in harness"}]


def test_a_type_that_fails_to_load_is_reported_not_hidden():
    facts = TranslatorContract(PluginScope.extension(HARNESS), points(explode_harness=True)).describe()
    assert facts["backend_types"] == []
    assert facts["unavailable"][0]["type"] == "signin" and "ImportError" in facts["unavailable"][0]["error"]


def test_describe_never_carries_settings():
    facts = TranslatorContract(PluginScope.builtin(), points()).describe()
    text = repr(facts)
    for word in ("fields", "settings\":", "default", "secret"):
        assert word not in text


def test_the_real_builtin_scope_serves_the_three_built_in_back_ends():
    facts = TranslatorContract(PluginScope.builtin()).describe()
    assert [t["type"] for t in facts["backend_types"]] == ["gemini", "openai_compatible", "openrouter"]
