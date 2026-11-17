# src/subtitle_translator/plugin/manifest.py
"""Generate the manifest's settings block from the back-end types. The signed manifest is the
only settings schema Bazarr+ reads; generating it keeps it equal to the code."""

import argparse
import json
import sys

from subtitle_translator.core.backends import FieldType, SettingField
from subtitle_translator.plugin.contract import SETTINGS_VERSION, TranslatorContract
from subtitle_translator.plugin.scope import PluginScope
from subtitle_translator.plugin.wire import ENGINE_LIMITS

# The bounds are the ones read_engine_settings enforces, so the manifest and a request agree.
ENGINE_FIELDS = [
    SettingField("request_timeout", FieldType.INTEGER, "Seconds to wait for one request", default=120,
                 minimum=ENGINE_LIMITS["request_timeout"][0], maximum=ENGINE_LIMITS["request_timeout"][1]),
    SettingField("max_retries", FieldType.INTEGER, "Retries of one batch", default=3,
                 minimum=ENGINE_LIMITS["max_retries"][0], maximum=ENGINE_LIMITS["max_retries"][1]),
]


def _field(spec: SettingField) -> dict:
    return {
        "key": spec.key,
        "type": str(spec.type),
        "label": spec.label,
        "default": spec.default,
        "required": spec.required,
        "secret": spec.secret,
        "destination": spec.destination,
        "help": spec.help,
        "help_url": spec.help_url,
        "choices": [
            {"value": c.value, "label": c.label, **({"fills": dict(c.fills)} if c.fills else {})}
            for c in spec.choices
        ],
        "min": spec.minimum,
        "max": spec.maximum,
        "group": spec.group,
        "show_when": {"field": spec.show_when[0], "equals": spec.show_when[1]} if spec.show_when else None,
    }


def settings_block(contract: TranslatorContract) -> dict:
    types = {}
    for type_id in contract.registry.types():
        backend = contract.registry.get(type_id)
        types[type_id] = {
            "name": backend.display_name,
            "sign_in": backend.uses_sign_in,
            "login": list(backend.login_argv) or None,
            "connection": [_field(spec) for spec in backend.fields()],
        }
    endpoints = sorted({host for type_id in contract.registry.types() for host in contract.registry.get(type_id).endpoints})
    return {
        "settings_version": SETTINGS_VERSION,
        "endpoints": endpoints,
        "engine": [_field(s) for s in ENGINE_FIELDS],
        "backend_types": types,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the plugin manifest's settings block")
    parser.add_argument("--scope", default="builtin", help="builtin, or extension:<distribution>")
    args = parser.parse_args()
    if args.scope == "builtin":
        scope = PluginScope.builtin()
    elif args.scope.startswith("extension:"):
        scope = PluginScope.extension(args.scope.split(":", 1)[1])
    else:
        parser.error("--scope is builtin or extension:<distribution>")
    json.dump(settings_block(TranslatorContract(scope)), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
