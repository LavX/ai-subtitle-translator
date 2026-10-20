"""Importing the core and the back ends must not load the server, and must not read the environment."""

import subprocess
import sys

CHECK = """
import os, sys
ours = []

class Env(dict):
    def _note(self, key):
        frame = sys._getframe(2)
        while frame is not None and frame.f_globals.get("__name__") == "os":
            frame = frame.f_back
        module = frame.f_globals.get("__name__", "") if frame is not None else ""
        if module.startswith("subtitle_translator"):
            ours.append((module, key))

    def __getitem__(self, key):
        self._note(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._note(key)
        return super().get(key, default)

os.environ = Env(os.environ)
import subtitle_translator.core.translator
import subtitle_translator.core.batch_processor
import subtitle_translator.core.batch_sizing
import subtitle_translator.core.srt_parser
import subtitle_translator.core.models
import subtitle_translator.core.settings
import subtitle_translator.backends.openrouter
import subtitle_translator.backends.smartfast
from subtitle_translator.core.settings import CoreSettings
from subtitle_translator.backends.openrouter import OpenRouterProvider
OpenRouterProvider(CoreSettings())
loaded = sorted(m for m in sys.modules if m.startswith("subtitle_translator.server"))
assert not loaded, loaded
for name in ("fastapi", "pydantic_settings", "dotenv", "cryptography"):
    assert name not in sys.modules, name
assert not ours, ours
"""


def test_core_and_backends_load_no_server_module_and_read_no_environment():
    result = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
