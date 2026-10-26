"""The distribution splits the core from the server and names itself correctly."""

import tomllib
from pathlib import Path

from subtitle_translator import __version__

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
CORE = {"httpx", "pydantic", "openrouter", "tzdata"}
SERVER = {"fastapi", "uvicorn", "pydantic-settings", "python-dotenv", "python-multipart", "cryptography"}


def _names(requirements):
    return {req.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower() for req in requirements}


def test_name_and_version_agree():
    assert PYPROJECT["project"]["name"] == "lavx-ai-subtitle-translator"
    assert PYPROJECT["project"]["version"] == __version__


def test_core_dependencies_are_exactly_the_core():
    deps = PYPROJECT["project"]["dependencies"]
    assert _names(deps) == CORE
    assert "openrouter==1.1.133" in deps
    assert any(dep.startswith("pydantic>=") and "<2.13" in dep for dep in deps)


def test_server_extra_and_dev_extra():
    extras = PYPROJECT["project"]["optional-dependencies"]
    assert _names(extras["server"]) == SERVER
    assert "lavx-ai-subtitle-translator[server]" in extras["dev"]


def test_console_script_and_no_requirements_file():
    assert PYPROJECT["project"]["scripts"]["ai-subtitle-translator"] == "subtitle_translator.server.cli:main"
    assert not (ROOT / "requirements.txt").exists()


def test_gui_assets_are_package_data():
    data = PYPROJECT["tool"]["setuptools"]["package-data"]["subtitle_translator"]
    for asset in ("index.html", "app.css", "app.js", "archive.mjs", "session.mjs", "backends.mjs", "outfit.woff2", "outfit-OFL.txt"):
        assert f"server/static/{asset}" in data, asset
    assert "_vendor/LICENSE.srt" in data
