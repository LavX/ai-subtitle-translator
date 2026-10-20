"""The documented entry points survive the move into subtitle_translator.server."""

import subprocess
import sys


def test_main_shim_exposes_the_server_app():
    import subtitle_translator.server.main as shim
    from subtitle_translator.server import main as server_main

    assert shim.app is server_main.app
    assert shim.create_app is server_main.create_app
    assert shim.lifespan is server_main.lifespan


def test_cli_module_still_runs_as_a_script():
    result = subprocess.run(
        [sys.executable, "-m", "subtitle_translator.server.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "regenerate-key" in result.stdout


def test_server_package_is_where_the_service_lives():
    from subtitle_translator.server import config, crypto, gui, ui_api, web  # noqa: F401
    from subtitle_translator.server.api import routes  # noqa: F401
    from subtitle_translator.server.queue import job_manager  # noqa: F401
