"""Tests for subtitle_translator.server.cli module."""

import os
import stat
from unittest.mock import MagicMock, patch

import pytest

from subtitle_translator.server.cli import main, regenerate_key


class TestRegenerateKey:
    def test_success(self, tmp_path, capsys):
        key_file = str(tmp_path / "test.key")
        with patch("subtitle_translator.server.cli.get_settings") as mock_get:
            settings = MagicMock()
            settings.encryption_key = ""
            settings.encryption_enabled = True
            settings.encryption_key_file = key_file
            mock_get.return_value = settings

            regenerate_key()

        captured = capsys.readouterr()
        assert "New encryption key generated" in captured.out
        assert f"Saved to: {key_file}" in captured.out
        assert os.path.exists(key_file)
        assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600

    def test_success_creates_parent_dirs(self, tmp_path, capsys):
        key_file = str(tmp_path / "nested" / "dir" / "test.key")
        with patch("subtitle_translator.server.cli.get_settings") as mock_get:
            settings = MagicMock()
            settings.encryption_key = ""
            settings.encryption_enabled = True
            settings.encryption_key_file = key_file
            mock_get.return_value = settings

            regenerate_key()

        assert os.path.exists(key_file)

    def test_error_encryption_key_env_set(self, capsys):
        with patch("subtitle_translator.server.cli.get_settings") as mock_get:
            settings = MagicMock()
            settings.encryption_key = "some-key-value"
            mock_get.return_value = settings

            with pytest.raises(SystemExit) as exc_info:
                regenerate_key()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "ENCRYPTION_KEY env var is set" in captured.out

    def test_error_encryption_disabled(self, capsys):
        with patch("subtitle_translator.server.cli.get_settings") as mock_get:
            settings = MagicMock()
            settings.encryption_key = ""
            settings.encryption_enabled = False
            mock_get.return_value = settings

            with pytest.raises(SystemExit) as exc_info:
                regenerate_key()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Encryption is disabled" in captured.out


class TestMain:
    def test_regenerate_key_command(self):
        with (
            patch("subtitle_translator.server.cli.regenerate_key") as mock_regen,
            patch("sys.argv", ["cli", "regenerate-key"]),
        ):
            main()
            mock_regen.assert_called_once()

    def test_no_command_prints_help(self, capsys):
        with patch("sys.argv", ["cli"]):
            main()

        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "AI Subtitle Translator CLI" in captured.out

    def test_unknown_command(self):
        with patch("sys.argv", ["cli", "bogus"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2


class TestServe:
    def test_serve_uses_settings_defaults(self):
        with patch("subtitle_translator.server.cli.get_settings") as mock_get, patch("uvicorn.run") as run:
            settings = MagicMock()
            settings.host = "127.0.0.1"
            settings.port = 9999
            mock_get.return_value = settings
            with patch("sys.argv", ["ai-subtitle-translator", "serve"]):
                main()
        run.assert_called_once_with("subtitle_translator.main:app", host="127.0.0.1", port=9999)

    def test_serve_flags_override_settings(self):
        with patch("subtitle_translator.server.cli.get_settings") as mock_get, patch("uvicorn.run") as run:
            settings = MagicMock()
            settings.host = "127.0.0.1"
            settings.port = 9999
            mock_get.return_value = settings
            with patch("sys.argv", ["ai-subtitle-translator", "serve", "--host", "0.0.0.0", "--port", "8765"]):
                main()
        run.assert_called_once_with("subtitle_translator.main:app", host="0.0.0.0", port=8765)
