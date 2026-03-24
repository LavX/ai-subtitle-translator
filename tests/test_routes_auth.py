"""Tests for auth middleware, encryption helpers, and uncovered routes in routes.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from subtitle_translator.api import routes
from subtitle_translator.api.models import SubtitleLine
from subtitle_translator.core.translator import ContentTranslationResult, FileTranslationResult
from subtitle_translator.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_settings():
    with patch("subtitle_translator.api.routes.get_settings") as mock:
        settings = MagicMock()
        settings.openrouter_api_key = "test-api-key"
        settings.openrouter_default_model = "google/gemini-2.5-flash-preview-09-2025"
        settings.openrouter_temperature = 0.3
        settings.batch_size = 25
        settings.parallel_batches_per_job = 4
        settings.admin_api_key = ""
        settings.encryption_enabled = False
        settings.encryption_strict = False
        mock.return_value = settings
        yield settings


@pytest.fixture(autouse=True)
def disable_auth():
    """Disable auth for most tests, enable explicitly when testing auth."""
    original_token = routes._auth_token
    original_key = routes._crypto_key
    routes._auth_token = None
    routes._crypto_key = None
    yield
    routes._auth_token = original_token
    routes._crypto_key = original_key


@pytest.fixture
def mock_translator():
    with patch("subtitle_translator.api.routes.get_translator") as mock_get:
        translator = AsyncMock()
        translator.health_check = AsyncMock(return_value=True)
        mock_get.return_value = translator
        yield translator


# ============================================================================
# Auth middleware: _verify_auth_token
# ============================================================================


class TestVerifyAuthToken:
    """Tests for the _verify_auth_token dependency."""

    def test_auth_disabled_passes_all_requests(self, client, mock_settings, mock_translator):
        """When _auth_token is None, all requests pass without a header."""
        routes._auth_token = None
        mock_translator.get_available_models = AsyncMock(return_value=[])
        response = client.get("/api/v1/models")
        assert response.status_code == 200

    def test_valid_token_passes(self, client, mock_settings, mock_translator):
        """Request with the correct X-Auth-Token passes on a protected endpoint."""
        routes._auth_token = "test-token"
        mock_translator.translate_content = AsyncMock(
            return_value=ContentTranslationResult(
                lines=[SubtitleLine(position=1, line="Szia")],
                model_used="test-model",
                tokens_used=10,
                success=True,
            )
        )
        response = client.post(
            "/api/v1/translate/content",
            json={
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [{"position": 1, "line": "Hello"}],
            },
            headers={"X-Auth-Token": "test-token"},
        )
        assert response.status_code == 200

    def test_wrong_token_returns_401(self, client, mock_settings, mock_translator):
        """Request with wrong token returns 401."""
        routes._auth_token = "test-token"
        response = client.post(
            "/api/v1/translate/content",
            json={
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [{"position": 1, "line": "Hello"}],
            },
            headers={"X-Auth-Token": "wrong-token"},
        )
        assert response.status_code == 401
        assert response.json()["detail"]["error"] == "unauthorized"

    def test_no_token_returns_401(self, client, mock_settings, mock_translator):
        """Request with no token header returns 401 when auth is enabled."""
        routes._auth_token = "test-token"
        response = client.post(
            "/api/v1/translate/content",
            json={
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [{"position": 1, "line": "Hello"}],
            },
        )
        assert response.status_code == 401
        assert response.json()["detail"]["error"] == "unauthorized"

    def test_jobs_router_requires_auth(self, client, mock_settings):
        """Jobs router has AuthDep as a router-level dependency."""
        routes._auth_token = "test-token"
        response = client.get("/api/v1/jobs")
        assert response.status_code == 401


# ============================================================================
# _decrypt_api_key
# ============================================================================


class TestDecryptApiKey:
    """Tests for the _decrypt_api_key helper."""

    def test_plaintext_key_returned_when_no_strict_mode(self, mock_settings):
        """Plaintext key is returned as-is when strict mode is off."""
        mock_settings.encryption_enabled = False
        mock_settings.encryption_strict = False
        result = routes._decrypt_api_key("sk-or-plaintext-key")
        assert result == "sk-or-plaintext-key"

    def test_plaintext_key_in_strict_mode_raises_400(self, mock_settings):
        """Plaintext key in strict mode raises 400."""
        mock_settings.encryption_enabled = True
        mock_settings.encryption_strict = True
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            routes._decrypt_api_key("sk-or-plaintext-key")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "encryption_required"

    def test_encrypted_key_when_encryption_disabled_raises_400(self):
        """Encrypted key (enc: prefix) when _crypto_key is None raises 400."""
        routes._crypto_key = None
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            routes._decrypt_api_key("enc:somebase64data")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "encryption_disabled"

    def test_decryption_failure_raises_400(self):
        """Decryption failure raises 400."""
        routes._crypto_key = b"\x00" * 32
        from fastapi import HTTPException

        with patch("subtitle_translator.crypto.decrypt", side_effect=ValueError("bad")):
            with pytest.raises(HTTPException) as exc_info:
                routes._decrypt_api_key("enc:baddata")
            assert exc_info.value.status_code == 400
            assert exc_info.value.detail["error"] == "decryption_failed"

    def test_successful_decryption(self):
        """Successful decryption returns decrypted key."""
        routes._crypto_key = b"\x00" * 32
        with patch("subtitle_translator.crypto.decrypt", return_value="sk-or-decrypted"):
            result = routes._decrypt_api_key("enc:validdata")
        assert result == "sk-or-decrypted"


# ============================================================================
# Health check: exception path (lines 147-149)
# ============================================================================


class TestHealthCheckFailure:
    """Test health check when translator.health_check raises an exception."""

    def test_health_check_exception_returns_unhealthy(self, client, mock_settings):
        """When health_check() raises, status should be unhealthy."""
        with patch("subtitle_translator.api.routes.get_translator") as mock_get:
            translator = AsyncMock()
            translator.health_check = AsyncMock(side_effect=RuntimeError("connection failed"))
            mock_get.return_value = translator

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["openrouterConfigured"] is True


# ============================================================================
# Service status health check failure (lines 828-830)
# ============================================================================


class TestServiceStatusHealthFailure:
    """Test service status when health check fails."""

    def test_service_status_health_check_exception(self, client, mock_settings):
        """When health_check() raises during /status, healthy should be False."""
        with patch("subtitle_translator.api.routes.get_translator") as mock_get:
            translator = AsyncMock()
            translator.health_check = AsyncMock(side_effect=RuntimeError("timeout"))
            mock_get.return_value = translator

            response = client.get("/api/v1/status")

            assert response.status_code == 200
            data = response.json()
            assert data["healthy"] is False


# ============================================================================
# Translate content: partial failure (lines 239-250, 274-280)
# ============================================================================


class TestTranslateContentPartialFailure:
    """Test translate_content when translation partially fails."""

    def test_partial_success_returns_partial_lines(self, client, mock_settings, mock_translator):
        """Translation returns success=False with partial lines."""
        partial_lines = [SubtitleLine(position=1, line="Translated")]
        mock_translator.translate_content = AsyncMock(
            return_value=ContentTranslationResult(
                lines=partial_lines,
                model_used="test-model",
                tokens_used=100,
                success=False,
                error="Batch 2 failed",
            )
        )

        response = client.post(
            "/api/v1/translate/content",
            json={
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [{"position": 1, "line": "Hello"}, {"position": 2, "line": "World"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["lines"]) == 1
        assert data["modelUsed"] == "test-model"

    def test_complete_failure_returns_500(self, client, mock_settings, mock_translator):
        """Translation returns success=False with no lines: 500 error."""
        mock_translator.translate_content = AsyncMock(
            return_value=ContentTranslationResult(
                lines=[],
                model_used="test-model",
                tokens_used=0,
                success=False,
                error="All batches failed",
            )
        )

        response = client.post(
            "/api/v1/translate/content",
            json={
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [{"position": 1, "line": "Hello"}],
            },
        )

        assert response.status_code == 500
        assert response.json()["detail"]["error"] == "translation_failed"

    def test_config_logging_with_api_key(self, client, mock_settings, mock_translator):
        """Config with api_key should be masked in logs (lines 239-245)."""
        mock_translator.translate_content = AsyncMock(
            return_value=ContentTranslationResult(
                lines=[SubtitleLine(position=1, line="Szia")],
                model_used="test-model",
                tokens_used=50,
                success=True,
            )
        )

        response = client.post(
            "/api/v1/translate/content",
            json={
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [{"position": 1, "line": "Hello"}],
                "config": {"apiKey": "sk-or-secret-key", "model": "test-model"},
            },
        )

        assert response.status_code == 200

    def test_config_override_api_key_has_api_key(self, client, mock_settings, mock_translator):
        """Request with config.apiKey sets has_api_key even without env key (line 250)."""
        mock_settings.openrouter_api_key = ""
        mock_translator.translate_content = AsyncMock(
            return_value=ContentTranslationResult(
                lines=[SubtitleLine(position=1, line="Szia")],
                model_used="test-model",
                tokens_used=50,
                success=True,
            )
        )

        response = client.post(
            "/api/v1/translate/content",
            json={
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "lines": [{"position": 1, "line": "Hello"}],
                "config": {"apiKey": "sk-or-request-key"},
            },
        )

        assert response.status_code == 200


# ============================================================================
# Translate file: error paths (lines 334, 337, 366-387)
# ============================================================================


class TestTranslateFileErrorPaths:
    """Test translate_file error branches."""

    def test_partial_file_translation(self, client, mock_settings, mock_translator):
        """File translation partial success returns content with 200."""
        mock_translator.translate_file = AsyncMock(
            return_value=FileTranslationResult(
                content="1\n00:00:01,000 --> 00:00:04,000\nTranslated\n\n",
                model_used="test-model",
                tokens_used=100,
                subtitle_count=1,
                success=False,
                error="Batch 2 failed",
            )
        )

        response = client.post(
            "/api/v1/translate/file",
            json={
                "content": "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n",
                "sourceLanguage": "en",
                "targetLanguage": "hu",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "Translated" in data["content"]

    def test_invalid_srt_error(self, client, mock_settings, mock_translator):
        """File translation with 'Invalid SRT' error returns 400."""
        mock_translator.translate_file = AsyncMock(
            return_value=FileTranslationResult(
                content="",
                model_used="test-model",
                tokens_used=0,
                subtitle_count=0,
                success=False,
                error="Invalid SRT format: missing timestamps",
            )
        )

        response = client.post(
            "/api/v1/translate/file",
            json={
                "content": "garbage data",
                "sourceLanguage": "en",
                "targetLanguage": "hu",
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "invalid_srt"

    def test_complete_file_failure(self, client, mock_settings, mock_translator):
        """File translation complete failure returns 500."""
        mock_translator.translate_file = AsyncMock(
            return_value=FileTranslationResult(
                content="",
                model_used="test-model",
                tokens_used=0,
                subtitle_count=0,
                success=False,
                error="API rate limit exceeded",
            )
        )

        response = client.post(
            "/api/v1/translate/file",
            json={
                "content": "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n",
                "sourceLanguage": "en",
                "targetLanguage": "hu",
            },
        )

        assert response.status_code == 500
        assert response.json()["detail"]["error"] == "translation_failed"

    def test_file_config_override_api_key(self, client, mock_settings, mock_translator):
        """Request with config.apiKey sets has_api_key for file endpoint (line 334)."""
        mock_settings.openrouter_api_key = ""
        mock_translator.translate_file = AsyncMock(
            return_value=FileTranslationResult(
                content="1\n00:00:01,000 --> 00:00:04,000\nSzia\n\n",
                model_used="test-model",
                tokens_used=50,
                subtitle_count=1,
                success=True,
            )
        )

        response = client.post(
            "/api/v1/translate/file",
            json={
                "content": "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n",
                "sourceLanguage": "en",
                "targetLanguage": "hu",
                "config": {"apiKey": "sk-or-request-key"},
            },
        )

        assert response.status_code == 200

    def test_file_no_api_key_returns_401(self, client, mock_settings, mock_translator):
        """File endpoint with no API key anywhere returns 401 (line 337)."""
        mock_settings.openrouter_api_key = ""

        response = client.post(
            "/api/v1/translate/file",
            json={
                "content": "1\n00:00:01,000 --> 00:00:04,000\nHello\n\n",
                "sourceLanguage": "en",
                "targetLanguage": "hu",
            },
        )

        assert response.status_code == 401
        assert response.json()["detail"]["error"] == "configuration_error"


# ============================================================================
# test_connection endpoint (lines 980-1030)
# ============================================================================


class TestTestConnection:
    """Tests for the /api/v1/test endpoint."""

    def test_successful_encryption_and_valid_key(self, client, mock_settings):
        """Encrypted key decrypts ok and OpenRouter returns 200."""
        routes._crypto_key = b"\x00" * 32

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "label": "My Key",
                "limit_remaining": 10.0,
                "usage": 5.0,
                "is_free_tier": False,
            }
        }

        with patch("subtitle_translator.crypto.decrypt", return_value="sk-or-decrypted"):
            with patch("httpx.AsyncClient") as mock_httpx:
                mock_client_instance = AsyncMock()
                mock_client_instance.get = AsyncMock(return_value=mock_response)
                mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
                mock_client_instance.__aexit__ = AsyncMock(return_value=False)
                mock_httpx.return_value = mock_client_instance

                response = client.post(
                    "/api/v1/test",
                    json={"apiKey": "enc:validdata"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["encryption"]["status"] == "ok"
        assert data["apiKey"]["status"] == "ok"
        assert data["apiKey"]["label"] == "My Key"

    def test_decryption_failure_returns_early(self, client, mock_settings):
        """When decryption fails, returns encryption error and null apiKey."""
        routes._crypto_key = b"\x00" * 32

        with patch("subtitle_translator.crypto.decrypt", side_effect=ValueError("bad")):
            response = client.post(
                "/api/v1/test",
                json={"apiKey": "enc:baddata"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["encryption"]["status"] == "error"
        assert data["apiKey"] is None

    def test_plaintext_key_openrouter_200(self, client, mock_settings):
        """Plaintext key skips encryption, OpenRouter returns 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"label": "Test", "limit_remaining": None, "usage": 0, "is_free_tier": True}
        }

        with patch("httpx.AsyncClient") as mock_httpx:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.return_value = mock_client_instance

            response = client.post(
                "/api/v1/test",
                json={"apiKey": "sk-or-plaintext"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["encryption"] is None
        assert data["apiKey"]["status"] == "ok"

    def test_openrouter_returns_401(self, client, mock_settings):
        """OpenRouter returns 401 for invalid key."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_httpx:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.return_value = mock_client_instance

            response = client.post(
                "/api/v1/test",
                json={"apiKey": "sk-or-invalid"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["apiKey"]["status"] == "error"
        assert data["apiKey"]["message"] == "Invalid API key"

    def test_openrouter_returns_other_status(self, client, mock_settings):
        """OpenRouter returns unexpected status code."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("httpx.AsyncClient") as mock_httpx:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.return_value = mock_client_instance

            response = client.post(
                "/api/v1/test",
                json={"apiKey": "sk-or-test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["apiKey"]["status"] == "error"
        assert "503" in data["apiKey"]["message"]

    def test_connection_error_to_openrouter(self, client, mock_settings):
        """Connection error when reaching OpenRouter."""
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(side_effect=Exception("DNS resolution failed"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.return_value = mock_client_instance

            response = client.post(
                "/api/v1/test",
                json={"apiKey": "sk-or-test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["apiKey"]["status"] == "error"
        assert "Connection failed" in data["apiKey"]["message"]

    def test_test_connection_requires_auth(self, client, mock_settings):
        """test_connection endpoint requires auth when enabled."""
        routes._auth_token = "secret-token"
        response = client.post(
            "/api/v1/test",
            json={"apiKey": "sk-or-test"},
        )
        assert response.status_code == 401
