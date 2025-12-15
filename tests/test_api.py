"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from subtitle_translator.main import app
from subtitle_translator.api.models import SubtitleLine
from subtitle_translator.core.translator import ContentTranslationResult, FileTranslationResult


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_settings():
    """Mock settings with API key configured."""
    with patch("subtitle_translator.api.routes.get_settings") as mock:
        settings = MagicMock()
        settings.openrouter_api_key = "test-api-key"
        settings.openrouter_default_model = "google/gemini-2.5-flash-preview-09-2025"
        mock.return_value = settings
        yield settings


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_healthy(self, client, mock_settings):
        """Test health check when service is healthy."""
        with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
            translator = AsyncMock()
            translator.health_check = AsyncMock(return_value=True)
            mock_translator.return_value = translator
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["version"] == "1.0.0"
            assert data["openrouter_configured"] is True

    def test_health_check_no_api_key(self, client):
        """Test health check when API key is not configured."""
        with patch("subtitle_translator.api.routes.get_settings") as mock_get:
            settings = MagicMock()
            settings.openrouter_api_key = ""
            mock_get.return_value = settings
            
            with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
                translator = AsyncMock()
                translator.health_check = AsyncMock(return_value=False)
                mock_translator.return_value = translator
                
                response = client.get("/health")
                
                assert response.status_code == 200
                data = response.json()
                assert data["openrouter_configured"] is False


class TestModelsEndpoint:
    """Tests for the models listing endpoint."""

    def test_list_models(self, client, mock_settings):
        """Test listing available models."""
        mock_models = [
            {
                "id": "google/gemini-2.5-flash-preview-09-2025",
                "name": "Gemini 2.5 Flash",
                "description": "Fast model",
                "context_length": 1048576,
                "is_default": True,
            },
            {
                "id": "anthropic/claude-sonnet-4.5",
                "name": "Claude Sonnet 4.5",
                "description": "Great quality",
                "context_length": 200000,
                "is_default": False,
            },
        ]
        
        with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
            translator = AsyncMock()
            translator.get_available_models = AsyncMock(return_value=mock_models)
            mock_translator.return_value = translator
            
            response = client.get("/api/v1/models")
            
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert len(data["models"]) == 2
            assert data["default_model"] == "google/gemini-2.5-flash-preview-09-2025"


class TestTranslateContentEndpoint:
    """Tests for the translate content endpoint."""

    def test_translate_content_success(self, client, mock_settings):
        """Test successful content translation."""
        request_data = {
            "sourceLanguage": "en",
            "targetLanguage": "es",
            "lines": [
                {"position": 1, "line": "Hello"},
                {"position": 2, "line": "World"},
            ],
        }
        
        translated_lines = [
            SubtitleLine(position=1, line="Hola"),
            SubtitleLine(position=2, line="Mundo"),
        ]
        
        with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
            translator = AsyncMock()
            translator.translate_content = AsyncMock(
                return_value=ContentTranslationResult(
                    lines=translated_lines,
                    model_used="google/gemini-2.5-flash-preview-09-2025",
                    tokens_used=100,
                    success=True,
                )
            )
            mock_translator.return_value = translator
            
            response = client.post("/api/v1/translate/content", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["lines"]) == 2
            assert data["lines"][0]["line"] == "Hola"
            assert data["model_used"] == "google/gemini-2.5-flash-preview-09-2025"

    def test_translate_content_empty_lines(self, client, mock_settings):
        """Test translation with empty lines."""
        request_data = {
            "sourceLanguage": "en",
            "targetLanguage": "es",
            "lines": [],
        }
        
        with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
            translator = AsyncMock()
            mock_translator.return_value = translator
            
            response = client.post("/api/v1/translate/content", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["lines"] == []

    def test_translate_content_no_api_key(self, client):
        """Test translation fails without API key."""
        with patch("subtitle_translator.api.routes.get_settings") as mock_get:
            settings = MagicMock()
            settings.openrouter_api_key = ""
            mock_get.return_value = settings
            
            with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
                translator = AsyncMock()
                mock_translator.return_value = translator
                
                request_data = {
                    "sourceLanguage": "en",
                    "targetLanguage": "es",
                    "lines": [{"position": 1, "line": "Hello"}],
                }
                
                response = client.post("/api/v1/translate/content", json=request_data)
                
                assert response.status_code == 401

    def test_translate_content_with_context(self, client, mock_settings):
        """Test translation with media context."""
        request_data = {
            "sourceLanguage": "en",
            "targetLanguage": "es",
            "title": "Breaking Bad",
            "mediaType": "Episode",
            "arrMediaId": 12345,
            "lines": [{"position": 1, "line": "Say my name"}],
        }
        
        with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
            translator = AsyncMock()
            translator.translate_content = AsyncMock(
                return_value=ContentTranslationResult(
                    lines=[SubtitleLine(position=1, line="Di mi nombre")],
                    model_used="google/gemini-2.5-flash-preview-09-2025",
                    tokens_used=50,
                    success=True,
                )
            )
            mock_translator.return_value = translator
            
            response = client.post("/api/v1/translate/content", json=request_data)
            
            assert response.status_code == 200


class TestTranslateFileEndpoint:
    """Tests for the translate file endpoint."""

    def test_translate_file_success(self, client, mock_settings):
        """Test successful file translation."""
        srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:00:08,000
How are you?
"""
        
        translated_srt = """1
00:00:01,000 --> 00:00:04,000
Hola mundo

2
00:00:05,000 --> 00:00:08,000
¿Cómo estás?
"""
        
        request_data = {
            "content": srt_content,
            "sourceLanguage": "en",
            "targetLanguage": "es",
        }
        
        with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
            translator = AsyncMock()
            translator.translate_file = AsyncMock(
                return_value=FileTranslationResult(
                    content=translated_srt,
                    model_used="google/gemini-2.5-flash-preview-09-2025",
                    tokens_used=200,
                    subtitle_count=2,
                    success=True,
                )
            )
            mock_translator.return_value = translator
            
            response = client.post("/api/v1/translate/file", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "Hola mundo" in data["content"]
            assert data["subtitle_count"] == 2

    def test_translate_file_empty_content(self, client, mock_settings):
        """Test translation with empty content."""
        request_data = {
            "content": "",
            "sourceLanguage": "en",
            "targetLanguage": "es",
        }
        
        with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
            translator = AsyncMock()
            mock_translator.return_value = translator
            
            response = client.post("/api/v1/translate/file", json=request_data)
            
            assert response.status_code == 400

    def test_translate_file_invalid_srt(self, client, mock_settings):
        """Test translation with invalid SRT content."""
        request_data = {
            "content": "This is not valid SRT content",
            "sourceLanguage": "en",
            "targetLanguage": "es",
        }
        
        with patch("subtitle_translator.api.routes.get_translator") as mock_translator:
            translator = AsyncMock()
            translator.translate_file = AsyncMock(
                return_value=FileTranslationResult(
                    content="",
                    model_used="google/gemini-2.5-flash-preview-09-2025",
                    success=False,
                    error="Invalid SRT content: parsing failed",
                )
            )
            mock_translator.return_value = translator
            
            response = client.post("/api/v1/translate/file", json=request_data)
            
            assert response.status_code == 400


class TestRequestValidation:
    """Tests for request validation."""

    def test_missing_required_fields(self, client, mock_settings):
        """Test that missing required fields return 422."""
        # Missing sourceLanguage
        request_data = {
            "targetLanguage": "es",
            "lines": [{"position": 1, "line": "Hello"}],
        }
        
        response = client.post("/api/v1/translate/content", json=request_data)
        assert response.status_code == 422

    def test_invalid_temperature(self, client, mock_settings):
        """Test that invalid temperature returns 422."""
        request_data = {
            "sourceLanguage": "en",
            "targetLanguage": "es",
            "lines": [{"position": 1, "line": "Hello"}],
            "temperature": 3.0,  # Invalid: max is 2.0
        }
        
        response = client.post("/api/v1/translate/content", json=request_data)
        assert response.status_code == 422