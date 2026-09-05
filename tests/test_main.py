"""Tests for main.py: create_app(), lifespan(), and CORS configuration."""

import os
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULE = "subtitle_translator.main"


def _make_settings(**overrides):
    """Return a mock Settings object with sensible defaults."""
    defaults = {
        "host": "0.0.0.0",
        "port": 8765,
        "debug": False,
        "openrouter_api_key": "sk-test-key",
        "openrouter_default_model": "test-model",
        "encryption_enabled": False,
        "encryption_strict": False,
        "encryption_key": "",
        "encryption_key_file": "/tmp/test.key",
        "db_path": "/tmp/test.db",
        "job_retention_hours": 24,
        "job_queue_max_concurrent": 2,
        "job_queue_max_jobs": 100,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


# ---------------------------------------------------------------------------
# create_app tests
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for the create_app() factory."""

    @patch(f"{MODULE}.get_settings", return_value=_make_settings())
    def test_returns_fastapi_instance(self, mock_gs):
        from subtitle_translator.main import create_app

        app = create_app()
        assert isinstance(app, FastAPI)

    @patch(f"{MODULE}.get_settings", return_value=_make_settings())
    def test_app_metadata(self, mock_gs):
        from subtitle_translator.main import create_app

        app = create_app()
        assert app.title == "AI Subtitle Translator"
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        assert app.openapi_url == "/openapi.json"

    @patch(f"{MODULE}.get_settings", return_value=_make_settings())
    def test_version_from_package(self, mock_gs):
        from subtitle_translator import __version__
        from subtitle_translator.main import create_app

        app = create_app()
        assert app.version == __version__

    @patch(f"{MODULE}.get_settings", return_value=_make_settings())
    def test_routers_included(self, mock_gs):
        from subtitle_translator.main import create_app

        app = create_app()

        # Newer FastAPI keeps an included router as a single entry that carries its
        # own routes instead of flattening them into app.routes, so walk both shapes
        # and back it with the OpenAPI schema, which is stable across versions.
        def _paths(routes):
            for route in routes:
                path = getattr(route, "path", None)
                if path is not None:
                    yield path
                nested = getattr(route, "routes", None)
                if nested is None:
                    nested = getattr(getattr(route, "router", None), "routes", None)
                yield from _paths(nested or [])

        paths = list(_paths(app.routes)) + list(app.openapi().get("paths", {}))
        # health, api, jobs, and config routers should each contribute routes
        assert "/health" in paths or any("/health" in p for p in paths)


# ---------------------------------------------------------------------------
# CORS configuration tests
# ---------------------------------------------------------------------------


class TestCorsConfiguration:
    """Tests for CORS middleware setup in create_app."""

    @patch(f"{MODULE}.get_settings", return_value=_make_settings())
    def test_default_wildcard_origins(self, mock_gs):
        """When CORS_ALLOWED_ORIGINS is unset, origins default to ['*']."""
        env = os.environ.copy()
        env.pop("CORS_ALLOWED_ORIGINS", None)
        with patch.dict(os.environ, env, clear=True):
            from subtitle_translator.main import create_app

            app = create_app()
            # Verify the CORS middleware was added with wildcard
            cors_middlewares = [
                m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"
            ]
            assert len(cors_middlewares) == 1
            assert cors_middlewares[0].kwargs["allow_origins"] == ["*"]
            assert cors_middlewares[0].kwargs["allow_credentials"] is False

    @patch(f"{MODULE}.get_settings", return_value=_make_settings())
    def test_custom_origins(self, mock_gs):
        """Custom CORS_ALLOWED_ORIGINS are parsed and credentials enabled."""
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "https://a.com, https://b.com"}):
            from subtitle_translator.main import create_app

            app = create_app()
            cors_middlewares = [
                m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"
            ]
            assert len(cors_middlewares) == 1
            assert cors_middlewares[0].kwargs["allow_origins"] == [
                "https://a.com",
                "https://b.com",
            ]
            assert cors_middlewares[0].kwargs["allow_credentials"] is True


# ---------------------------------------------------------------------------
# lifespan tests
# ---------------------------------------------------------------------------


def _standard_lifespan_patches(settings=None):
    """Return a dict of patches commonly needed for lifespan tests."""
    if settings is None:
        settings = _make_settings()
    return {
        f"{MODULE}.get_settings": MagicMock(return_value=settings),
        f"{MODULE}.set_crypto_key": MagicMock(),
        f"{MODULE}.set_auth_token": MagicMock(),
        f"{MODULE}.derive_auth_token": MagicMock(return_value="deadbeef" * 8),
        f"{MODULE}.JobStore": MagicMock(return_value=MagicMock(close=MagicMock())),
        f"{MODULE}.job_manager": MagicMock(
            set_store=MagicMock(),
            recover_jobs=AsyncMock(return_value=0),
            start_workers=AsyncMock(),
            stop_workers=AsyncMock(),
            set_worker_handler=MagicMock(),
        ),
        f"{MODULE}.close_translator": AsyncMock(),
    }


@pytest.mark.asyncio
class TestLifespanEncryptionDisabled:
    """Lifespan when encryption_enabled=False."""

    async def test_startup_and_shutdown_no_encryption(self):
        settings = _make_settings(encryption_enabled=False)
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.set_crypto_key", patches[f"{MODULE}.set_crypto_key"]) as mock_set_ck,
            patch(f"{MODULE}.set_auth_token", patches[f"{MODULE}.set_auth_token"]) as mock_set_at,
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]) as mock_jm,
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]) as mock_ct,
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                # During startup: crypto_key should be None
                mock_set_ck.assert_called_once_with(None)
                mock_set_at.assert_called_once_with(None)
                mock_jm.start_workers.assert_awaited_once()

            # After shutdown
            mock_jm.stop_workers.assert_awaited_once()
            mock_ct.assert_awaited_once()

    async def test_encryption_strict_warning_when_disabled(self):
        """encryption_strict=True but encryption_enabled=False logs a warning."""
        settings = _make_settings(encryption_enabled=False, encryption_strict=True)
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.set_crypto_key", patches[f"{MODULE}.set_crypto_key"]),
            patch(f"{MODULE}.set_auth_token", patches[f"{MODULE}.set_auth_token"]),
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]),
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                pass
            # Check that the strict warning was logged
            mock_logger.warning.assert_any_call(
                "ENCRYPTION_STRICT is set but encryption is disabled. Strict mode has no effect."
            )


@pytest.mark.asyncio
class TestLifespanEncryptionEnabled:
    """Lifespan when encryption_enabled=True."""

    async def test_key_loaded(self):
        """Existing key loaded (was_generated=False)."""
        fake_key = b"\x00" * 32
        settings = _make_settings(encryption_enabled=True)
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.load_or_generate_key", return_value=(fake_key, False)) as mock_load,
            patch(f"{MODULE}.set_crypto_key", patches[f"{MODULE}.set_crypto_key"]) as mock_set_ck,
            patch(f"{MODULE}.set_auth_token", patches[f"{MODULE}.set_auth_token"]) as mock_set_at,
            patch(f"{MODULE}.derive_auth_token", return_value="tok123") as mock_derive,
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]) as mock_store_cls,
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]),
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                mock_load.assert_called_once_with(
                    settings.encryption_key, settings.encryption_key_file
                )
                mock_set_ck.assert_called_once_with(fake_key)
                mock_derive.assert_called_once_with(fake_key)
                mock_set_at.assert_called_once_with("tok123")
                # JobStore receives the crypto key
                mock_store_cls.assert_called_once_with(
                    db_path=settings.db_path, crypto_key=fake_key
                )

    async def test_key_generated(self):
        """Brand-new key generated (was_generated=True) logs banner."""
        fake_key = b"\xab" * 32
        settings = _make_settings(encryption_enabled=True)
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.load_or_generate_key", return_value=(fake_key, True)),
            patch(f"{MODULE}.set_crypto_key", patches[f"{MODULE}.set_crypto_key"]),
            patch(f"{MODULE}.set_auth_token", patches[f"{MODULE}.set_auth_token"]),
            patch(f"{MODULE}.derive_auth_token", return_value="tok456"),
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]),
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                pass
            # The "NEW ENCRYPTION KEY GENERATED" banner should appear
            mock_logger.info.assert_any_call("NEW ENCRYPTION KEY GENERATED")

    async def test_key_init_failure_raises(self):
        """If load_or_generate_key raises, lifespan propagates the error."""
        settings = _make_settings(encryption_enabled=True)
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(
                f"{MODULE}.load_or_generate_key",
                side_effect=ValueError("bad key"),
            ),
            patch(f"{MODULE}.set_crypto_key", patches[f"{MODULE}.set_crypto_key"]),
            patch(f"{MODULE}.set_auth_token", patches[f"{MODULE}.set_auth_token"]),
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]),
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
        ):
            mock_app = MagicMock()
            with pytest.raises(ValueError, match="bad key"):
                async with lifespan(mock_app):
                    pass


@pytest.mark.asyncio
class TestLifespanJobRecovery:
    """Lifespan job recovery and worker configuration."""

    async def test_jobs_recovered_logged(self):
        """When recover_jobs returns > 0, a log message is emitted."""
        settings = _make_settings(encryption_enabled=False)
        patches = _standard_lifespan_patches(settings)
        patches[f"{MODULE}.job_manager"].recover_jobs = AsyncMock(return_value=5)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.set_crypto_key", patches[f"{MODULE}.set_crypto_key"]),
            patch(f"{MODULE}.set_auth_token", patches[f"{MODULE}.set_auth_token"]),
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]) as mock_jm,
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                pass
            mock_jm.recover_jobs.assert_awaited_once()
            mock_logger.info.assert_any_call(f"Recovered {5} jobs from previous session")

    async def test_worker_config_applied(self):
        """max_concurrent, max_jobs, and worker handler are set before start."""
        settings = _make_settings(
            encryption_enabled=False,
            job_queue_max_concurrent=4,
            job_queue_max_jobs=200,
        )
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.set_crypto_key", patches[f"{MODULE}.set_crypto_key"]),
            patch(f"{MODULE}.set_auth_token", patches[f"{MODULE}.set_auth_token"]),
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]) as mock_jm,
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
            patch(f"{MODULE}.job_worker_handler") as mock_handler,
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                assert mock_jm.max_concurrent == 4
                assert mock_jm.max_jobs == 200
                mock_jm.set_worker_handler.assert_called_once_with(mock_handler)

    async def test_job_ttl_set_from_settings(self):
        """job_manager.job_ttl is set to timedelta from settings."""
        settings = _make_settings(encryption_enabled=False, job_retention_hours=48)
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.set_crypto_key", patches[f"{MODULE}.set_crypto_key"]),
            patch(f"{MODULE}.set_auth_token", patches[f"{MODULE}.set_auth_token"]),
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]) as mock_jm,
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                assert mock_jm.job_ttl == timedelta(hours=48)


@pytest.mark.asyncio
class TestLifespanAuthToken:
    """Auth token derivation during lifespan startup."""

    async def test_auth_token_derived_when_crypto_key_present(self):
        """When encryption is enabled, auth token is derived and set."""
        fake_key = b"\x01" * 32
        settings = _make_settings(encryption_enabled=True)
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.load_or_generate_key", return_value=(fake_key, False)),
            patch(f"{MODULE}.set_crypto_key"),
            patch(f"{MODULE}.set_auth_token") as mock_set_at,
            patch(f"{MODULE}.derive_auth_token", return_value="derived_token") as mock_derive,
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]),
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                pass
            mock_derive.assert_called_once_with(fake_key)
            mock_set_at.assert_called_once_with("derived_token")
            mock_logger.info.assert_any_call(
                "Auth token derived from encryption key (X-Auth-Token header required)"
            )

    async def test_no_auth_token_when_no_crypto_key(self):
        """When encryption is disabled, auth token is set to None."""
        settings = _make_settings(encryption_enabled=False)
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.set_crypto_key"),
            patch(f"{MODULE}.set_auth_token") as mock_set_at,
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]),
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                pass
            mock_set_at.assert_called_once_with(None)
            mock_logger.warning.assert_any_call(
                "No encryption key, auth token disabled. All endpoints are open."
            )


@pytest.mark.asyncio
class TestLifespanApiKeyWarning:
    """Lifespan logs a warning when the OpenRouter API key is missing."""

    async def test_warning_when_api_key_empty(self):
        settings = _make_settings(encryption_enabled=False, openrouter_api_key="")
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.set_crypto_key"),
            patch(f"{MODULE}.set_auth_token"),
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]),
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                pass
            mock_logger.warning.assert_any_call("OpenRouter API key is not configured!")

    async def test_info_when_api_key_set(self):
        settings = _make_settings(encryption_enabled=False, openrouter_api_key="sk-real")
        patches = _standard_lifespan_patches(settings)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.set_crypto_key"),
            patch(f"{MODULE}.set_auth_token"),
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]),
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]),
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                pass
            mock_logger.info.assert_any_call("OpenRouter API key is configured")


@pytest.mark.asyncio
class TestLifespanShutdown:
    """Verify shutdown cleans up resources."""

    async def test_store_closed_on_shutdown(self):
        settings = _make_settings(encryption_enabled=False)
        patches = _standard_lifespan_patches(settings)
        mock_store = MagicMock()
        patches[f"{MODULE}.JobStore"] = MagicMock(return_value=mock_store)

        from subtitle_translator.main import lifespan

        with (
            patch(f"{MODULE}.get_settings", patches[f"{MODULE}.get_settings"]),
            patch(f"{MODULE}.set_crypto_key"),
            patch(f"{MODULE}.set_auth_token"),
            patch(f"{MODULE}.JobStore", patches[f"{MODULE}.JobStore"]),
            patch(f"{MODULE}.job_manager", patches[f"{MODULE}.job_manager"]) as mock_jm,
            patch(f"{MODULE}.close_translator", patches[f"{MODULE}.close_translator"]) as mock_ct,
        ):
            mock_app = MagicMock()
            async with lifespan(mock_app):
                pass

            mock_jm.stop_workers.assert_awaited_once()
            mock_store.close.assert_called_once()
            mock_ct.assert_awaited_once()
