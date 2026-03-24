"""FastAPI application entry point for AI Subtitle Translator service by LavX."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from subtitle_translator.api.routes import (
    api_router,
    config_router,
    health_router,
    jobs_router,
    set_crypto_key,
)
from subtitle_translator.config import get_settings
from subtitle_translator.core.translator import close_translator
from subtitle_translator.crypto import load_or_generate_key
from subtitle_translator.queue.job_manager import job_manager
from subtitle_translator.queue.job_store import JobStore
from subtitle_translator.queue.worker import job_worker_handler

# Configure logging - use LOG_LEVEL env var (default: INFO)
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Set debug logger level explicitly for the OpenRouter provider
if log_level == logging.DEBUG:
    logging.getLogger("subtitle_translator.providers.openrouter.debug").setLevel(logging.DEBUG)
    logging.getLogger("subtitle_translator.core.batch_processor.debug").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    logger.info(f"Starting AI Subtitle Translator service on {settings.host}:{settings.port}")
    logger.info(f"Default model: {settings.openrouter_default_model}")

    if not settings.openrouter_api_key:
        logger.warning("OpenRouter API key is not configured!")
    else:
        logger.info("OpenRouter API key is configured")

    # Encryption setup
    crypto_key = None
    if settings.encryption_enabled:
        try:
            crypto_key, was_generated = load_or_generate_key(
                settings.encryption_key, settings.encryption_key_file
            )
            if was_generated:
                logger.info("=" * 70)
                logger.info("NEW ENCRYPTION KEY GENERATED")
                logger.info("Retrieve your key: cat %s", settings.encryption_key_file)
                logger.info("Or via CLI: python -m subtitle_translator.cli regenerate-key")
                logger.info("Copy this key to your Bazarr settings.")
                logger.info("=" * 70)
            else:
                logger.info("Encryption enabled (key loaded)")
        except Exception:
            logger.critical("Encryption is enabled but key initialization failed")
            raise
    else:
        logger.info("Encryption disabled")
        if settings.encryption_strict:
            logger.warning("ENCRYPTION_STRICT is set but encryption is disabled. Strict mode has no effect.")
    set_crypto_key(crypto_key)

    # Job persistence setup
    store = JobStore(db_path=settings.db_path, crypto_key=crypto_key)
    job_manager.set_store(store)
    job_manager.job_ttl = timedelta(hours=settings.job_retention_hours)

    # Recover jobs from previous run
    recovered = await job_manager.recover_jobs()
    if recovered:
        logger.info(f"Recovered {recovered} jobs from previous session")

    # Start job queue workers
    logger.info(f"Starting job queue with {settings.job_queue_max_concurrent} workers")
    job_manager.max_concurrent = settings.job_queue_max_concurrent
    job_manager.max_jobs = settings.job_queue_max_jobs
    job_manager.set_worker_handler(job_worker_handler)
    await job_manager.start_workers()

    yield

    # Shutdown
    logger.info("Shutting down AI Subtitle Translator service")
    await job_manager.stop_workers()
    store.close()
    await close_translator()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    get_settings()

    app = FastAPI(
        title="AI Subtitle Translator",
        description=(
            "LLM-powered subtitle translation microservice by LavX. "
            "Originally created for Bazarr but designed as a flexible API for any integration. "
            "Powered by OpenRouter for access to multiple AI models."
        ),
        version="1.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "*")
    if cors_origins_env == "*":
        cors_origins = ["*"]
        cors_credentials = False
    else:
        cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
        cors_credentials = True
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(api_router)
    app.include_router(jobs_router)
    app.include_router(config_router)

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "subtitle_translator.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
