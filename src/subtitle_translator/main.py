"""FastAPI application entry point for AI Subtitle Translator service by LavX."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from subtitle_translator.api.routes import api_router, health_router, jobs_router
from subtitle_translator.config import get_settings
from subtitle_translator.core.translator import close_translator
from subtitle_translator.queue.job_manager import job_manager
from subtitle_translator.queue.worker import job_worker_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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
    
    # Start job queue workers
    logger.info(f"Starting job queue with {settings.job_queue_max_concurrent} workers")
    job_manager.max_concurrent = settings.job_queue_max_concurrent
    job_manager.max_jobs = settings.job_queue_max_jobs
    job_manager.job_ttl = settings.job_queue_ttl
    job_manager.set_worker_handler(job_worker_handler)
    await job_manager.start_workers()
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Subtitle Translator service")
    await job_manager.stop_workers()
    await close_translator()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title="AI Subtitle Translator",
        description=(
            "LLM-powered subtitle translation microservice by LavX. "
            "Originally created for Bazarr but designed as a flexible API for any integration. "
            "Powered by OpenRouter for access to multiple AI models."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(api_router)
    app.include_router(jobs_router)
    
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