"""FastAPI application setup and configuration."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.auth import require_api_key
from api.rate_limit import RateLimitMiddleware
from database import init_db
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for app startup and shutdown.
    """
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Deep Research Agent API",
        description="Advanced research agent with semantic search, source filtering, and confidence scoring",
        version="2.0.0",
        lifespan=lifespan,
    )
    
    # Rate limiting: added before CORS so CORS remains the outermost middleware
    # (its headers then apply even to 429 / error responses). Set
    # RATE_LIMIT_REQUESTS=0 to disable.
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=int(os.getenv("RATE_LIMIT_REQUESTS", "120")),
        window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
    )

    # CORS middleware. Origins are configurable via ALLOWED_ORIGINS (comma-separated);
    # defaults to common local dev ports for the Streamlit UI / frontend.
    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000"
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers. Health is always open (used for container health checks);
    # everything under /api/v1 is gated by require_api_key (no-op unless API_KEY is set).
    from .routes import research, sources, validation, health, conversation

    protected = [Depends(require_api_key)]
    app.include_router(health.router, tags=["health"])
    app.include_router(conversation.router, prefix="/api/v1", tags=["conversations"], dependencies=protected)
    app.include_router(research.router, prefix="/api/v1", tags=["research"], dependencies=protected)
    app.include_router(sources.router, prefix="/api/v1", tags=["sources"], dependencies=protected)
    app.include_router(validation.router, prefix="/api/v1", tags=["validation"], dependencies=protected)
    
    # Global error handler
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
