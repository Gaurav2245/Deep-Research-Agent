"""Health check endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Response

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "deep-research-agent-api",
        "version": "2.0.0"
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        from database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        return Response(
            status_code=503,
            content=f'{{"status": "not ready", "error": "{str(e)}"}}',
            media_type="application/json"
        )
