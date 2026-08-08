"""Database connection and initialization."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import get_database_config
from utils.logger import get_logger

logger = get_logger(__name__)

_db_config = get_database_config()


def get_database_url() -> str:
    """Build PostgreSQL connection URL from centralized config."""
    return _db_config.url


# Create engine with connection pooling
engine = create_engine(
    get_database_url(),
    echo=False,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=_db_config.pool_size,
    max_overflow=_db_config.max_overflow,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for a DB session outside of FastAPI's dependency
    injection (e.g. Streamlit, CLI scripts): `with session_scope() as db: ...`."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def apply_research_schema_patches() -> None:
    """
    Add columns that were introduced after the DB was first created.
    SQLAlchemy create_all() does not ALTER existing tables.
    Safe to call on every API / app startup.
    """
    from sqlalchemy import text

    if engine.dialect.name != "postgresql":
        return
        
    patches = [
        # v2.3 Conversational Cognition fields
        ("understood_intent", "TEXT"),
        ("query_reasoning", "TEXT"),
        ("active_topic", "VARCHAR(255)"),
        ("is_follow_up", "BOOLEAN DEFAULT FALSE"),
        ("entities_resolved", "JSON DEFAULT '{}'::json"),
        ("conversational_knowledge", "JSON DEFAULT '{}'::json"),
    ]
    
    for col_name, col_type in patches:
        patch = text(
            f"""
            DO $body$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'research'
                      AND column_name = '{col_name}'
                ) THEN
                    ALTER TABLE research
                    ADD COLUMN {col_name} {col_type};
                END IF;
            END
            $body$;
            """
        )
        try:
            with engine.begin() as conn:
                conn.execute(patch)
            logger.info(f"Schema patch: research.{col_name} verified/added")
        except Exception as exc:
            logger.warning(f"Schema patch {col_name} failed (non-fatal): {exc}")


def init_db() -> None:
    """Initialize database tables. Run with retries at startup."""
    from .models import Base
    import time
    import sqlalchemy
    
    max_retries = 5
    retry_delay = 5
    
    for i in range(max_retries):
        try:
            logger.info(f"Database initialization attempt {i+1}/{max_retries}...")
            Base.metadata.create_all(bind=engine)
            apply_research_schema_patches()
            logger.info("Database tables initialized successfully")
            return
        except sqlalchemy.exc.OperationalError as e:
            if i < max_retries - 1:
                logger.warning(f"Database not ready ({e}). Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Could not initialize database.")
                raise
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {e}")
            raise
