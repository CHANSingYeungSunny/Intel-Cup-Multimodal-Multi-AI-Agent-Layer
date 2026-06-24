"""
Database layer — async SQLAlchemy engine, session factory, and helpers.

Uses ``asyncpg`` for PostgreSQL and provides a ``get_db`` FastAPI dependency
that yields an ``AsyncSession`` per request.

The engine is created lazily on first access so that the module can be
imported without a running database or the ``asyncpg`` driver.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config import DATABASE_URL

# ---------------------------------------------------------------------------
# Lazy engine & session (created on first call to get_engine / get_session)
# ---------------------------------------------------------------------------

_async_engine: Optional = None
_AsyncSessionLocal: Optional = None


def get_async_engine():
    """Return the module-level async engine, creating it on first call."""
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _async_engine


def _get_session_factory():
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _AsyncSessionLocal


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db():
    """Yield an ``AsyncSession`` and close it after the request completes."""
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Table creation (called at lifespan startup)
# ---------------------------------------------------------------------------

async def create_tables():
    """Create all tables that do not yet exist (safe for repeated calls)."""
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine():
    """Gracefully dispose of the async engine at shutdown."""
    global _async_engine, _AsyncSessionLocal
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _AsyncSessionLocal = None


# Lazily-exported session factory for lifespan usage
def AsyncSessionLocal():
    """Lazy accessor for the async session factory."""
    return _get_session_factory()()
