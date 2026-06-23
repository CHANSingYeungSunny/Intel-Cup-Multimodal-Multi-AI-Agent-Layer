"""
Extended database layer for the Multi-AI Agent Layer.

Provides a SEPARATE async engine and DeclarativeBase (MultiBase)
for Multi-layer tables, sharing the same PostgreSQL database as the
Single AI Agent Layer but isolated in metadata.

Uses importlib to load Single-layer DB module to avoid name collisions.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from base import MultiBase  # noqa: E402

# ---------------------------------------------------------------------------
# Locate the Single AI Agent Layer
# ---------------------------------------------------------------------------
_SINGLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "intel multimodal (AI_Agent_Single_layer)",
)

# Load Single-layer database module via importlib
_single_db_path = os.path.join(_SINGLE_DIR, "database.py")
_spec = importlib.util.spec_from_file_location("_single_database", _single_db_path)
_single_db = importlib.util.module_from_spec(_spec)
sys.modules["_single_database"] = _single_db
_spec.loader.exec_module(_single_db)

# Re-export Single-layer DB utilities for backward-compatible routes
Base = _single_db.Base
get_db = _single_db.get_db
create_tables = _single_db.create_tables
dispose_engine = _single_db.dispose_engine
AsyncSessionLocal = _single_db.AsyncSessionLocal

# At module load time, replace 'database' in sys.modules with this Multi
# version (which re-exports everything from the Single version + adds
# MultiBase, get_multi_db, etc.).  Single-layer modules that ``import
# database`` will transparently get Base/get_db from here.
sys.modules["database"] = sys.modules[__name__]

from config import MULTI_AGENT_DB_URL  # noqa: E402

# ---------------------------------------------------------------------------
# Lazy engine & session for Multi-layer tables
# ---------------------------------------------------------------------------

_multi_async_engine: Optional = None
_MultiAsyncSessionLocal: Optional = None


def get_multi_async_engine():
    """Return the Multi-layer async engine, creating it on first call."""
    global _multi_async_engine
    if _multi_async_engine is None:
        _multi_async_engine = create_async_engine(
            MULTI_AGENT_DB_URL,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _multi_async_engine


def _get_multi_session_factory():
    """Lazily create the Multi-layer session factory."""
    global _MultiAsyncSessionLocal
    if _MultiAsyncSessionLocal is None:
        _MultiAsyncSessionLocal = async_sessionmaker(
            bind=get_multi_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _MultiAsyncSessionLocal


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_multi_db():
    """Yield an AsyncSession for Multi-layer tables."""
    factory = _get_multi_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Table creation / engine disposal
# ---------------------------------------------------------------------------

async def create_multi_tables():
    """Create all Multi-layer tables that do not yet exist."""
    engine = get_multi_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(MultiBase.metadata.create_all)


async def dispose_multi_engine():
    """Gracefully dispose of the Multi-layer async engine."""
    global _multi_async_engine, _MultiAsyncSessionLocal
    if _multi_async_engine is not None:
        await _multi_async_engine.dispose()
        _multi_async_engine = None
        _MultiAsyncSessionLocal = None


def MultiAsyncSessionLocal():
    """Lazy accessor for the Multi-layer async session factory."""
    return _get_multi_session_factory()()
