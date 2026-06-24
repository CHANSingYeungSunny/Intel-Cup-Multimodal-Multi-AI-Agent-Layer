"""
Shared fixtures for the Single AI Agent Layer test suite.

Pure unit tests (trend_analyzer, decision_engine, advice_generator) run
without any external dependencies.  Integration tests (test_api) require
SQLAlchemy + aiosqlite; they are skipped when those are not installed.
"""

import sys
import os

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Try to import SQLAlchemy (optional — only needed for integration tests)
# ---------------------------------------------------------------------------
_SQLALCHEMY_AVAILABLE = False
try:
    import sqlalchemy  # noqa: F401
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    _SQLALCHEMY_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

_AIOHTTP_AVAILABLE = False
try:
    from httpx import ASGITransport, AsyncClient
    _AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

# Import test helper — requires the app to be importable
_APP_AVAILABLE = False
try:
    import models  # noqa: F401 — registers ORM classes with Base.metadata
    from database import Base, get_db
    _APP_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Test database (SQLite, unique file per test session via tmp_path_factory)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sqlalchemy_available():
    """True when SQLAlchemy + aiosqlite are importable."""
    return _SQLALCHEMY_AVAILABLE and _APP_AVAILABLE


@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    """Create a unique temporary directory for the test database."""
    return str(tmp_path_factory.mktemp("test_db") / "test_agent.db")


# Check for pytest_asyncio
_pytest_asyncio_available = False
try:
    import pytest_asyncio
    _pytest_asyncio_available = True
except ImportError:
    pass


# Only define async fixtures when the required packages are installed.
if _SQLALCHEMY_AVAILABLE and _APP_AVAILABLE and _pytest_asyncio_available:

    @pytest_asyncio.fixture  # function-scoped to avoid ScopeMismatch
    async def async_engine(test_db_path):
        """Create a function-scoped async SQLite engine with a unique file."""
        url = f"sqlite+aiosqlite:///{test_db_path}"
        engine = create_async_engine(url, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield engine
        await engine.dispose()

    @pytest_asyncio.fixture
    async def async_session(async_engine):
        """Create a fresh async session per test (rolled back after)."""
        factory = async_sessionmaker(
            bind=async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with factory() as session:
            async with session.begin():
                yield session
                await session.rollback()

    @pytest_asyncio.fixture
    async def async_client(async_engine):
        """Return an httpx AsyncClient pointed at the FastAPI app."""
        if not _AIOHTTP_AVAILABLE:
            pytest.skip("httpx not installed")

        from main import create_app

        app = create_app()

        # Replace the DB dependency with the test session
        async def override_get_db():
            factory = async_sessionmaker(
                bind=async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            async with factory() as session:
                try:
                    yield session
                finally:
                    await session.close()

        app.dependency_overrides[get_db] = override_get_db

        # Manually initialise app.state with HealthAgent (bypass the lifespan)
        from agent_orchestrator import HealthAgent

        # Create a session factory bound to the test engine
        def test_db_factory():
            return async_sessionmaker(
                bind=async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )()

        agent = HealthAgent(db_session_factory=test_db_factory)
        app.state.agent = agent
        app.state.db_connected = True

        # Seed rules directly into the in-memory list
        from config import DEFAULT_DECISION_RULES
        agent.decision_engine._rules = list(DEFAULT_DECISION_RULES)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

else:
    # When SQLAlchemy is not available, provide a dummy async_client that skips
    @pytest.fixture
    def async_client(sqlalchemy_available):
        if not sqlalchemy_available:
            pytest.skip("SQLAlchemy + aiosqlite not installed — skipping integration tests")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_trend_summary(
    trend="stable",
    unhealthy_ratio=0.0,
    healthy_ratio=0.0,
    hr_slope=0.0,
    spo2_slope=0.0,
    rr_slope=0.0,
):
    """Build a minimal trend_summary dict for testing."""
    return {
        "trend": trend,
        "history_size": 10,
        "trend_window_size": 10,
        "unhealthy_ratio": unhealthy_ratio,
        "healthy_ratio": healthy_ratio,
        "hr_slope": hr_slope,
        "spo2_slope": spo2_slope,
        "rr_slope": rr_slope,
        "recent_predictions": [0, 0, 1, 2, 1, 0, 0, 1, 0, 0],
    }
