"""
Shared fixtures for the Multi-AI Agent Layer test suite.

Sets up:
- sys.path injection for both Single and Multi layer imports
- Two aiosqlite engines (Single-layer Base + Multi-layer MultiBase)
- async_client fixture for integration tests
- Standalone fixtures: mcp_server, coordinator, skills
"""

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Path injection — Multi layer root + Single layer
# ---------------------------------------------------------------------------
_MULTI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SINGLE_DIR = os.path.join(
    _MULTI_DIR, "intel multimodal (AI_Agent_Single_layer)"
)

if _SINGLE_DIR not in sys.path:
    sys.path.insert(0, _SINGLE_DIR)
if _MULTI_DIR not in sys.path:
    sys.path.insert(0, _MULTI_DIR)  # Multi layer must be first for its config/database/schemas to shadow

# Force-load Multi-layer shadow modules FIRST before any other Multi imports
import _ensure_imports  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Check availability of optional packages
# ---------------------------------------------------------------------------
_SQLALCHEMY_AVAILABLE = False
try:
    import sqlalchemy  # noqa: F401
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    pass

_AIOHTTP_AVAILABLE = False
try:
    from httpx import ASGITransport, AsyncClient

    _AIOHTTP_AVAILABLE = True
except ImportError:
    pass

_pytest_asyncio_available = False
try:
    import pytest_asyncio

    _pytest_asyncio_available = True
except ImportError:
    pass

_APP_AVAILABLE = False
try:
    from base import MultiBase  # noqa: F401
    from database import Base as SingleBase  # noqa: F401

    _APP_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Test database paths
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sqlalchemy_available():
    return _SQLALCHEMY_AVAILABLE and _APP_AVAILABLE


@pytest.fixture(scope="session")
def test_db_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("multi_test_db")


# ===========================================================================
# Async Database Fixtures
# ===========================================================================
if _SQLALCHEMY_AVAILABLE and _APP_AVAILABLE and _pytest_asyncio_available:

    @pytest_asyncio.fixture
    async def single_engine(test_db_dir):
        """SQLite engine for Single-layer tables."""
        url = f"sqlite+aiosqlite:///{test_db_dir}/single_test.db"
        engine = create_async_engine(url, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SingleBase.metadata.create_all)
        yield engine
        await engine.dispose()

    @pytest_asyncio.fixture
    async def multi_engine(test_db_dir):
        """SQLite engine for Multi-layer tables."""
        url = f"sqlite+aiosqlite:///{test_db_dir}/multi_test.db"
        engine = create_async_engine(url, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(MultiBase.metadata.create_all)
        yield engine
        await engine.dispose()

    @pytest_asyncio.fixture
    async def single_session(single_engine):
        """Async session for Single-layer tables."""
        factory = async_sessionmaker(
            bind=single_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with factory() as session:
            async with session.begin():
                yield session
                await session.rollback()

    @pytest_asyncio.fixture
    async def multi_session(multi_engine):
        """Async session for Multi-layer tables."""
        factory = async_sessionmaker(
            bind=multi_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with factory() as session:
            async with session.begin():
                yield session
                await session.rollback()

    @pytest_asyncio.fixture
    async def async_client(single_engine, multi_engine):
        """httpx AsyncClient pointed at the Multi-AI Agent Layer FastAPI app."""
        if not _AIOHTTP_AVAILABLE:
            pytest.skip("httpx not installed")

        # Must import models to register ORM mappings
        import models as _multi_models  # noqa: F401
        import models as _single_models_ref  # noqa: F401

        from main import create_multi_app
        from database import get_db as original_get_db
        from database import get_multi_db as original_get_multi_db
        from database import Base as SingleBaseLocal

        app = create_multi_app()

        # Override Single-layer get_db → SQLite
        async def override_get_db():
            factory = async_sessionmaker(
                bind=single_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            async with factory() as session:
                try:
                    yield session
                finally:
                    await session.close()

        # Override Multi-layer get_multi_db → SQLite
        async def override_get_multi_db():
            factory = async_sessionmaker(
                bind=multi_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            async with factory() as session:
                try:
                    yield session
                finally:
                    await session.close()

        app.dependency_overrides[original_get_db] = override_get_db
        app.dependency_overrides[original_get_multi_db] = override_get_multi_db

        # Initialize app.state manually (bypass PostgreSQL lifespan)
        from agent_orchestrator import HealthAgent
        from mcp_server import MCPServer
        from agent_coordinator import AgentCoordinator
        from skills import AnomalyDetector, AdvancedTrendAnalyzer
        from config import DEFAULT_DECISION_RULES

        # Single-layer DB factory
        def single_db_factory():
            return async_sessionmaker(
                bind=single_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )()

        # Multi-layer DB factory
        def multi_db_factory():
            return async_sessionmaker(
                bind=multi_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )()

        # HealthAgent
        agent = HealthAgent(db_session_factory=single_db_factory)
        agent.decision_engine._rules = list(DEFAULT_DECISION_RULES)
        app.state.agent = agent

        # MCPServer
        mcp = MCPServer(db_session_factory=multi_db_factory)
        app.state.mcp_server = mcp

        # Skills
        skills = {
            "anomaly_detector": AnomalyDetector(window_size=15, zscore_threshold=2.5),
            "advanced_trend_analyzer": AdvancedTrendAnalyzer(
                window_sizes=[5, 10, 30], forecast_horizon=3
            ),
        }
        app.state.skills = skills

        # AgentCoordinator
        coordinator = AgentCoordinator(
            mcp_server=mcp,
            skills=skills,
            db_session_factory=multi_db_factory,
            single_agent=agent,
            llm_advice_generator=None,
        )
        app.state.coordinator = coordinator

        # Register agents
        await mcp.register_agent(
            agent_id="health_agent",
            agent_type="health",
            capabilities=["health_assessment", "advice_generation"],
        )
        await mcp.register_agent(
            agent_id="anomaly_detector",
            agent_type="anomaly_detector",
            capabilities=["anomaly_detector"],
        )

        app.state.db_connected = True
        app.state.multi_db_connected = True

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client

else:
    @pytest.fixture
    def async_client(sqlalchemy_available):
        if not sqlalchemy_available:
            pytest.skip(
                "SQLAlchemy + aiosqlite not installed — skipping integration tests"
            )


# ===========================================================================
# Standalone fixtures (no HTTP needed)
# ===========================================================================

@pytest.fixture
def mcp_server():
    """Standalone MCPServer (no DB persistence)."""
    from mcp_server import MCPServer
    return MCPServer(db_session_factory=None)


@pytest.fixture
def anomaly_detector():
    """Standalone AnomalyDetector for unit tests."""
    from skills import AnomalyDetector
    return AnomalyDetector(window_size=15, zscore_threshold=2.5)


@pytest.fixture
def advanced_trend_analyzer():
    """Standalone AdvancedTrendAnalyzer for unit tests."""
    from skills import AdvancedTrendAnalyzer
    return AdvancedTrendAnalyzer(
        window_sizes=[5, 10, 30], forecast_horizon=3
    )


@pytest.fixture
def llm_generator():
    """Standalone LLMAdviceGenerator (passthrough mode)."""
    from skills import LLMAdviceGenerator
    return LLMAdviceGenerator(backend="none")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tick_payload(
    prediction=0,
    subject_id="test_subject",
    hr_sim=80.0,
    spo2_sim=97.0,
    rr_sim=0.85,
):
    """Build a minimal TickRequest payload dict."""
    return {
        "prediction": prediction,
        "subject_id": subject_id,
        "hr_sim": hr_sim,
        "spo2_sim": spo2_sim,
        "rr_sim": rr_sim,
    }


def make_rule_payload(
    rule_id="test_rule",
    name="Test Rule",
    condition=None,
    result_severity="medium",
    result_condition="Test Condition",
    result_advice="Test advice text.",
    result_actions=None,
    priority=99,
):
    """Build a RuleCreateRequest payload dict."""
    return {
        "rule_id": rule_id,
        "name": name,
        "condition": condition or {"current_prediction": 1, "trend": "stable"},
        "result_severity": result_severity,
        "result_condition": result_condition,
        "result_advice": result_advice,
        "result_actions": result_actions or ["test_action"],
        "priority": priority,
    }
