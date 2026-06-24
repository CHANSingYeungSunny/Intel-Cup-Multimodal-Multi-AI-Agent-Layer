"""
Single AI Agent Layer — FastAPI entry point.

A standalone, production-ready microservice that ingests health observations,
detects trends, evaluates configurable decision rules, and emits structured
natural-language health advice.

Quick start::

    uvicorn main:create_app --factory --reload
    # OpenAPI docs at http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    APP_TITLE,
    APP_VERSION,
    API_V1_PREFIX,
    CORS_ORIGINS,
    HISTORY_WINDOW_SIZE,
)
from database import AsyncSessionLocal, get_db, create_tables, dispose_engine
from agent_orchestrator import HealthAgent
from models import AdviceLog
from schemas import (
    TickRequest,
    AdviceResponse,
    AdviceHistoryResponse,
    TrendResponse,
    TrendHistoryResponse,
    RuleResponse,
    RuleCreateRequest,
    StatusResponse,
    HealthResponse,
    utc_now_iso,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent_layer")

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan:

    - **Startup**: Create DB tables, instantiate the HealthAgent orchestrator,
      seed default decision rules from config if the DB is empty.
    - **Shutdown**: Gracefully dispose of the async engine.
    """
    # --- Startup ------------------------------------------------------------
    logger.info("Starting Single AI Agent Layer v%s", APP_VERSION)
    db_available = False

    # 1. Try to create tables (PostgreSQL/asyncpg may not be installed)
    try:
        await create_tables()
        logger.info("Database tables verified / created")
        db_available = True
    except Exception as exc:
        logger.warning(
            "Database not available — running in in-memory-only mode (%s)", exc
        )

    # 2. Instantiate the HealthAgent orchestrator
    #    Use DB session factory only when PostgreSQL is reachable.
    agent = HealthAgent(
        db_session_factory=AsyncSessionLocal if db_available else None
    )
    app.state.agent = agent

    # 3. Load / seed decision rules from DB (or use config defaults)
    if db_available:
        try:
            async with AsyncSessionLocal() as db:
                await agent.decision_engine.load_rules_from_db(db)
                logger.info(
                    "Loaded %d decision rules",
                    agent.decision_engine.get_rule_count(),
                )
        except Exception as exc:
            logger.warning(
                "Could not load rules from DB: %s — using config defaults", exc
            )
    else:
        # Load default rules directly into memory (no DB)
        from config import DEFAULT_DECISION_RULES
        agent.decision_engine._rules = list(DEFAULT_DECISION_RULES)
        logger.info(
            "Loaded %d default rules (in-memory, no database)",
            agent.decision_engine.get_rule_count(),
        )

    # 4. DB connectivity flag
    app.state.db_connected = db_available

    logger.info("Startup complete — API ready at %s", API_V1_PREFIX)

    yield

    # --- Shutdown -----------------------------------------------------------
    logger.info("Shutting down — disposing database engine")
    await dispose_engine()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description=(
            "Standalone AI agent for multimodal health monitoring. "
            "Ingests health observations, detects trends, evaluates "
            "configurable decision rules, and emits structured "
            "natural-language health advice."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=API_V1_PREFIX)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_agent(request: Request) -> HealthAgent:
    """Return the HealthAgent singleton from app state."""
    return request.app.state.agent


# ---------------------------------------------------------------------------
# POST /tick — process one health observation
# ---------------------------------------------------------------------------


@router.post(
    "/tick",
    response_model=Optional[AdviceResponse],
    summary="Process a health observation tick",
    response_description="Advice dict, or null when deduplicated (unchanged)",
)
async def process_tick(
    request: Request,
    body: TickRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Process one health observation tick through the full agent pipeline.

    1. Compute vital-sign proxies from *feature_vector* when explicit
       hr / spo2 / rr are not provided.
    2. Feed the observation into TrendAnalyzer (in-memory + DB).
    3. Evaluate decision rules against the current trend summary.
    4. Generate structured advice via AdviceGenerator.
    5. Deduplicate — return ``null`` when the matched rule is unchanged.
    6. Persist advice to AdviceLog and save a trend snapshot.

    Returns the advice dict, or ``null`` when deduplicated (HTTP 200).
    """
    agent: HealthAgent = _get_agent(request)

    advice = await agent.process_tick(
        prediction=body.prediction,
        subject_id=body.subject_id,
        feature_vector=body.feature_vector,
        hr_sim=body.hr_sim,
        spo2_sim=body.spo2_sim,
        rr_sim=body.rr_sim,
    )

    # Update DB connectivity flag
    request.app.state.db_connected = (
        agent.trend_analyzer.get_history_size() > 0
    )

    return advice


# -------------------------------------------------------------------
# POST /reset — clear agent state
# -------------------------------------------------------------------


@router.post(
    "/reset",
    status_code=200,
    summary="Reset agent state",
    response_description="Confirmation message",
)
async def reset_agent(request: Request):
    """
    Clear all agent state: history buffer, advice history, dedup key,
    and trend data.  Does **not** affect database tables.
    """
    agent: HealthAgent = _get_agent(request)
    agent.reset()
    return {"message": "Agent state has been reset", "status": "ok"}


# -------------------------------------------------------------------
# GET /advice/current
# -------------------------------------------------------------------


@router.get(
    "/advice/current",
    response_model=Optional[AdviceResponse],
    summary="Get the latest advice",
)
async def get_current_advice(request: Request):
    """Return the most recently generated advice, or ``null``."""
    agent: HealthAgent = _get_agent(request)
    return agent.get_current_advice()


# -------------------------------------------------------------------
# GET /advice/history
# -------------------------------------------------------------------


@router.get(
    "/advice/history",
    response_model=AdviceHistoryResponse,
    summary="Get recent advice history",
)
async def get_advice_history(
    request: Request,
    n: int = Query(20, ge=1, le=100, description="Number of entries (1-100)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the most recent *n* advice entries.

    Falls back to the in-memory buffer; queries PostgreSQL when the DB is
    available and the in-memory buffer is insufficient.
    """
    agent: HealthAgent = _get_agent(request)
    result = agent.get_advice_history(n)

    # If in-memory history is too small, supplement from DB
    if len(result) < n and request.app.state.db_connected:
        try:
            from sqlalchemy import select as sa_select

            stmt = (
                sa_select(AdviceLog)
                .order_by(AdviceLog.timestamp.desc())
                .limit(n)
            )
            db_result = await db.execute(stmt)
            db_rows = list(db_result.scalars().all())
            result = [
                {
                    "matched_rule_id": row.matched_rule_id,
                    "matched_rule_name": row.matched_rule_name,
                    "severity": row.severity,
                    "possible_condition": row.possible_condition,
                    "advice": row.advice,
                    "actions": row.actions,
                    "context": row.context,
                    "timestamp": row.timestamp.isoformat()
                    if hasattr(row.timestamp, "isoformat")
                    else str(row.timestamp),
                }
                for row in reversed(db_rows[:n])
            ]
        except Exception as exc:
            logger.warning("Failed to query advice history from DB: %s", exc)

    return AdviceHistoryResponse(history=result, count=len(result))


# -------------------------------------------------------------------
# GET /trends/current
# -------------------------------------------------------------------


@router.get(
    "/trends/current",
    response_model=TrendResponse,
    summary="Get current trend summary",
)
async def get_current_trends(request: Request):
    """Return the current trend-analyzer summary."""
    agent: HealthAgent = _get_agent(request)
    return agent.get_trend_summary()


# -------------------------------------------------------------------
# GET /trends/history
# -------------------------------------------------------------------


@router.get(
    "/trends/history",
    response_model=TrendHistoryResponse,
    summary="Get historical trend snapshots",
)
async def get_trend_history(
    request: Request,
    window: int = Query(100, ge=1, le=1000, description="Max snapshots (1-1000)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return historical trend snapshots from PostgreSQL.

    Returns an empty list when the database is not connected.
    """
    agent: HealthAgent = _get_agent(request)
    try:
        snapshots = await agent.trend_analyzer.get_trend_history(
            db, limit=window
        )
        result = [
            TrendResponse(
                trend=s.trend,
                history_size=s.history_size,
                trend_window_size=s.trend_window_size,
                unhealthy_ratio=s.unhealthy_ratio,
                healthy_ratio=s.healthy_ratio,
                hr_slope=s.hr_slope,
                spo2_slope=s.spo2_slope,
                rr_slope=s.rr_slope,
                recent_predictions=s.recent_predictions or [],
                timestamp=s.timestamp.isoformat()
                if hasattr(s.timestamp, "isoformat")
                else str(s.timestamp),
            )
            for s in reversed(snapshots)
        ]
        return TrendHistoryResponse(snapshots=result, count=len(result))
    except Exception as exc:
        logger.warning("Failed to query trend history: %s", exc)
        return TrendHistoryResponse(snapshots=[], count=0)


# -------------------------------------------------------------------
# GET /rules
# -------------------------------------------------------------------


@router.get(
    "/rules",
    response_model=list[RuleResponse],
    summary="List all decision rules",
)
async def get_rules(request: Request):
    """Return metadata for all active decision rules."""
    agent: HealthAgent = _get_agent(request)
    return agent.get_rules()


# -------------------------------------------------------------------
# POST /rules
# -------------------------------------------------------------------


@router.post(
    "/rules",
    response_model=RuleResponse,
    status_code=201,
    summary="Create a new decision rule",
)
async def create_rule(
    request: Request,
    body: RuleCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Add a new decision rule and persist it to PostgreSQL.

    The rule takes effect immediately after creation.
    """
    agent: HealthAgent = _get_agent(request)
    try:
        rule_dict = await agent.decision_engine.add_rule(
            db, body.model_dump()
        )
        return rule_dict
    except Exception as exc:
        logger.error("Failed to create rule: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# -------------------------------------------------------------------
# DELETE /rules/{rule_id}
# -------------------------------------------------------------------


@router.delete(
    "/rules/{rule_id}",
    status_code=204,
    summary="Delete a decision rule",
)
async def delete_rule(
    request: Request,
    rule_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Remove a decision rule by its *rule_id*.

    Returns 204 on success, 404 if the rule does not exist.
    """
    agent: HealthAgent = _get_agent(request)
    try:
        deleted = await agent.decision_engine.remove_rule(db, rule_id)
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"Rule '{rule_id}' not found"
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete rule: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# -------------------------------------------------------------------
# GET /status
# -------------------------------------------------------------------


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Get agent status",
)
async def get_status(request: Request):
    """
    Return lightweight agent status for system heartbeat integration.
    """
    agent: HealthAgent = _get_agent(request)
    status = agent.get_status()
    return StatusResponse(
        enabled=status["enabled"],
        rules_count=status["rules_count"],
        history_size=status["history_size"],
        latest_severity=status["latest_severity"],
        latest_condition=status["latest_condition"],
        trend=status["trend"],
        db_connected=request.app.state.db_connected,
    )


# -------------------------------------------------------------------
# GET /health
# -------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Health check endpoint.

    Verifies the application is running and tests database connectivity.
    """
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
        request.app.state.db_connected = True
    except Exception:
        request.app.state.db_connected = False

    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        db_connected=db_ok,
    )


# ---------------------------------------------------------------------------
# Main entry point (for ``python main.py``)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:create_app",
        host="0.0.0.0",
        port=8000,
        factory=True,
        reload=True,
    )
