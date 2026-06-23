"""
Multi AI Agent Layer — FastAPI entry point.

Evolves the Single AI Agent Layer into a multi-agent system with
MCP (Memory-Control-Planning) orchestration, specialized skills,
and agent coordination.

Quick start::

    uvicorn main:create_multi_app --factory --reload
    # OpenAPI docs at http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Inject Single AI Agent Layer into sys.path for imports
# ---------------------------------------------------------------------------
_SINGLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "intel multimodal (AI_Agent_Single_layer)",
)
if _SINGLE_DIR not in sys.path:
    sys.path.insert(0, _SINGLE_DIR)

# Force-load Multi-layer shadow modules (config, database, schemas, models)
# so that they transparently re-export Single-layer equivalents plus
# Multi-specific additions.
import _ensure_imports  # noqa: E402, F401 — must be first

# ---------------------------------------------------------------------------
# Multi-layer imports
# ---------------------------------------------------------------------------
from config import (  # noqa: E402
    APP_TITLE,
    APP_VERSION,
    API_V1_PREFIX,
    MULTI_API_PREFIX,
    MCP_API_PREFIX,
    CORS_ORIGINS,
    MCP_ENABLED,
    MCP_MEMORY_MAX_ENTRIES,
    MCP_DEFAULT_TTL_SECONDS,
    AGENT_COORDINATOR_MAX_WORKERS,
    AGENT_TIMEOUT_SECONDS,
    SKILLS_ENABLED,
    LLM_BACKEND,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_LOCAL_ENDPOINT,
    ANOMALY_DETECTOR_ZSCORE_THRESHOLD,
    ANOMALY_DETECTOR_WINDOW,
    ADVANCED_TREND_WINDOWS,
    FORECAST_HORIZON,
)

from database import (  # noqa: E402
    get_db,
    create_tables,
    dispose_engine,
    AsyncSessionLocal,
    get_multi_db,
    create_multi_tables,
    dispose_multi_engine,
    MultiAsyncSessionLocal,
)

from mcp_server import MCPServer  # noqa: E402
from agent_coordinator import AgentCoordinator  # noqa: E402
from schemas import (  # noqa: E402
    TickRequest,
    AdviceResponse,
    AdviceHistoryResponse,
    TrendResponse,
    TrendHistoryResponse,
    RuleResponse,
    RuleCreateRequest,
    StatusResponse,
    HealthResponse,
    MultiAdviceResponse,
    MultiTrendResponse,
    AnomalyResponse,
    SkillsResponse,
    AgentsResponse,
    MCPStatusResponse,
    MultiTickResponse,
    WorkflowRequest,
    WorkflowResponse,
    AgentRegisterRequest,
    SkillsExecuteRequest,
    utc_now_iso,
)

# Single-layer imports (after path injection)
# noinspection PyUnresolvedReferences
from agent_orchestrator import HealthAgent  # noqa: E402
from models import AdviceLog  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("multi_agent_layer")

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
router_v1 = APIRouter()   # Backward-compatible single-agent routes
router_multi = APIRouter()  # Multi-agent extended routes
router_mcp = APIRouter()    # MCP server routes

# ===========================================================================
# Lifespan — startup / shutdown
# ===========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Extended application lifespan.

    Startup:
    1. Create Single-layer DB tables.
    2. Create Multi-layer DB tables.
    3. Instantiate HealthAgent (Single layer).
    4. Seed decision rules.
    5. Instantiate MCPServer.
    6. Instantiate skills.
    7. Instantiate AgentCoordinator.
    8. Register skills as logical agents in MCP.

    Shutdown:
    Dispose both engines.
    """
    # --- Startup ------------------------------------------------------------
    logger.info("Starting Multi AI Agent Layer v%s", APP_VERSION)
    db_available = False

    # 1. Single-layer DB tables
    try:
        await create_tables()
        logger.info("Single-layer database tables verified / created")
        db_available = True
    except Exception as exc:
        logger.warning(
            "Single-layer database not available — in-memory mode (%s)", exc
        )

    # 2. Multi-layer DB tables
    multi_db_available = False
    try:
        await create_multi_tables()
        logger.info("Multi-layer database tables verified / created")
        multi_db_available = True
    except Exception as exc:
        logger.warning(
            "Multi-layer database not available (%s)", exc
        )

    # 3. HealthAgent (Single layer)
    agent = HealthAgent(
        db_session_factory=AsyncSessionLocal if db_available else None
    )
    app.state.agent = agent

    # 4. Seed decision rules
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
        from config import DEFAULT_DECISION_RULES
        agent.decision_engine._rules = list(DEFAULT_DECISION_RULES)
        logger.info(
            "Loaded %d default rules (in-memory)",
            agent.decision_engine.get_rule_count(),
        )

    # 5. MCPServer
    mcp_db_factory = MultiAsyncSessionLocal if multi_db_available else None
    mcp_server = MCPServer(
        db_session_factory=mcp_db_factory,
        memory_max_entries=MCP_MEMORY_MAX_ENTRIES,
        memory_default_ttl=MCP_DEFAULT_TTL_SECONDS,
        agent_timeout=AGENT_TIMEOUT_SECONDS,
    )
    app.state.mcp_server = mcp_server

    # 6. Skills
    skills = {}
    llm_gen = None

    if "anomaly_detector" in SKILLS_ENABLED:
        from skills import AnomalyDetector
        skills["anomaly_detector"] = AnomalyDetector(
            window_size=ANOMALY_DETECTOR_WINDOW,
            zscore_threshold=ANOMALY_DETECTOR_ZSCORE_THRESHOLD,
        )
        logger.info("AnomalyDetector skill loaded")

    if "advanced_trend_analyzer" in SKILLS_ENABLED:
        from skills import AdvancedTrendAnalyzer
        skills["advanced_trend_analyzer"] = AdvancedTrendAnalyzer(
            window_sizes=ADVANCED_TREND_WINDOWS,
            forecast_horizon=FORECAST_HORIZON,
        )
        logger.info("AdvancedTrendAnalyzer skill loaded")

    if "llm_advice_generator" in SKILLS_ENABLED:
        from skills import LLMAdviceGenerator
        llm_gen = LLMAdviceGenerator(
            backend=LLM_BACKEND,
            api_key=LLM_API_KEY if LLM_API_KEY else None,
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            local_endpoint=LLM_LOCAL_ENDPOINT,
        )
        skills["llm_advice_generator"] = llm_gen
        logger.info("LLMAdviceGenerator skill loaded (backend=%s)", LLM_BACKEND)

    # 7. AgentCoordinator
    coordinator = AgentCoordinator(
        mcp_server=mcp_server,
        skills=skills,
        db_session_factory=mcp_db_factory,
        single_agent=agent,
        llm_advice_generator=llm_gen,
    )
    app.state.coordinator = coordinator

    # 8. Register skills as logical agents in MCP
    for skill_name, skill_obj in skills.items():
        await mcp_server.register_agent(
            agent_id=skill_name,
            agent_type=skill_name,
            capabilities=[skill_name],
            metadata={"internal": True, "class": type(skill_obj).__name__},
        )

    # Register the health agent
    await mcp_server.register_agent(
        agent_id="health_agent",
        agent_type="health",
        capabilities=[
            "health_assessment",
            "trend_analysis",
            "advice_generation",
            "rule_evaluation",
        ],
        metadata={"internal": True, "class": "HealthAgent"},
    )

    # DB connectivity flags
    app.state.db_connected = db_available
    app.state.multi_db_connected = multi_db_available

    logger.info(
        "Startup complete — API ready at %s | Multi at %s | MCP at %s",
        API_V1_PREFIX,
        MULTI_API_PREFIX,
        MCP_API_PREFIX,
    )

    yield

    # --- Shutdown -----------------------------------------------------------
    logger.info("Shutting down — disposing database engines")
    await dispose_engine()
    await dispose_multi_engine()


# ===========================================================================
# App factory
# ===========================================================================


def create_multi_app() -> FastAPI:
    """Create and configure the Multi-AI Agent Layer FastAPI application."""
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description=(
            "Multi-AI Agent Layer — evolves the Single AI Agent Layer "
            "into a multi-agent system with MCP (Memory-Control-Planning) "
            "orchestration, specialized skills (anomaly detection, advanced "
            "trend analysis, LLM advice generation), and agent coordination. "
            "Fully backward-compatible with the Single AI Agent Layer API."
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

    app.include_router(router_v1, prefix=API_V1_PREFIX)
    app.include_router(router_multi, prefix=MULTI_API_PREFIX)
    app.include_router(router_mcp, prefix=MCP_API_PREFIX)

    return app


# ===========================================================================
# Helpers
# ===========================================================================


def _get_agent(request: Request) -> HealthAgent:
    """Return the HealthAgent singleton from app state."""
    return request.app.state.agent


def _get_coordinator(request: Request) -> AgentCoordinator:
    """Return the AgentCoordinator singleton from app state."""
    return request.app.state.coordinator


def _get_mcp(request: Request) -> MCPServer:
    """Return the MCPServer singleton from app state."""
    return request.app.state.mcp_server


# ===========================================================================
# ROUTER A — /api/v1  (11 backward-compatible endpoints)
#
# These mirror the Single AI Agent Layer exactly. All call
# request.app.state.agent (HealthAgent) directly.
# ===========================================================================


# -------------------------------------------------------------------
# POST /tick
# -------------------------------------------------------------------

@router_v1.post(
    "/tick",
    response_model=Optional[AdviceResponse],
    summary="Process a health observation tick",
    response_description="Advice dict, or null when deduplicated (unchanged)",
)
async def process_tick(
    request: Request,
    body: TickRequest,
    db=Depends(get_db),
):
    """
    Process one health observation tick through the agent pipeline.

    Backward-compatible: returns the Single-layer HealthAgent advice.
    Multi-agent results are computed asynchronously and available at
    ``/api/v1/multi/advice``.
    """
    agent: HealthAgent = _get_agent(request)

    # Run the full multi-agent pipeline in the background
    coordinator: AgentCoordinator = _get_coordinator(request)
    import asyncio
    asyncio.create_task(
        coordinator.process_tick_multi(
            prediction=body.prediction,
            subject_id=body.subject_id,
            feature_vector=body.feature_vector,
            hr_sim=body.hr_sim,
            spo2_sim=body.spo2_sim,
            rr_sim=body.rr_sim,
        )
    )

    # Return the Single-layer advice for backward compatibility
    advice = await agent.process_tick(
        prediction=body.prediction,
        subject_id=body.subject_id,
        feature_vector=body.feature_vector,
        hr_sim=body.hr_sim,
        spo2_sim=body.spo2_sim,
        rr_sim=body.rr_sim,
    )

    request.app.state.db_connected = (
        agent.trend_analyzer.get_history_size() > 0
    )

    return advice


# -------------------------------------------------------------------
# POST /reset
# -------------------------------------------------------------------

@router_v1.post(
    "/reset",
    status_code=200,
    summary="Reset agent state",
    response_description="Confirmation message",
)
async def reset_agent(request: Request):
    """Clear all agent state including multi-agent caches."""
    agent: HealthAgent = _get_agent(request)
    agent.reset()

    # Also reset multi-agent state
    coordinator: AgentCoordinator = _get_coordinator(request)
    for skill in coordinator._skills.values():
        if hasattr(skill, "reset"):
            skill.reset()
    coordinator._latest_multi_advice = None
    coordinator._latest_anomalies = []
    coordinator._latest_multi_trend = None

    return {"message": "Agent state has been reset", "status": "ok"}


# -------------------------------------------------------------------
# GET /advice/current
# -------------------------------------------------------------------

@router_v1.get(
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

@router_v1.get(
    "/advice/history",
    response_model=AdviceHistoryResponse,
    summary="Get recent advice history",
)
async def get_advice_history(
    request: Request,
    n: int = Query(20, ge=1, le=100, description="Number of entries (1-100)"),
    db=Depends(get_db),
):
    """Return the most recent *n* advice entries."""
    agent: HealthAgent = _get_agent(request)
    result = agent.get_advice_history(n)

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

@router_v1.get(
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

@router_v1.get(
    "/trends/history",
    response_model=TrendHistoryResponse,
    summary="Get historical trend snapshots",
)
async def get_trend_history(
    request: Request,
    window: int = Query(100, ge=1, le=1000, description="Max snapshots (1-1000)"),
    db=Depends(get_db),
):
    """Return historical trend snapshots from PostgreSQL."""
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

@router_v1.get(
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

@router_v1.post(
    "/rules",
    response_model=RuleResponse,
    status_code=201,
    summary="Create a new decision rule",
)
async def create_rule(
    request: Request,
    body: RuleCreateRequest,
    db=Depends(get_db),
):
    """Add a new decision rule and persist it to PostgreSQL."""
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

@router_v1.delete(
    "/rules/{rule_id}",
    status_code=204,
    summary="Delete a decision rule",
)
async def delete_rule(
    request: Request,
    rule_id: str,
    db=Depends(get_db),
):
    """Remove a decision rule by its *rule_id*."""
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

@router_v1.get(
    "/status",
    response_model=StatusResponse,
    summary="Get agent status",
)
async def get_status(request: Request):
    """Return lightweight agent status for system heartbeat integration."""
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

@router_v1.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check(
    request: Request,
    db=Depends(get_db),
):
    """Health check — verifies app + DB connectivity."""
    db_ok = False
    try:
        from sqlalchemy import text
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


# ===========================================================================
# ROUTER B — /api/v1/multi  (5 extended multi-agent endpoints)
# ===========================================================================


# -------------------------------------------------------------------
# GET /multi/advice
# -------------------------------------------------------------------

@router_multi.get(
    "/advice",
    response_model=MultiAdviceResponse,
    summary="Get aggregated multi-agent advice",
)
async def get_multi_advice(request: Request):
    """
    Return aggregated advice from all agents including the Single-layer
    HealthAgent, skills, and any registered external agents.
    """
    coordinator: AgentCoordinator = _get_coordinator(request)
    advice = coordinator.get_aggregated_advice()
    if advice:
        return MultiAdviceResponse(**advice)

    # Fallback: build from single-agent advice
    agent: HealthAgent = _get_agent(request)
    single = agent.get_current_advice()
    return MultiAdviceResponse(
        aggregated_advice=single,
        agent_contributions=[],
        consensus_severity=single.get("severity", "low") if single else "low",
    )


# -------------------------------------------------------------------
# GET /multi/trends
# -------------------------------------------------------------------

@router_multi.get(
    "/trends",
    response_model=MultiTrendResponse,
    summary="Get combined multi-scale trend analysis",
)
async def get_multi_trends(request: Request):
    """
    Return combined trend analysis: Single-layer TrendAnalyzer output
    plus multi-scale trends and forecasts from AdvancedTrendAnalyzer.
    """
    agent: HealthAgent = _get_agent(request)
    single_trend = agent.get_trend_summary()

    coordinator: AgentCoordinator = _get_coordinator(request)
    multi_data = coordinator.get_multi_trend()

    if multi_data:
        return MultiTrendResponse(
            single_agent_trend=single_trend,
            multi_scale_trends=multi_data.get("multi_scale_trends", {}),
            forecast=multi_data.get("forecast", {}),
            cross_scale_insight=multi_data.get("cross_scale_insight", ""),
        )

    return MultiTrendResponse(
        single_agent_trend=single_trend,
        multi_scale_trends={},
        forecast={},
        cross_scale_insight="Advanced trend analyzer has insufficient data.",
    )


# -------------------------------------------------------------------
# GET /multi/anomalies
# -------------------------------------------------------------------

@router_multi.get(
    "/anomalies",
    response_model=AnomalyResponse,
    summary="Get recent anomaly events",
)
async def get_anomalies(
    request: Request,
    n: int = Query(20, ge=1, le=100, description="Number of events (1-100)"),
    multi_db=Depends(get_multi_db),
):
    """
    Return recent anomaly detection events.

    First checks in-memory cache, then falls back to the database.
    """
    coordinator: AgentCoordinator = _get_coordinator(request)
    anomalies = coordinator.get_anomalies(n)

    # Supplement from DB if needed
    if len(anomalies) < n and request.app.state.multi_db_connected:
        try:
            from sqlalchemy import select, desc
            from models import AnomalyEvent

            stmt = (
                select(AnomalyEvent)
                .order_by(desc(AnomalyEvent.timestamp))
                .limit(n)
            )
            result = await multi_db.execute(stmt)
            db_anomalies = list(result.scalars().all())
            anomalies = [a.to_dict() for a in reversed(db_anomalies[:n])]
        except Exception as exc:
            logger.warning("Failed to query anomalies from DB: %s", exc)

    return AnomalyResponse(
        detected=len(anomalies) > 0,
        anomalies=anomalies,
    )


# -------------------------------------------------------------------
# POST /multi/skills
# -------------------------------------------------------------------

@router_multi.post(
    "/skills",
    response_model=SkillsResponse,
    summary="Execute skills on demand",
)
async def execute_skills(request: Request, body: SkillsExecuteRequest):
    """
    Execute one or more skills with provided input data.

    Available skills: anomaly_detector, advanced_trend_analyzer,
    llm_advice_generator.
    """
    coordinator: AgentCoordinator = _get_coordinator(request)
    results = await coordinator.execute_skills(body.skill_names, body.input)

    # Build aggregate summary
    success_count = sum(1 for r in results if r["status"] == "success")
    summary = (
        f"Executed {len(results)} skill(s): "
        f"{success_count} succeeded, {len(results) - success_count} failed."
    )

    return SkillsResponse(
        skill_results=results,
        aggregate_summary=summary,
    )


# -------------------------------------------------------------------
# GET /multi/agents
# -------------------------------------------------------------------

@router_multi.get(
    "/agents",
    response_model=AgentsResponse,
    summary="List registered agents",
)
async def list_agents(
    request: Request,
    agent_type: Optional[str] = Query(
        None, description="Filter by agent type"
    ),
):
    """Return all agents registered with the MCP server."""
    mcp: MCPServer = _get_mcp(request)
    agents = await mcp.list_agents(agent_type)
    return AgentsResponse(agents=agents, count=len(agents))


# ===========================================================================
# ROUTER C — /api/v1/mcp  (5 MCP server endpoints)
# ===========================================================================


# -------------------------------------------------------------------
# GET /mcp/status
# -------------------------------------------------------------------

@router_mcp.get(
    "/status",
    response_model=MCPStatusResponse,
    summary="MCP server status",
)
async def get_mcp_status(request: Request):
    """Return MCP server status summary."""
    mcp: MCPServer = _get_mcp(request)
    status = mcp.get_status()
    return MCPStatusResponse(**status)


# -------------------------------------------------------------------
# POST /mcp/agents
# -------------------------------------------------------------------

@router_mcp.post(
    "/agents",
    response_model=dict,
    status_code=201,
    summary="Register a new agent",
)
async def register_agent(request: Request, body: AgentRegisterRequest):
    """Register a new agent with the MCP server."""
    mcp: MCPServer = _get_mcp(request)
    agent_info = await mcp.register_agent(
        agent_id=body.agent_id,
        agent_type=body.agent_type,
        capabilities=body.capabilities,
        endpoint_url=body.endpoint_url,
        metadata=body.metadata,
    )
    return agent_info


# -------------------------------------------------------------------
# DELETE /mcp/agents/{agent_id}
# -------------------------------------------------------------------

@router_mcp.delete(
    "/agents/{agent_id}",
    status_code=204,
    summary="Deregister an agent",
)
async def deregister_agent(request: Request, agent_id: str):
    """Remove an agent from the registry."""
    mcp: MCPServer = _get_mcp(request)
    existed = await mcp.deregister_agent(agent_id)
    if not existed:
        raise HTTPException(
            status_code=404, detail=f"Agent '{agent_id}' not found"
        )


# -------------------------------------------------------------------
# POST /mcp/workflow
# -------------------------------------------------------------------

@router_mcp.post(
    "/workflow",
    response_model=WorkflowResponse,
    summary="Start a multi-agent workflow",
)
async def start_workflow(request: Request, body: WorkflowRequest):
    """
    Plan and initiate a multi-agent workflow for a given goal.

    The MCP Planner decomposes the goal into subtasks, assigns agents,
    resolves dependencies, and creates a workflow session.
    """
    mcp: MCPServer = _get_mcp(request)
    session = await mcp.start_workflow(body.goal, body.context)
    return WorkflowResponse(
        session_id=session["session_id"],
        status=session["status"],
    )


# -------------------------------------------------------------------
# GET /mcp/workflow/{session_id}
# -------------------------------------------------------------------

@router_mcp.get(
    "/workflow/{session_id}",
    summary="Get workflow status",
)
async def get_workflow_status(request: Request, session_id: str):
    """Return the current status of a workflow session."""
    mcp: MCPServer = _get_mcp(request)
    session = await mcp.get_workflow_status(session_id)
    if not session:
        raise HTTPException(
            status_code=404, detail=f"Workflow '{session_id}' not found"
        )
    return session


# ===========================================================================
# Main entry point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:create_multi_app",
        host="0.0.0.0",
        port=8000,
        factory=True,
        reload=True,
    )
