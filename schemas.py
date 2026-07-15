"""
Extended Pydantic v2 request and response schemas for the Multi-AI Agent Layer.

Uses importlib to load Single-layer schemas to avoid module name collision.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Locate the Single AI Agent Layer
# ---------------------------------------------------------------------------
_SINGLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "intel multimodal (AI_Agent_Single_layer)",
)

# Load Single-layer schemas via importlib
_single_schemas_path = os.path.join(_SINGLE_DIR, "schemas.py")
_spec = importlib.util.spec_from_file_location(
    "_single_schemas", _single_schemas_path
)
_single_schemas = importlib.util.module_from_spec(_spec)
sys.modules["_single_schemas"] = _single_schemas
_spec.loader.exec_module(_single_schemas)

# Also pre-cache the Single-layer schemas as 'schemas' in sys.modules
# so that Single-layer modules find the correct version when they
# ``from schemas import ...``.
if "schemas" not in sys.modules:
    sys.modules["schemas"] = _single_schemas

# Replace 'schemas' in sys.modules with this Multi-layer version
# (which re-exports all Single-layer schemas + adds Multi schemas).
sys.modules["schemas"] = sys.modules[__name__]

# Re-export all Single-layer schemas
TickRequest = _single_schemas.TickRequest
AdviceResponse = _single_schemas.AdviceResponse
AdviceHistoryResponse = _single_schemas.AdviceHistoryResponse
TrendResponse = _single_schemas.TrendResponse
TrendHistoryResponse = _single_schemas.TrendHistoryResponse
RuleResponse = _single_schemas.RuleResponse
RuleCreateRequest = _single_schemas.RuleCreateRequest
StatusResponse = _single_schemas.StatusResponse
HealthResponse = _single_schemas.HealthResponse
TickResponse = _single_schemas.TickResponse
utc_now_iso = _single_schemas.utc_now_iso


# ===========================================================================
# Helper
# ===========================================================================

def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Multi-agent schemas
# ===========================================================================

class AgentContribution(BaseModel):
    """Contribution from a single specialized agent."""
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Agent type: health/anomaly/trend/llm")
    advice: dict = Field(..., description="Structured advice dict from this agent")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score (0–1)"
    )


class MultiAdviceResponse(BaseModel):
    """Aggregated advice from all agents."""
    aggregated_advice: Optional[AdviceResponse] = Field(
        None, description="Primary advice from the Single-layer HealthAgent"
    )
    agent_contributions: list[AgentContribution] = Field(
        default_factory=list,
        description="Contributions from each specialized agent",
    )
    consensus_severity: str = Field(
        default="low", description="Consensus severity across all agents"
    )
    timestamp: str = Field(default_factory=_now_iso)


class ForecastData(BaseModel):
    horizon: int = Field(..., description="Forecast horizon")
    predicted_values: list[float] = Field(default_factory=list)
    confidence_interval: tuple[float, float] = Field(default=(0.0, 0.0))


class MultiTrendResponse(BaseModel):
    single_agent_trend: Optional[TrendResponse] = None
    multi_scale_trends: dict = Field(default_factory=dict)
    forecast: dict = Field(default_factory=dict)
    cross_scale_insight: str = ""
    timestamp: str = Field(default_factory=_now_iso)


class AnomalyEventData(BaseModel):
    metric_name: str = Field(...)
    z_score: float = Field(...)
    severity: str = Field(...)
    observed_value: float = Field(...)
    expected_value: float = Field(...)
    subject_id: str = ""
    timestamp: str = Field(default_factory=_now_iso)


class AnomalyResponse(BaseModel):
    detected: bool = False
    anomalies: list[AnomalyEventData] = Field(default_factory=list)
    timestamp: str = Field(default_factory=_now_iso)


class SkillResult(BaseModel):
    skill_name: str = Field(...)
    status: str = Field(...)
    output: dict = Field(default_factory=dict)
    duration_ms: float = 0.0


class SkillsResponse(BaseModel):
    skill_results: list[SkillResult] = Field(default_factory=list)
    aggregate_summary: str = ""


class AgentInfo(BaseModel):
    agent_id: str = Field(...)
    agent_type: str = Field(...)
    status: str = "active"
    capabilities: list[str] = Field(default_factory=list)
    last_heartbeat: Optional[str] = None


class AgentsResponse(BaseModel):
    agents: list[AgentInfo] = Field(default_factory=list)
    count: int = 0


class MCPStatusResponse(BaseModel):
    memory_entries: int = 0
    control_active: bool = True
    planning_queue_size: int = 0
    uptime_seconds: float = 0.0
    version: str = "2.0.0"


class MultiTickResponse(BaseModel):
    single_agent_advice: Optional[AdviceResponse] = None
    multi_agent_advice: Optional[MultiAdviceResponse] = None
    anomalies: Optional[AnomalyResponse] = None
    skills_executed: list[str] = Field(default_factory=list)


class WorkflowRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    context: dict = Field(default_factory=dict)


class WorkflowResponse(BaseModel):
    session_id: str = Field(...)
    status: str = Field(...)


class AgentRegisterRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=128)
    agent_type: str = Field(..., min_length=1, max_length=64)
    capabilities: list[str] = Field(default_factory=list)
    endpoint_url: Optional[str] = Field(None, max_length=512)
    metadata: dict = Field(default_factory=dict)


class SkillsExecuteRequest(BaseModel):
    skill_names: list[str] = Field(..., min_length=1)
    input: dict = Field(default_factory=dict)


class LiveInferenceResponse(BaseModel):
    """Real-time live inference result from hardware → AI pipeline."""
    prediction: Optional[int] = Field(None, description="Predicted health class (0/1/2)")
    health_state: str = Field("Initializing...", description="Human-readable health state")
    confidence: Optional[float] = Field(None, description="Model confidence (0-1)")
    advice: Optional[dict] = Field(None, description="AI agent advice dict")
    anomalies: list[dict] = Field(default_factory=list, description="Detected anomalies")
    timestamp: Optional[str] = Field(None, description="ISO 8601 UTC timestamp")
    status: str = Field("initializing", description="ok / degraded / initializing / error")
    mode: str = Field("mock", description="live / mock")
