"""
Extended ORM models for the Multi-AI Agent Layer.

Defines five new SQLAlchemy tables using MultiBase (separate metadata
from the Single-layer Base).  These tables live in the same PostgreSQL
database but are managed independently.

Tables
------
- AgentRegistry  — catalog of registered specialized agents
- AnomalyEvent   — detected health anomalies
- SkillExecution — audit log of skill executions
- MCPState       — persistent key-value state for the MCP server
- MultiAgentSession — multi-agent workflow sessions
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    JSON,
    Index,
)
from sqlalchemy.sql import func

from base import MultiBase

# ---------------------------------------------------------------------------
# Load and re-export ALL Single-layer ORM models (Observation, AdviceLog,
# DecisionRule, TrendSnapshot) so that Single-layer modules that ``from
# models import Observation`` still work transparently.
# ---------------------------------------------------------------------------
_SINGLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "intel multimodal (AI_Agent_Single_layer)",
)
_single_models_path = os.path.join(_SINGLE_DIR, "models.py")
_spec = importlib.util.spec_from_file_location("_single_models", _single_models_path)
_single_models = importlib.util.module_from_spec(_spec)
sys.modules["_single_models"] = _single_models
_spec.loader.exec_module(_single_models)

# Copy all public names from Single-layer models
for _name in dir(_single_models):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_single_models, _name)

# Replace 'models' in sys.modules with this Multi-layer version so that
# both Single and Multi ORM classes are available from ``import models``.
sys.modules["models"] = sys.modules[__name__]


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


# ===========================================================================
# Agent Registry
# ===========================================================================

class AgentRegistry(MultiBase):
    """Catalog of specialized agents registered with the MCP server."""

    __tablename__ = "agent_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(128), unique=True, nullable=False, index=True)
    agent_type = Column(
        String(64), nullable=False
    )  # health / anomaly / trend / llm / external
    status = Column(
        String(32), nullable=False, default="active"
    )  # active / inactive / error
    capabilities = Column(JSON, nullable=False, default=list)
    endpoint_url = Column(String(512), nullable=True)
    metadata_ = Column("metadata", JSON, nullable=False, default=dict)
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_agent_registry_type", "agent_type"),
        Index("ix_agent_registry_status", "status"),
    )

    def to_dict(self) -> dict:
        """Return a lightweight metadata dict for API responses."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "capabilities": self.capabilities or [],
            "last_heartbeat": (
                self.last_heartbeat.isoformat() if self.last_heartbeat else None
            ),
        }


# ===========================================================================
# Anomaly Events
# ===========================================================================

class AnomalyEvent(MultiBase):
    """Detected health anomaly persisted for audit and retrieval."""

    __tablename__ = "anomaly_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(String(128), nullable=False, index=True)
    z_score = Column(Float, nullable=False)
    metric_name = Column(
        String(32), nullable=False
    )  # hr / spo2 / rr / prediction
    observed_value = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    severity = Column(String(32), nullable=False, default="warning")  # warning / critical
    timestamp = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_anomaly_events_timestamp", "timestamp"),
        Index("ix_anomaly_events_subject_timestamp", "subject_id", "timestamp"),
    )

    def to_dict(self) -> dict:
        """Return a dict suitable for AnomalyEventData schema."""
        return {
            "metric_name": self.metric_name,
            "z_score": self.z_score,
            "severity": self.severity,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "subject_id": self.subject_id,
            "timestamp": (
                self.timestamp.isoformat() if self.timestamp else ""
            ),
        }


# ===========================================================================
# Skill Executions
# ===========================================================================

class SkillExecution(MultiBase):
    """Audit log recording each skill execution."""

    __tablename__ = "skill_executions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    skill_name = Column(String(64), nullable=False, index=True)
    agent_id = Column(String(128), nullable=True)
    input_summary = Column(JSON, nullable=False, default=dict)
    output_summary = Column(JSON, nullable=False, default=dict)
    status = Column(
        String(32), nullable=False, default="success"
    )  # success / error / timeout
    duration_ms = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_skill_executions_name_time", "skill_name", "timestamp"),
    )


# ===========================================================================
# MCP State (persistent key-value store)
# ===========================================================================

class MCPState(MultiBase):
    """Persistent key-value store for the MCP Memory component."""

    __tablename__ = "mcp_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(256), unique=True, nullable=False, index=True)
    value = Column(JSON, nullable=False, default=dict)
    namespace = Column(
        String(64), nullable=False, default="memory"
    )  # memory / control / planning
    ttl_seconds = Column(Integer, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )

    __table_args__ = (
        Index("ix_mcp_state_namespace", "namespace"),
    )


# ===========================================================================
# Multi-Agent Sessions
# ===========================================================================

class MultiAgentSession(MultiBase):
    """Tracks multi-agent workflow sessions."""

    __tablename__ = "multi_agent_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    workflow_plan = Column(JSON, nullable=False, default=dict)
    status = Column(
        String(32), nullable=False, default="planned"
    )  # planned / running / completed / failed
    agents_involved = Column(JSON, nullable=False, default=list)
    results = Column(JSON, nullable=False, default=dict)
    started_at = Column(DateTime(timezone=True), nullable=False, default=_utc_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)
