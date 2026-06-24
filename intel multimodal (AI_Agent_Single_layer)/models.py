"""
SQLAlchemy ORM models for the Single AI Agent Layer.

Four tables:

* **observations**     — every health tick ingested by the agent
* **advice_log**       — every piece of advice generated
* **decision_rules**   — configurable decision rules (dynamic CRUD)
* **trend_snapshots**  — periodic trend-state snapshots for historical queries
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    Boolean,
    DateTime,
    JSON,
    Index,
)
from sqlalchemy.sql import func

from database import Base


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(Base):
    """A single health observation ingested by the agent."""

    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(String(128), nullable=False, index=True)
    prediction = Column(Integer, nullable=False, comment="0=Healthy, 1=Sub-healthy, 2=Unhealthy")
    hr = Column(Float, nullable=True, comment="Heart rate (bpm)")
    spo2 = Column(Float, nullable=True, comment="Oxygen saturation (%)")
    rr = Column(Float, nullable=True, comment="RR interval (s)")
    feature_vector = Column(JSON, nullable=True, comment="256-dim fusion embedding")
    timestamp = Column(DateTime(timezone=True), nullable=False, comment="Observation time (UTC)")
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_observations_timestamp", "timestamp"),
        Index("ix_observations_subject_timestamp", "subject_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<Observation id={self.id} subject={self.subject_id!r} "
            f"pred={self.prediction} ts={self.timestamp}>"
        )


# ---------------------------------------------------------------------------
# AdviceLog
# ---------------------------------------------------------------------------

class AdviceLog(Base):
    """A record of every piece of advice emitted by the agent."""

    __tablename__ = "advice_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    matched_rule_id = Column(String(64), nullable=True, comment="null for default advice")
    matched_rule_name = Column(String(128), nullable=False)
    severity = Column(String(16), nullable=False, comment="low / medium / high")
    possible_condition = Column(Text, nullable=False)
    advice = Column(Text, nullable=False)
    actions = Column(JSON, nullable=False, comment="list of action tag strings")
    context = Column(JSON, nullable=False, comment="evaluation context dict")
    timestamp = Column(DateTime(timezone=True), nullable=False, comment="Advice generation time (UTC)")
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_advice_log_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<AdviceLog id={self.id} rule={self.matched_rule_id!r} "
            f"severity={self.severity!r} ts={self.timestamp}>"
        )


# ---------------------------------------------------------------------------
# DecisionRule
# ---------------------------------------------------------------------------

class DecisionRule(Base):
    """A single configurable decision rule persisted in PostgreSQL."""

    __tablename__ = "decision_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    condition = Column(JSON, nullable=False, comment="dict of condition keys")
    result_severity = Column(String(16), nullable=False, comment="low / medium / high")
    result_condition = Column(String(256), nullable=False, comment="possible_condition text")
    result_advice = Column(Text, nullable=False)
    result_actions = Column(JSON, nullable=False, comment="list of action tag strings")
    priority = Column(Integer, nullable=False, default=0, comment="lower = higher priority")
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_rule_dict(self) -> dict:
        """Convert the ORM row back to the in-memory rule dict format."""
        return {
            "id": self.rule_id,
            "name": self.name,
            "priority": self.priority,
            "enabled": self.enabled,
            "condition": self.condition or {},
            "result": {
                "severity": self.result_severity,
                "possible_condition": self.result_condition,
                "advice": self.result_advice,
                "actions": self.result_actions or [],
            },
        }

    def to_api_dict(self) -> dict:
        """Convert to a flat dict matching the RuleResponse Pydantic schema."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "condition": self.condition or {},
            "result_severity": self.result_severity,
            "result_condition": self.result_condition,
            "result_advice": self.result_advice,
            "result_actions": self.result_actions or [],
            "priority": self.priority,
            "enabled": self.enabled,
        }

    def __repr__(self) -> str:
        return (
            f"<DecisionRule id={self.rule_id!r} name={self.name!r} "
            f"priority={self.priority} enabled={self.enabled}>"
        )


# ---------------------------------------------------------------------------
# TrendSnapshot
# ---------------------------------------------------------------------------

class TrendSnapshot(Base):
    """A point-in-time snapshot of the trend-analyzer summary."""

    __tablename__ = "trend_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trend = Column(String(16), nullable=False, comment="degrading / improving / stable")
    history_size = Column(Integer, nullable=False)
    trend_window_size = Column(Integer, nullable=False)
    unhealthy_ratio = Column(Float, nullable=False)
    healthy_ratio = Column(Float, nullable=False)
    hr_slope = Column(Float, nullable=False)
    spo2_slope = Column(Float, nullable=False)
    rr_slope = Column(Float, nullable=False)
    recent_predictions = Column(JSON, nullable=False, comment="list of recent prediction ints")
    timestamp = Column(DateTime(timezone=True), nullable=False, comment="Snapshot time (UTC)")
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_trend_snapshots_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<TrendSnapshot id={self.id} trend={self.trend!r} "
            f"size={self.history_size} ts={self.timestamp}>"
        )
