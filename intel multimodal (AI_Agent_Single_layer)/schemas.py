"""
Pydantic v2 request and response schemas for the Single AI Agent Layer API.

These models define the API contract and drive FastAPI's automatic OpenAPI
documentation and request validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TickRequest(BaseModel):
    """
    A single health-observation tick submitted to the agent for processing.

    Either provide explicit vital signs (*hr_sim*, *spo2_sim*, *rr_sim*) or
    pass a *feature_vector* (256-dim fusion embedding) from which lightweight
    vital-sign proxies will be computed.
    """

    prediction: int = Field(
        ...,
        ge=0,
        le=2,
        description="Predicted health class: 0=Healthy, 1=Sub-healthy, 2=Unhealthy",
        examples=[2],
    )
    subject_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Subject / patient identifier",
        examples=["subject14"],
    )
    feature_vector: Optional[list[float]] = Field(
        None,
        description="256-dim fusion embedding (or str JSON); auto-parsed if string",
        examples=[[0.123, -0.456, 0.789]],
    )
    hr_sim: Optional[float] = Field(
        None,
        ge=30.0,
        le=220.0,
        description="Heart rate in bpm (bypasses proxy computation)",
    )
    spo2_sim: Optional[float] = Field(
        None,
        ge=50.0,
        le=100.0,
        description="Oxygen saturation in % (bypasses proxy computation)",
    )
    rr_sim: Optional[float] = Field(
        None,
        ge=0.2,
        le=3.0,
        description="RR interval in seconds (bypasses proxy computation)",
    )

    @field_validator("feature_vector", mode="before")
    @classmethod
    def _coerce_feature_vector(cls, v):
        """Accept a JSON-encoded string or a plain list."""
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v


class RuleCreateRequest(BaseModel):
    """Payload for creating a new decision rule."""

    rule_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Unique rule identifier",
        examples=["custom_bradycardia"],
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Human-readable rule name",
    )
    condition: dict = Field(
        ...,
        description="Condition keys that must all match for the rule to fire",
        examples=[{"current_prediction": 1, "hr_trend_max": -8.0}],
    )
    result_severity: Literal["low", "medium", "high"] = Field(
        ...,
        description="Severity level when rule fires",
    )
    result_condition: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Possible condition name displayed to the user",
    )
    result_advice: str = Field(
        ...,
        min_length=1,
        description="Natural-language advice text",
    )
    result_actions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up action tags",
    )
    priority: int = Field(
        0,
        description="Evaluation order (lower = higher priority)",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class AdviceResponse(BaseModel):
    """Structured health advice returned by the agent."""

    matched_rule_id: Optional[str] = Field(
        None,
        description="ID of the matched decision rule, or null for default advice",
    )
    matched_rule_name: str = Field(
        ...,
        description="Name of the matched rule, or 'default'",
    )
    severity: str = Field(
        ...,
        description="Severity level: low, medium, or high",
    )
    possible_condition: str = Field(
        ...,
        description="Human-readable possible condition",
    )
    advice: str = Field(
        ...,
        description="Natural-language recommendation text",
    )
    actions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up actions",
    )
    context: dict = Field(
        ...,
        description="Evaluation context (prediction, trend, slopes, ratios)",
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of advice generation",
    )


class TrendResponse(BaseModel):
    """Current health trend summary."""

    trend: str = Field(
        ...,
        description="Trend direction: degrading, improving, or stable",
    )
    history_size: int = Field(
        ...,
        description="Number of observations in the rolling buffer",
    )
    trend_window_size: int = Field(
        ...,
        description="Number of observations used for trend computation",
    )
    unhealthy_ratio: float = Field(
        ...,
        description="Fraction of unhealthy predictions in the trend window",
    )
    healthy_ratio: float = Field(
        ...,
        description="Fraction of healthy predictions in the trend window",
    )
    hr_slope: float = Field(
        ...,
        description="Heart-rate slope (bpm / observation)",
    )
    spo2_slope: float = Field(
        ...,
        description="SpO₂ slope (% / observation)",
    )
    rr_slope: float = Field(
        ...,
        description="RR-interval slope (s / observation)",
    )
    recent_predictions: list[int] = Field(
        default_factory=list,
        description="Most recent prediction values (0/1/2)",
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 UTC timestamp",
    )


class TrendHistoryResponse(BaseModel):
    """Paginated list of trend snapshots."""

    snapshots: list[TrendResponse] = Field(
        default_factory=list,
    )
    count: int = Field(..., description="Number of snapshots returned")


class RuleResponse(BaseModel):
    """Metadata for a single decision rule."""

    rule_id: str
    name: str
    condition: dict
    result_severity: str
    result_condition: str
    result_advice: str
    result_actions: list[str]
    priority: int
    enabled: bool


class StatusResponse(BaseModel):
    """Lightweight agent status for system heartbeats."""

    enabled: bool = True
    rules_count: int = 0
    history_size: int = 0
    latest_severity: str = "none"
    latest_condition: str = ""
    trend: str = "stable"
    db_connected: bool = False


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    version: str = "1.0.0"
    db_connected: bool = False


class AdviceHistoryResponse(BaseModel):
    """Paginated advice history."""

    history: list[AdviceResponse] = Field(default_factory=list)
    count: int = 0


class TickResponse(BaseModel):
    """Wrapper for tick response — advice or null on dedup."""

    advice: Optional[AdviceResponse] = None
    deduplicated: bool = False


# ---------------------------------------------------------------------------
# Helper to generate a UTC timestamp string
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
