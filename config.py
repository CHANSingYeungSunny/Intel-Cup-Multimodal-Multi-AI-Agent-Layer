"""
Extended configuration for the Multi-AI Agent Layer.

Re-exports all settings from the Single AI Agent Layer and adds
Multi-specific keys for MCP, Skills, and Agent Coordination.

Uses importlib to load the Single-layer config module explicitly,
avoiding the Python module-name collision between the two config.py files.
"""

import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# Locate the Single AI Agent Layer
# ---------------------------------------------------------------------------
_SINGLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "intel multimodal (AI_Agent_Single_layer)",
)

# ---------------------------------------------------------------------------
# Load Single-layer config via importlib (avoids name collision with self)
# ---------------------------------------------------------------------------
_single_config_path = os.path.join(_SINGLE_DIR, "config.py")
_spec = importlib.util.spec_from_file_location("_single_config", _single_config_path)
_single_config = importlib.util.module_from_spec(_spec)
sys.modules["_single_config"] = _single_config
_spec.loader.exec_module(_single_config)

# Copy all public names from the Single-layer config into this namespace
for _name in dir(_single_config):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_single_config, _name)

# Replace 'config' in sys.modules with this Multi-layer version (which
# re-exports everything from the Single-layer config + adds Multi keys).
# Single-layer modules that ``import config`` will transparently get
# all the values they expect.
sys.modules["config"] = sys.modules[__name__]

# ---------------------------------------------------------------------------
# Multi-layer specific overrides
# ---------------------------------------------------------------------------
APP_TITLE: str = os.environ.get("APP_TITLE", "Multi AI Agent Layer")
APP_VERSION: str = os.environ.get("APP_VERSION", "2.0.0")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
MULTI_AGENT_DB_URL: str = os.environ.get(
    "MULTI_AGENT_DB_URL",
    os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_layer",
    ),
)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
MCP_ENABLED: bool = os.environ.get("MCP_ENABLED", "true").lower() in (
    "1", "true", "yes", "on",
)
MCP_MEMORY_MAX_ENTRIES: int = int(os.environ.get("MCP_MEMORY_MAX_ENTRIES", "1000"))
MCP_DEFAULT_TTL_SECONDS: int = int(os.environ.get("MCP_DEFAULT_TTL_SECONDS", "3600"))

# ---------------------------------------------------------------------------
# Agent Coordinator
# ---------------------------------------------------------------------------
AGENT_COORDINATOR_MAX_WORKERS: int = int(
    os.environ.get("AGENT_COORDINATOR_MAX_WORKERS", "4")
)
AGENT_TIMEOUT_SECONDS: float = float(
    os.environ.get("AGENT_TIMEOUT_SECONDS", "10.0")
)

# ---------------------------------------------------------------------------
# Anomaly Detector skill
# ---------------------------------------------------------------------------
ANOMALY_DETECTOR_ZSCORE_THRESHOLD: float = float(
    os.environ.get("ANOMALY_DETECTOR_ZSCORE_THRESHOLD", "2.5")
)
ANOMALY_DETECTOR_WINDOW: int = int(
    os.environ.get("ANOMALY_DETECTOR_WINDOW", "30")
)
ANOMALY_PERSISTENCE_COUNT: int = int(
    os.environ.get("ANOMALY_PERSISTENCE_COUNT", "3")
)

# ---------------------------------------------------------------------------
# Advanced Trend Analyzer skill
# ---------------------------------------------------------------------------
ADVANCED_TREND_WINDOWS: list[int] = [
    int(w)
    for w in os.environ.get("ADVANCED_TREND_WINDOWS", "5,10,30,60").split(",")
]
FORECAST_HORIZON: int = int(os.environ.get("FORECAST_HORIZON", "5"))

# ---------------------------------------------------------------------------
# LLM Advice Generator skill
# ---------------------------------------------------------------------------
LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "none")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o")
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_LOCAL_ENDPOINT: str = os.environ.get(
    "LLM_LOCAL_ENDPOINT", "http://localhost:11434/v1/chat/completions"
)

# ---------------------------------------------------------------------------
# Skills registry
# ---------------------------------------------------------------------------
SKILLS_ENABLED: list[str] = [
    s.strip()
    for s in os.environ.get(
        "SKILLS_ENABLED",
        "anomaly_detector,advanced_trend_analyzer,llm_advice_generator",
    ).split(",")
    if s.strip()
]

# ---------------------------------------------------------------------------
# API prefixes
# ---------------------------------------------------------------------------
MULTI_API_PREFIX: str = os.environ.get("MULTI_API_PREFIX", "/api/v1/multi")
MCP_API_PREFIX: str = os.environ.get("MCP_API_PREFIX", "/api/v1/mcp")
