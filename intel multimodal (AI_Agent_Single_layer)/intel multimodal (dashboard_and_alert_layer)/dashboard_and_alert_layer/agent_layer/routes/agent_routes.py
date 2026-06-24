"""
Flask blueprint for AI Agent Layer REST endpoints.

Registered at ``/api/`` by :func:`dashboard.backend.app.create_app`.

Endpoints
---------
GET /api/agent_advice     — current advice + trend summary
GET /api/agent_history    — recent advice history  (?n=<int>)
GET /api/agent_rules      — active decision rules metadata
GET /api/agent_status     — lightweight agent status

╔═══════════════════════════════════════════════════════════════════════════╗
║ DEPRECATED — Replaced by ../../../main.py REST API (FastAPI service).    ║
║ See ../../../README.md for migration guide.  Kept for backward compat.   ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os

import requests
from flask import Blueprint, jsonify, request

agent_bp = Blueprint("agent", __name__)

# Module-level reference set by set_agent_instance() during startup
_agent_instance = None

# External agent API URL (set via AGENT_API_URL env var / --agent-api-url)
_EXTERNAL_API = os.environ.get("AGENT_API_URL", "")


def set_agent_instance(agent):
    """Store the HealthAgent singleton for route handlers."""
    global _agent_instance
    _agent_instance = agent


def _get_agent():
    """Return the current agent instance or ``None``."""
    return _agent_instance


def _proxy_get(path, timeout=5):
    """Proxy a GET request to the external agent API.  Returns (data, status)."""
    if not _EXTERNAL_API:
        return {"enabled": False, "error": "AI Agent is not enabled"}, 503
    try:
        resp = requests.get(f"{_EXTERNAL_API}{path}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json(), 200
        return {"enabled": False, "error": f"Agent API returned {resp.status_code}"}, 502
    except requests.RequestException as exc:
        return {"enabled": False, "error": str(exc)}, 503


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@agent_bp.route("/api/agent_advice", methods=["GET"])
def get_agent_advice():
    """
    Return the latest AI agent advice including trend summary.

    Falls back to the external FastAPI agent when no internal agent is set.
    """
    agent = _get_agent()
    if agent is None:
        data, status = _proxy_get("/advice/current")
        if status == 200 and data:
            # The external API returns an AdviceResponse dict directly.
            # Wrap it to match the Dashboard's expected shape.
            return jsonify({
                "latest_advice": data,
                "trend_summary": None,
                "active_rules_count": 0,
            })
        return jsonify(data), status

    return jsonify(
        {
            "latest_advice": agent.get_current_advice(),
            "trend_summary": agent.get_trend_summary(),
            "active_rules_count": agent.get_status().get("rules_count", 0),
        }
    )


@agent_bp.route("/api/agent_history", methods=["GET"])
def get_agent_history():
    """Return recent advice history.  Proxies to external API if needed."""
    agent = _get_agent()
    if agent is None:
        try:
            n = int(request.args.get("n", 20))
        except (ValueError, TypeError):
            n = 20
        n = max(1, min(n, 50))
        data, status = _proxy_get(f"/advice/history?n={n}")
        if status == 200:
            return jsonify(data)
        return jsonify(data), status

    try:
        n = int(request.args.get("n", 20))
    except (ValueError, TypeError):
        n = 20
    n = max(1, min(n, 50))

    history = agent.get_advice_history(n)
    return jsonify({"history": history, "count": len(history)})


@agent_bp.route("/api/agent_rules", methods=["GET"])
def get_agent_rules():
    """Return decision rules metadata.  Proxies to external API if needed."""
    agent = _get_agent()
    if agent is None:
        data, status = _proxy_get("/rules")
        if status == 200:
            # The external API returns a list of rule objects.
            # Wrap it to match the Dashboard's expected shape.
            return jsonify({"rules": data if isinstance(data, list) else [], "count": len(data) if isinstance(data, list) else 0})
        return jsonify(data), status

    rules = agent.get_rules()
    return jsonify({"rules": rules, "count": len(rules)})


@agent_bp.route("/api/agent_status", methods=["GET"])
def get_agent_status():
    """Return agent status.  Proxies to external API if needed."""
    agent = _get_agent()
    if agent is None:
        data, status = _proxy_get("/status")
        if status == 200 and isinstance(data, dict):
            return jsonify(data)
        return jsonify(data), status

    return jsonify(agent.get_status())
