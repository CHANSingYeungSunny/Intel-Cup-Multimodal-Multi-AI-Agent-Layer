"""
Demo Mode Routes — health state override for competition demonstration.

Only active when DEMO_MODE_ENABLED=true (environment variable).
Allows the frontend to toggle between Healthy / Semi-healthy / Unhealthy
simulated states without modifying CSV files or restarting services.

Endpoints
---------
POST   /api/demo/override   — activate a health state override
DELETE /api/demo/override   — clear override (resume normal data)
GET    /api/demo/status     — current override status
"""

import os
import threading

import numpy as np
from flask import Blueprint, jsonify, request

demo_bp = Blueprint("demo", __name__)

# ---------------------------------------------------------------------------
# Thread-safe in-memory override store
# ---------------------------------------------------------------------------
_override_lock = threading.Lock()
_override_state: dict | None = None  # None = no override active

# ---------------------------------------------------------------------------
# Pre-generated feature vectors for each health state
# (256-dim, biased toward their respective class)
# ---------------------------------------------------------------------------
_FEATURE_TEMPLATES = {
    "healthy": {
        "prediction": 0,
        "label": 0,
        "label_name": "Healthy",
        "description": "Healthy subject — normal vitals, clear audio, stable PPG",
        # Feature vector biased positive in first segments (healthy indicators)
        "feature_vector": (
            list(np.random.RandomState(42).randn(256).astype(float) * 0.3 + 0.5)
        ),
    },
    "semi_healthy": {
        "prediction": 1,
        "label": 1,
        "label_name": "Sub-healthy",
        "description": "Sub-healthy subject — slightly elevated HR, mild cough patterns",
        "feature_vector": (
            list(np.random.RandomState(99).randn(256).astype(float) * 0.5 + 0.0)
        ),
    },
    "unhealthy": {
        "prediction": 2,
        "label": 2,
        "label_name": "Unhealthy",
        "description": "Unhealthy subject — high HR, low SpO₂, strong cough signals",
        "feature_vector": (
            list(np.random.RandomState(7).randn(256).astype(float) * 0.4 - 0.4)
        ),
    },
}


def get_override():
    """Thread-safe read of current override state. Returns None if inactive."""
    with _override_lock:
        return dict(_override_state) if _override_state else None


# ---------------------------------------------------------------------------
# Routes (only registered when DEMO_MODE_ENABLED=true)
# ---------------------------------------------------------------------------


@demo_bp.route("/api/demo/override", methods=["POST"])
def set_override():
    """
    Activate a health state override.

    Request JSON:  {"state": "healthy" | "semi_healthy" | "unhealthy"}
    Response:      current override state with feature vector summary
    """
    global _override_state

    body = request.get_json(silent=True) or {}
    state = body.get("state", "").lower().strip()

    if state not in _FEATURE_TEMPLATES:
        return (
            jsonify({
                "error": f"Invalid state '{state}'. "
                f"Use: {', '.join(_FEATURE_TEMPLATES.keys())}",
                "active_override": _override_state is not None,
            }),
            400,
        )

    template = _FEATURE_TEMPLATES[state]
    with _override_lock:
        _override_state = {
            "state": state,
            "prediction": template["prediction"],
            "label": template["label"],
            "feature_vector": template["feature_vector"],
            "description": template["description"],
            "activated_at": str(np.datetime64("now")),
            "subject_id": "demo_subject",
            "filename": f"demo://{state}",
        }

    return jsonify({
        "status": "ok",
        "override": {
            "state": state,
            "prediction": template["prediction"],
            "label_name": template["label_name"],
            "description": template["description"],
        },
    })


@demo_bp.route("/api/demo/override", methods=["DELETE"])
def clear_override():
    """
    Clear any active override.  The simulator resumes reading from
    predictions.csv.
    """
    global _override_state
    was_active = _override_state is not None
    with _override_lock:
        _override_state = None

    return jsonify({
        "status": "ok",
        "was_active": was_active,
        "message": "Override cleared — simulator resumes normal data flow",
    })


@demo_bp.route("/api/demo/status", methods=["GET"])
def demo_status():
    """Return current demo mode status."""
    override = get_override()
    return jsonify({
        "demo_mode_enabled": True,
        "override_active": override is not None,
        "override": override,
        "available_states": list(_FEATURE_TEMPLATES.keys()),
    })
