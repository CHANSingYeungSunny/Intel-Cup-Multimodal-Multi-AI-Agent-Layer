"""REST endpoints for health state summary."""
import json
import subprocess
import sys
from pathlib import Path

from flask import Blueprint, jsonify
from dashboard.backend.data_loader import store

health_bp = Blueprint("health", __name__)


def _find_live_sensor_script():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tools" / "sensors" / "read_live_sensors.py"
        if candidate.is_file():
            return candidate
    return None


@health_bp.route("/api/health_state", methods=["GET"])
def health_state():
    """Aggregated health state across all predictions."""
    exp = store.get_experiment(store.get_active_experiment_id())
    counts = store.get_counts()

    return jsonify({
        **counts,
        "active_experiment_id": store.get_active_experiment_id(),
        "active_experiment_label": exp.get("config_label", "") if exp else "",
        "accuracy": round(float(exp["test_accuracy"]) * 100, 2) if exp else None,
        "f1_macro": round(float(exp["test_f1_macro"]) * 100, 2) if exp else None,
    })


@health_bp.route("/api/health_history", methods=["GET"])
def health_history():
    """Returns the full list of predictions as a history log."""
    predictions = store.get_predictions_list()
    return jsonify({
        "total": len(predictions),
        "predictions": predictions,
    })


@health_bp.route("/api/live_sensors", methods=["GET"])
def live_sensors():
    """Runs the on-device sensor probe script and returns its JSON output."""
    script_path = _find_live_sensor_script()
    if script_path is None:
        return jsonify({"error": "live_sensor_script_not_found"}), 500

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(script_path.parents[2]),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return jsonify({
            "error": "live_sensor_timeout",
            "timeout_seconds": 15,
        }), 504
    except Exception as exc:
        return jsonify({
            "error": "live_sensor_execution_failed",
            "details": str(exc),
        }), 500

    if result.returncode != 0:
        return jsonify({
            "error": "live_sensor_script_failed",
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }), 500

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return jsonify({
            "error": "live_sensor_invalid_json",
            "details": str(exc),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }), 500

    return jsonify(payload)