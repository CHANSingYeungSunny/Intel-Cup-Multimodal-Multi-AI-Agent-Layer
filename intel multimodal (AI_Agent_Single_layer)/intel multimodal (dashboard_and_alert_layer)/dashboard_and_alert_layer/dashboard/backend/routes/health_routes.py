"""REST endpoints for health state summary."""
from flask import Blueprint, jsonify
from dashboard.backend.data_loader import store

health_bp = Blueprint("health", __name__)


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
