"""REST endpoints for experiment metadata and selection."""
from flask import Blueprint, jsonify
from dashboard.backend.data_loader import store

experiment_bp = Blueprint("experiments", __name__)


@experiment_bp.route("/api/experiments", methods=["GET"])
def list_experiments():
    """List all experiments with summary fields."""
    exps = store.get_all_experiments()
    return jsonify({
        "experiments": exps,
        "active_experiment_id": store.get_active_experiment_id(),
    })


@experiment_bp.route("/api/experiments/<int:exp_id>", methods=["GET"])
def get_experiment(exp_id):
    """Full experiment row with deserialized curves."""
    exp = store.get_experiment(exp_id)
    if exp is None:
        return jsonify({"error": f"Experiment {exp_id} not found"}), 404

    # Convert numpy values to Python native types
    clean = {}
    for k, v in exp.items():
        if hasattr(v, "item"):      # numpy scalar
            clean[k] = v.item()
        elif isinstance(v, (list, dict)):
            clean[k] = v
        else:
            try:
                clean[k] = float(v) if "." in str(v) else int(v)
            except (ValueError, TypeError):
                clean[k] = str(v)

    return jsonify(clean)


@experiment_bp.route("/api/experiments/switch/<int:exp_id>", methods=["POST"])
def switch_experiment(exp_id):
    """Switch the active experiment via REST."""
    ok = store.set_active_experiment(exp_id)
    if not ok:
        return jsonify({"error": f"Experiment {exp_id} not found"}), 404
    exp = store.get_experiment(exp_id)
    return jsonify({
        "success": True,
        "active_experiment_id": exp_id,
        "label": exp.get("config_label", "") if exp else "",
    })
