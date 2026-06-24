"""REST endpoints for disease classification results."""
import numpy as np
from flask import Blueprint, jsonify
from dashboard.backend.data_loader import store
from config import LABEL_NAMES

disease_bp = Blueprint("disease", __name__)


def _native(val):
    """Convert numpy scalar to Python native type."""
    if hasattr(val, "item"):
        return val.item()
    return val


@disease_bp.route("/api/disease_classification", methods=["GET"])
def disease_classification():
    """Full classification results with metrics and confusion matrix."""
    exp_id = store.get_active_experiment_id()
    exp = store.get_experiment(exp_id)

    if exp is None:
        return jsonify({"error": f"Experiment {exp_id} not found"}), 404

    num_classes = store.get_num_classes()
    class_names = [LABEL_NAMES[i] for i in range(num_classes)]
    cm = exp.get("confusion_matrix", [[0]*num_classes]*num_classes)

    # Per-class metrics from confusion matrix
    cm_arr = np.array(cm, dtype=np.float64)
    per_class = []
    for i in range(num_classes):
        tp = cm_arr[i, i]
        fp = cm_arr[:, i].sum() - tp
        fn = cm_arr[i, :].sum() - tp
        support = int(cm_arr[i, :].sum())
        precision = round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else 0.0
        recall = round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else 0.0
        f1 = round(float(2 * precision * recall / (precision + recall)), 4) if (precision + recall) > 0 else 0.0
        per_class.append({
            "class": class_names[i],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })

    predictions = store.get_predictions_list()

    # Convert confusion matrix to native ints
    cm_native = [[int(v) for v in row] for row in cm]

    return jsonify({
        "experiment_id": exp_id,
        "num_classes": num_classes,
        "class_names": class_names,
        "experiment_label": str(exp.get("config_label", "")),
        "metrics": {
            "accuracy": round(_native(exp.get("test_accuracy", 0)) * 100, 2),
            "precision_macro": round(_native(exp.get("test_precision_macro", 0)) * 100, 2),
            "recall_macro": round(_native(exp.get("test_recall_macro", 0)) * 100, 2),
            "f1_macro": round(_native(exp.get("test_f1_macro", 0)) * 100, 2),
            "precision_weighted": round(_native(exp.get("test_precision_weighted", 0)) * 100, 2),
            "recall_weighted": round(_native(exp.get("test_recall_weighted", 0)) * 100, 2),
            "f1_weighted": round(_native(exp.get("test_f1_weighted", 0)) * 100, 2),
            "test_loss": float(_native(exp.get("test_loss", 0))),
        },
        "confusion_matrix": cm_native,
        "per_class": per_class,
        "predictions": predictions,
    })
