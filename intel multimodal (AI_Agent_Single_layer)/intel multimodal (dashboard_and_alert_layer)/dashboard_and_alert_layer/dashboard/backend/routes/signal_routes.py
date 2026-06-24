"""REST endpoints for cough/respiration curves and physiological trends."""
from flask import Blueprint, jsonify, request
from dashboard.backend.data_loader import store
from dashboard.backend.feature_analyzer import FeatureAnalyzer

signal_bp = Blueprint("signals", __name__)

# Lazily initialized; set after startup
_analyzer = None


def init_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = FeatureAnalyzer(store.get_feature_matrix())
        _analyzer.fit_pca()
    return _analyzer


@signal_bp.route("/api/cough_curve", methods=["GET"])
def cough_curve():
    """Simulated respiratory waveform for a given subject."""
    subject_id = request.args.get("subject", store.get_unique_subjects()[0])
    data = store.get_feature_vectors_for_subject(subject_id)

    if not data["feature_vectors"]:
        return jsonify({"error": f"No data for subject {subject_id}"}), 404

    # Use the first window's feature vector
    vec = data["feature_vectors"][0]
    meta = {
        "prediction": data["predictions"][0] if data["predictions"] else 0,
        "label": data["labels"][0] if data["labels"] else -1,
    }
    result = FeatureAnalyzer.generate_cough_curve(vec, sample_metadata=meta)
    result["subject_id"] = subject_id
    return jsonify(result)


@signal_bp.route("/api/physio_trend", methods=["GET"])
def physio_trend():
    """Multi-vital-sign trend lines for a given subject."""
    subject_id = request.args.get("subject", store.get_unique_subjects()[0])
    data = store.get_feature_vectors_for_subject(subject_id)

    if not data["feature_vectors"]:
        return jsonify({"error": f"No data for subject {subject_id}"}), 404

    result = FeatureAnalyzer.generate_physio_trends(data)
    return jsonify(result)
