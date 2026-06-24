"""REST endpoints for feature visualization (PCA / t-SNE)."""
from flask import Blueprint, jsonify, request
from dashboard.backend.data_loader import store
from dashboard.backend.routes.signal_routes import init_analyzer

feature_bp = Blueprint("features", __name__)


@feature_bp.route("/api/feature_viz", methods=["GET"])
def feature_viz():
    """PCA or t-SNE scatterplot coordinates."""
    method = request.args.get("method", "pca").lower()
    components = int(request.args.get("components", 2))
    components = max(2, min(3, components))

    analyzer = init_analyzer()

    if method == "tsne":
        coords = analyzer.get_tsne_coordinates(n_components=components)
        explained_variance = None
    else:
        coords = analyzer.get_pca_coordinates(n_components=components)
        explained_variance = analyzer.get_explained_variance()[:components]

    # Build point list with metadata
    df = store.get_predictions_df()
    subjects = store.get_subjects()
    points = []
    for i, c in enumerate(coords):
        if i >= len(df):
            break
        point = {
            "x": round(c[0], 6),
            "y": round(c[1], 6) if len(c) > 1 else 0,
            "z": round(c[2], 6) if len(c) > 2 else 0,
            "label": int(df.iloc[i]["label"]),
            "prediction": int(df.iloc[i]["prediction"]),
            "subject": subjects[i] if i < len(subjects) else "unknown",
            "filename": str(df.iloc[i]["filename"]),
        }
        points.append(point)

    return jsonify({
        "method": method,
        "explained_variance": explained_variance,
        "n_points": len(points),
        "points": points,
    })
