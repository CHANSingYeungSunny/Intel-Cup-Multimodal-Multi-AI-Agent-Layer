"""Tests for FeatureAnalyzer (PCA, signal generation)."""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.backend.data_loader import DataStore
from dashboard.backend.feature_analyzer import FeatureAnalyzer


def _make_analyzer():
    store = DataStore()
    store.load_all()
    return FeatureAnalyzer(store.get_feature_matrix())


def test_pca_fit():
    analyzer = _make_analyzer()
    coords = analyzer.get_pca_coordinates(n_components=2)
    assert len(coords) == 92
    assert len(coords[0]) == 2
    variance = analyzer.get_explained_variance()
    assert len(variance) >= 2
    assert all(0 < v < 1 for v in variance)


def test_tsne_fit():
    analyzer = _make_analyzer()
    coords = analyzer.get_tsne_coordinates(n_components=2)
    assert len(coords) == 92
    assert len(coords[0]) == 2


def test_generate_cough_curve():
    analyzer = _make_analyzer()
    rng = np.random.RandomState(42)
    vec = rng.randn(256).tolist()
    result = FeatureAnalyzer.generate_cough_curve(vec, {"prediction": 0, "label": 0})
    assert "timestamps" in result
    assert "amplitude" in result
    assert len(result["timestamps"]) == 50
    assert len(result["amplitude"]) == 50
    assert "peak_count" in result
    assert "respiratory_rate_bpm" in result


def test_cough_curve_unhealthy():
    rng = np.random.RandomState(42)
    vec = rng.randn(256).tolist()
    result = FeatureAnalyzer.generate_cough_curve(vec, {"prediction": 2, "label": 2})
    assert result["prediction"] == 2


def test_generate_physio_trends():
    rng = np.random.RandomState(42)
    data = {
        "subject_id": "test_subject",
        "windows": [0, 1, 2],
        "predictions": [0, 1, 2],
        "labels": [0, 1, 2],
        "feature_vectors": [rng.randn(256).tolist() for _ in range(3)],
    }
    result = FeatureAnalyzer.generate_physio_trends(data)
    assert len(result["heart_rate_sim"]) == 3
    assert len(result["spo2_sim"]) == 3
    assert len(result["rr_interval_sim"]) == 3
    assert result["predictions"] == [0, 1, 2]


def test_physio_ranges():
    """Generated vitals should be in plausible physiological ranges."""
    rng = np.random.RandomState(0)
    data = {
        "subject_id": "test",
        "windows": list(range(20)),
        "predictions": [0] * 20,
        "labels": [0] * 20,
        "feature_vectors": [rng.randn(256).tolist() for _ in range(20)],
    }
    result = FeatureAnalyzer.generate_physio_trends(data)
    hr = result["heart_rate_sim"]
    spo2 = result["spo2_sim"]
    rr = result["rr_interval_sim"]

    assert all(50 <= h <= 120 for h in hr), f"HR out of range: {hr}"
    assert all(80 <= s <= 100 for s in spo2), f"SpO2 out of range: {spo2}"
    assert all(0.4 <= r <= 1.4 for r in rr), f"RR interval out of range: {rr}"
