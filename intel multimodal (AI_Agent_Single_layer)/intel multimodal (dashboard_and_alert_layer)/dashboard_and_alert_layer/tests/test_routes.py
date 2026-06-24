"""Tests for Flask REST API endpoints."""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dashboard.backend.data_loader import store
from dashboard.backend.app import create_app


@pytest.fixture
def client():
    store.load_all()
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_state(client):
    resp = client.get("/api/health_state")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "total_samples" in data
    assert "prediction_counts" in data
    assert "current_state" in data
    assert data["total_samples"] == 92


def test_experiments_list(client):
    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "experiments" in data
    assert len(data["experiments"]) == 5


def test_experiment_detail(client):
    resp = client.get("/api/experiments/1")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "confusion_matrix" in data
    assert "train_loss_curve" in data


def test_experiment_not_found(client):
    resp = client.get("/api/experiments/99")
    assert resp.status_code == 404


def test_disease_classification(client):
    resp = client.get("/api/disease_classification")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "metrics" in data
    assert "confusion_matrix" in data
    assert "per_class" in data
    assert "predictions" in data


def test_feature_viz_pca(client):
    resp = client.get("/api/feature_viz?method=pca&components=2")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["method"] == "pca"
    assert "points" in data
    assert data["n_points"] == 92
    assert "explained_variance" in data


def test_feature_viz_tsne(client):
    resp = client.get("/api/feature_viz?method=tsne&components=2")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["method"] == "tsne"
    assert len(data["points"]) == 92


def test_cough_curve(client):
    # Get a valid subject first
    resp = client.get("/api/health_history")
    data = json.loads(resp.data)
    first = data["predictions"][0]
    import re
    m = re.search(r"subject(\d+)", first["filename"])
    subject = f"subject{m.group(1)}" if m else "subject14"

    resp = client.get(f"/api/cough_curve?subject={subject}")
    assert resp.status_code == 200
    curve = json.loads(resp.data)
    assert "timestamps" in curve
    assert "amplitude" in curve
    assert len(curve["timestamps"]) == 50


def test_physio_trend(client):
    import re
    resp = client.get("/api/health_history")
    data = json.loads(resp.data)
    first = data["predictions"][0]
    m = re.search(r"subject(\d+)", first["filename"])
    subject = f"subject{m.group(1)}" if m else "subject14"

    resp = client.get(f"/api/physio_trend?subject={subject}")
    assert resp.status_code == 200
    trend = json.loads(resp.data)
    assert "heart_rate_sim" in trend
    assert "spo2_sim" in trend
