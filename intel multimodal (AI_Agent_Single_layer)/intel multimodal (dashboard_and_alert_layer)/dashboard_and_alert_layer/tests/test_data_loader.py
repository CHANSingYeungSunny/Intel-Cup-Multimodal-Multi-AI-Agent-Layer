"""Tests for DataStore (CSV loading, filename parsing, feature extraction)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.backend.data_loader import DataStore


def test_load_predictions():
    store = DataStore()
    store.load_all()
    assert store.get_prediction_count() > 0, "Should load predictions"
    assert store.get_prediction_count() == 92, f"Expected 92 predictions, got {store.get_prediction_count()}"


def test_feature_matrix():
    store = DataStore()
    store.load_all()
    mat = store.get_feature_matrix()
    assert mat.shape == (92, 256), f"Expected (92, 256), got {mat.shape}"


def test_subjects():
    store = DataStore()
    store.load_all()
    subs = store.get_subjects()
    assert len(subs) == 92
    unique = store.get_unique_subjects()
    assert len(unique) > 0
    # Subject IDs are extracted from filenames; may include UBFC1 subjects too
    assert all(s != "unknown" for s in unique)


def test_counts():
    store = DataStore()
    store.load_all()
    counts = store.get_counts()
    assert "prediction_counts" in counts
    assert "current_state" in counts
    total = sum(counts["prediction_counts"].values())
    assert total == 92


def test_experiments():
    store = DataStore()
    store.load_all()
    exps = store.get_all_experiments()
    assert len(exps) == 5, f"Expected 5 experiments, got {len(exps)}"
    assert exps[0]["id"] == 1

    exp = store.get_experiment(1)
    assert exp is not None
    assert "confusion_matrix" in exp
    assert "train_loss_curve" in exp


def test_subject_predictions():
    store = DataStore()
    store.load_all()
    unique = store.get_unique_subjects()
    data = store.get_feature_vectors_for_subject(unique[0])
    assert "feature_vectors" in data
    assert "predictions" in data
    assert data["subject_id"] == unique[0]


def test_filename_parsing():
    store = DataStore()
    assert store._extract_win_number("v:UBFC2/subject14/win_000292|a:...|p:...") == 292
    assert store._extract_win_number("v:UBFC2/subject1/win_000000|...") == 0
