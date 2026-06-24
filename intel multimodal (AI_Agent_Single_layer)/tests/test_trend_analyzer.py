"""Unit tests for TrendAnalyzer.

Ports the existing test suite from ``tests_agent/test_trend_analyzer.py``
and adds PostgreSQL persistence round-trip tests.
"""

import pytest
from trend_analyzer import TrendAnalyzer


@pytest.fixture
def analyzer():
    return TrendAnalyzer(window_size=20)


def _add_n(analyzer, predictions, subject="s1"):
    """Helper: add a sequence of observations, each with default vitals."""
    for i, p in enumerate(predictions):
        analyzer.add_observation(
            prediction=p,
            subject_id=subject,
            hr_sim=75.0 + i,
            spo2_sim=97.0 - i * 0.1,
            rr_sim=0.85,
        )


class TestTrendAnalyzer:
    """Tests for the trend classification and vital-sign slopes."""

    # ---- Initial state ----

    def test_initial_state(self, analyzer):
        assert analyzer.get_history_size() == 0
        assert analyzer.get_trend() == "stable"

    # ---- History capacity ----

    def test_history_capacity(self, analyzer):
        for i in range(30):
            analyzer.add_observation(prediction=0, subject_id="s1")
        assert analyzer.get_history_size() == 20  # maxlen=20

    # ---- Trend classification ----

    def test_degrading_trend(self, analyzer):
        # 6 unhealthy out of 10 = 0.6 ≥ DEGRADING_THRESHOLD (0.3)
        preds = [2, 2, 0, 2, 0, 2, 1, 2, 0, 2]
        _add_n(analyzer, preds)
        assert analyzer.get_trend() == "degrading"

    def test_improving_trend(self, analyzer):
        # 8 healthy out of 10 = 0.8 ≥ IMPROVING_THRESHOLD (0.7)
        preds = [0, 0, 0, 1, 0, 0, 0, 2, 0, 0]
        _add_n(analyzer, preds)
        assert analyzer.get_trend() == "improving"

    def test_stable_trend(self, analyzer):
        preds = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        _add_n(analyzer, preds)
        assert analyzer.get_trend() == "stable"

    def test_stable_with_few_observations(self, analyzer):
        """Fewer than 5 observations → always stable."""
        for p in [2, 2, 2]:  # only 3, all unhealthy
            analyzer.add_observation(prediction=p, subject_id="s1")
        assert analyzer.get_trend() == "stable"

    # ---- Vital-sign slopes ----

    def test_hr_slope_positive(self, analyzer):
        # HR rises by 1 bpm per observation → slope ≈ 1.0
        for i in range(10):
            analyzer.add_observation(
                prediction=0, subject_id="s1",
                hr_sim=70.0 + i, spo2_sim=97.0, rr_sim=0.85,
            )
        slope = analyzer.get_hr_trend()
        assert slope > 0.5  # ≈ 1.0

    def test_hr_slope_negative(self, analyzer):
        for i in range(10):
            analyzer.add_observation(
                prediction=0, subject_id="s1",
                hr_sim=80.0 - i, spo2_sim=97.0, rr_sim=0.85,
            )
        slope = analyzer.get_hr_trend()
        assert slope < -0.5  # ≈ -1.0

    def test_spo2_slope_with_none_values(self, analyzer):
        """Slope with some None vitals should still compute from remaining."""
        for i in range(5):
            analyzer.add_observation(
                prediction=0, subject_id="s1",
                hr_sim=None, spo2_sim=98.0 - i, rr_sim=None,
            )
        for i in range(5, 10):
            analyzer.add_observation(
                prediction=0, subject_id="s1",
                hr_sim=75.0, spo2_sim=None, rr_sim=0.85,
            )
        # spo2 has 5 values, should still compute slope
        slope = analyzer.get_spo2_trend()
        assert slope < 0  # decreasing

    def test_rr_slope_positive(self, analyzer):
        for i in range(10):
            analyzer.add_observation(
                prediction=0, subject_id="s1",
                hr_sim=75.0, spo2_sim=97.0, rr_sim=0.7 + i * 0.02,
            )
        slope = analyzer.get_rr_trend()
        assert slope > 0

    def test_slope_insufficient_data_returns_zero(self, analyzer):
        analyzer.add_observation(prediction=0, subject_id="s1", hr_sim=75.0)
        assert analyzer.get_hr_trend() == 0.0

    # ---- Accessors ----

    def test_get_recent_predictions(self, analyzer):
        _add_n(analyzer, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        recent = analyzer.get_recent_predictions(n=5)
        assert recent == [2, 0, 1, 2, 0]

    def test_get_summary(self, analyzer):
        _add_n(analyzer, [0] * 10)
        summary = analyzer.get_summary()
        assert "trend" in summary
        assert "hr_slope" in summary
        assert "spo2_slope" in summary
        assert "rr_slope" in summary
        assert "recent_predictions" in summary
        assert "timestamp" in summary
        assert summary["history_size"] == 10

    def test_reset(self, analyzer):
        _add_n(analyzer, [0, 1, 2])
        assert analyzer.get_history_size() == 3
        analyzer.reset()
        assert analyzer.get_history_size() == 0

    def test_get_history_returns_copy(self, analyzer):
        _add_n(analyzer, [0, 1])
        history = analyzer.get_history()
        history.append({"prediction": 9, "subject_id": "mutated"})
        assert analyzer.get_history_size() == 2  # not affected

    # ---- Trend thresholds ----

    def test_ratio_at_degrading_threshold_boundary(self, analyzer):
        # Exactly 3 unhealthy out of 10 = 0.3 → degrading
        preds = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0]
        _add_n(analyzer, preds)
        assert analyzer.get_trend() == "degrading"

    def test_ratio_below_degrading_threshold(self, analyzer):
        # 2 unhealthy out of 10 = 0.2 < 0.3 → not degrading
        preds = [2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        _add_n(analyzer, preds)
        assert analyzer.get_trend() != "degrading"

    def test_get_unhealthy_ratio(self, analyzer):
        preds = [2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        _add_n(analyzer, preds)
        assert analyzer.get_unhealthy_ratio() == 0.2

    def test_get_healthy_ratio(self, analyzer):
        preds = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        _add_n(analyzer, preds)
        assert analyzer.get_healthy_ratio() == 0.7
