"""Unit tests for TrendAnalyzer — history buffer and trend detection."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_layer.trend_analyzer import TrendAnalyzer


def _obs(prediction, hr=None, spo2=None, rr=None):
    return {
        "prediction": prediction,
        "subject_id": "test_subject",
        "hr": hr,
        "spo2": spo2,
        "rr": rr,
        "timestamp": "2026-06-23T00:00:00Z",
    }


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer history buffer and trend classification."""

    def test_initial_state(self):
        ta = TrendAnalyzer(window_size=20)
        assert ta.get_history_size() == 0
        assert ta.get_trend() == "stable"
        assert ta.get_unhealthy_ratio() == 0.0
        assert ta.get_healthy_ratio() == 0.0
        assert ta.get_hr_trend() == 0.0
        assert ta.get_spo2_trend() == 0.0

    def test_history_capacity(self):
        ta = TrendAnalyzer(window_size=5)
        for i in range(10):
            ta.add_observation(prediction=i % 3, subject_id="s1")
        assert ta.get_history_size() == 5

    def test_degrading_trend(self):
        """≥30% unhealthy → degrading."""
        ta = TrendAnalyzer(window_size=10)
        # 6 unhealthy, 2 sub-healthy, 2 healthy
        for _ in range(6):
            ta.add_observation(prediction=2, subject_id="s1")
        for _ in range(2):
            ta.add_observation(prediction=1, subject_id="s1")
        for _ in range(2):
            ta.add_observation(prediction=0, subject_id="s1")
        assert ta.get_trend() == "degrading"
        assert ta.get_unhealthy_ratio() == 0.6

    def test_improving_trend(self):
        """≥70% healthy → improving."""
        ta = TrendAnalyzer(window_size=10)
        for _ in range(8):
            ta.add_observation(prediction=0, subject_id="s1")
        for _ in range(2):
            ta.add_observation(prediction=1, subject_id="s1")
        assert ta.get_trend() == "improving"
        assert ta.get_healthy_ratio() == 0.8

    def test_stable_trend(self):
        """Mixed predictions with no clear dominance (< 30% unhealthy, < 70% healthy)."""
        ta = TrendAnalyzer(window_size=10)
        # 2 unhealthy, 4 healthy, 4 sub-healthy → unhealthy=0.2, healthy=0.4 → stable
        for p in [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]:
            ta.add_observation(prediction=p, subject_id="s1")
        assert ta.get_trend() == "stable"

    def test_stable_with_few_observations(self):
        """Trend is 'stable' when buffer is too small."""
        ta = TrendAnalyzer(window_size=20)
        ta.add_observation(prediction=2, subject_id="s1")
        ta.add_observation(prediction=2, subject_id="s1")
        assert ta.get_trend() == "stable"

    def test_hr_slope_positive(self):
        """Verify positive HR slope detection."""
        ta = TrendAnalyzer(window_size=10)
        for i in range(10):
            ta.add_observation(prediction=0, subject_id="s1", hr_sim=70.0 + i * 2.0)
        slope = ta.get_hr_trend()
        assert slope > 1.5  # roughly 2 bpm/tick

    def test_hr_slope_negative(self):
        """Verify negative HR slope detection."""
        ta = TrendAnalyzer(window_size=10)
        for i in range(10):
            ta.add_observation(prediction=0, subject_id="s1", hr_sim=90.0 - i * 1.5)
        slope = ta.get_hr_trend()
        assert slope < -1.0

    def test_hr_slope_insufficient_data(self):
        """Slope is 0 when fewer than 2 HR values exist."""
        ta = TrendAnalyzer(window_size=10)
        ta.add_observation(prediction=0, subject_id="s1", hr_sim=80.0)
        assert ta.get_hr_trend() == 0.0

    def test_spo2_slope_with_none_values(self):
        """Slope computed only from non-None values."""
        ta = TrendAnalyzer(window_size=10)
        for i in range(5):
            ta.add_observation(prediction=0, subject_id="s1")  # no spo2
        for i in range(5):
            ta.add_observation(prediction=0, subject_id="s1", spo2_sim=97.0 - i * 1.0)
        slope = ta.get_spo2_trend()
        # Only 5 valid points → slope should be computed
        assert slope < 0

    def test_get_recent_predictions(self):
        ta = TrendAnalyzer(window_size=20)
        for i in range(15):
            ta.add_observation(prediction=i % 3, subject_id="s1")
        preds = ta.get_recent_predictions(5)
        assert len(preds) == 5
        # 15 iterations of i%3: [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2] → last 5 = [1,2,0,1,2]
        assert preds == [1, 2, 0, 1, 2]

    def test_get_summary(self):
        ta = TrendAnalyzer(window_size=10)
        for i in range(10):
            ta.add_observation(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0, rr_sim=0.85)
        summary = ta.get_summary()
        assert "trend" in summary
        assert "history_size" in summary
        assert "hr_slope" in summary
        assert "spo2_slope" in summary
        assert "recent_predictions" in summary
        assert summary["history_size"] == 10

    def test_reset(self):
        ta = TrendAnalyzer()
        for _ in range(5):
            ta.add_observation(prediction=1, subject_id="s1")
        ta.reset()
        assert ta.get_history_size() == 0
        assert ta.get_trend() == "stable"

    def test_get_history_returns_copy(self):
        ta = TrendAnalyzer(window_size=5)
        ta.add_observation(prediction=0, subject_id="s1")
        hist = ta.get_history()
        hist.append({"fake": True})
        assert ta.get_history_size() == 1  # original unchanged

    # ------------------------------------------------------------------
    # New tests — boundary coverage & RR trend
    # ------------------------------------------------------------------

    def test_rr_trend_computation(self):
        """Verify get_rr_trend() returns a numeric slope."""
        ta = TrendAnalyzer(window_size=10)
        for i in range(10):
            ta.add_observation(prediction=0, subject_id="s1", rr_sim=0.85 + i * 0.02)
        slope = ta.get_rr_trend()
        assert slope > 0.01  # positive slope from increasing RR values

    def test_rr_trend_with_none_values(self):
        """Slope computed from non-None RR values only."""
        ta = TrendAnalyzer(window_size=10)
        for _ in range(4):
            ta.add_observation(prediction=0, subject_id="s1")  # no rr
        for i in range(6):
            ta.add_observation(prediction=0, subject_id="s1", rr_sim=0.80 + i * 0.04)
        slope = ta.get_rr_trend()
        assert slope > 0.02

    def test_degrading_threshold_boundary(self):
        """Exactly 3/10 unhealthy (= 0.3, DEGRADING_THRESHOLD) → degrading."""
        ta = TrendAnalyzer(window_size=10)
        for p in [2, 2, 2, 1, 1, 1, 1, 0, 0, 0]:
            ta.add_observation(prediction=p, subject_id="s1")
        assert ta.get_unhealthy_ratio() == 0.3
        assert ta.get_trend() == "degrading"

    def test_improving_threshold_boundary(self):
        """Exactly 7/10 healthy (= 0.7, IMPROVING_THRESHOLD) → improving."""
        ta = TrendAnalyzer(window_size=10)
        for p in [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]:
            ta.add_observation(prediction=p, subject_id="s1")
        assert ta.get_healthy_ratio() == 0.7
        assert ta.get_trend() == "improving"

    def test_degrading_below_threshold(self):
        """2/10 unhealthy = 0.2 < 0.3 → not degrading."""
        ta = TrendAnalyzer(window_size=10)
        for p in [2, 2, 1, 1, 1, 1, 0, 0, 0, 0]:
            ta.add_observation(prediction=p, subject_id="s1")
        assert ta.get_unhealthy_ratio() == 0.2
        assert ta.get_trend() != "degrading"
