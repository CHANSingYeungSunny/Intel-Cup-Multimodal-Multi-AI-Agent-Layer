"""Unit tests for HealthAgent — orchestration of TrendAnalyzer + DecisionEngine."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent_layer.health_agent import HealthAgent


class TestHealthAgent:
    """Tests for HealthAgent orchestration logic."""

    def test_process_tick_returns_advice(self):
        agent = HealthAgent(data_store=None)
        # Push enough varied ticks to build a trend that will produce advice
        for i in range(7):
            agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        # Reset dedup key so the next call always returns fresh advice
        agent._last_advice_key = None
        result = agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        assert result is not None
        assert "severity" in result
        assert "possible_condition" in result
        assert "advice" in result
        assert "timestamp" in result

    def test_process_tick_detects_degrading(self):
        agent = HealthAgent(data_store=None)
        # Feed mostly unhealthy to create a degrading trend (≥30% unhealthy)
        for _ in range(6):
            agent.process_tick(prediction=2, subject_id="s1", hr_sim=95.0, spo2_sim=92.0)
        for _ in range(4):
            agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        # Verify the trend was detected
        summary = agent.get_trend_summary()
        assert summary["trend"] == "degrading"
        assert summary["unhealthy_ratio"] >= 0.3
        # Advice should have been generated (check via get_current_advice)
        assert agent.get_current_advice() is not None

    def test_advice_deduplication(self):
        """Same rule match on consecutive ticks returns None (no re-emit)."""
        agent = HealthAgent(data_store=None)
        # First call with a fresh agent returns advice (no prior key to match)
        first = agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        assert first is not None, "first advice should not be None"
        # Identical second call should be deduplicated
        second = agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        assert second is None, "identical advice should be deduplicated"
        # get_current_advice still returns the stored advice
        assert agent.get_current_advice() is not None

    def test_get_current_advice(self):
        agent = HealthAgent(data_store=None)
        assert agent.get_current_advice() is None
        for _ in range(5):
            agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        # Force fresh advice on the next tick
        agent._last_advice_key = None
        result = agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        assert result is not None
        assert agent.get_current_advice() is not None

    def test_advice_history_capped(self):
        agent = HealthAgent(data_store=None)
        # Force many unique advice entries by alternating predictions
        for i in range(60):
            pred = i % 3
            # Reset agent after each advice to force uniqueness
            agent._last_advice_key = None
            agent.process_tick(prediction=pred, subject_id="s1", hr_sim=75.0 + i, spo2_sim=97.0 - i * 0.1)
        history = agent.get_advice_history(n=100)
        assert len(history) <= 50

    def test_get_advice_history_limited(self):
        agent = HealthAgent(data_store=None)
        for _ in range(15):
            agent._last_advice_key = None
            agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        history = agent.get_advice_history(n=5)
        assert len(history) == 5
        history_all = agent.get_advice_history()
        assert len(history_all) <= 20  # default n=20

    def test_get_trend_summary(self):
        agent = HealthAgent(data_store=None)
        for _ in range(5):
            agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        summary = agent.get_trend_summary()
        assert "trend" in summary
        assert "history_size" in summary
        assert summary["history_size"] == 5

    def test_get_rules(self):
        agent = HealthAgent(data_store=None)
        rules = agent.get_rules()
        assert len(rules) > 0
        assert "id" in rules[0]
        assert "name" in rules[0]

    def test_get_status(self):
        agent = HealthAgent(data_store=None)
        status = agent.get_status()
        assert status["enabled"] is True
        assert status["rules_count"] > 0
        assert "latest_severity" in status
        assert "trend" in status

    def test_reset(self):
        agent = HealthAgent(data_store=None)
        for _ in range(10):
            agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        agent.reset()
        assert agent.get_current_advice() is None
        assert agent.get_trend_summary()["history_size"] == 0
        assert len(agent.get_advice_history()) == 0

    def test_vital_proxies_from_feature_vector(self):
        """_compute_vital_proxies returns sensible clamped values."""
        agent = HealthAgent(data_store=None)
        # Random 256-dim vector
        vec = list(np.random.randn(256).astype(float))
        hr, spo2, rr = agent._compute_vital_proxies(vec)
        assert 50 <= hr <= 120
        assert 85 <= spo2 <= 100
        assert 0.5 <= rr <= 1.3

    def test_vital_proxies_from_json_string(self):
        """_compute_vital_proxies handles JSON string feature vectors."""
        agent = HealthAgent(data_store=None)
        import json
        vec = json.dumps([0.0] * 256)
        hr, spo2, rr = agent._compute_vital_proxies(vec)
        assert 50 <= hr <= 120
        assert 85 <= spo2 <= 100
        assert 0.5 <= rr <= 1.3

    def test_vital_proxies_none_input(self):
        """_compute_vital_proxies returns defaults for None input."""
        agent = HealthAgent(data_store=None)
        hr, spo2, rr = agent._compute_vital_proxies(None)
        assert hr == 80.0
        assert spo2 == 97.0
        assert rr == 0.85

    def test_process_tick_uses_provided_vitals(self):
        """Explicit hr_sim/spo2_sim override proxy computation."""
        agent = HealthAgent(data_store=None)
        for _ in range(10):
            agent.process_tick(prediction=0, subject_id="s1", hr_sim=60.0, spo2_sim=99.0)
        # Check that trend summary reflects the explicit values
        summary = agent.get_trend_summary()
        assert summary["history_size"] == 10

    # ------------------------------------------------------------------
    # New tests — edge cases
    # ------------------------------------------------------------------

    def test_vital_proxies_malformed_json(self):
        """Invalid JSON feature vector string falls back to defaults."""
        agent = HealthAgent(data_store=None)
        hr, spo2, rr = agent._compute_vital_proxies("not valid json")
        assert hr == 80.0
        assert spo2 == 97.0
        assert rr == 0.85

    def test_reset_then_process_tick(self):
        """After reset, process_tick generates fresh advice (not deduplicated)."""
        agent = HealthAgent(data_store=None)
        for _ in range(10):
            agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        agent.reset()
        result = agent.process_tick(prediction=0, subject_id="s1", hr_sim=75.0, spo2_sim=97.0)
        assert result is not None, "first tick after reset should produce advice"
        assert agent.get_trend_summary()["history_size"] == 1
