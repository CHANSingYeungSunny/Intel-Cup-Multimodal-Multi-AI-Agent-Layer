"""Unit tests for DecisionEngine — rule matching and advice generation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_layer.decision_engine import DecisionEngine


# Minimal rule set for testing, independent of agent_config.py
TEST_RULES = [
    {
        "id": "test_001",
        "name": "test_influenza",
        "condition": {
            "current_prediction": 2,
            "trend": "degrading",
            "hr_trend_min": 5.0,
        },
        "result": {
            "severity": "high",
            "possible_condition": "Test Influenza",
            "advice": "Test influenza advice.",
            "actions": ["action_a"],
        },
    },
    {
        "id": "test_002",
        "name": "test_respiratory",
        "condition": {
            "current_prediction": 2,
            "spo2_trend_max": -3.0,
        },
        "result": {
            "severity": "high",
            "possible_condition": "Test Respiratory",
            "advice": "Test respiratory advice.",
            "actions": ["action_b"],
        },
    },
    {
        "id": "test_003",
        "name": "test_early_warning",
        "condition": {
            "current_prediction": 1,
            "trend": "degrading",
        },
        "result": {
            "severity": "medium",
            "possible_condition": "Test Early Warning",
            "advice": "Test early warning advice.",
            "actions": ["action_c"],
        },
    },
    {
        "id": "test_004",
        "name": "test_stable_healthy",
        "condition": {
            "current_prediction": 0,
            "trend": "improving",
        },
        "result": {
            "severity": "low",
            "possible_condition": "Test Recovery",
            "advice": "Test recovery advice.",
            "actions": ["action_d"],
        },
    },
]


def _trend_summary(trend="stable", unhealthy_ratio=0.0, healthy_ratio=0.0):
    return {
        "trend": trend,
        "unhealthy_ratio": unhealthy_ratio,
        "healthy_ratio": healthy_ratio,
        "hr_slope": 0.0,
        "spo2_slope": 0.0,
    }


class TestDecisionEngine:
    """Tests for rule-matching logic."""

    def test_first_match_priority(self):
        """Rule 001 matches before Rule 002 when both conditions hold."""
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _trend_summary(trend="degrading", unhealthy_ratio=0.6)
        result = engine.evaluate(
            trend_summary=summary,
            current_prediction=2,
            hr_slope=6.0,
            spo2_slope=-4.0,  # both rule 001 and 002 would match
        )
        assert result["matched_rule_id"] == "test_001"
        assert result["severity"] == "high"
        assert result["possible_condition"] == "Test Influenza"

    def test_second_rule_when_first_does_not_match(self):
        """Rule 002 fires when Rule 001's hr_trend_min is not met."""
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _trend_summary(trend="stable")
        result = engine.evaluate(
            trend_summary=summary,
            current_prediction=2,
            hr_slope=1.0,     # does NOT meet ≥ 5.0
            spo2_slope=-4.0,   # meets ≤ -3.0
        )
        assert result["matched_rule_id"] == "test_002"
        assert result["possible_condition"] == "Test Respiratory"

    def test_subhealthy_rule(self):
        """Rule 003 matches sub-healthy + degrading."""
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _trend_summary(trend="degrading", unhealthy_ratio=0.4)
        result = engine.evaluate(
            trend_summary=summary,
            current_prediction=1,
            hr_slope=1.0,
            spo2_slope=0.0,
        )
        assert result["matched_rule_id"] == "test_003"
        assert result["severity"] == "medium"

    def test_healthy_recovery_rule(self):
        """Rule 004 matches healthy + improving."""
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _trend_summary(trend="improving", healthy_ratio=0.8)
        result = engine.evaluate(
            trend_summary=summary,
            current_prediction=0,
        )
        assert result["matched_rule_id"] == "test_004"
        assert result["severity"] == "low"

    def test_default_advice_when_no_rule_matches(self):
        """Fall back to default advice when every rule fails."""
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _trend_summary(trend="stable")
        result = engine.evaluate(
            trend_summary=summary,
            current_prediction=1,  # sub-healthy, but no rule for stable + sub-healthy
        )
        assert result["matched_rule_id"] is None
        assert result["matched_rule_name"] == "default"
        assert "severity" in result
        assert "advice" in result
        assert "context" in result
        assert "timestamp" in result

    def test_result_includes_context(self):
        """Every result dict contains the evaluation context."""
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _trend_summary(trend="degrading")
        result = engine.evaluate(
            trend_summary=summary,
            current_prediction=1,
            hr_slope=2.0,
            spo2_slope=-1.0,
        )
        ctx = result.get("context", {})
        assert ctx["current_prediction"] == 1
        assert ctx["trend"] == "degrading"
        assert ctx["hr_slope"] == 2.0
        assert ctx["spo2_slope"] == -1.0

    def test_threshold_boundary_exact(self):
        """Threshold conditions are inclusive (≥, ≤)."""
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _trend_summary(trend="degrading")
        result = engine.evaluate(
            trend_summary=summary,
            current_prediction=2,
            hr_slope=5.0,  # exactly at threshold
            spo2_slope=0.0,
        )
        assert result["matched_rule_id"] == "test_001"

    def test_get_all_rules(self):
        engine = DecisionEngine(rules=TEST_RULES)
        rules = engine.get_all_rules()
        assert len(rules) == 4
        assert rules[0]["id"] == "test_001"
        assert "condition" in rules[0]

    def test_add_rule(self):
        engine = DecisionEngine(rules=[])
        engine.add_rule(TEST_RULES[0])
        assert engine.get_rule_count() == 1
        # new rule should now match
        summary = _trend_summary(trend="degrading")
        result = engine.evaluate(
            trend_summary=summary, current_prediction=2, hr_slope=6.0
        )
        assert result["matched_rule_id"] == "test_001"

    def test_remove_rule(self):
        engine = DecisionEngine(rules=list(TEST_RULES))
        assert engine.remove_rule("test_001") is True
        assert engine.get_rule_count() == 3
        assert engine.remove_rule("nonexistent") is False
        assert engine.get_rule_count() == 3

    def test_empty_rules_always_returns_default(self):
        engine = DecisionEngine(rules=[])
        result = engine.evaluate(
            trend_summary=_trend_summary(),
            current_prediction=2,
        )
        assert result["matched_rule_id"] is None
        assert result["matched_rule_name"] == "default"

    def test_condition_without_trend_matches_any_trend(self):
        """A rule that omits 'trend' in its condition should match any trend."""
        rules = [
            {
                "id": "no_trend_rule",
                "name": "any_trend",
                "condition": {"current_prediction": 2},
                "result": {"severity": "high", "possible_condition": "Any", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        for trend in ("degrading", "improving", "stable"):
            result = engine.evaluate(
                trend_summary=_trend_summary(trend=trend),
                current_prediction=2,
            )
            assert result["matched_rule_id"] == "no_trend_rule", f"failed for trend={trend}"

    # ------------------------------------------------------------------
    # New tests — threshold keys, empty conditions, rr_slope
    # ------------------------------------------------------------------

    def test_unhealthy_ratio_min(self):
        """Rule with unhealthy_ratio_min threshold."""
        rules = [
            {
                "id": "r1", "name": "t", "condition": {"current_prediction": 2, "unhealthy_ratio_min": 0.5},
                "result": {"severity": "high", "possible_condition": "X", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        # 0.6 ≥ 0.5 → match
        r = engine.evaluate(_trend_summary(unhealthy_ratio=0.6), 2)
        assert r["matched_rule_id"] == "r1"
        # 0.4 < 0.5 → no match (default)
        r2 = engine.evaluate(_trend_summary(unhealthy_ratio=0.4), 2)
        assert r2["matched_rule_id"] is None

    def test_healthy_ratio_min(self):
        """Rule with healthy_ratio_min threshold."""
        rules = [
            {
                "id": "r1", "name": "t", "condition": {"current_prediction": 0, "healthy_ratio_min": 0.6},
                "result": {"severity": "low", "possible_condition": "X", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        r = engine.evaluate(_trend_summary(healthy_ratio=0.7), 0)
        assert r["matched_rule_id"] == "r1"
        r2 = engine.evaluate(_trend_summary(healthy_ratio=0.5), 0)
        assert r2["matched_rule_id"] is None

    def test_hr_trend_max(self):
        """Rule with hr_trend_max (upper bound)."""
        rules = [
            {
                "id": "r1", "name": "t", "condition": {"current_prediction": 1, "hr_trend_max": 3.0},
                "result": {"severity": "medium", "possible_condition": "X", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        # 2.0 ≤ 3.0 → match
        r = engine.evaluate(_trend_summary(), 1, hr_slope=2.0)
        assert r["matched_rule_id"] == "r1"
        # 4.0 > 3.0 → no match
        r2 = engine.evaluate(_trend_summary(), 1, hr_slope=4.0)
        assert r2["matched_rule_id"] is None

    def test_spo2_trend_min(self):
        """Rule with spo2_trend_min (lower bound)."""
        rules = [
            {
                "id": "r1", "name": "t", "condition": {"current_prediction": 2, "spo2_trend_min": 2.0},
                "result": {"severity": "high", "possible_condition": "X", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        # 3.0 ≥ 2.0 → match
        r = engine.evaluate(_trend_summary(), 2, spo2_slope=3.0)
        assert r["matched_rule_id"] == "r1"
        # 1.0 < 2.0 → no match
        r2 = engine.evaluate(_trend_summary(), 2, spo2_slope=1.0)
        assert r2["matched_rule_id"] is None

    def test_empty_condition_dict(self):
        """Rule with {} condition matches anything (always fires first)."""
        rules = [
            {
                "id": "catch_all", "name": "t", "condition": {},
                "result": {"severity": "low", "possible_condition": "C", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        r = engine.evaluate(_trend_summary(), 2, hr_slope=0.0)
        assert r["matched_rule_id"] == "catch_all"

    def test_rr_trend_min_rule(self):
        """Rule can condition on rr_slope via rr_trend_min."""
        rules = [
            {
                "id": "r1", "name": "t", "condition": {"current_prediction": 1, "rr_trend_min": 0.05},
                "result": {"severity": "medium", "possible_condition": "X", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        r = engine.evaluate(_trend_summary(), 1, rr_slope=0.06)
        assert r["matched_rule_id"] == "r1"
        r2 = engine.evaluate(_trend_summary(), 1, rr_slope=0.03)
        assert r2["matched_rule_id"] is None
