"""Unit tests for DecisionEngine.

Ports the existing test suite from ``tests_agent/test_decision_engine.py``
with the same rule-matching logic verified.
"""

import pytest
from unittest.mock import AsyncMock

from decision_engine import DecisionEngine


# Minimal rule set for testing, independent of config.py
TEST_RULES = [
    {
        "id": "test_001",
        "name": "test_influenza",
        "priority": 0,
        "enabled": True,
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
        "priority": 1,
        "enabled": True,
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
        "priority": 2,
        "enabled": True,
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
        "priority": 3,
        "enabled": True,
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


def _ts(trend="stable", unhealthy_ratio=0.0, healthy_ratio=0.0,
        hr_slope=0.0, spo2_slope=0.0, rr_slope=0.0):
    return {
        "trend": trend,
        "unhealthy_ratio": unhealthy_ratio,
        "healthy_ratio": healthy_ratio,
        "hr_slope": hr_slope,
        "spo2_slope": spo2_slope,
        "rr_slope": rr_slope,
    }


class TestDecisionEngine:
    """Tests for rule-matching logic and rule management."""

    def test_first_match_priority(self):
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _ts(trend="degrading", unhealthy_ratio=0.6)
        result = engine.evaluate(
            trend_summary=summary, current_prediction=2,
            hr_slope=6.0, spo2_slope=-4.0,
        )
        assert result["matched_rule_id"] == "test_001"
        assert result["severity"] == "high"
        assert result["possible_condition"] == "Test Influenza"

    def test_second_rule_when_first_does_not_match(self):
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _ts(trend="stable")
        result = engine.evaluate(
            trend_summary=summary, current_prediction=2,
            hr_slope=1.0, spo2_slope=-4.0,
        )
        assert result["matched_rule_id"] == "test_002"
        assert result["possible_condition"] == "Test Respiratory"

    def test_subhealthy_rule(self):
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _ts(trend="degrading", unhealthy_ratio=0.4)
        result = engine.evaluate(
            trend_summary=summary, current_prediction=1,
            hr_slope=1.0, spo2_slope=0.0,
        )
        assert result["matched_rule_id"] == "test_003"
        assert result["severity"] == "medium"

    def test_healthy_recovery_rule(self):
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _ts(trend="improving", healthy_ratio=0.8)
        result = engine.evaluate(
            trend_summary=summary, current_prediction=0,
        )
        assert result["matched_rule_id"] == "test_004"
        assert result["severity"] == "low"

    def test_default_advice_when_no_rule_matches(self):
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _ts(trend="stable")
        result = engine.evaluate(
            trend_summary=summary, current_prediction=1,
        )
        assert result["matched_rule_id"] is None
        assert result["matched_rule_name"] == "default"
        assert "severity" in result
        assert "advice" in result
        assert "context" in result
        assert "timestamp" in result

    def test_result_includes_context(self):
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _ts(trend="degrading")
        result = engine.evaluate(
            trend_summary=summary, current_prediction=1,
            hr_slope=2.0, spo2_slope=-1.0,
        )
        ctx = result.get("context", {})
        assert ctx["current_prediction"] == 1
        assert ctx["trend"] == "degrading"
        assert ctx["hr_slope"] == 2.0
        assert ctx["spo2_slope"] == -1.0

    def test_threshold_boundary_exact(self):
        engine = DecisionEngine(rules=TEST_RULES)
        summary = _ts(trend="degrading")
        result = engine.evaluate(
            trend_summary=summary, current_prediction=2,
            hr_slope=5.0, spo2_slope=0.0,
        )
        assert result["matched_rule_id"] == "test_001"

    def test_get_all_rules(self):
        engine = DecisionEngine(rules=TEST_RULES)
        rules = engine.get_all_rules()
        assert len(rules) == 4
        assert rules[0]["rule_id"] == "test_001"
        assert "condition" in rules[0]

    def test_get_rule_count(self):
        engine = DecisionEngine(rules=TEST_RULES)
        assert engine.get_rule_count() == 4

    def test_empty_rules_always_returns_default(self):
        engine = DecisionEngine(rules=[])
        result = engine.evaluate(
            trend_summary=_ts(), current_prediction=2,
        )
        assert result["matched_rule_id"] is None
        assert result["matched_rule_name"] == "default"

    def test_condition_without_trend_matches_any_trend(self):
        rules = [
            {
                "id": "no_trend_rule",
                "name": "any_trend",
                "priority": 0,
                "enabled": True,
                "condition": {"current_prediction": 2},
                "result": {
                    "severity": "high",
                    "possible_condition": "Any",
                    "advice": "",
                    "actions": [],
                },
            }
        ]
        engine = DecisionEngine(rules=rules)
        for trend in ("degrading", "improving", "stable"):
            result = engine.evaluate(
                trend_summary=_ts(trend=trend), current_prediction=2,
            )
            assert result["matched_rule_id"] == "no_trend_rule", f"failed for trend={trend}"

    # ---- Threshold key tests ----

    def test_unhealthy_ratio_min(self):
        rules = [
            {
                "id": "r1", "name": "t", "priority": 0, "enabled": True,
                "condition": {"current_prediction": 2, "unhealthy_ratio_min": 0.5},
                "result": {"severity": "high", "possible_condition": "X", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        r = engine.evaluate(_ts(unhealthy_ratio=0.6), 2)
        assert r["matched_rule_id"] == "r1"
        r2 = engine.evaluate(_ts(unhealthy_ratio=0.4), 2)
        assert r2["matched_rule_id"] is None

    def test_healthy_ratio_min(self):
        rules = [
            {
                "id": "r1", "name": "t", "priority": 0, "enabled": True,
                "condition": {"current_prediction": 0, "healthy_ratio_min": 0.6},
                "result": {"severity": "low", "possible_condition": "X", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        r = engine.evaluate(_ts(healthy_ratio=0.7), 0)
        assert r["matched_rule_id"] == "r1"
        r2 = engine.evaluate(_ts(healthy_ratio=0.5), 0)
        assert r2["matched_rule_id"] is None

    def test_hr_trend_max(self):
        rules = [
            {
                "id": "r1", "name": "t", "priority": 0, "enabled": True,
                "condition": {"current_prediction": 1, "hr_trend_max": 3.0},
                "result": {"severity": "medium", "possible_condition": "X", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        r = engine.evaluate(_ts(), 1, hr_slope=2.0)
        assert r["matched_rule_id"] == "r1"
        r2 = engine.evaluate(_ts(), 1, hr_slope=4.0)
        assert r2["matched_rule_id"] is None

    def test_spo2_trend_min(self):
        rules = [
            {
                "id": "r1", "name": "t", "priority": 0, "enabled": True,
                "condition": {"current_prediction": 2, "spo2_trend_min": 2.0},
                "result": {"severity": "high", "possible_condition": "X", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        r = engine.evaluate(_ts(), 2, spo2_slope=3.0)
        assert r["matched_rule_id"] == "r1"
        r2 = engine.evaluate(_ts(), 2, spo2_slope=1.0)
        assert r2["matched_rule_id"] is None

    def test_empty_condition_dict(self):
        rules = [
            {
                "id": "catch_all", "name": "t", "priority": 0, "enabled": True,
                "condition": {},
                "result": {"severity": "low", "possible_condition": "C", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        r = engine.evaluate(_ts(), 2, hr_slope=0.0)
        assert r["matched_rule_id"] == "catch_all"

    def test_rr_trend_min_rule(self):
        rules = [
            {
                "id": "r1", "name": "t", "priority": 0, "enabled": True,
                "condition": {"current_prediction": 1, "rr_trend_min": 0.05},
                "result": {"severity": "medium", "possible_condition": "X", "advice": "", "actions": []},
            }
        ]
        engine = DecisionEngine(rules=rules)
        r = engine.evaluate(_ts(), 1, rr_slope=0.06)
        assert r["matched_rule_id"] == "r1"
        r2 = engine.evaluate(_ts(), 1, rr_slope=0.03)
        assert r2["matched_rule_id"] is None

    def test_disabled_rule_is_skipped(self):
        rules = [
            {
                "id": "disabled_rule",
                "name": "t",
                "priority": 0,
                "enabled": False,
                "condition": {"current_prediction": 2},
                "result": {"severity": "high", "possible_condition": "X", "advice": "", "actions": []},
            },
            {
                "id": "enabled_rule",
                "name": "t2",
                "priority": 1,
                "enabled": True,
                "condition": {"current_prediction": 2},
                "result": {"severity": "medium", "possible_condition": "Y", "advice": "", "actions": []},
            },
        ]
        engine = DecisionEngine(rules=rules)
        r = engine.evaluate(_ts(), 2)
        assert r["matched_rule_id"] == "enabled_rule"
