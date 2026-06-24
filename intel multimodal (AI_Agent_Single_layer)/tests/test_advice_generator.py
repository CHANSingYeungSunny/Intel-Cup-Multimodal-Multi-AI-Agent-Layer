"""Unit tests for AdviceGenerator."""

import pytest
from advice_generator import AdviceGenerator


@pytest.fixture
def generator():
    return AdviceGenerator()


MATCHED_RULE = {
    "id": "rule_001",
    "name": "test_influenza",
    "result": {
        "severity": "high",
        "possible_condition": "Possible Influenza / Severe Systemic Infection",
        "advice": "Immediate medical consultation is recommended.",
        "actions": ["notify_physician", "continuous_vitals_monitoring"],
    },
}

CONTEXT = {
    "current_prediction": 2,
    "trend": "degrading",
    "unhealthy_ratio": 0.6,
    "healthy_ratio": 0.2,
    "hr_slope": 7.2,
    "spo2_slope": -1.5,
    "rr_slope": 0.01,
}


class TestAdviceGenerator:
    """Tests for advice dict assembly."""

    def test_generate_has_all_required_keys(self, generator):
        advice = generator.generate(MATCHED_RULE, CONTEXT)
        assert "matched_rule_id" in advice
        assert "matched_rule_name" in advice
        assert "severity" in advice
        assert "possible_condition" in advice
        assert "advice" in advice
        assert "actions" in advice
        assert "context" in advice
        assert "timestamp" in advice

    def test_generate_passes_rule_id_and_name(self, generator):
        advice = generator.generate(MATCHED_RULE, CONTEXT)
        assert advice["matched_rule_id"] == "rule_001"
        assert advice["matched_rule_name"] == "test_influenza"

    def test_generate_passes_severity(self, generator):
        advice = generator.generate(MATCHED_RULE, CONTEXT)
        assert advice["severity"] == "high"

    def test_generate_passes_possible_condition(self, generator):
        advice = generator.generate(MATCHED_RULE, CONTEXT)
        assert "Influenza" in advice["possible_condition"]

    def test_generate_passes_advice_text(self, generator):
        advice = generator.generate(MATCHED_RULE, CONTEXT)
        assert "medical consultation" in advice["advice"]

    def test_generate_passes_actions(self, generator):
        advice = generator.generate(MATCHED_RULE, CONTEXT)
        assert advice["actions"] == ["notify_physician", "continuous_vitals_monitoring"]

    def test_generate_includes_context(self, generator):
        advice = generator.generate(MATCHED_RULE, CONTEXT)
        ctx = advice["context"]
        assert ctx["current_prediction"] == 2
        assert ctx["trend"] == "degrading"

    def test_generate_timestamp_is_iso8601(self, generator):
        advice = generator.generate(MATCHED_RULE, CONTEXT)
        assert "T" in advice["timestamp"]  # ISO 8601 format

    def test_generate_custom_timestamp(self, generator):
        ts = "2026-01-01T00:00:00.000Z"
        advice = generator.generate(MATCHED_RULE, CONTEXT, timestamp=ts)
        assert advice["timestamp"] == ts

    def test_generate_default_has_null_rule_id(self, generator):
        advice = generator.generate_default(CONTEXT)
        assert advice["matched_rule_id"] is None
        assert advice["matched_rule_name"] == "default"

    def test_generate_default_has_all_required_keys(self, generator):
        advice = generator.generate_default(CONTEXT)
        for key in [
            "matched_rule_id", "matched_rule_name", "severity",
            "possible_condition", "advice", "actions", "context", "timestamp",
        ]:
            assert key in advice, f"Missing key: {key}"

    def test_generate_deep_copies_to_prevent_mutation(self, generator):
        """Mutating the returned dict should not affect subsequent calls."""
        a1 = generator.generate(MATCHED_RULE, CONTEXT)
        a1["actions"].append("extra_action")
        a2 = generator.generate(MATCHED_RULE, CONTEXT)
        assert len(a2["actions"]) == 2  # original count, not 3

    def test_generate_with_empty_actions(self, generator):
        rule = {
            "id": "r1",
            "name": "t",
            "result": {
                "severity": "low",
                "possible_condition": "",
                "advice": "",
                "actions": [],
            },
        }
        advice = generator.generate(rule, CONTEXT)
        assert advice["actions"] == []

    def test_all_severity_levels(self, generator):
        for sev in ("low", "medium", "high"):
            rule = {
                "id": "r1",
                "name": "t",
                "result": {
                    "severity": sev,
                    "possible_condition": "",
                    "advice": "",
                    "actions": [],
                },
            }
            advice = generator.generate(rule, CONTEXT)
            assert advice["severity"] == sev
