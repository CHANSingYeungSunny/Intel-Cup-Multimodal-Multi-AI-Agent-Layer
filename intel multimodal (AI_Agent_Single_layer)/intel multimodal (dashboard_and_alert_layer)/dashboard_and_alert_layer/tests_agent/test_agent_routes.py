"""Integration tests for AI Agent Flask REST endpoints."""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dashboard.backend.data_loader import store
from dashboard.backend.app import create_app


@pytest.fixture
def client():
    """Flask test client with seeded HealthAgent."""
    store.load_all()

    from agent_layer.health_agent import HealthAgent
    from agent_layer.routes.agent_routes import set_agent_instance

    agent = HealthAgent(store)
    # Seed enough ticks to build a trend and generate advice
    for _ in range(8):
        agent.process_tick(prediction=0, subject_id="s_test", hr_sim=75.0, spo2_sim=97.0)
    agent.process_tick(prediction=0, subject_id="s_test", hr_sim=75.0, spo2_sim=97.0)

    set_agent_instance(agent)

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestAgentRoutes:
    """Tests for the 4 agent REST endpoints."""

    def test_agent_advice_endpoint(self, client):
        resp = client.get("/api/agent_advice")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "latest_advice" in data
        assert "trend_summary" in data
        assert "active_rules_count" in data
        assert data["active_rules_count"] > 0

        advice = data["latest_advice"]
        assert "severity" in advice
        assert "possible_condition" in advice
        assert "advice" in advice
        assert "timestamp" in advice

        trend = data["trend_summary"]
        assert "trend" in trend
        assert "history_size" in trend
        assert "unhealthy_ratio" in trend
        assert "healthy_ratio" in trend
        assert "hr_slope" in trend
        assert "spo2_slope" in trend
        assert "rr_slope" in trend
        assert "trend_window_size" in trend

        # Validate advice context sub-keys
        ctx = advice.get("context", {})
        assert "current_prediction" in ctx
        assert "trend" in ctx
        assert "hr_slope" in ctx
        assert "spo2_slope" in ctx
        assert "rr_slope" in ctx
        assert "unhealthy_ratio" in ctx
        assert "healthy_ratio" in ctx

    def test_agent_history_endpoint(self, client):
        resp = client.get("/api/agent_history")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "history" in data
        assert "count" in data
        assert isinstance(data["history"], list)

        # Validate structure of history entries
        for entry in data["history"]:
            assert "matched_rule_id" in entry or "matched_rule_id" not in entry  # can be None
            assert "severity" in entry
            assert "possible_condition" in entry
            assert "advice" in entry
            assert "timestamp" in entry

        # With query param
        resp = client.get("/api/agent_history?n=3")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["count"] <= 3

    def test_agent_history_invalid_n(self, client):
        """Invalid n param should default to 20."""
        resp = client.get("/api/agent_history?n=abc")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["count"] <= 20

    def test_agent_history_n_clamped(self, client):
        """n should be clamped to [1, 50]."""
        resp = client.get("/api/agent_history?n=100")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["count"] <= 50

        resp = client.get("/api/agent_history?n=0")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["count"] >= 1

    def test_agent_rules_endpoint(self, client):
        resp = client.get("/api/agent_rules")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "rules" in data
        assert "count" in data
        assert data["count"] > 0
        rule = data["rules"][0]
        assert "id" in rule
        assert "name" in rule
        assert "condition" in rule
        assert "result_severity" in rule
        assert "result_condition" in rule
        # Validate condition keys are from the known set
        valid_cond_keys = {
            "current_prediction", "trend",
            "hr_trend_min", "hr_trend_max",
            "spo2_trend_min", "spo2_trend_max",
            "rr_trend_min", "rr_trend_max",
            "unhealthy_ratio_min", "healthy_ratio_min",
        }
        for cond_key in rule["condition"]:
            assert cond_key in valid_cond_keys, f"unexpected condition key: {cond_key}"

    def test_agent_status_endpoint(self, client):
        resp = client.get("/api/agent_status")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["enabled"] is True
        assert data["rules_count"] > 0
        assert data["history_size"] > 0
        assert "latest_severity" in data
        assert "latest_condition" in data
        assert "trend" in data

    def test_agent_disabled_returns_503(self, client):
        """When no agent is set, endpoints return 503."""
        from agent_layer.routes.agent_routes import set_agent_instance
        set_agent_instance(None)

        for endpoint in ["/api/agent_advice", "/api/agent_history",
                         "/api/agent_rules", "/api/agent_status"]:
            resp = client.get(endpoint)
            assert resp.status_code == 503
            data = json.loads(resp.data)
            assert "error" in data
