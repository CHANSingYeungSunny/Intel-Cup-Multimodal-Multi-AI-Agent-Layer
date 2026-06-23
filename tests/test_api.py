"""
Integration tests for the Multi-AI Agent Layer REST API.

Covers:
- All 11 backward-compatible /api/v1/ endpoints
- All 5 new /api/v1/multi/ endpoints
- All 5 new /api/v1/mcp/ endpoints
- Response schema validation
- Error states (422, 404)
"""

import pytest

from tests.conftest import make_tick_payload, make_rule_payload


# ===========================================================================
# Backward-Compatible Endpoints (/api/v1)
# ===========================================================================

class TestBackwardCompatAPI:
    """All 11 original Single-layer endpoints must work identically."""

    # --- POST /tick ---

    @pytest.mark.asyncio
    async def test_tick_returns_advice(self, async_client):
        """POST /tick returns an AdviceResponse."""
        resp = await async_client.post(
            "/api/v1/tick", json=make_tick_payload(prediction=2)
        )
        assert resp.status_code == 200
        data = resp.json()
        # Either advice or null (dedup on first call may still match)
        if data is not None:
            assert "severity" in data
            assert "advice" in data
            assert "possible_condition" in data

    @pytest.mark.asyncio
    async def test_tick_deduplication(self, async_client):
        """Identical tick twice → second returns null."""
        payload = make_tick_payload(prediction=0)
        resp1 = await async_client.post("/api/v1/tick", json=payload)
        resp2 = await async_client.post("/api/v1/tick", json=payload)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Second should be null if deduplicated
        if resp1.json() is not None:
            assert resp2.json() is None

    @pytest.mark.asyncio
    async def test_tick_validation_error(self, async_client):
        """Missing required field → 422."""
        resp = await async_client.post(
            "/api/v1/tick", json={"subject_id": "test"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_tick_invalid_prediction(self, async_client):
        """Prediction out of range → 422."""
        resp = await async_client.post(
            "/api/v1/tick",
            json=make_tick_payload(prediction=99),
        )
        assert resp.status_code == 422

    # --- GET /advice/current ---

    @pytest.mark.asyncio
    async def test_advice_current(self, async_client):
        """GET /advice/current returns advice or null."""
        # First send a tick to generate advice
        await async_client.post(
            "/api/v1/tick", json=make_tick_payload(prediction=1)
        )
        resp = await async_client.get("/api/v1/advice/current")
        assert resp.status_code == 200

    # --- GET /advice/history ---

    @pytest.mark.asyncio
    async def test_advice_history(self, async_client):
        """GET /advice/history returns history list."""
        resp = await async_client.get("/api/v1/advice/history?n=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "history" in data
        assert "count" in data

    # --- GET /trends/current ---

    @pytest.mark.asyncio
    async def test_trends_current(self, async_client):
        """GET /trends/current returns trend summary."""
        await async_client.post(
            "/api/v1/tick", json=make_tick_payload(prediction=0)
        )
        resp = await async_client.get("/api/v1/trends/current")
        assert resp.status_code == 200
        data = resp.json()
        assert "trend" in data
        assert data["trend"] in ("stable", "degrading", "improving")

    # --- GET /trends/history ---

    @pytest.mark.asyncio
    async def test_trends_history(self, async_client):
        """GET /trends/history returns snapshot list."""
        resp = await async_client.get("/api/v1/trends/history?window=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "snapshots" in data
        assert "count" in data

    # --- Rules CRUD ---

    @pytest.mark.asyncio
    async def test_get_rules(self, async_client):
        """GET /rules returns list of rules."""
        resp = await async_client.get("/api/v1/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_create_rule(self, async_client):
        """POST /rules creates a new rule."""
        payload = make_rule_payload(rule_id="api_test_rule")
        resp = await async_client.post("/api/v1/rules", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["rule_id"] == "api_test_rule"

    @pytest.mark.asyncio
    async def test_create_rule_validation(self, async_client):
        """Invalid rule → 422."""
        resp = await async_client.post(
            "/api/v1/rules",
            json={"rule_id": "bad", "name": "Bad Rule"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_rule(self, async_client):
        """DELETE /rules/{id} removes a rule."""
        # Create then delete
        await async_client.post(
            "/api/v1/rules",
            json=make_rule_payload(rule_id="to_delete"),
        )
        resp = await async_client.delete("/api/v1/rules/to_delete")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_rule_not_found(self, async_client):
        """Deleting nonexistent rule → 404."""
        resp = await async_client.delete("/api/v1/rules/nonexistent_xyz")
        assert resp.status_code == 404

    # --- Status & Health ---

    @pytest.mark.asyncio
    async def test_status(self, async_client):
        """GET /status returns agent status."""
        resp = await async_client.get("/api/v1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert data["enabled"] is True

    @pytest.mark.asyncio
    async def test_health(self, async_client):
        """GET /health returns ok."""
        resp = await async_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    # --- POST /reset ---

    @pytest.mark.asyncio
    async def test_reset(self, async_client):
        """POST /reset clears state."""
        await async_client.post(
            "/api/v1/tick", json=make_tick_payload(prediction=2)
        )
        resp = await async_client.post("/api/v1/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    # --- OpenAPI docs ---

    @pytest.mark.asyncio
    async def test_docs_endpoint(self, async_client):
        """GET /docs returns HTML."""
        resp = await async_client.get("/docs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ===========================================================================
# Multi-Agent Endpoints (/api/v1/multi)
# ===========================================================================

class TestMultiAPI:
    """Tests for the 5 new /api/v1/multi/ endpoints."""

    @pytest.mark.asyncio
    async def test_multi_advice(self, async_client):
        """GET /multi/advice returns aggregated advice."""
        # Generate some data first
        await async_client.post(
            "/api/v1/tick", json=make_tick_payload(prediction=0)
        )
        resp = await async_client.get("/api/v1/multi/advice")
        assert resp.status_code == 200
        data = resp.json()
        assert "aggregated_advice" in data
        assert "agent_contributions" in data
        assert "consensus_severity" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_multi_trends(self, async_client):
        """GET /multi/trends returns multi-scale trend data."""
        resp = await async_client.get("/api/v1/multi/trends")
        assert resp.status_code == 200
        data = resp.json()
        assert "single_agent_trend" in data
        assert "multi_scale_trends" in data
        assert "forecast" in data
        assert "cross_scale_insight" in data

    @pytest.mark.asyncio
    async def test_multi_anomalies(self, async_client):
        """GET /multi/anomalies returns anomaly events."""
        resp = await async_client.get("/api/v1/multi/anomalies?n=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "detected" in data
        assert "anomalies" in data
        assert isinstance(data["anomalies"], list)

    @pytest.mark.asyncio
    async def test_multi_skills_execute(self, async_client):
        """POST /multi/skills executes skills on demand."""
        payload = {
            "skill_names": ["anomaly_detector"],
            "input": {
                "hr": 80.0, "spo2": 97.0, "rr": 0.85,
                "prediction": 0, "subject_id": "test",
            },
        }
        resp = await async_client.post(
            "/api/v1/multi/skills", json=payload
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "skill_results" in data
        assert "aggregate_summary" in data
        assert len(data["skill_results"]) == 1
        assert data["skill_results"][0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_multi_skills_unknown(self, async_client):
        """Requesting unknown skill returns error in results."""
        payload = {
            "skill_names": ["nonexistent"],
            "input": {"data": "test"},
        }
        resp = await async_client.post(
            "/api/v1/multi/skills", json=payload
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["skill_results"][0]["status"] == "error"

    @pytest.mark.asyncio
    async def test_multi_agents(self, async_client):
        """GET /multi/agents lists registered agents."""
        resp = await async_client.get("/api/v1/multi/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert "count" in data
        # Health agent and skills should be registered
        assert data["count"] >= 2


# ===========================================================================
# MCP Endpoints (/api/v1/mcp)
# ===========================================================================

class TestMCPAPI:
    """Tests for the 5 new /api/v1/mcp/ endpoints."""

    @pytest.mark.asyncio
    async def test_mcp_status(self, async_client):
        """GET /mcp/status returns MCP server status."""
        resp = await async_client.get("/api/v1/mcp/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "memory_entries" in data
        assert "control_active" in data
        assert data["control_active"] is True
        assert data["version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_mcp_register_agent(self, async_client):
        """POST /mcp/agents registers a new agent."""
        payload = {
            "agent_id": "test_external_agent",
            "agent_type": "external",
            "capabilities": ["custom_analysis"],
            "endpoint_url": "http://example.com/api",
            "metadata": {"version": "1.0"},
        }
        resp = await async_client.post(
            "/api/v1/mcp/agents", json=payload
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["agent_id"] == "test_external_agent"
        assert data["agent_type"] == "external"
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_mcp_register_agent_validation(self, async_client):
        """Missing required fields → 422."""
        resp = await async_client.post(
            "/api/v1/mcp/agents", json={"agent_type": "test"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_mcp_deregister_agent(self, async_client):
        """DELETE /mcp/agents/{id} removes the agent."""
        # Register first
        await async_client.post(
            "/api/v1/mcp/agents",
            json={
                "agent_id": "to_deregister",
                "agent_type": "test",
                "capabilities": [],
            },
        )
        resp = await async_client.delete(
            "/api/v1/mcp/agents/to_deregister"
        )
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_mcp_deregister_not_found(self, async_client):
        """Deleting nonexistent agent → 404."""
        resp = await async_client.delete(
            "/api/v1/mcp/agents/ghost_agent"
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_mcp_create_workflow(self, async_client):
        """POST /mcp/workflow creates a workflow session."""
        payload = {
            "goal": "monitor health and detect anomalies",
            "context": {"subject_id": "test_subject"},
        }
        resp = await async_client.post(
            "/api/v1/mcp/workflow", json=payload
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["status"] == "planned"
        assert data["session_id"].startswith("wf_")

    @pytest.mark.asyncio
    async def test_mcp_get_workflow_status(self, async_client):
        """GET /mcp/workflow/{session_id} returns status."""
        # Create workflow first
        create_resp = await async_client.post(
            "/api/v1/mcp/workflow",
            json={"goal": "test workflow", "context": {}},
        )
        session_id = create_resp.json()["session_id"]

        resp = await async_client.get(
            f"/api/v1/mcp/workflow/{session_id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["status"] == "planned"

    @pytest.mark.asyncio
    async def test_mcp_workflow_not_found(self, async_client):
        """Querying nonexistent workflow → 404."""
        resp = await async_client.get(
            "/api/v1/mcp/workflow/no_such_session"
        )
        assert resp.status_code == 404
