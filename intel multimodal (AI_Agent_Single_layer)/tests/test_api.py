"""Integration tests for the FastAPI REST endpoints."""

import pytest


# ---- POST /api/v1/tick ----


@pytest.mark.asyncio
async def test_tick_returns_advice(async_client):
    """A valid tick returns a full advice dict."""
    payload = {
        "prediction": 2,
        "subject_id": "subject14",
        "hr_sim": 95.0,
        "spo2_sim": 94.0,
        "rr_sim": 0.72,
    }
    response = await async_client.post("/api/v1/tick", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data is not None
    assert "matched_rule_id" in data
    assert "severity" in data
    assert "advice" in data
    assert "context" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_tick_deduplicates(async_client):
    """Two identical ticks → second returns null."""
    payload = {
        "prediction": 0,
        "subject_id": "s1",
        "hr_sim": 72.0,
        "spo2_sim": 98.0,
        "rr_sim": 0.88,
    }
    r1 = await async_client.post("/api/v1/tick", json=payload)
    assert r1.status_code == 200
    assert r1.json() is not None

    r2 = await async_client.post("/api/v1/tick", json=payload)
    assert r2.status_code == 200
    assert r2.json() is None  # deduplicated


@pytest.mark.asyncio
async def test_tick_missing_prediction_returns_422(async_client):
    response = await async_client.post("/api/v1/tick", json={"subject_id": "s1"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_tick_out_of_range_prediction_returns_422(async_client):
    response = await async_client.post(
        "/api/v1/tick",
        json={"prediction": 5, "subject_id": "s1"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_tick_with_feature_vector(async_client):
    """Providing a feature_vector triggers proxy computation."""
    payload = {
        "prediction": 1,
        "subject_id": "s1",
        "feature_vector": [0.1] * 256,
    }
    response = await async_client.post("/api/v1/tick", json=payload)
    assert response.status_code == 200
    assert response.json() is not None


# ---- GET /api/v1/advice/current ----


@pytest.mark.asyncio
async def test_advice_current_after_tick(async_client):
    payload = {"prediction": 2, "subject_id": "s1", "hr_sim": 100.0}
    await async_client.post("/api/v1/tick", json=payload)

    response = await async_client.get("/api/v1/advice/current")
    assert response.status_code == 200
    data = response.json()
    assert data is not None
    assert data["severity"] is not None


# ---- GET /api/v1/advice/history ----


@pytest.mark.asyncio
async def test_advice_history(async_client):
    # Send a few ticks with different severities
    await async_client.post(
        "/api/v1/tick",
        json={"prediction": 2, "subject_id": "s1", "hr_sim": 100.0},
    )
    await async_client.post(
        "/api/v1/tick",
        json={"prediction": 0, "subject_id": "s1", "hr_sim": 72.0},
    )

    response = await async_client.get("/api/v1/advice/history?n=10")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert "count" in data
    assert data["count"] >= 1


@pytest.mark.asyncio
async def test_advice_history_respects_limit(async_client):
    response = await async_client.get("/api/v1/advice/history?n=3")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] <= 3


# ---- GET /api/v1/trends/current ----


@pytest.mark.asyncio
async def test_trends_current(async_client):
    await async_client.post(
        "/api/v1/tick",
        json={"prediction": 1, "subject_id": "s1", "hr_sim": 75.0},
    )
    response = await async_client.get("/api/v1/trends/current")
    assert response.status_code == 200
    data = response.json()
    assert "trend" in data
    assert "hr_slope" in data
    assert "history_size" in data


# ---- GET /api/v1/trends/history ----


@pytest.mark.asyncio
async def test_trends_history(async_client):
    response = await async_client.get("/api/v1/trends/history?window=5")
    assert response.status_code == 200
    data = response.json()
    assert "snapshots" in data
    assert "count" in data


# ---- GET /api/v1/rules ----


@pytest.mark.asyncio
async def test_get_rules(async_client):
    response = await async_client.get("/api/v1/rules")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "rule_id" in data[0]
    assert "condition" in data[0]


# ---- POST /api/v1/rules ----


@pytest.mark.asyncio
async def test_create_rule(async_client):
    new_rule = {
        "rule_id": "test_custom",
        "name": "Test Custom Rule",
        "condition": {"current_prediction": 2},
        "result_severity": "high",
        "result_condition": "Custom Condition",
        "result_advice": "Custom advice text.",
        "result_actions": ["action_x"],
        "priority": 99,
    }
    response = await async_client.post("/api/v1/rules", json=new_rule)
    # 201 on success
    assert response.status_code in (201, 200)
    data = response.json()
    assert data["rule_id"] == "test_custom"


@pytest.mark.asyncio
async def test_create_rule_validation(async_client):
    """Invalid payload → 422."""
    response = await async_client.post("/api/v1/rules", json={"rule_id": "bad"})
    assert response.status_code == 422


# ---- DELETE /api/v1/rules/{rule_id} ----


@pytest.mark.asyncio
async def test_delete_rule(async_client):
    # First create a rule
    new_rule = {
        "rule_id": "to_delete",
        "name": "Will Delete",
        "condition": {},
        "result_severity": "low",
        "result_condition": "X",
        "result_advice": "Y",
        "result_actions": [],
        "priority": 200,
    }
    await async_client.post("/api/v1/rules", json=new_rule)

    # Then delete it
    response = await async_client.delete("/api/v1/rules/to_delete")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_delete_nonexistent_rule(async_client):
    response = await async_client.delete("/api/v1/rules/nonexistent_xyz")
    assert response.status_code == 404


# ---- GET /api/v1/status ----


@pytest.mark.asyncio
async def test_status(async_client):
    response = await async_client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert "rules_count" in data
    assert "history_size" in data
    assert "trend" in data


# ---- POST /api/v1/reset ----


@pytest.mark.asyncio
async def test_reset(async_client):
    """Reset clears agent state."""
    # First add a tick to build up state
    await async_client.post(
        "/api/v1/tick",
        json={"prediction": 2, "subject_id": "s1", "hr_sim": 100.0},
    )
    # Verify state exists
    status_before = await async_client.get("/api/v1/status")
    assert status_before.json()["history_size"] > 0

    # Reset
    response = await async_client.post("/api/v1/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

    # Verify state is cleared
    status_after = await async_client.get("/api/v1/status")
    assert status_after.json()["history_size"] == 0
    assert status_after.json()["latest_severity"] == "none"


# ---- GET /api/v1/health ----


@pytest.mark.asyncio
async def test_health(async_client):
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "db_connected" in data
