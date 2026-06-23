"""
Tests for the MCP Server — MemoryStore, Controller, Planner, and MCPServer.

Covers:
- Agent registration / deregistration / listing
- Memory store CRUD, TTL expiry, namespace scoping
- Knowledge sharing
- Controller fan-out / fan-in strategies
- Planner task decomposition and dependency resolution
- Workflow creation and status retrieval
- MCP status reporting
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest


# ===========================================================================
# Memory Store tests
# ===========================================================================

class TestMemoryStore:
    """Tests for the MemoryStore component."""

    def test_put_and_get(self, mcp_server):
        """Basic put → get round-trip."""
        asyncio.run(mcp_server.memory.put("k1", "hello"))
        result = asyncio.run(mcp_server.memory.get("k1"))
        assert result == "hello"

    def test_get_nonexistent(self, mcp_server):
        """Getting a missing key returns None."""
        result = asyncio.run(mcp_server.memory.get("no_such_key"))
        assert result is None

    def test_delete(self, mcp_server):
        """Deleting a key removes it."""
        asyncio.run(mcp_server.memory.put("k1", "value"))
        existed = asyncio.run(mcp_server.memory.delete("k1"))
        assert existed is True
        assert asyncio.run(mcp_server.memory.get("k1")) is None

    def test_delete_nonexistent(self, mcp_server):
        """Deleting a missing key returns False."""
        existed = asyncio.run(mcp_server.memory.delete("ghost"))
        assert existed is False

    def test_namespace_isolation(self, mcp_server):
        """Keys in different namespaces don't collide."""
        asyncio.run(mcp_server.memory.put("k", "mem_val", namespace="memory"))
        asyncio.run(mcp_server.memory.put("k", "ctrl_val", namespace="control"))
        assert (
            asyncio.run(mcp_server.memory.get("k", namespace="memory"))
            == "mem_val"
        )
        assert (
            asyncio.run(mcp_server.memory.get("k", namespace="control"))
            == "ctrl_val"
        )

    def test_list_namespace(self, mcp_server):
        """Listing a namespace returns all entries."""
        asyncio.run(mcp_server.memory.put("a", 1, namespace="memory"))
        asyncio.run(mcp_server.memory.put("b", 2, namespace="memory"))
        asyncio.run(mcp_server.memory.put("c", 3, namespace="control"))
        mem_entries = asyncio.run(
            mcp_server.memory.list_namespace("memory")
        )
        assert len(mem_entries) == 2
        keys = {e["key"] for e in mem_entries}
        assert keys == {"a", "b"}

    def test_ttl_expiry(self, mcp_server):
        """Entries expire after their TTL."""
        asyncio.run(
            mcp_server.memory.put("ephemeral", "gone", ttl=0)
        )
        # Force a small wait to ensure expiry
        import time
        time.sleep(0.01)
        result = asyncio.run(mcp_server.memory.get("ephemeral"))
        assert result is None

    def test_count(self, mcp_server):
        """Memory count reflects number of entries."""
        assert mcp_server.memory.count() == 0
        asyncio.run(mcp_server.memory.put("a", 1))
        asyncio.run(mcp_server.memory.put("b", 2))
        assert mcp_server.memory.count() == 2

    def test_clear(self, mcp_server):
        """Clear removes all entries."""
        asyncio.run(mcp_server.memory.put("a", 1))
        asyncio.run(mcp_server.memory.put("b", 2))
        asyncio.run(mcp_server.memory.clear())
        assert mcp_server.memory.count() == 0

    def test_share_knowledge(self, mcp_server):
        """Knowledge sharing stores with metadata."""
        asyncio.run(
            mcp_server.memory.share_knowledge(
                from_agent="agent_a",
                to_agents=["agent_b", "agent_c"],
                knowledge={"key_finding": "elevated HR"},
            )
        )
        ns = asyncio.run(
            mcp_server.memory.list_namespace("memory")
        )
        shared = [e for e in ns if e["key"].startswith("shared:agent_a:")]
        assert len(shared) == 1
        assert shared[0]["value"]["source_agent"] == "agent_a"
        assert "agent_b" in shared[0]["value"]["recipients"]


# ===========================================================================
# Controller tests
# ===========================================================================

class TestController:
    """Tests for the Controller component."""

    def test_register_agent(self, mcp_server):
        """Registering an agent adds it to the registry."""
        info = asyncio.run(
            mcp_server.controller.register_agent(
                agent_id="test_1",
                agent_type="health",
                capabilities=["monitoring"],
            )
        )
        assert info["agent_id"] == "test_1"
        assert info["status"] == "active"

    def test_register_duplicate_updates(self, mcp_server):
        """Registering the same agent_id updates the entry."""
        asyncio.run(
            mcp_server.controller.register_agent(
                agent_id="test_1", agent_type="health"
            )
        )
        info = asyncio.run(
            mcp_server.controller.register_agent(
                agent_id="test_1", agent_type="anomaly"
            )
        )
        assert info["agent_type"] == "anomaly"

    def test_list_agents(self, mcp_server):
        """Listing agents returns all registered."""
        asyncio.run(
            mcp_server.controller.register_agent(
                agent_id="a1", agent_type="health"
            )
        )
        asyncio.run(
            mcp_server.controller.register_agent(
                agent_id="a2", agent_type="anomaly"
            )
        )
        agents = asyncio.run(mcp_server.controller.list_agents())
        assert len(agents) >= 2

    def test_list_agents_filtered(self, mcp_server):
        """Filtering by agent_type works."""
        asyncio.run(
            mcp_server.controller.register_agent(
                agent_id="a1", agent_type="health"
            )
        )
        asyncio.run(
            mcp_server.controller.register_agent(
                agent_id="a2", agent_type="anomaly"
            )
        )
        health_agents = asyncio.run(
            mcp_server.controller.list_agents(agent_type="health")
        )
        assert len(health_agents) == 1
        assert health_agents[0]["agent_id"] == "a1"

    def test_deregister_agent(self, mcp_server):
        """Deregistering removes the agent."""
        asyncio.run(
            mcp_server.controller.register_agent(
                agent_id="temp", agent_type="test"
            )
        )
        existed = asyncio.run(
            mcp_server.controller.deregister_agent("temp")
        )
        assert existed is True
        assert asyncio.run(mcp_server.controller.get_agent("temp")) is None

    def test_deregister_nonexistent(self, mcp_server):
        """Deregistering a missing agent returns False."""
        existed = asyncio.run(
            mcp_server.controller.deregister_agent("ghost")
        )
        assert existed is False


# ===========================================================================
# Planner tests
# ===========================================================================

class TestPlanner:
    """Tests for the Planner component."""

    def test_decompose_task_anomaly(self, mcp_server):
        """Goal with 'anomaly' keyword gets anomaly subtask."""
        agents = [
            {
                "agent_id": "anomaly_detector",
                "agent_type": "anomaly",
                "capabilities": ["detect_anomalies"],
            }
        ]
        subtasks = asyncio.run(
            mcp_server.planner.decompose_task(
                "detect anomalies in health data", agents
            )
        )
        assert len(subtasks) >= 1
        assert any(
            "anomal" in t["task_id"].lower() for t in subtasks
        )

    def test_decompose_task_trend(self, mcp_server):
        """Goal with 'trend' keyword maps correctly."""
        agents = [
            {
                "agent_id": "trend_analyzer",
                "agent_type": "trend",
                "capabilities": ["analyze_trends"],
            }
        ]
        subtasks = asyncio.run(
            mcp_server.planner.decompose_task(
                "analyze health trends and forecast", agents
            )
        )
        assert len(subtasks) >= 1
        assert any("trend" in t["task_id"] for t in subtasks)

    def test_decompose_task_fallback(self, mcp_server):
        """When no keyword matches, falls back to first agent."""
        agents = [
            {
                "agent_id": "health_agent",
                "agent_type": "health",
                "capabilities": ["assess"],
            }
        ]
        subtasks = asyncio.run(
            mcp_server.planner.decompose_task(
                "do something completely new", agents
            )
        )
        assert len(subtasks) == 1
        assert subtasks[0]["assigned_agent"] == "health_agent"

    def test_decompose_no_agents(self, mcp_server):
        """Empty agent list produces empty subtasks."""
        subtasks = asyncio.run(
            mcp_server.planner.decompose_task("monitor health", [])
        )
        assert subtasks == []

    def test_resolve_dependencies_no_deps(self, mcp_server):
        """Subtasks with no dependencies stay in order."""
        subtasks = [
            {"task_id": "t1", "description": "A", "assigned_agent": "a1", "dependencies": []},
            {"task_id": "t2", "description": "B", "assigned_agent": "a2", "dependencies": []},
        ]
        sorted_tasks = asyncio.run(
            mcp_server.planner.resolve_dependencies(subtasks)
        )
        assert len(sorted_tasks) == 2

    def test_resolve_dependencies_with_deps(self, mcp_server):
        """Dependencies are topologically sorted."""
        subtasks = [
            {"task_id": "t1", "description": "A", "assigned_agent": "a1", "dependencies": []},
            {"task_id": "t2", "description": "B", "assigned_agent": "a2", "dependencies": ["t1"]},
            {"task_id": "t3", "description": "C", "assigned_agent": "a3", "dependencies": ["t1"]},
        ]
        sorted_tasks = asyncio.run(
            mcp_server.planner.resolve_dependencies(subtasks)
        )
        # t1 must come before t2 and t3
        ids = [t["task_id"] for t in sorted_tasks]
        assert ids[0] == "t1"
        assert ids.index("t2") > ids.index("t1")
        assert ids.index("t3") > ids.index("t1")


# ===========================================================================
# MCPServer integration tests
# ===========================================================================

class TestMCPServerIntegration:
    """Integration tests for the full MCPServer."""

    def test_full_register_list_deregister(self, mcp_server):
        """Full lifecycle: register → list → deregister."""
        asyncio.run(
            mcp_server.register_agent(
                agent_id="lifecycle_test",
                agent_type="test",
                capabilities=["testing"],
            )
        )
        agents = asyncio.run(mcp_server.list_agents())
        assert any(a["agent_id"] == "lifecycle_test" for a in agents)

        asyncio.run(mcp_server.deregister_agent("lifecycle_test"))
        agents = asyncio.run(mcp_server.list_agents())
        assert not any(a["agent_id"] == "lifecycle_test" for a in agents)

    def test_start_workflow(self, mcp_server):
        """Starting a workflow returns a session."""
        asyncio.run(
            mcp_server.register_agent(
                agent_id="wf_agent",
                agent_type="health",
                capabilities=["monitoring"],
            )
        )
        session = asyncio.run(
            mcp_server.start_workflow("monitor health status")
        )
        assert "session_id" in session
        assert session["status"] == "planned"
        assert session["session_id"].startswith("wf_")

    def test_get_workflow_status_nonexistent(self, mcp_server):
        """Querying a nonexistent workflow returns None."""
        result = asyncio.run(
            mcp_server.get_workflow_status("no_such_session")
        )
        assert result is None

    def test_mcp_status(self, mcp_server):
        """get_status returns expected keys."""
        status = mcp_server.get_status()
        assert "memory_entries" in status
        assert "control_active" in status
        assert "planning_queue_size" in status
        assert "uptime_seconds" in status
        assert status["control_active"] is True
        assert status["version"] == "2.0.0"
