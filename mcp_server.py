"""
MCP Server — Memory, Control, Planning orchestration for the Multi-AI Agent Layer.

Provides three integrated components:

**MemoryStore**
    In-memory LRU cache + PostgreSQL persistence for cross-agent knowledge
    sharing, with optional TTL-based expiry.

**Controller**
    Agent lifecycle management (register, deregister, health checks),
    fan-out request dispatch, and fan-in response aggregation.

**Planner**
    Task decomposition into subtask DAGs, dependency resolution via
    topological sort, and workflow execution management.

The ``MCPServer`` class composes all three and serves as the main
interface used by ``AgentCoordinator`` and the FastAPI routes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Inject Single AI Agent Layer into sys.path
# ---------------------------------------------------------------------------
_SINGLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "intel multimodal (AI_Agent_Single_layer)",
)
if _SINGLE_DIR not in sys.path:
    sys.path.insert(0, _SINGLE_DIR)

logger = logging.getLogger(__name__)


# ===========================================================================
# Memory Store
# ===========================================================================


class MemoryStore:
    """
    In-memory LRU cache with optional PostgreSQL persistence.

    Supports TTL-based expiry and namespace scoping (memory / control / planning).
    When a DB session factory is provided, entries are also persisted to the
    ``mcp_state`` table for durability across restarts.

    Parameters
    ----------
    max_entries : int
        Maximum number of in-memory entries (LRU eviction).
    default_ttl : int
        Default TTL in seconds for entries without an explicit TTL.
    db_session_factory : callable or None
        Async session factory for persisting state to PostgreSQL.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: int = 3600,
        db_session_factory=None,
    ):
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._db_session_factory = db_session_factory

        # OrderedDict for LRU: most recently used at the end
        self._store: OrderedDict[str, dict] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def put(
        self,
        key: str,
        value: Any,
        namespace: str = "memory",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a value under *key* in *namespace*.

        If *ttl* is provided it overrides the default TTL.
        """
        full_key = f"{namespace}:{key}"
        ttl_val = ttl if ttl is not None else self._default_ttl
        # ttl=0 means immediate expiry; ttl>0 sets a future expires_at
        if ttl_val == 0:
            expires_at = 0.0  # epoch — always expired
        elif ttl_val > 0:
            expires_at = (
                datetime.now(timezone.utc).timestamp() + ttl_val
            )
        else:
            expires_at = None  # negative → never expires

        entry = {
            "key": key,
            "namespace": namespace,
            "value": value,
            "ttl_seconds": ttl_val,
            "expires_at": expires_at,
        }

        # LRU eviction
        if full_key in self._store:
            del self._store[full_key]
        elif len(self._store) >= self._max_entries:
            # Evict oldest (first item in OrderedDict)
            self._store.popitem(last=False)

        self._store[full_key] = entry

        # Persist to DB
        if self._db_session_factory:
            await self._persist(entry)

    async def get(self, key: str, namespace: str = "memory") -> Optional[Any]:
        """
        Retrieve a value by *key* and *namespace*.

        Returns ``None`` if not found or expired.
        """
        full_key = f"{namespace}:{key}"
        entry = self._store.get(full_key)

        if entry is None:
            # Try loading from DB
            if self._db_session_factory:
                entry = await self._load_from_db(key, namespace)
                if entry:
                    self._store[full_key] = entry

        if entry is None:
            return None

        # Check TTL expiry
        if entry.get("expires_at") is not None:
            now_ts = datetime.now(timezone.utc).timestamp()
            if now_ts > entry["expires_at"]:
                del self._store[full_key]
                return None

        # Move to end (most recently used)
        self._store.move_to_end(full_key)
        return entry["value"]

    async def delete(self, key: str, namespace: str = "memory") -> bool:
        """Delete an entry. Returns True if it existed."""
        full_key = f"{namespace}:{key}"
        existed = full_key in self._store
        self._store.pop(full_key, None)
        return existed

    async def list_namespace(self, namespace: str = "memory") -> list[dict]:
        """List all non-expired keys in a namespace."""
        now_ts = datetime.now(timezone.utc).timestamp()
        result = []
        for full_key, entry in self._store.items():
            if entry["namespace"] != namespace:
                continue
            if entry.get("expires_at") and now_ts > entry["expires_at"]:
                continue
            result.append({"key": entry["key"], "value": entry["value"]})
        return result

    async def share_knowledge(
        self,
        from_agent: str,
        to_agents: list[str],
        knowledge: dict,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Share knowledge from one agent to others.

        Stores under ``shared:<from_agent>`` in the memory namespace with
        metadata about recipients.
        """
        payload = {
            "source_agent": from_agent,
            "recipients": to_agents,
            "knowledge": knowledge,
            "shared_at": datetime.now(timezone.utc).isoformat(),
        }
        await self.put(
            key=f"shared:{from_agent}:{uuid.uuid4().hex[:8]}",
            value=payload,
            namespace="memory",
            ttl=ttl,
        )

    def count(self) -> int:
        """Return the number of in-memory entries."""
        return len(self._store)

    async def clear(self) -> None:
        """Clear all in-memory entries."""
        self._store.clear()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def _persist(self, entry: dict) -> None:
        """Persist an entry to the mcp_state table."""
        try:
            from models import MCPState

            async with self._db_session_factory() as db:
                stmt = await db.execute(
                    __import__("sqlalchemy").select(MCPState).where(
                        MCPState.key == entry["key"],
                        MCPState.namespace == entry["namespace"],
                    )
                )
                existing = stmt.scalar_one_or_none()

                if existing:
                    existing.value = entry["value"]
                    existing.ttl_seconds = entry.get("ttl_seconds")
                    exp = entry.get("expires_at")
                    if exp:
                        existing.expires_at = datetime.fromtimestamp(
                            exp, tz=timezone.utc
                        )
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    exp = entry.get("expires_at")
                    db.add(
                        MCPState(
                            key=entry["key"],
                            value=entry["value"],
                            namespace=entry["namespace"],
                            ttl_seconds=entry.get("ttl_seconds"),
                            expires_at=(
                                datetime.fromtimestamp(exp, tz=timezone.utc)
                                if exp
                                else None
                            ),
                        )
                    )
                await db.commit()
        except Exception as exc:
            logger.debug("Failed to persist MCP state: %s", exc)

    async def _load_from_db(
        self, key: str, namespace: str
    ) -> Optional[dict]:
        """Try to load an entry from PostgreSQL."""
        try:
            from models import MCPState
            from sqlalchemy import select

            async with self._db_session_factory() as db:
                stmt = (
                    select(MCPState)
                    .where(
                        MCPState.key == key,
                        MCPState.namespace == namespace,
                    )
                )
                result = await db.execute(stmt)
                row = result.scalar_one_or_none()
                if row:
                    return {
                        "key": row.key,
                        "namespace": row.namespace,
                        "value": row.value,
                        "ttl_seconds": row.ttl_seconds,
                        "expires_at": (
                            row.expires_at.timestamp()
                            if row.expires_at
                            else None
                        ),
                    }
        except Exception as exc:
            logger.debug("Failed to load MCP state from DB: %s", exc)
        return None


# ===========================================================================
# Controller
# ===========================================================================


class Controller:
    """
    Agent lifecycle manager and request router.

    Handles agent registration, deregistration, health checks, and
    fan-out/fan-in patterns for parallel agent execution.

    Parameters
    ----------
    db_session_factory : callable or None
        Async session factory for persisting agent registry.
    agent_timeout : float
        Timeout in seconds for agent requests (default 10.0).
    """

    def __init__(
        self,
        db_session_factory=None,
        agent_timeout: float = 10.0,
    ):
        self._db_session_factory = db_session_factory
        self._agent_timeout = agent_timeout

        # In-memory agent registry: agent_id -> dict
        self._agents: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: list[str] | None = None,
        endpoint_url: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Register a new agent (or update an existing one).

        Returns the agent info dict.
        """
        now = datetime.now(timezone.utc)
        agent_info = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "status": "active",
            "capabilities": capabilities or [],
            "endpoint_url": endpoint_url,
            "metadata": metadata or {},
            "last_heartbeat": now.isoformat(),
        }
        self._agents[agent_id] = agent_info

        # Persist to DB
        if self._db_session_factory:
            try:
                await self._persist_agent(agent_info)
            except Exception as exc:
                logger.warning(
                    "Failed to persist agent %s: %s", agent_id, exc
                )

        logger.info(
            "Registered agent '%s' (%s) — %d capabilities",
            agent_id,
            agent_type,
            len(capabilities or []),
        )
        return agent_info

    async def deregister_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the registry.

        Returns True if the agent existed.
        """
        existed = agent_id in self._agents
        self._agents.pop(agent_id, None)
        if existed:
            logger.info("Deregistered agent '%s'", agent_id)
        return existed

    async def get_agent(self, agent_id: str) -> Optional[dict]:
        """Return agent info or None."""
        return self._agents.get(agent_id)

    async def list_agents(
        self, agent_type: Optional[str] = None
    ) -> list[dict]:
        """List all registered agents, optionally filtered by type."""
        result = list(self._agents.values())
        if agent_type:
            result = [
                a for a in result if a["agent_type"] == agent_type
            ]
        return result

    async def health_check_all(self) -> dict[str, str]:
        """
        Check health of all registered agents.

        For external agents (with endpoint_url), attempts an HTTP GET
        to their /health endpoint.  For internal agents, marks as active.

        Returns dict mapping agent_id -> status.
        """
        import httpx

        results = {}
        for agent_id, info in list(self._agents.items()):
            url = info.get("endpoint_url")
            if url:
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        resp = await client.get(f"{url.rstrip('/')}/health")
                        if resp.status_code == 200:
                            info["status"] = "active"
                        else:
                            info["status"] = "error"
                except Exception:
                    info["status"] = "error"
            else:
                info["status"] = "active"

            info["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
            results[agent_id] = info["status"]

        return results

    async def fan_out(
        self,
        request: dict,
        agent_ids: list[str],
    ) -> list[dict]:
        """
        Send a request to multiple agents in parallel.

        Each agent receives the request dict via HTTP POST to its
        endpoint_url (if configured).  Internal agents are skipped.

        Returns a list of response dicts (one per agent, in order).
        Failed/timeout agents produce an error entry.
        """
        import httpx

        async def _call_one(agent_id: str) -> dict:
            info = self._agents.get(agent_id)
            if not info:
                return {
                    "agent_id": agent_id,
                    "status": "error",
                    "error": "agent not found",
                }

            url = info.get("endpoint_url")
            if not url:
                return {
                    "agent_id": agent_id,
                    "status": "skipped",
                    "reason": "no endpoint_url (internal agent)",
                }

            try:
                async with httpx.AsyncClient(
                    timeout=self._agent_timeout
                ) as client:
                    resp = await client.post(url, json=request)
                    resp.raise_for_status()
                    return {
                        "agent_id": agent_id,
                        "status": "success",
                        "data": resp.json(),
                    }
            except Exception as exc:
                logger.warning(
                    "Fan-out to agent '%s' failed: %s", agent_id, exc
                )
                return {
                    "agent_id": agent_id,
                    "status": "error",
                    "error": str(exc),
                }

        tasks = [_call_one(aid) for aid in agent_ids]
        return await asyncio.gather(*tasks)

    async def fan_in(
        self,
        responses: list[dict],
        strategy: str = "majority",
    ) -> dict:
        """
        Aggregate multiple agent responses.

        Strategies:
        - ``"all"`` — return all responses
        - ``"first"`` — return the first successful response
        - ``"majority"`` — return the most common response (for categorical)
        - ``"average"`` — average numeric fields
        """
        successful = [r for r in responses if r.get("status") == "success"]

        if strategy == "first":
            return successful[0] if successful else {}

        elif strategy == "majority":
            if not successful:
                return {}
            # Pick the most common severity among agent advice
            severities = [
                r.get("data", {}).get("severity", "low")
                for r in successful
            ]
            from collections import Counter
            majority_sev = Counter(severities).most_common(1)[0][0]
            return {
                "consensus_severity": majority_sev,
                "agent_count": len(successful),
                "all_responses": responses,
            }

        elif strategy == "average":
            if not successful:
                return {}
            numeric_fields = {}
            for r in successful:
                data = r.get("data", {})
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        numeric_fields.setdefault(k, []).append(v)
            averaged = {
                k: sum(vals) / len(vals)
                for k, vals in numeric_fields.items()
            }
            return {"averaged": averaged, "agent_count": len(successful)}

        else:  # "all"
            return {"responses": responses, "agent_count": len(successful)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _persist_agent(self, info: dict) -> None:
        """Persist agent registration to the agent_registry table."""
        from models import AgentRegistry
        from sqlalchemy import select

        async with self._db_session_factory() as db:
            stmt = select(AgentRegistry).where(
                AgentRegistry.agent_id == info["agent_id"]
            )
            result = await db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.agent_type = info["agent_type"]
                existing.status = info["status"]
                existing.capabilities = info["capabilities"]
                existing.endpoint_url = info.get("endpoint_url")
                existing.metadata_ = info.get("metadata", {})
                existing.last_heartbeat = datetime.fromisoformat(
                    info["last_heartbeat"]
                )
            else:
                db.add(
                    AgentRegistry(
                        agent_id=info["agent_id"],
                        agent_type=info["agent_type"],
                        status=info["status"],
                        capabilities=info["capabilities"],
                        endpoint_url=info.get("endpoint_url"),
                        metadata_=info.get("metadata", {}),
                        last_heartbeat=datetime.fromisoformat(
                            info["last_heartbeat"]
                        ),
                    )
                )
            await db.commit()


# ===========================================================================
# Planner
# ===========================================================================


class Planner:
    """
    Workflow planner — task decomposition, dependency resolution,
    and execution management.

    Parameters
    ----------
    db_session_factory : callable or None
        Async session factory for persisting workflow sessions.
    """

    def __init__(self, db_session_factory=None):
        self._db_session_factory = db_session_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decompose_task(
        self,
        goal: str,
        available_agents: list[dict],
    ) -> list[dict]:
        """
        Break a high-level goal into subtasks assigned to agents.

        Uses a simple keyword-matching heuristic to map goals to
        agent capabilities.  Returns a list of subtask dicts with
        ``task_id``, ``description``, ``assigned_agent``, and
        ``dependencies``.
        """
        subtasks = []
        goal_lower = goal.lower()

        # Simple keyword → subtask mapping
        keyword_map = {
            "anomal": ("detect_anomalies", "Detect anomalies in the data stream"),
            "trend": ("analyze_trends", "Analyze multi-scale health trends"),
            "forecast": ("forecast_vitals", "Forecast vital signs"),
            "advice": ("generate_advice", "Generate health advice"),
            "health": ("assess_health", "Assess overall health status"),
            "monitor": ("monitor_vitals", "Monitor vital signs continuously"),
            "classify": ("classify_condition", "Classify the health condition"),
        }

        assigned_tasks = set()
        for keyword, (task_id, desc) in keyword_map.items():
            if keyword in goal_lower and task_id not in assigned_tasks:
                # Find an agent capable of this task
                agent = self._find_agent_for_task(
                    keyword, available_agents
                )
                assigned_tasks.add(task_id)
                subtasks.append({
                    "task_id": task_id,
                    "description": desc,
                    "assigned_agent": agent["agent_id"] if agent else None,
                    "agent_type": agent["agent_type"] if agent else None,
                    "dependencies": [],
                })

        if not subtasks and available_agents:
            # Default: assign to first available agent
            agent = available_agents[0]
            subtasks.append({
                "task_id": "execute_goal",
                "description": goal,
                "assigned_agent": agent["agent_id"],
                "agent_type": agent["agent_type"],
                "dependencies": [],
            })
        elif not available_agents:
            # No agents at all — return empty
            return []

        return subtasks

    async def resolve_dependencies(
        self, subtasks: list[dict]
    ) -> list[dict]:
        """
        Topologically sort subtasks by their declared dependencies.

        Returns the sorted list (Kahn's algorithm).
        """
        # Build adjacency and in-degree
        task_ids = {t["task_id"] for t in subtasks}
        in_degree: dict[str, int] = {tid: 0 for tid in task_ids}
        adj: dict[str, list[str]] = {tid: [] for tid in task_ids}

        for t in subtasks:
            for dep in t.get("dependencies", []):
                if dep in task_ids:
                    adj[dep].append(t["task_id"])
                    in_degree[t["task_id"]] += 1

        # Kahn's algorithm
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        sorted_ids = []
        while queue:
            node = queue.pop(0)
            sorted_ids.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Map back to full subtask dicts
        task_map = {t["task_id"]: t for t in subtasks}
        return [task_map[tid] for tid in sorted_ids if tid in task_map]

    async def create_workflow(
        self, session_id: str, subtasks: list[dict]
    ) -> dict:
        """
        Create a new workflow session and persist to DB.

        Returns the session dict.
        """
        now = datetime.now(timezone.utc)
        session = {
            "session_id": session_id,
            "workflow_plan": subtasks,
            "status": "planned",
            "agents_involved": list(
                {t["assigned_agent"] for t in subtasks if t.get("assigned_agent")}
            ),
            "results": {},
            "started_at": now.isoformat(),
            "completed_at": None,
        }

        # Persist to DB
        if self._db_session_factory:
            try:
                from models import MultiAgentSession

                async with self._db_session_factory() as db:
                    db.add(
                        MultiAgentSession(
                            session_id=session_id,
                            workflow_plan=subtasks,
                            status="planned",
                            agents_involved=session["agents_involved"],
                            results={},
                        )
                    )
                    await db.commit()
            except Exception as exc:
                logger.warning(
                    "Failed to persist workflow session %s: %s",
                    session_id,
                    exc,
                )

        return session

    async def get_workflow_status(self, session_id: str) -> Optional[dict]:
        """Return the current status of a workflow session."""
        if self._db_session_factory:
            try:
                from models import MultiAgentSession
                from sqlalchemy import select

                async with self._db_session_factory() as db:
                    stmt = select(MultiAgentSession).where(
                        MultiAgentSession.session_id == session_id
                    )
                    result = await db.execute(stmt)
                    row = result.scalar_one_or_none()
                    if row:
                        return {
                            "session_id": row.session_id,
                            "status": row.status,
                            "workflow_plan": row.workflow_plan,
                            "agents_involved": row.agents_involved,
                            "results": row.results,
                            "started_at": (
                                row.started_at.isoformat()
                                if row.started_at
                                else None
                            ),
                            "completed_at": (
                                row.completed_at.isoformat()
                                if row.completed_at
                                else None
                            ),
                        }
            except Exception as exc:
                logger.warning(
                    "Failed to load workflow %s: %s", session_id, exc
                )
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _find_agent_for_task(
        keyword: str, agents: list[dict]
    ) -> Optional[dict]:
        """Find the first agent whose capabilities or type match the keyword."""
        for agent in agents:
            caps = [
                c.lower()
                for c in agent.get("capabilities", [])
            ]
            agent_type = agent.get("agent_type", "").lower()
            if keyword in agent_type or any(keyword in c for c in caps):
                return agent
        return agents[0] if agents else None


# ===========================================================================
# MCPServer — top-level composition
# ===========================================================================


class MCPServer:
    """
    MCP orchestration server composing MemoryStore, Controller, and Planner.

    This is the main interface used by AgentCoordinator and the FastAPI
    MCP routes.

    Parameters
    ----------
    db_session_factory : callable or None
        Async session factory for database persistence.
    memory_max_entries : int
        Max in-memory cache entries (default 1000).
    memory_default_ttl : int
        Default TTL in seconds (default 3600).
    agent_timeout : float
        Timeout for external agent requests (default 10.0).
    """

    def __init__(
        self,
        db_session_factory=None,
        memory_max_entries: int = 1000,
        memory_default_ttl: int = 3600,
        agent_timeout: float = 10.0,
    ):
        self.memory = MemoryStore(
            max_entries=memory_max_entries,
            default_ttl=memory_default_ttl,
            db_session_factory=db_session_factory,
        )
        self.controller = Controller(
            db_session_factory=db_session_factory,
            agent_timeout=agent_timeout,
        )
        self.planner = Planner(
            db_session_factory=db_session_factory,
        )
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Convenience delegates
    # ------------------------------------------------------------------

    async def register_agent(self, **kwargs) -> dict:
        """Register an agent with the controller."""
        return await self.controller.register_agent(**kwargs)

    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent."""
        return await self.controller.deregister_agent(agent_id)

    async def get_agent(self, agent_id: str) -> Optional[dict]:
        """Get agent info."""
        return await self.controller.get_agent(agent_id)

    async def list_agents(
        self, agent_type: Optional[str] = None
    ) -> list[dict]:
        """List registered agents."""
        return await self.controller.list_agents(agent_type)

    async def route_request(
        self,
        request_type: str,
        payload: dict,
        preferred_agent: Optional[str] = None,
    ) -> dict:
        """
        Route a request to the appropriate agent(s).

        If *preferred_agent* is specified, routes only to that agent.
        Otherwise, fans out to all active agents matching the request type.
        """
        if preferred_agent:
            responses = await self.controller.fan_out(
                payload, [preferred_agent]
            )
        else:
            matching = [
                a["agent_id"]
                for a in await self.controller.list_agents()
                if request_type in a.get("capabilities", [])
                or a.get("agent_type") == request_type
            ]
            if not matching:
                matching = [
                    a["agent_id"]
                    for a in await self.controller.list_agents()
                ]
            responses = await self.controller.fan_out(payload, matching)

        return await self.controller.fan_in(responses)

    async def start_workflow(
        self, goal: str, context: Optional[dict] = None
    ) -> dict:
        """
        Plan and start a multi-agent workflow for a given goal.

        Returns the workflow session dict.
        """
        agents = await self.controller.list_agents()
        subtasks = await self.planner.decompose_task(goal, agents)
        sorted_tasks = await self.planner.resolve_dependencies(subtasks)

        session_id = f"wf_{uuid.uuid4().hex[:12]}"
        session = await self.planner.create_workflow(
            session_id, sorted_tasks
        )

        logger.info(
            "Workflow %s planned — %d subtasks, %d agents",
            session_id,
            len(sorted_tasks),
            len(session.get("agents_involved", [])),
        )
        return session

    async def get_workflow_status(self, session_id: str) -> Optional[dict]:
        """Get workflow session status."""
        return await self.planner.get_workflow_status(session_id)

    def get_status(self) -> dict:
        """Return MCP server status summary."""
        return {
            "memory_entries": self.memory.count(),
            "control_active": True,
            "planning_queue_size": 0,  # simplified — no async queue
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "version": "2.0.0",
        }
