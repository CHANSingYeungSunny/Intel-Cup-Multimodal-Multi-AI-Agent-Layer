"""
Agent Coordinator — multi-agent fan-out/fan-in orchestration.

Wraps the Single-layer ``HealthAgent`` together with the MCP server
and skills modules to deliver aggregated multi-agent outputs while
preserving full backward compatibility.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Inject Single AI Agent Layer into sys.path
# ---------------------------------------------------------------------------
_SINGLE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "intel multimodal (AI_Agent_Single_layer)",
)
if _SINGLE_DIR not in sys.path:
    sys.path.insert(0, _SINGLE_DIR)

from agent_orchestrator import HealthAgent  # noqa: E402

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Multi-agent coordinator — wraps the Single-layer HealthAgent and
    orchestrates skills + external agents through the MCP server.

    Parameters
    ----------
    mcp_server : MCPServer
        The MCP orchestration server instance.
    skills : dict[str, object]
        Mapping of skill name to instantiated skill object, e.g.
        ``{"anomaly_detector": AnomalyDetector(...), ...}``.
    db_session_factory : callable or None
        Multi-layer async session factory for persisting anomaly events
        and skill executions.
    single_agent : HealthAgent
        The Single-layer HealthAgent instance (for backward compatibility).
    llm_advice_generator : LLMAdviceGenerator or None
        Optional LLM enrichment skill.
    """

    def __init__(
        self,
        mcp_server,
        skills: dict,
        db_session_factory=None,
        single_agent: Optional[HealthAgent] = None,
        llm_advice_generator=None,
    ):
        self._mcp = mcp_server
        self._skills = skills
        self._db_session_factory = db_session_factory
        self._single_agent = single_agent
        self._llm = llm_advice_generator

        # Latest multi-agent results cache
        self._latest_multi_advice: Optional[dict] = None
        self._latest_anomalies: list[dict] = []
        self._latest_multi_trend: Optional[dict] = None

    # ------------------------------------------------------------------
    # Core pipeline — process_tick_multi
    # ------------------------------------------------------------------

    async def process_tick_multi(
        self,
        prediction: int,
        subject_id: str,
        feature_vector=None,
        hr_sim: Optional[float] = None,
        spo2_sim: Optional[float] = None,
        rr_sim: Optional[float] = None,
    ) -> dict:
        """
        Process one health observation tick through the full multi-agent
        pipeline.

        Returns a dict with keys:
        - ``single_agent_advice`` — the Single-layer HealthAgent's advice
        - ``multi_agent_advice`` — aggregated MultiAdviceResponse data
        - ``anomalies`` — list of detected anomaly events
        - ``skills_executed`` — names of skills that ran
        """
        skills_executed: list[str] = []
        multi_advice_data: Optional[dict] = None
        anomalies: list[dict] = []
        multi_trend_data: Optional[dict] = None

        # --- Resolve vital signs (same proxy computation as Single layer) ---
        hr = hr_sim
        spo2 = spo2_sim
        rr_val = rr_sim

        if self._single_agent and (hr is None or spo2 is None):
            proxy_hr, proxy_spo2, proxy_rr = (
                self._single_agent._compute_vital_proxies(feature_vector)
            )
            if hr is None:
                hr = proxy_hr
            if spo2 is None:
                spo2 = proxy_spo2
            if rr_val is None:
                rr_val = proxy_rr

        hr = hr or 80.0
        spo2 = spo2 or 97.0
        rr_val = rr_val or 0.85

        # --- 1. Single-layer HealthAgent (always — backward compat) ---
        single_advice = None
        if self._single_agent:
            try:
                single_advice = await self._single_agent.process_tick(
                    prediction=prediction,
                    subject_id=subject_id,
                    feature_vector=feature_vector,
                    hr_sim=hr,
                    spo2_sim=spo2,
                    rr_sim=rr_val,
                )
            except Exception as exc:
                logger.warning("Single-agent process_tick failed: %s", exc)

        # --- 2. Anomaly Detection skill ---
        anomaly_detector = self._skills.get("anomaly_detector")
        if anomaly_detector:
            try:
                t0 = time.perf_counter()
                anomalies = anomaly_detector.update(
                    hr=hr,
                    spo2=spo2,
                    rr=rr_val,
                    prediction=prediction,
                    subject_id=subject_id,
                )
                duration = (time.perf_counter() - t0) * 1000
                skills_executed.append("anomaly_detector")

                # Persist anomaly events
                if anomalies and self._db_session_factory:
                    await self._persist_anomalies(anomalies)

                # Log skill execution
                if self._db_session_factory:
                    await self._log_skill_execution(
                        "anomaly_detector",
                        "success",
                        duration,
                        {"prediction": prediction, "subject_id": subject_id},
                        {"anomalies_found": len(anomalies)},
                    )
            except Exception as exc:
                logger.warning("Anomaly detector failed: %s", exc)
        self._latest_anomalies = anomalies

        # --- 3. Advanced Trend Analysis skill ---
        advanced_trend = self._skills.get("advanced_trend_analyzer")
        if advanced_trend:
            try:
                t0 = time.perf_counter()
                advanced_trend.update({
                    "prediction": prediction,
                    "hr_sim": hr,
                    "spo2_sim": spo2,
                    "rr_sim": rr_val,
                })
                multi_trend_data = advanced_trend.get_summary()
                duration = (time.perf_counter() - t0) * 1000
                skills_executed.append("advanced_trend_analyzer")

                if self._db_session_factory:
                    await self._log_skill_execution(
                        "advanced_trend_analyzer",
                        "success",
                        duration,
                        {"prediction": prediction},
                        {"trends": multi_trend_data.get("multi_scale_trends", {})},
                    )
            except Exception as exc:
                logger.warning("Advanced trend analyzer failed: %s", exc)
        self._latest_multi_trend = multi_trend_data

        # --- 4. LLM enrichment (if configured) ---
        llm_enriched_advice = single_advice
        if self._llm and single_advice:
            try:
                # Build clinical context from trend data
                trend_summary = (
                    self._single_agent.get_trend_summary()
                    if self._single_agent
                    else {}
                )
                context = {
                    "current_prediction": prediction,
                    **trend_summary,
                }
                llm_enriched_advice = await self._llm.enrich(
                    single_advice, context
                )
                if single_advice != llm_enriched_advice:
                    skills_executed.append("llm_advice_generator")
            except Exception as exc:
                logger.warning("LLM enrichment failed: %s", exc)

        # --- 5. Fan-out to external agents ---
        agent_contributions: list[dict] = []
        external_agents = await self._mcp.list_agents()
        external_agents = [
            a
            for a in external_agents
            if a.get("endpoint_url")  # only agents with HTTP endpoints
        ]

        if external_agents:
            try:
                payload = {
                    "prediction": prediction,
                    "subject_id": subject_id,
                    "hr_sim": hr,
                    "spo2_sim": spo2,
                    "rr_sim": rr_val,
                }
                responses = await self._mcp.controller.fan_out(
                    payload,
                    [a["agent_id"] for a in external_agents],
                )
                for resp in responses:
                    if resp.get("status") == "success":
                        agent_contributions.append({
                            "agent_id": resp["agent_id"],
                            "agent_type": "external",
                            "advice": resp.get("data", {}),
                            "confidence": 1.0,
                        })
            except Exception as exc:
                logger.warning("External agent fan-out failed: %s", exc)

        # --- 6. Build multi-agent advice ---
        if single_advice or agent_contributions:
            # Determine consensus severity
            severities = []
            if single_advice:
                severities.append(single_advice.get("severity", "low"))
            for contrib in agent_contributions:
                sev = (
                    contrib.get("advice", {}).get("severity", "low")
                )
                severities.append(sev)

            from collections import Counter
            consensus = (
                Counter(severities).most_common(1)[0][0]
                if severities
                else "low"
            )

            # Build contributions including the health agent
            all_contributions = []
            if self._single_agent:
                all_contributions.append({
                    "agent_id": "health_agent",
                    "agent_type": "health",
                    "advice": llm_enriched_advice or {},
                    "confidence": 1.0,
                })
            all_contributions.extend(agent_contributions)

            multi_advice_data = {
                "aggregated_advice": llm_enriched_advice,
                "agent_contributions": all_contributions,
                "consensus_severity": consensus,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        self._latest_multi_advice = multi_advice_data

        return {
            "single_agent_advice": single_advice,
            "multi_agent_advice": multi_advice_data,
            "anomalies": anomalies,
            "skills_executed": skills_executed,
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_aggregated_advice(self) -> Optional[dict]:
        """Return the latest aggregated multi-agent advice."""
        return self._latest_multi_advice

    def get_anomalies(self, n: int = 20) -> list[dict]:
        """Return the most recent anomalies (up to *n*)."""
        return self._latest_anomalies[-n:] if self._latest_anomalies else []

    def get_multi_trend(self) -> Optional[dict]:
        """Return the latest multi-scale trend data."""
        return self._latest_multi_trend

    async def execute_skills(
        self,
        skill_names: list[str],
        input_data: dict,
    ) -> list[dict]:
        """
        Execute specified skills on demand.

        Returns a list of skill result dicts.
        """
        results: list[dict] = []
        for name in skill_names:
            skill = self._skills.get(name)
            if not skill:
                results.append({
                    "skill_name": name,
                    "status": "error",
                    "output": {"error": f"Skill '{name}' not found"},
                    "duration_ms": 0.0,
                })
                continue

            t0 = time.perf_counter()
            try:
                if name == "anomaly_detector":
                    output = skill.detect_on_stream(
                        [input_data]
                        if not isinstance(input_data, list)
                        else input_data
                    )
                    results.append({
                        "skill_name": name,
                        "status": "success",
                        "output": {"anomalies": output},
                        "duration_ms": (time.perf_counter() - t0) * 1000,
                    })
                elif name == "advanced_trend_analyzer":
                    skill.update(input_data)
                    output = skill.get_summary()
                    results.append({
                        "skill_name": name,
                        "status": "success",
                        "output": output,
                        "duration_ms": (time.perf_counter() - t0) * 1000,
                    })
                elif name == "llm_advice_generator" and self._llm:
                    output = await self._llm.enrich(
                        input_data.get("advice", {}),
                        input_data.get("context", {}),
                    )
                    results.append({
                        "skill_name": name,
                        "status": "success",
                        "output": output,
                        "duration_ms": (time.perf_counter() - t0) * 1000,
                    })
                else:
                    results.append({
                        "skill_name": name,
                        "status": "error",
                        "output": {"error": f"No handler for skill '{name}'"},
                        "duration_ms": 0.0,
                    })
            except Exception as exc:
                results.append({
                    "skill_name": name,
                    "status": "error",
                    "output": {"error": str(exc)},
                    "duration_ms": (time.perf_counter() - t0) * 1000,
                })

        # Persist skill executions
        if self._db_session_factory:
            for r in results:
                await self._log_skill_execution(
                    r["skill_name"],
                    r["status"],
                    r["duration_ms"],
                    input_data,
                    r["output"],
                )

        return results

    def get_coordinator_status(self) -> dict:
        """Return coordinator + MCP status."""
        return {
            "mcp_status": self._mcp.get_status(),
            "skills_loaded": list(self._skills.keys()),
            "single_agent_available": self._single_agent is not None,
            "llm_enabled": self._llm is not None
            and self._llm._backend != "none",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _persist_anomalies(self, anomalies: list[dict]) -> None:
        """Persist anomaly events to the database."""
        try:
            from models import AnomalyEvent

            async with self._db_session_factory() as db:
                for a in anomalies:
                    db.add(
                        AnomalyEvent(
                            subject_id=a.get("subject_id", ""),
                            z_score=a.get("z_score", 0.0),
                            metric_name=a.get("metric_name", "unknown"),
                            observed_value=a.get("observed_value", 0.0),
                            expected_value=a.get("expected_value", 0.0),
                            severity=a.get("severity", "warning"),
                            timestamp=datetime.fromisoformat(
                                a["timestamp"]
                            )
                            if a.get("timestamp")
                            else datetime.now(timezone.utc),
                        )
                    )
                await db.commit()
        except Exception as exc:
            logger.warning("Failed to persist anomalies: %s", exc)

    async def _log_skill_execution(
        self,
        skill_name: str,
        status: str,
        duration_ms: float,
        input_summary: dict,
        output_summary: dict,
    ) -> None:
        """Log a skill execution to the database."""
        try:
            from models import SkillExecution

            async with self._db_session_factory() as db:
                db.add(
                    SkillExecution(
                        skill_name=skill_name,
                        status=status,
                        duration_ms=duration_ms,
                        input_summary=input_summary,
                        output_summary=output_summary,
                    )
                )
                await db.commit()
        except Exception as exc:
            logger.debug("Failed to log skill execution: %s", exc)
