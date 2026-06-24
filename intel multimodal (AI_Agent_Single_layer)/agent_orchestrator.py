"""
HealthAgent — central orchestrator for the Single AI Agent Layer.

Wraps :class:`TrendAnalyzer`, :class:`DecisionEngine`, and
:class:`AdviceGenerator` into a clean singleton that mirrors the
existing ``agent_layer/health_agent.py`` interface while using the
enhanced PostgreSQL-backed modules.

This class is the primary entry point for:
* The FastAPI route handlers (see ``main.py``)
* Direct programmatic use (e.g. from the Dashboard simulator bridge)
* Future multi-agent orchestration
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

import numpy as np

from config import (
    HISTORY_WINDOW_SIZE,
    ADVICE_HISTORY_MAX,
)
from trend_analyzer import TrendAnalyzer, _ensure_datetime
from decision_engine import DecisionEngine
from advice_generator import AdviceGenerator
from schemas import utc_now_iso

logger = logging.getLogger(__name__)


class HealthAgent:
    """
    Lightweight rule-based AI agent for health monitoring advice.

    Parameters
    ----------
    db_session_factory : callable, optional
        Async session factory for DB persistence.  When provided, each
        ``process_tick`` call persists observations, advice, and trend
        snapshots to PostgreSQL.
    """

    def __init__(self, db_session_factory=None):
        self._trend_analyzer = TrendAnalyzer(window_size=HISTORY_WINDOW_SIZE)
        self._decision_engine = DecisionEngine()
        self._advice_generator = AdviceGenerator()
        self._db_session_factory = db_session_factory

        self._latest_advice: Optional[dict] = None
        self._advice_history: list[dict] = []
        self._last_advice_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Main entry point — process one health observation tick
    # ------------------------------------------------------------------

    async def process_tick(
        self,
        prediction: int,
        subject_id: str,
        feature_vector=None,
        hr_sim: Optional[float] = None,
        spo2_sim: Optional[float] = None,
        rr_sim: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Process one simulation tick through the full agent pipeline.

        1. Compute vital-sign proxies from *feature_vector* when explicit
           hr / spo2 / rr are not provided.
        2. Feed the observation into :class:`TrendAnalyzer`.
        3. Persist observation to PostgreSQL (when DB is available).
        4. Evaluate decision rules against the current trend summary.
        5. Generate structured advice via :class:`AdviceGenerator`.
        6. Deduplicate — return ``None`` when the matched rule is unchanged.
        7. Persist advice log and trend snapshot.

        Returns the advice dict, or ``None`` when deduplicated.
        """
        # --- 1. Resolve vital signs -------------------------------------
        hr = hr_sim
        spo2 = spo2_sim
        rr_val = rr_sim

        if hr is None or spo2 is None:
            proxy_hr, proxy_spo2, proxy_rr = self._compute_vital_proxies(
                feature_vector
            )
            if hr is None:
                hr = proxy_hr
            if spo2 is None:
                spo2 = proxy_spo2
            if rr_val is None:
                rr_val = proxy_rr

        # --- 2. Update trend buffer ------------------------------------
        observation = self._trend_analyzer.add_observation(
            prediction=prediction,
            subject_id=subject_id,
            hr_sim=hr,
            spo2_sim=spo2,
            rr_sim=rr_val,
            feature_vector=feature_vector,
        )

        # --- 3. Persist observation to DB ------------------------------
        db_connected = False
        if self._db_session_factory:
            try:
                async with self._db_session_factory() as db:
                    await self._trend_analyzer.persist_observation(db, observation)
                    db_connected = True
            except Exception as exc:
                logger.warning("Failed to persist observation: %s", exc)

        # --- 4. Evaluate decision rules --------------------------------
        trend_summary = self._trend_analyzer.get_summary()
        matched_rule = self._decision_engine.evaluate(
            trend_summary=trend_summary,
            current_prediction=prediction,
            hr_slope=trend_summary["hr_slope"],
            spo2_slope=trend_summary["spo2_slope"],
            rr_slope=trend_summary["rr_slope"],
        )

        # --- 5. Generate advice ----------------------------------------
        if matched_rule.get("matched_rule_id") is not None:
            advice = self._advice_generator.generate(
                matched_rule=matched_rule,
                context=matched_rule.get("context", {}),
                timestamp=matched_rule.get("timestamp"),
            )
        else:
            advice = self._advice_generator.generate_default(
                context=matched_rule.get("context", {}),
                timestamp=matched_rule.get("timestamp"),
            )

        # --- 6. Deduplicate --------------------------------------------
        advice_key = (
            advice.get("matched_rule_id")
            or advice.get("matched_rule_name", "none")
        )
        if advice_key == self._last_advice_key:
            return None  # unchanged

        self._last_advice_key = advice_key
        self._latest_advice = advice

        # Manage ring buffer
        self._advice_history.append(advice)
        if len(self._advice_history) > ADVICE_HISTORY_MAX:
            self._advice_history = self._advice_history[-ADVICE_HISTORY_MAX:]

        # --- 7. Persist advice log & trend snapshot --------------------
        if self._db_session_factory:
            try:
                from models import AdviceLog

                async with self._db_session_factory() as db:
                    log_entry = AdviceLog(
                        matched_rule_id=advice.get("matched_rule_id"),
                        matched_rule_name=advice.get("matched_rule_name", "default"),
                        severity=advice.get("severity", "low"),
                        possible_condition=advice.get("possible_condition", ""),
                        advice=advice.get("advice", ""),
                        actions=advice.get("actions", []),
                        context=advice.get("context", {}),
                        timestamp=_ensure_datetime(
                            advice.get("timestamp", utc_now_iso())
                        ),
                    )
                    db.add(log_entry)
                    await db.commit()

                    await self._trend_analyzer.save_trend_snapshot(db)
            except Exception as exc:
                logger.warning(
                    "Failed to persist advice / trend snapshot: %s", exc
                )

        return advice

    # ------------------------------------------------------------------
    # Accessors (mirror existing HealthAgent interface)
    # ------------------------------------------------------------------

    def get_current_advice(self) -> Optional[dict]:
        """Return the most recently generated advice, or ``None``."""
        return self._latest_advice

    def get_advice_history(self, n: int = 20) -> list[dict]:
        """Return the most recent *n* advice entries."""
        n = max(1, min(n, ADVICE_HISTORY_MAX))
        return self._advice_history[-n:]

    def get_trend_summary(self) -> dict:
        """Return the current trend-analyzer summary."""
        return self._trend_analyzer.get_summary()

    def get_rules(self) -> list[dict]:
        """Return metadata for all active decision rules."""
        return self._decision_engine.get_all_rules()

    def get_status(self) -> dict:
        """Return lightweight agent status for system heartbeats."""
        latest = self._latest_advice
        return {
            "enabled": True,
            "rules_count": self._decision_engine.get_rule_count(),
            "history_size": self._trend_analyzer.get_history_size(),
            "latest_severity": (
                latest.get("severity", "none") if latest else "none"
            ),
            "latest_condition": (
                latest.get("possible_condition", "") if latest else ""
            ),
            "trend": self._trend_analyzer.get_trend(),
        }

    def reset(self) -> None:
        """Clear all agent state (history, advice buffer, dedup key)."""
        self._trend_analyzer.reset()
        self._latest_advice = None
        self._advice_history.clear()
        self._last_advice_key = None
        logger.info("HealthAgent state has been reset")

    # ------------------------------------------------------------------
    # Decision engine delegation (for DB-backed rule CRUD)
    # ------------------------------------------------------------------

    @property
    def decision_engine(self) -> DecisionEngine:
        """Expose the decision engine for rule CRUD operations."""
        return self._decision_engine

    @property
    def trend_analyzer(self) -> TrendAnalyzer:
        """Expose the trend analyzer for DB history queries."""
        return self._trend_analyzer

    # ------------------------------------------------------------------
    # Lightweight vital-sign proxies from feature vectors
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_vital_proxies(
        feature_vector,
    ) -> Tuple[float, float, float]:
        """
        Compute proxy HR, SpO₂, and RR from a feature vector.

        Uses the same PCA-projection rationale as the original
        ``health_agent.py``: the first third of the fusion embedding
        maps to vision features (HR), the middle to audio (SpO₂), and
        the final to physiological (RR interval).
        """
        try:
            if isinstance(feature_vector, str):
                vec = np.array(json.loads(feature_vector), dtype=np.float64)
            elif feature_vector is not None:
                vec = np.array(feature_vector, dtype=np.float64).flatten()
            else:
                return (80.0, 97.0, 0.85)

            n = len(vec)
            third = max(1, n // 3)

            vision_proxy = float(np.mean(vec[:third]))
            hr = 75.0 + vision_proxy * 10.0

            audio_proxy = float(np.mean(vec[third : 2 * third]))
            spo2 = 97.0 - abs(audio_proxy) * 5.0

            physio_proxy = float(np.mean(vec[2 * third :]))
            rr = 0.85 + physio_proxy * 0.2

            return (
                round(float(np.clip(hr, 50, 120)), 1),
                round(float(np.clip(spo2, 85, 100)), 1),
                round(float(np.clip(rr, 0.5, 1.3)), 3),
            )
        except Exception as exc:
            logger.warning(
                "Vital proxy computation failed, using defaults: %s", exc
            )
            return (80.0, 97.0, 0.85)
