"""
HealthAgent — central orchestrator for the AI Agent Layer.

Wires together :class:`TrendAnalyzer` and :class:`DecisionEngine` and
provides the primary interface for the HealthSimulator (``process_tick``)
and Flask routes (``get_current_advice``, etc.).

╔═══════════════════════════════════════════════════════════════════════════╗
║ DEPRECATED — Replaced by ../../agent_orchestrator.py (FastAPI service).  ║
║ See ../../README.md for migration guide.  Kept for backward compat.      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import json
import logging
from typing import Optional, Tuple

import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from agent_layer.trend_analyzer import TrendAnalyzer
from agent_layer.decision_engine import DecisionEngine
from agent_layer.agent_config import HISTORY_WINDOW_SIZE


class HealthAgent:
    """
    Lightweight rule-based AI agent for health monitoring advice.

    Parameters
    ----------
    data_store : DataStore
        The singleton DataStore instance (used to access predictions and
        feature vectors for vital-sign proxy computation).
    """

    def __init__(self, data_store=None):
        # data_store is accepted for forward compatibility (future data-backed rules)
        self._trend_analyzer = TrendAnalyzer(window_size=HISTORY_WINDOW_SIZE)
        self._decision_engine = DecisionEngine()
        self._latest_advice: Optional[dict] = None
        self._advice_history = []  # list of dicts, max 50 entries
        self._last_advice_key: Optional[str] = None  # for deduplication

    # ------------------------------------------------------------------
    # Main entry point — called by HealthSimulator on every tick
    # ------------------------------------------------------------------
    def process_tick(
        self,
        prediction: int,
        subject_id: str,
        feature_vector=None,
        hr_sim: float = None,
        spo2_sim: float = None,
        rr_sim: float = None,
    ) -> Optional[dict]:
        """
        Process one simulation tick.

        1. Compute lightweight vital-sign proxies from *feature_vector*
           when explicit *hr_sim* / *spo2_sim* are not provided.
        2. Feed the observation into :class:`TrendAnalyzer`.
        3. Evaluate decision rules.
        4. Store and return the advice dict (or ``None`` when unchanged).

        Returns
        -------
        dict or None
            Advice dict, or ``None`` when the advice is unchanged from the
            previous tick (deduplication).
        """
        # --- 1. Resolve vital signs -----------------------------------------
        hr = hr_sim
        spo2 = spo2_sim
        rr = rr_sim

        if hr is None or spo2 is None:
            proxy_hr, proxy_spo2, proxy_rr = self._compute_vital_proxies(feature_vector)
            if hr is None:
                hr = proxy_hr
            if spo2 is None:
                spo2 = proxy_spo2
            if rr is None:
                rr = proxy_rr

        # --- 2. Update trend buffer -----------------------------------------
        self._trend_analyzer.add_observation(
            prediction=prediction,
            subject_id=subject_id,
            hr_sim=hr,
            spo2_sim=spo2,
            rr_sim=rr,
        )

        # --- 3. Evaluate decision rules -------------------------------------
        trend_summary = self._trend_analyzer.get_summary()
        advice = self._decision_engine.evaluate(
            trend_summary=trend_summary,
            current_prediction=prediction,
            hr_slope=trend_summary["hr_slope"],
            spo2_slope=trend_summary["spo2_slope"],
            rr_slope=trend_summary["rr_slope"],
        )

        # --- 4. Deduplicate & store -----------------------------------------
        advice_key = advice.get("matched_rule_id", "none")
        if advice_key == self._last_advice_key:
            return None  # unchanged — no SocketIO emission needed

        self._last_advice_key = advice_key
        self._latest_advice = advice
        self._advice_history.append(advice)
        if len(self._advice_history) > 50:
            self._advice_history = self._advice_history[-50:]

        return advice

    # ------------------------------------------------------------------
    # Accessors for REST routes & SocketIO handlers
    # ------------------------------------------------------------------
    def get_current_advice(self) -> Optional[dict]:
        """Return the most recent advice, or ``None``."""
        return self._latest_advice

    def get_advice_history(self, n: int = 20) -> list[dict]:
        """Return the most recent *n* advice entries."""
        return self._advice_history[-n:]

    def get_trend_summary(self) -> dict:
        """Return the current trend-analyzer summary."""
        return self._trend_analyzer.get_summary()

    def get_rules(self) -> list[dict]:
        """Return metadata for all active decision rules."""
        return self._decision_engine.get_all_rules()

    def get_status(self) -> dict:
        """Return agent status for the system-status heartbeat."""
        advice = self._latest_advice
        return {
            "enabled": True,
            "rules_count": self._decision_engine.get_rule_count(),
            "history_size": self._trend_analyzer.get_history_size(),
            "latest_severity": advice.get("severity", "none") if advice else "none",
            "latest_condition": advice.get("possible_condition", "") if advice else "",
            "trend": self._trend_analyzer.get_trend(),
        }

    def reset(self):
        """Clear agent state (history, advice, dedup key)."""
        self._trend_analyzer.reset()
        self._latest_advice = None
        self._advice_history.clear()
        self._last_advice_key = None

    # ------------------------------------------------------------------
    # Lightweight vital-sign proxies from feature vectors
    #
    # Uses the same PCA-projection rationale as FeatureAnalyzer: the
    # first third of the 256-dim fusion embedding roughly corresponds to
    # vision features (correlated with HR), the middle third to audio
    # features (correlated with SpO₂), and the final third to physio
    # features (correlated with RR interval).
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_vital_proxies(feature_vector) -> Tuple[float, float, float]:
        """
        Compute proxy HR, SpO₂, and RR from a 256-dim feature vector.

        Returns (hr, spo2, rr) each as a float clamped to realistic ranges.
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

            # Vision-third proxy → HR  (baseline ~75, range 50–120)
            vision_proxy = float(np.mean(vec[:third]))
            hr = 75.0 + vision_proxy * 10.0

            # Audio-third proxy → SpO₂  (baseline ~97, range 85–100)
            audio_proxy = float(np.mean(vec[third : 2 * third]))
            spo2 = 97.0 - abs(audio_proxy) * 5.0

            # Physio-third proxy → RR interval  (baseline ~0.85, range 0.5–1.3)
            physio_proxy = float(np.mean(vec[2 * third :]))
            rr = 0.85 + physio_proxy * 0.2

            return (
                round(float(np.clip(hr, 50, 120)), 1),
                round(float(np.clip(spo2, 85, 100)), 1),
                round(float(np.clip(rr, 0.5, 1.3)), 3),
            )
        except Exception as e:
            logger.warning("Vital proxy computation failed, using defaults: %s", e)
            return (80.0, 97.0, 0.85)
