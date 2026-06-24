"""
TrendAnalyzer — maintains a rolling history buffer of health observations
and computes trend direction, vital-sign slopes, and summary statistics.

Used by HealthAgent to provide context for decision-rule evaluation.

╔═══════════════════════════════════════════════════════════════════════════╗
║ DEPRECATED — Replaced by ../../trend_analyzer.py (FastAPI service).      ║
║ See ../../README.md for migration guide.  Kept for backward compat.      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

from collections import deque
from typing import Optional

import numpy as np
from datetime import datetime, timezone

from agent_layer.agent_config import (
    HISTORY_WINDOW_SIZE,
    TREND_WINDOW_SIZE,
    DEGRADING_THRESHOLD,
    IMPROVING_THRESHOLD,
)


class TrendAnalyzer:
    """Rolling buffer of health observations with trend-detection logic."""

    def __init__(self, window_size=HISTORY_WINDOW_SIZE):
        self._history = deque(maxlen=window_size)
        self._window_size = window_size

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def add_observation(
        self,
        prediction: int,
        subject_id: str,
        hr_sim: float = None,
        spo2_sim: float = None,
        rr_sim: float = None,
        timestamp: str = None,
    ):
        """Push a new observation; deque maxlen handles automatic eviction."""
        self._history.append(
            {
                "prediction": int(prediction),
                "subject_id": str(subject_id),
                "hr": float(hr_sim) if hr_sim is not None else None,
                "spo2": float(spo2_sim) if spo2_sim is not None else None,
                "rr": float(rr_sim) if rr_sim is not None else None,
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            }
        )

    # ------------------------------------------------------------------
    # Trend classification
    # ------------------------------------------------------------------
    def get_trend(self) -> str:
        """
        Classify the recent prediction trend.

        Returns one of: ``"degrading"``, ``"improving"``, ``"stable"``.

        Uses the last *TREND_WINDOW_SIZE* observations.  If fewer
        observations are available the trend is ``"stable"``.
        """
        window = self._get_trend_window()
        if len(window) < max(2, TREND_WINDOW_SIZE // 2):
            return "stable"

        preds = [obs["prediction"] for obs in window]
        unhealthy_frac = preds.count(2) / len(preds)
        healthy_frac = preds.count(0) / len(preds)

        if unhealthy_frac >= DEGRADING_THRESHOLD:
            return "degrading"
        if healthy_frac >= IMPROVING_THRESHOLD:
            return "improving"
        return "stable"

    def get_unhealthy_ratio(self) -> float:
        """Fraction of Unhealthy (2) predictions in the trend window."""
        window = self._get_trend_window()
        if not window:
            return 0.0
        return sum(1 for o in window if o["prediction"] == 2) / len(window)

    def get_healthy_ratio(self) -> float:
        """Fraction of Healthy (0) predictions in the trend window."""
        window = self._get_trend_window()
        if not window:
            return 0.0
        return sum(1 for o in window if o["prediction"] == 0) / len(window)

    # ------------------------------------------------------------------
    # Vital-sign slopes  (simple linear regression via numpy.polyfit)
    # ------------------------------------------------------------------
    def get_hr_trend(self) -> float:
        """Slope of heart rate across the trend window (bpm / observation)."""
        return self._vital_slope("hr")

    def get_spo2_trend(self) -> float:
        """Slope of SpO₂ across the trend window (% / observation)."""
        return self._vital_slope("spo2")

    def get_rr_trend(self) -> float:
        """Slope of RR interval across the trend window (s / observation)."""
        return self._vital_slope("rr")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_recent_predictions(self, n: Optional[int] = None) -> list:
        """Return the last *n* prediction values (0/1/2)."""
        if n is None:
            n = TREND_WINDOW_SIZE
        items = list(self._history)
        return [obs["prediction"] for obs in items[-n:]]

    def get_history(self) -> list[dict]:
        """Return a shallow copy of the full history buffer."""
        return list(self._history)

    def get_history_size(self) -> int:
        return len(self._history)

    def get_summary(self) -> dict:
        """One-shot summary of all computed metrics."""
        return {
            "trend": self.get_trend(),
            "history_size": len(self._history),
            "trend_window_size": min(len(self._history), TREND_WINDOW_SIZE),
            "unhealthy_ratio": round(self.get_unhealthy_ratio(), 3),
            "healthy_ratio": round(self.get_healthy_ratio(), 3),
            "hr_slope": round(self.get_hr_trend(), 3),
            "spo2_slope": round(self.get_spo2_trend(), 3),
            "rr_slope": round(self.get_rr_trend(), 3),
            "recent_predictions": self.get_recent_predictions(),
        }

    def reset(self):
        """Clear the history buffer."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_trend_window(self):
        """Return the most recent TREND_WINDOW_SIZE observations as a list."""
        items = list(self._history)
        return items[-TREND_WINDOW_SIZE:]

    def _vital_slope(self, key: str) -> float:
        """
        Compute slope via linear regression over the trend window.

        Returns 0.0 when fewer than 2 data points with non-None values exist.
        """
        window = self._get_trend_window()
        values = [(i, obs[key]) for i, obs in enumerate(window) if obs.get(key) is not None]
        if len(values) < 2:
            return 0.0
        x = np.array([v[0] for v in values], dtype=np.float64)
        y = np.array([v[1] for v in values], dtype=np.float64)
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
