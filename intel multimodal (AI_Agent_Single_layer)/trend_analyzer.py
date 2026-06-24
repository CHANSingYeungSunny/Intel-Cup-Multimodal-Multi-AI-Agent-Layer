"""
TrendAnalyzer — maintains a rolling history buffer of health observations
and computes trend direction, vital-sign slopes, and summary statistics.

Evolved from the existing ``agent_layer/trend_analyzer.py`` with added
PostgreSQL persistence for observations and trend snapshots.

Algorithm (preserved from the original):
    * Fraction of Unhealthy predictions ≥ ``DEGRADING_THRESHOLD`` → "degrading"
    * Fraction of Healthy predictions   ≥ ``IMPROVING_THRESHOLD`` → "improving"
    * Otherwise → "stable"
    * Vital-sign slopes via ``numpy.polyfit`` over the trend window.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
from sqlalchemy import select, func as sqlfunc
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    HISTORY_WINDOW_SIZE,
    TREND_WINDOW_SIZE,
    DEGRADING_THRESHOLD,
    IMPROVING_THRESHOLD,
)
from models import Observation, TrendSnapshot
from schemas import utc_now_iso

logger = logging.getLogger(__name__)


def _ensure_datetime(ts: Union[str, datetime, None]) -> Optional[datetime]:
    """Convert an ISO 8601 string to a timezone-aware datetime, or pass through."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


class TrendAnalyzer:
    """
    Rolling buffer of health observations with trend-detection logic.

    Parameters
    ----------
    window_size : int
        Maximum number of observations retained in the in-memory deque.
    """

    def __init__(self, window_size: int = HISTORY_WINDOW_SIZE):
        self._history: deque[dict] = deque(maxlen=window_size)
        self._window_size = window_size

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_observation(
        self,
        prediction: int,
        subject_id: str,
        hr_sim: Optional[float] = None,
        spo2_sim: Optional[float] = None,
        rr_sim: Optional[float] = None,
        feature_vector: Optional[list[float]] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Push a new observation into the rolling buffer.

        The deque's *maxlen* handles automatic eviction of old entries.

        Parameters
        ----------
        prediction : int
            Predicted class (0=Healthy, 1=Sub-healthy, 2=Unhealthy).
        subject_id : str
            Subject / patient identifier.
        hr_sim : float, optional
            Heart rate in bpm.
        spo2_sim : float, optional
            Oxygen saturation in %.
        rr_sim : float, optional
            RR interval in seconds.
        feature_vector : list[float], optional
            Raw feature vector (stored as JSON in the DB).
        timestamp : str, optional
            ISO 8601 UTC timestamp.

        Returns
        -------
        dict
            The observation dict that was added.
        """
        obs = {
            "prediction": int(prediction),
            "subject_id": str(subject_id),
            "hr": float(hr_sim) if hr_sim is not None else None,
            "spo2": float(spo2_sim) if spo2_sim is not None else None,
            "rr": float(rr_sim) if rr_sim is not None else None,
            "feature_vector": feature_vector,
            "timestamp": timestamp or utc_now_iso(),
        }
        self._history.append(obs)
        return obs

    async def persist_observation(
        self,
        db: AsyncSession,
        observation: dict,
    ) -> Observation:
        """
        Write an observation dict to the ``observations`` table.

        This is separated from ``add_observation`` so callers can choose to
        defer or skip persistence (e.g. in tests).
        """
        orm_obs = Observation(
            subject_id=observation["subject_id"],
            prediction=observation["prediction"],
            hr=observation.get("hr"),
            spo2=observation.get("spo2"),
            rr=observation.get("rr"),
            feature_vector=observation.get("feature_vector"),
            timestamp=_ensure_datetime(observation["timestamp"]),
        )
        db.add(orm_obs)
        await db.commit()
        return orm_obs

    # ------------------------------------------------------------------
    # Trend classification
    # ------------------------------------------------------------------

    def get_trend(self) -> str:
        """
        Classify the recent prediction trend.

        Returns one of ``"degrading"``, ``"improving"``, ``"stable"``.

        Uses the last *TREND_WINDOW_SIZE* observations.  If fewer than
        ``max(2, TREND_WINDOW_SIZE // 2)`` observations are available the
        trend is ``"stable"``.
        """
        window = self._get_trend_window()
        min_required = max(2, TREND_WINDOW_SIZE // 2)
        if len(window) < min_required:
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
    # Vital-sign slopes  (linear regression via numpy.polyfit)
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

    def get_recent_predictions(self, n: Optional[int] = None) -> list[int]:
        """Return the last *n* prediction values (0/1/2)."""
        if n is None:
            n = TREND_WINDOW_SIZE
        items = list(self._history)
        return [obs["prediction"] for obs in items[-n:]]

    def get_history(self) -> list[dict]:
        """Return a shallow copy of the full history buffer."""
        return list(self._history)

    def get_history_size(self) -> int:
        """Return the number of observations currently buffered."""
        return len(self._history)

    def get_summary(self) -> dict:
        """One-shot summary of all computed trend metrics."""
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
            "timestamp": utc_now_iso(),
        }

    def reset(self) -> None:
        """Clear the in-memory history buffer."""
        self._history.clear()

    # ------------------------------------------------------------------
    # PostgreSQL-backed queries
    # ------------------------------------------------------------------

    async def save_trend_snapshot(self, db: AsyncSession) -> TrendSnapshot:
        """
        Persist the current trend summary to the ``trend_snapshots`` table.

        Call this after each tick to build a queryable trend history.
        """
        summary = self.get_summary()
        snapshot = TrendSnapshot(
            trend=summary["trend"],
            history_size=summary["history_size"],
            trend_window_size=summary["trend_window_size"],
            unhealthy_ratio=summary["unhealthy_ratio"],
            healthy_ratio=summary["healthy_ratio"],
            hr_slope=summary["hr_slope"],
            spo2_slope=summary["spo2_slope"],
            rr_slope=summary["rr_slope"],
            recent_predictions=summary["recent_predictions"],
            timestamp=_ensure_datetime(summary["timestamp"]),
        )
        db.add(snapshot)
        await db.commit()
        return snapshot

    @staticmethod
    async def get_history_from_db(
        db: AsyncSession,
        limit: int = 100,
        subject_id: Optional[str] = None,
    ) -> list[Observation]:
        """
        Query the ``observations`` table for historical data.

        Parameters
        ----------
        db : AsyncSession
        limit : int
            Max rows to return (most recent first).
        subject_id : str, optional
            Filter to a specific subject.

        Returns
        -------
        list[Observation]
        """
        stmt = select(Observation).order_by(Observation.timestamp.desc())
        if subject_id:
            stmt = stmt.where(Observation.subject_id == subject_id)
        stmt = stmt.limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_trend_history(
        db: AsyncSession,
        limit: int = 100,
    ) -> list[TrendSnapshot]:
        """
        Query the ``trend_snapshots`` table for historical trend state.

        Parameters
        ----------
        db : AsyncSession
        limit : int
            Max rows to return (most recent first).

        Returns
        -------
        list[TrendSnapshot]
        """
        stmt = (
            select(TrendSnapshot)
            .order_by(TrendSnapshot.timestamp.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_observation_count(db: AsyncSession) -> int:
        """Return the total number of rows in ``observations``."""
        stmt = select(sqlfunc.count(Observation.id))
        result = await db.execute(stmt)
        return result.scalar() or 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_trend_window(self) -> list[dict]:
        """Return the most recent *TREND_WINDOW_SIZE* observations."""
        items = list(self._history)
        return items[-TREND_WINDOW_SIZE:]

    def _vital_slope(self, key: str) -> float:
        """
        Compute slope via linear regression over the trend window.

        Returns 0.0 when fewer than 2 data points with non-None values exist.
        """
        window = self._get_trend_window()
        values = [
            (i, obs[key])
            for i, obs in enumerate(window)
            if obs.get(key) is not None
        ]
        if len(values) < 2:
            return 0.0
        x = np.array([v[0] for v in values], dtype=np.float64)
        y = np.array([v[1] for v in values], dtype=np.float64)
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
