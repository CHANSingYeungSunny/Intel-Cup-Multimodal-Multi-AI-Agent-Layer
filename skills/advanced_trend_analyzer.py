"""
Advanced Trend Analyzer Skill — multi-timescale trend classification
with linear+exponential smoothing forecasting.

Extends the Single-layer TrendAnalyzer by tracking trends at multiple
window sizes simultaneously and providing short-term forecasts for HR,
SpO₂, and RR interval.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Trend classification thresholds
_DEGRADING_THRESHOLD = 0.3
_IMPROVING_THRESHOLD = 0.7


class AdvancedTrendAnalyzer:
    """
    Multi-timescale trend analyzer with forecasting.

    Maintains separate rolling buffers at each configured window size.
    On each update, classifies the trend (degrading / improving / stable)
    at every timescale.  Also provides linear + exponential smoothing
    forecasts for vital-sign metrics.

    Parameters
    ----------
    window_sizes : list[int]
        Window sizes for multi-scale analysis (default [5, 10, 30, 60]).
    forecast_horizon : int
        Number of future steps to predict (default 5).
    degrading_threshold : float
        Unhealthy ratio below which trend is "degrading" (default 0.3).
    improving_threshold : float
        Healthy ratio above which trend is "improving" (default 0.7).
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        forecast_horizon: int = 5,
        degrading_threshold: float = _DEGRADING_THRESHOLD,
        improving_threshold: float = _IMPROVING_THRESHOLD,
    ):
        self._window_sizes = window_sizes or [5, 10, 30, 60]
        self._forecast_horizon = forecast_horizon
        self._degrading_threshold = degrading_threshold
        self._improving_threshold = improving_threshold

        # Multi-scale observation buffers
        self._prediction_buffers: dict[int, deque[int]] = {
            w: deque(maxlen=w) for w in self._window_sizes
        }
        # Vital-sign buffers (use the largest window for forecasting)
        self._max_window = max(self._window_sizes) if self._window_sizes else 60
        self._hr_buffer: deque[float] = deque(maxlen=self._max_window)
        self._spo2_buffer: deque[float] = deque(maxlen=self._max_window)
        self._rr_buffer: deque[float] = deque(maxlen=self._max_window)

        # Latest results cache
        self._latest_multi_scale: dict = {}
        self._latest_forecast: dict = {}
        self._latest_cross_scale_insight: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, observation_dict: dict) -> None:
        """
        Ingest a new observation.

        *observation_dict* must contain keys:
        ``prediction`` (int), ``hr_sim`` (float), ``spo2_sim`` (float),
        ``rr_sim`` (float).
        """
        prediction = observation_dict.get("prediction", 0)
        hr = observation_dict.get("hr_sim", 80.0)
        spo2 = observation_dict.get("spo2_sim", 97.0)
        rr = observation_dict.get("rr_sim", 0.85)

        # Update prediction buffers at each window size
        for buf in self._prediction_buffers.values():
            buf.append(prediction)

        # Update vital-sign buffers
        self._hr_buffer.append(float(hr))
        self._spo2_buffer.append(float(spo2))
        self._rr_buffer.append(float(rr))

        # Recompute classifications
        self._latest_multi_scale = self._classify_all_windows()

        # Recompute forecast
        self._latest_forecast = self._compute_forecasts()

        # Cross-scale insight
        self._latest_cross_scale_insight = self._detect_cross_scale_divergence()

    def get_multi_scale_trends(self) -> dict:
        """
        Return trend classification per window size.

        Returns dict mapping window size (as str) to trend direction,
        e.g. ``{"5": "degrading", "10": "stable", "30": "improving"}``.
        """
        return self._latest_multi_scale

    def forecast(
        self,
        metric: str = "hr",
        horizon: int | None = None,
    ) -> dict:
        """
        Forecast a metric for *horizon* steps ahead.

        Uses linear regression on the full vital-sign buffer combined
        with simple exponential smoothing (alpha=0.3).  Returns the
        average of both methods.

        Returns dict with keys: ``horizon``, ``predicted_values``,
        ``confidence_interval``.
        """
        if horizon is None:
            horizon = self._forecast_horizon

        buffers = {"hr": self._hr_buffer, "spo2": self._spo2_buffer, "rr": self._rr_buffer}
        buf = buffers.get(metric, self._hr_buffer)

        if len(buf) < 3:
            return {
                "horizon": horizon,
                "predicted_values": [],
                "confidence_interval": (0.0, 0.0),
            }

        arr = np.array(list(buf), dtype=np.float64)
        n = len(arr)
        x = np.arange(n, dtype=np.float64)

        # Linear regression
        slope, intercept = np.polyfit(x, arr, 1)
        x_future = np.arange(n, n + horizon, dtype=np.float64)
        lin_pred = intercept + slope * x_future

        # Exponential smoothing (Holt's linear trend)
        alpha = 0.3
        beta = 0.1
        level = arr[0]
        trend = arr[1] - arr[0] if n > 1 else 0.0
        for val in arr[1:]:
            new_level = alpha * val + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level, trend = new_level, new_trend

        es_pred = np.array(
            [level + trend * (i + 1) for i in range(horizon)],
            dtype=np.float64,
        )

        # Average predictions
        predicted = ((lin_pred + es_pred) / 2.0).tolist()

        # Confidence interval from regression standard error
        residuals = arr - (intercept + slope * x)
        se = np.std(residuals, ddof=2) if len(residuals) > 2 else 0.0
        ci_factor = 1.96 * se
        ci_lower = float(predicted[-1] - ci_factor) if predicted else 0.0
        ci_upper = float(predicted[-1] + ci_factor) if predicted else 0.0

        return {
            "horizon": horizon,
            "predicted_values": [round(v, 3) for v in predicted],
            "confidence_interval": (round(ci_lower, 3), round(ci_upper, 3)),
        }

    def get_cross_scale_insight(self) -> str:
        """
        Return a human-readable insight about short-term vs long-term
        trend divergences.
        """
        return self._latest_cross_scale_insight

    def get_summary(self) -> dict:
        """One-shot summary: multi-scale trends + forecasts + insight."""
        return {
            "multi_scale_trends": self._latest_multi_scale,
            "forecast": self._latest_forecast,
            "cross_scale_insight": self._latest_cross_scale_insight,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def reset(self) -> None:
        """Clear all buffers."""
        for buf in self._prediction_buffers.values():
            buf.clear()
        self._hr_buffer.clear()
        self._spo2_buffer.clear()
        self._rr_buffer.clear()
        self._latest_multi_scale = {}
        self._latest_forecast = {}
        self._latest_cross_scale_insight = ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify_all_windows(self) -> dict:
        """Classify trend at each window size."""
        result = {}
        for w, buf in self._prediction_buffers.items():
            if len(buf) < max(3, w // 2):
                result[str(w)] = "insufficient_data"
                continue

            arr = np.array(list(buf), dtype=np.float64)
            healthy_ratio = float(np.sum(arr == 0) / len(arr))
            unhealthy_ratio = float(np.sum(arr == 2) / len(arr))

            if unhealthy_ratio >= self._degrading_threshold:
                trend = "degrading"
            elif healthy_ratio >= self._improving_threshold:
                trend = "improving"
            else:
                trend = "stable"

            result[str(w)] = trend

        return result

    def _compute_forecasts(self) -> dict:
        """Compute forecasts for all three vital-sign metrics."""
        return {
            "hr": self.forecast("hr"),
            "spo2": self.forecast("spo2"),
            "rr": self.forecast("rr"),
        }

    def _detect_cross_scale_divergence(self) -> str:
        """Detect meaningful divergences between short- and long-term trends."""
        if not self._latest_multi_scale:
            return ""

        # Find smallest and largest window with enough data
        small_key = None
        large_key = None
        for w in sorted(self._window_sizes):
            key = str(w)
            if (
                key in self._latest_multi_scale
                and self._latest_multi_scale[key] != "insufficient_data"
            ):
                if small_key is None:
                    small_key = key
                large_key = key

        if small_key is None or large_key is None or small_key == large_key:
            return ""

        short_trend = self._latest_multi_scale[small_key]
        long_trend = self._latest_multi_scale[large_key]

        if short_trend == "degrading" and long_trend == "stable":
            return (
                "Early warning: short-term trend is degrading while "
                "long-term remains stable. Increased monitoring advised."
            )
        elif short_trend == "degrading" and long_trend == "improving":
            return (
                "Short-term degradation detected against a long-term "
                "improving backdrop. Monitor closely — may be transient."
            )
        elif short_trend == "stable" and long_trend == "degrading":
            return (
                "Chronic concern: long-term degrading trend persists "
                "despite recent short-term stability."
            )
        elif short_trend == "improving" and long_trend == "degrading":
            return (
                "Recovery signal: short-term improvement detected "
                "against a longer-term degraded baseline."
            )
        elif short_trend == "improving" and long_trend == "stable":
            return (
                "Positive: short-term improvement trend observed. "
                "Continue current regimen."
            )

        return ""
