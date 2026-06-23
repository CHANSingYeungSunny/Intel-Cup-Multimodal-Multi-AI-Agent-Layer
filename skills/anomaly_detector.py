"""
Anomaly Detection Skill — rolling z-score based health anomaly detection.

Detects point anomalies (single high-z-score observations) and persistence
anomalies (consecutive same-class predictions deviating from recent pattern)
across HR, SpO₂, RR interval, and prediction metrics.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default metrics tracked
DEFAULT_METRICS = ["hr", "spo2", "rr", "prediction"]

# z-score bands for severity classification
_WARNING_THRESHOLD = 2.5
_CRITICAL_THRESHOLD = 3.5


class AnomalyDetector:
    """
    Statistical anomaly detector using rolling z-scores.

    Maintains a sliding window per metric.  On each ``update()``,
    computes the z-score of the new value against the window mean/std.
    Flags the observation when |z_score| exceeds the threshold.

    Also tracks consecutive predictions for persistence anomaly
    detection (e.g. 3+ Unhealthy in a row when recent history was Healthy).

    Parameters
    ----------
    window_size : int
        Number of observations in the rolling window (default 30).
    zscore_threshold : float
        Z-score beyond which a value is flagged (default 2.5).
    persistence_count : int
        Consecutive same-class predictions needed for persistence flag
        (default 3).
    """

    def __init__(
        self,
        window_size: int = 30,
        zscore_threshold: float = 2.5,
        persistence_count: int = 3,
    ):
        self._window_size = max(5, window_size)
        self._zscore_threshold = zscore_threshold
        self._persistence_count = persistence_count

        # Rolling deques per metric
        self._buffers: dict[str, deque[float]] = {
            m: deque(maxlen=self._window_size) for m in DEFAULT_METRICS
        }
        # Prediction history (longer window for persistence detection)
        self._prediction_history: deque[int] = deque(maxlen=self._window_size * 2)

        # Cached stats
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        hr: float,
        spo2: float,
        rr: float,
        prediction: int,
        subject_id: str = "",
    ) -> list[dict]:
        """
        Process a new observation and return any detected anomalies.

        Returns a list of anomaly dicts (empty if everything is normal).
        """
        anomalies: list[dict] = []
        now = datetime.now(timezone.utc)

        values = {"hr": hr, "spo2": spo2, "rr": rr, "prediction": float(prediction)}

        for metric, val in values.items():
            self._buffers[metric].append(val)

            # Need enough data for meaningful stats
            if len(self._buffers[metric]) < 5:
                continue

            arr = np.array(list(self._buffers[metric]), dtype=np.float64)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1))  # sample std

            self._means[metric] = mean
            self._stds[metric] = std

            if std < 1e-9:
                continue  # no variance → skip

            z = (val - mean) / std

            if abs(z) >= self._zscore_threshold:
                severity = "critical" if abs(z) >= _CRITICAL_THRESHOLD else "warning"
                anomalies.append({
                    "metric_name": metric,
                    "z_score": round(z, 3),
                    "severity": severity,
                    "observed_value": val,
                    "expected_value": round(mean, 3),
                    "subject_id": subject_id,
                    "timestamp": now.isoformat(),
                })

        # Persistence anomaly check
        self._prediction_history.append(prediction)
        persist_anomaly = self._check_persistence(subject_id, now)
        if persist_anomaly:
            anomalies.append(persist_anomaly)

        return anomalies

    def detect_on_stream(self, observations: list[dict]) -> list[dict]:
        """
        Batch detection over a list of observation dicts.

        Each dict must have keys: hr, spo2, rr, prediction, subject_id (optional).
        """
        all_anomalies: list[dict] = []
        for obs in observations:
            anomalies = self.update(
                hr=obs.get("hr", 80.0),
                spo2=obs.get("spo2", 97.0),
                rr=obs.get("rr", 0.85),
                prediction=obs.get("prediction", 0),
                subject_id=obs.get("subject_id", ""),
            )
            all_anomalies.extend(anomalies)
        return all_anomalies

    def get_statistics(self) -> dict:
        """Return current mean and std for each metric."""
        return {
            m: {
                "mean": self._means.get(m, 0.0),
                "std": self._stds.get(m, 0.0),
                "buffer_size": len(self._buffers[m]),
            }
            for m in DEFAULT_METRICS
        }

    def reset(self) -> None:
        """Clear all rolling windows and cached stats."""
        for buf in self._buffers.values():
            buf.clear()
        self._prediction_history.clear()
        self._means.clear()
        self._stds.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_persistence(
        self, subject_id: str, now: datetime
    ) -> Optional[dict]:
        """
        Detect persistence anomalies.

        Flags when the last N predictions are all the same class AND that
        class differs from the majority of the preceding window.
        """
        if len(self._prediction_history) < self._persistence_count + 3:
            return None

        recent = list(self._prediction_history)[-self._persistence_count:]
        if len(set(recent)) != 1:
            return None  # not all the same

        current_class = recent[0]
        older = list(self._prediction_history)[
            : -self._persistence_count
        ]
        if not older:
            return None

        older_mode = int(max(set(older), key=older.count))

        if current_class != older_mode:
            # Class shift detected — flag if moving to worse state
            if current_class > older_mode:
                return {
                    "metric_name": "prediction",
                    "z_score": 0.0,
                    "severity": "warning",
                    "observed_value": float(current_class),
                    "expected_value": float(older_mode),
                    "subject_id": subject_id,
                    "timestamp": now.isoformat(),
                }

        return None
