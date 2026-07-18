"""
HealthSimulator: background thread that cycles through predictions.csv
to simulate a real-time health-monitoring data stream.

On each tick it reads a prediction row, feeds it to the AlertManager,
and emits a 'health_update' SocketIO event to connected clients.
"""
import time
import threading
import random
from datetime import datetime, timezone
from config import SIMULATION_INTERVAL_SECONDS, SIMULATION_SHUFFLE, LABEL_NAMES


class HealthSimulator:
    """Cycles through predictions and emits SocketIO events."""

    def __init__(self, data_store, socketio, alert_manager=None, agent=None, agent_api_url=None):
        self._store = data_store
        self._socketio = socketio
        self._alert_manager = alert_manager
        self._agent = agent
        self._agent_api_url = agent_api_url  # external FastAPI service URL
        self._thread = None
        self._running = False
        self._paused = False
        self._cursor = 0
        self._interval = SIMULATION_INTERVAL_SECONDS
        self._speed_multiplier = 1.0
        self._predictions_processed = 0
        self._alerts_triggered = 0
        self._order = list(range(data_store.get_prediction_count()))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[HealthSimulator] Started (interval={self._interval}s, "
              f"shuffle={SIMULATION_SHUFFLE})")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def set_speed(self, multiplier):
        """Adjust simulation speed. 1.0 = normal, 2.0 = double speed."""
        self._speed_multiplier = max(0.1, min(5.0, float(multiplier)))

    def set_paused(self, paused):
        self._paused = bool(paused)

    def is_paused(self):
        return self._paused

    def get_stats(self):
        return {
            "predictions_processed": self._predictions_processed,
            "alerts_triggered": self._alerts_triggered,
            "cursor": self._cursor,
            "running": self._running,
            "paused": self._paused,
            "speed": self._speed_multiplier,
        }

    def increment_alert_count(self):
        self._alerts_triggered += 1

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------
    def _run(self):
        df = self._store.get_predictions_df()
        while self._running:
            if self._paused:
                time.sleep(0.5)
                continue

            # ── Demo Mode Override Check ──────────────────────────────
            # If a demo override is active, use its fixed prediction +
            # feature_vector instead of reading from the CSV DataFrame.
            override = None
            try:
                from dashboard.backend.routes.demo_routes import get_override
                override = get_override()
            except Exception:
                pass  # demo routes not registered → no override

            if override:
                prediction = override["prediction"]
                label = override["label"]
                fv = override["feature_vector"]
                subject_id = override.get("subject_id", "demo_subject")
                fname = override.get("filename", "demo://override")
                # Rotate the feature vector slightly each tick so the
                # agent sees varying data (more realistic demo).
                import numpy as _np
                noise = _np.random.randn(len(fv)).astype(float) * 0.05
                fv = [float(v + n) for v, n in zip(fv, noise)]
            else:
                # Get current prediction row from CSV
                idx = self._order[self._cursor % len(self._order)]
                row = df.iloc[idx]
                prediction = int(row["prediction"])
                label = int(row["label"])
                fv = row["feature_vector"]
                # Parse subject from filename
                import re
                fname = str(row["filename"])
                subj_match = re.search(r"subject(\d+)", fname)
                subject_id = f"subject{subj_match.group(1)}" if subj_match else "unknown"
            # ──────────────────────────────────────────────────────────

            # Build event payload
            payload = {
                "prediction": prediction,
                "prediction_name": LABEL_NAMES.get(prediction, "Unknown"),
                "label": label,
                "label_name": LABEL_NAMES.get(label, "Unknown"),
                "subject": subject_id,
                "filename": fname,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_active": False,
                "demo_override": override is not None,
            }

            # Feed to AlertManager
            if self._alert_manager:
                try:
                    alert_result = self._alert_manager.evaluate(
                        prediction=prediction,
                        subject_id=subject_id,
                        feature_vector=fv
                    )
                    if alert_result:
                        payload["alert_active"] = True
                        self._alerts_triggered += 1
                        self._socketio.emit("alert_triggered", alert_result)
                except Exception as e:
                    self._socketio.emit("alert_error", {"error": str(e)})

            # Feed to AI Agent
            if self._agent_api_url:
                # External FastAPI service
                try:
                    import requests
                    import json as _json
                    if isinstance(fv, str):
                        fv = _json.loads(fv)
                    resp = requests.post(
                        f"{self._agent_api_url}/tick",
                        json={
                            "prediction": prediction,
                            "subject_id": subject_id,
                            "feature_vector": fv,
                        },
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data:
                            self._socketio.emit("agent_advice", data)
                except Exception as e:
                    self._socketio.emit("agent_error", {"error": str(e)})
            elif self._agent:
                try:
                    if isinstance(fv, str):
                        import json as _json
                        fv = _json.loads(fv)
                    agent_advice = self._agent.process_tick(
                        prediction=prediction,
                        subject_id=subject_id,
                        feature_vector=fv,
                    )
                    if agent_advice:
                        self._socketio.emit("agent_advice", agent_advice)
                except Exception as e:
                    self._socketio.emit("agent_error", {"error": str(e)})

            # Emit health update
            self._socketio.emit("health_update", payload)
            self._predictions_processed += 1
            self._cursor += 1

            # Sleep for the interval, adjusted by speed
            sleep_time = self._interval / self._speed_multiplier
            time.sleep(sleep_time)
