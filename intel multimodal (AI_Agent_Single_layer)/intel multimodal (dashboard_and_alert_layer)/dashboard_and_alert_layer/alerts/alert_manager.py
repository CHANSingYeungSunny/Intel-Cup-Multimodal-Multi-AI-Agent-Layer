"""
AlertManager: central orchestrator for the alert system.

Implements a state machine:
    NORMAL --(prediction==2)--> ALERTING
    ALERTING --(5x healthy)--> NORMAL

Integrates LED simulator, Buzzer simulator, and Telegram Bot.
"""
import time
import json
from datetime import datetime, timezone
from alerts.alert_rules import (
    UNHEALTHY_PREDICTION_THRESHOLD,
    CONSECUTIVE_HEALTHY_TO_CLEAR,
    MIN_ALERT_DURATION_SECONDS,
    TELEGRAM_COOLDOWN_SECONDS,
)
from alerts.led_simulator import LEDSimulator
from alerts.buzzer_simulator import BuzzerSimulator
from alerts.telegram_bot import TelegramAlertBot


class AlertEvent:
    """Represents an alert event."""
    def __init__(self, event_type, prediction=None, subject_id=None, message="",
                 level="critical"):
        self.type = event_type          # "triggered" or "cleared"
        self.prediction = prediction
        self.subject_id = subject_id
        self.message = message
        self.level = level
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self):
        return {
            "type": self.type,
            "level": self.level,
            "prediction": self.prediction,
            "subject": self.subject_id,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class AlertManager:
    """Orchestrates LED, Buzzer, and Telegram alerts with debounce logic."""

    def __init__(self):
        self._led = LEDSimulator()
        self._buzzer = BuzzerSimulator()
        self._telegram = TelegramAlertBot()
        self._callbacks = []

        # State
        self._state = "NORMAL"           # NORMAL | ALERTING
        self._healthy_counter = 0
        self._alert_start_time = None
        self._last_telegram_time = None
        self._alert_history = []          # list of AlertEvent dicts (max 50)

        print("[AlertManager] Initialized (state=NORMAL)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, prediction, subject_id, feature_vector=None):
        """
        Evaluate a new prediction and transition state if needed.

        Args:
            prediction: int (0=Healthy, 1=Sub-healthy, 2=Unhealthy)
            subject_id: str
            feature_vector: str (JSON) or list — used for context

        Returns:
            AlertEvent | None — non-None when state changes or alert fires
        """
        fv_summary = None
        if feature_vector:
            try:
                if isinstance(feature_vector, str):
                    vec = json.loads(feature_vector)
                else:
                    vec = feature_vector
                fv_summary = {
                    "dim": len(vec),
                    "mean": round(sum(vec) / len(vec), 4),
                    "max": round(max(vec), 4),
                    "min": round(min(vec), 4),
                }
            except Exception:
                pass

        if prediction == UNHEALTHY_PREDICTION_THRESHOLD:
            return self._on_unhealthy(subject_id, prediction, fv_summary)
        else:
            return self._on_healthy(subject_id, prediction)

    def clear_alerts(self):
        """Force-clear all alerts."""
        self._state = "NORMAL"
        self._healthy_counter = 0
        self._alert_start_time = None
        self._led.turn_off()
        self._buzzer.stop_beep()
        return AlertEvent("cleared", message="Alerts manually cleared")

    def trigger_test_alert(self):
        """Manually trigger a test alert for verification."""
        return self._fire_alert("test_subject", 2, {"test": True})

    def get_status(self):
        """Return current alert system status."""
        return {
            "state": self._state,
            "active": self._state == "ALERTING",
            "healthy_counter": self._healthy_counter,
            "led": self._led.get_state(),
            "buzzer": self._buzzer.get_state(),
            "telegram_configured": self._telegram.is_configured(),
            "alert_history": self._alert_history[-20:],
        }

    def register_callback(self, callback):
        """Register a callback to be invoked on alert state changes."""
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _on_unhealthy(self, subject_id, prediction, fv_summary):
        self._healthy_counter = 0

        if self._state == "NORMAL":
            return self._enter_alerting(subject_id, prediction, fv_summary)
        elif self._state == "ALERTING":
            # Already alerting — could re-fire if enough time passed
            pass
        return None

    def _on_healthy(self, subject_id, prediction):
        if self._state == "ALERTING":
            self._healthy_counter += 1
            if self._healthy_counter >= CONSECUTIVE_HEALTHY_TO_CLEAR:
                return self._exit_alerting()
        else:
            self._healthy_counter = 0
        return None

    def _enter_alerting(self, subject_id, prediction, fv_summary):
        """Transition NORMAL -> ALERTING."""
        self._state = "ALERTING"
        self._alert_start_time = time.time()
        self._healthy_counter = 0
        return self._fire_alert(subject_id, prediction, fv_summary)

    def _exit_alerting(self):
        """Transition ALERTING -> NORMAL."""
        # Enforce minimum alert duration
        if self._alert_start_time:
            elapsed = time.time() - self._alert_start_time
            if elapsed < MIN_ALERT_DURATION_SECONDS:
                return None

        self._state = "NORMAL"
        self._healthy_counter = 0
        self._alert_start_time = None
        self._led.turn_off()
        self._buzzer.stop_beep()

        event = AlertEvent("cleared", message="Alert cleared — system returned to healthy")
        self._add_to_history(event)
        self._invoke_callbacks(event)
        return event.to_dict()

    def _fire_alert(self, subject_id, prediction, fv_summary):
        """Activate all alert channels."""
        # LED
        self._led.turn_on(color="red", blinking=True, pattern="rapid")
        # Buzzer
        self._buzzer.start_beep(pattern="continuous", frequency=1000)

        # Telegram (with cooldown)
        now = time.time()
        if self._last_telegram_time is None or \
           (now - self._last_telegram_time) >= TELEGRAM_COOLDOWN_SECONDS:
            self._telegram.send_alert(prediction, subject_id, fv_summary)
            self._last_telegram_time = now

        message = f"Unhealthy state detected for {subject_id}"
        event = AlertEvent(
            "triggered",
            prediction=prediction,
            subject_id=subject_id,
            message=message,
            level="critical",
        )
        self._add_to_history(event)
        self._invoke_callbacks(event)
        return event.to_dict()

    def _add_to_history(self, event):
        self._alert_history.append(event.to_dict())
        if len(self._alert_history) > 50:
            self._alert_history = self._alert_history[-50:]

    def _invoke_callbacks(self, event):
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass
