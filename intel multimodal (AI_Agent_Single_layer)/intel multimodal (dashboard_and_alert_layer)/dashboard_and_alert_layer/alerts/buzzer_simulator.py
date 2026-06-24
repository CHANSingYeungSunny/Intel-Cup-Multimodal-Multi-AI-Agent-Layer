"""
Buzzer Simulator: simulates alarm sound via console output and log file.

Optionally uses winsound.Beep() on Windows for actual audio output.
"""
import os
import sys
import time
from datetime import datetime, timezone
from config import ALERT_LOG_FILE


class BuzzerSimulator:
    """Simulates a buzzer alarm with console patterns."""

    def __init__(self):
        self._active = False
        self._frequency = 1000    # Hz (nominal)
        self._pattern = "off"
        self._log_file = ALERT_LOG_FILE
        os.makedirs(os.path.dirname(self._log_file), exist_ok=True)

    def start_beep(self, pattern="continuous", frequency=1000):
        """Start simulated beeping."""
        self._active = True
        self._frequency = frequency
        self._pattern = pattern
        self._log("START", f"frequency={frequency}Hz pattern={pattern}")
        self._console_beep()

        # On Windows, attempt real beep
        if sys.platform == "win32":
            try:
                import winsound
                winsound.Beep(frequency, 300)
            except Exception:
                pass
        else:
            print("\a", end="", flush=True)  # BEL character

    def stop_beep(self):
        """Stop simulated beeping."""
        if self._active:
            self._log("STOP", "Buzzer silenced")
        self._active = False
        self._pattern = "off"
        print("[BUZZER] OFF — Alarm Silenced")

    def get_state(self):
        """Return current buzzer state for frontend display."""
        return {
            "active": self._active,
            "frequency": self._frequency,
            "pattern": self._pattern,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _console_beep(self):
        if self._pattern == "continuous":
            print("[BUZZER] █████ BEEP BEEP BEEP █████")
        elif self._pattern == "intermittent":
            print("[BUZZER] ████ BEEP --- BEEP --- BEEP ████")
        elif self._pattern == "sos":
            print("[BUZZER] ···---··· SOS ···---···")
        else:
            print(f"[BUZZER] ACTIVE — {self._pattern}")

    def _log(self, action, detail):
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] BUZZER {action}: {detail}\n")
        except Exception:
            pass
