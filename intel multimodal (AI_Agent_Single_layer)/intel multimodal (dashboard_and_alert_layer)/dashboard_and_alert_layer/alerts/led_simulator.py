"""
LED Simulator: simulates GPIO LED behavior via console output and log file.
"""
import os
import time
from datetime import datetime, timezone
from config import ALERT_LOG_FILE


class LEDSimulator:
    """Simulates an LED indicator with ANSI console colors and log output."""

    # ANSI escape codes
    RED = "\033[41m\033[97m"       # red background, white text
    YELLOW = "\033[43m\033[30m"    # yellow background, black text
    GREEN = "\033[42m\033[30m"     # green background, black text
    RESET = "\033[0m"
    BLINK = "\033[5m"              # blink (not always supported)

    def __init__(self):
        self._active = False
        self._color = "red"
        self._blinking = False
        self._pattern = "off"
        self._log_file = ALERT_LOG_FILE
        os.makedirs(os.path.dirname(self._log_file), exist_ok=True)

    def turn_on(self, color="red", blinking=True, pattern="rapid"):
        """Activate the simulated LED."""
        self._active = True
        self._color = color
        self._blinking = blinking
        self._pattern = pattern
        self._log("ON", f"color={color} blinking={blinking} pattern={pattern}")
        self._render_console()

    def turn_off(self):
        """Deactivate the simulated LED."""
        if self._active:
            self._log("OFF", "LED deactivated")
        self._active = False
        self._blinking = False
        self._pattern = "off"
        print(f"{self.GREEN}[LED] OFF — System Normal{self.RESET}")

    def get_state(self):
        """Return current LED state for frontend display."""
        return {
            "active": self._active,
            "color": self._color,
            "blinking": self._blinking,
            "pattern": self._pattern,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _render_console(self):
        if self._color == "red":
            color_code = self.RED
        elif self._color == "yellow":
            color_code = self.YELLOW
        else:
            color_code = self.GREEN

        blink = self.BLINK if self._blinking else ""
        print(f"{blink}{color_code}[LED] ALERT — {self._color.upper()} {self._pattern}{self.RESET}")

    def _log(self, action, detail):
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] LED {action}: {detail}\n")
        except Exception:
            pass
