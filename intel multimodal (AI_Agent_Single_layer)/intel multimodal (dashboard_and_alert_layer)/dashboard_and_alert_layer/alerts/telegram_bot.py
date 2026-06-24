"""
Telegram Bot for sending push notifications when Unhealthy state is detected.

Uses python-telegram-bot library. Operates in "dry-run" mode when
TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are not configured.
"""
import os
import json
from datetime import datetime, timezone
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ALERT_LOG_FILE


class TelegramAlertBot:
    """Sends alert and health-report messages via Telegram Bot API."""

    def __init__(self):
        self._token = TELEGRAM_BOT_TOKEN
        self._chat_id = TELEGRAM_CHAT_ID
        self._configured = bool(self._token and self._chat_id)
        self._last_alert_time = None

        if self._configured:
            print(f"[TelegramBot] Configured — will send real messages to chat {self._chat_id}")
        else:
            print("[TelegramBot] DRY-RUN mode — set TELEGRAM_BOT_TOKEN and "
                  "TELEGRAM_CHAT_ID to enable real messages")

    def is_configured(self):
        return self._configured

    def send_alert(self, prediction, subject_id, feature_summary=None):
        """
        Send an urgent alert when Unhealthy state is detected.

        Args:
            prediction: int (2 = Unhealthy)
            subject_id: str
            feature_summary: dict with optional extra context

        Returns:
            dict with delivery status
        """
        now = datetime.now(timezone.utc)
        self._last_alert_time = now
        ts = now.strftime("%Y-%m-%d %H:%M:%S UTC")

        message = (
            "🚨 HEALTH ALERT — Unhealthy State Detected\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 Subject: {subject_id}\n"
            f"🔴 State: UNHEALTHY (Class 2)\n"
            f"⏰ Time: {ts}\n"
        )
        if feature_summary:
            message += f"📊 Context: {json.dumps(feature_summary)}\n"
        message += (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🔗 Dashboard: http://localhost:5000"
        )

        if self._configured:
            return self._send_via_api(message)
        else:
            return self._dry_run(message, "ALERT")

    def send_clear(self):
        """Send notification that alert has been cleared."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        message = (
            "✅ ALERT CLEARED\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ Time: {ts}\n"
            "Status: System returned to normal\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        if self._configured:
            return self._send_via_api(message)
        else:
            return self._dry_run(message, "CLEAR")

    def send_health_report(self, health_state):
        """Send a periodic health summary report."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        counts = health_state.get("prediction_counts", {})
        message = (
            "📊 HEALTH REPORT\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ Time: {ts}\n"
            f"💚 Healthy:      {counts.get('Healthy', 0)}\n"
            f"💛 Sub-healthy:  {counts.get('Sub-healthy', 0)}\n"
            f"❤️ Unhealthy:    {counts.get('Unhealthy', 0)}\n"
            f"📋 Total Samples: {health_state.get('total_samples', 0)}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        if self._configured:
            return self._send_via_api(message)
        else:
            return self._dry_run(message, "REPORT")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _send_via_api(self, message):
        """Send message via Telegram Bot API using raw HTTP request."""
        import urllib.request
        import urllib.parse

        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
            ok = result.get("ok", False)
            status = "sent" if ok else f"failed: {result.get('description', 'unknown')}"
        except Exception as e:
            status = f"error: {e}"

        self._log(status, message)
        return {"delivered": ok if isinstance(ok, bool) else False, "status": status}

    def _dry_run(self, message, tag):
        """Log message to console and file instead of sending."""
        safe_message = message.encode("ascii", errors="replace").decode("ascii")
        print(f"\n[TelegramBot DRY-RUN] {tag}")
        print("-" * 40)
        try:
            print(message)
        except UnicodeEncodeError:
            print(safe_message)
        print("-" * 40)
        self._log(f"DRY-RUN {tag}", message)
        return {"delivered": False, "status": "dry-run", "message": message}

    def _log(self, status, message):
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            with open(ALERT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] TELEGRAM {status}:\n{message}\n\n")
        except Exception:
            pass
