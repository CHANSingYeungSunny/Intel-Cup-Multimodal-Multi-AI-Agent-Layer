"""Tests for AlertManager state machine and alert components."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alerts.alert_manager import AlertManager, AlertEvent
from alerts.led_simulator import LEDSimulator
from alerts.buzzer_simulator import BuzzerSimulator
from alerts.telegram_bot import TelegramAlertBot


class TestAlertManager:
    def setup_method(self):
        self.mgr = AlertManager()

    def test_initial_state(self):
        status = self.mgr.get_status()
        assert status["state"] == "NORMAL"
        assert status["active"] is False

    def test_unhealthy_triggers_alert(self):
        result = self.mgr.evaluate(prediction=2, subject_id="test_subject")
        assert result is not None
        assert result["type"] == "triggered"
        assert result["level"] == "critical"
        assert self.mgr.get_status()["state"] == "ALERTING"

    def test_healthy_does_not_trigger(self):
        result = self.mgr.evaluate(prediction=0, subject_id="test_subject")
        assert result is None
        assert self.mgr.get_status()["state"] == "NORMAL"

    def test_alert_clears_after_consecutive_healthy(self):
        # Trigger alert
        self.mgr.evaluate(prediction=2, subject_id="test")
        assert self.mgr.get_status()["state"] == "ALERTING"
        # Bypass minimum alert duration for testing
        self.mgr._alert_start_time = 0  # simulate alert that started long ago

        # Send 5 healthy predictions to clear
        result = None
        for i in range(5):
            result = self.mgr.evaluate(prediction=0, subject_id="test")
        assert result is not None
        assert result["type"] == "cleared"
        assert self.mgr.get_status()["state"] == "NORMAL"

    def test_healthy_during_alerting_resets_counter(self):
        # Trigger
        self.mgr.evaluate(prediction=2, subject_id="test")
        # 3 healthy
        for _ in range(3):
            self.mgr.evaluate(prediction=0, subject_id="test")
        # Another unhealthy resets counter
        self.mgr.evaluate(prediction=2, subject_id="test")
        # Need 5 more healthy
        for _ in range(4):
            self.mgr.evaluate(prediction=0, subject_id="test")
        assert self.mgr.get_status()["state"] == "ALERTING"

    def test_test_alert(self):
        event = self.mgr.trigger_test_alert()
        assert event is not None
        assert event["type"] == "triggered"
        assert event["subject"] == "test_subject"

    def test_clear_alerts(self):
        self.mgr.evaluate(prediction=2, subject_id="test")
        assert self.mgr.get_status()["state"] == "ALERTING"
        event = self.mgr.clear_alerts()
        assert event.type == "cleared"
        assert self.mgr.get_status()["state"] == "NORMAL"

    def test_alert_history(self):
        self.mgr.evaluate(prediction=2, subject_id="test")
        status = self.mgr.get_status()
        assert len(status["alert_history"]) >= 1


class TestLEDSimulator:
    def test_initial_state(self):
        led = LEDSimulator()
        assert led.get_state()["active"] is False

    def test_turn_on_off(self):
        led = LEDSimulator()
        led.turn_on(color="red", blinking=True, pattern="rapid")
        assert led.get_state()["active"] is True
        assert led.get_state()["color"] == "red"

        led.turn_off()
        assert led.get_state()["active"] is False


class TestBuzzerSimulator:
    def test_initial_state(self):
        buzzer = BuzzerSimulator()
        assert buzzer.get_state()["active"] is False

    def test_start_stop(self):
        buzzer = BuzzerSimulator()
        buzzer.start_beep(pattern="continuous", frequency=1000)
        assert buzzer.get_state()["active"] is True

        buzzer.stop_beep()
        assert buzzer.get_state()["active"] is False


class TestTelegramBot:
    def test_dry_run_mode(self):
        bot = TelegramAlertBot()
        assert not bot.is_configured()

    def test_send_alert_dry_run(self):
        bot = TelegramAlertBot()
        result = bot.send_alert(2, "test_subject")
        assert result["delivered"] is False
        assert result["status"] == "dry-run"
        assert "message" in result

    def test_send_clear_dry_run(self):
        bot = TelegramAlertBot()
        result = bot.send_clear()
        assert result["delivered"] is False

    def test_send_health_report_dry_run(self):
        bot = TelegramAlertBot()
        result = bot.send_health_report({
            "total_samples": 92,
            "prediction_counts": {"Healthy": 50, "Sub-healthy": 30, "Unhealthy": 12},
        })
        assert result["delivered"] is False
