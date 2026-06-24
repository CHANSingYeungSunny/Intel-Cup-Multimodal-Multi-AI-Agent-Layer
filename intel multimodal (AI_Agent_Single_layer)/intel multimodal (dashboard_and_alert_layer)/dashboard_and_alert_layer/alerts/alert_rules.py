"""
Alert rule definitions and threshold configurations.
"""
from config import ALERT_RULES

# Re-export for convenience
UNHEALTHY_PREDICTION_THRESHOLD = ALERT_RULES["unhealthy_prediction_threshold"]
CONSECUTIVE_HEALTHY_TO_CLEAR = ALERT_RULES["consecutive_healthy_to_clear"]
MIN_ALERT_DURATION_SECONDS = ALERT_RULES["min_alert_duration_seconds"]
TELEGRAM_COOLDOWN_SECONDS = ALERT_RULES["telegram_cooldown_seconds"]
ENABLE_TELEGRAM = ALERT_RULES["enable_telegram"]
ENABLE_CONSOLE_ALERTS = ALERT_RULES["enable_console_alerts"]

# Alert severity levels
SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"
