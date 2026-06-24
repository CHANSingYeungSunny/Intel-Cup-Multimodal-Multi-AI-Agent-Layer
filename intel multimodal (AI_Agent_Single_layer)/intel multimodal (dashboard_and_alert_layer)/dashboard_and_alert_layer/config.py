"""
Centralized configuration for Dashboard & Alerts Layer.
"""
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_OUTPUT_DIR = os.path.join(BASE_DIR, "fusion_layer", "outputs")
PREDICTIONS_CSV = os.path.join(FUSION_OUTPUT_DIR, "predictions.csv")
EXPERIMENT_CSV = os.path.join(FUSION_OUTPUT_DIR, "experiment_results_with_accuracy.csv")
ALERT_LOG_FILE = os.path.join(BASE_DIR, "alerts", "alert_log.txt")

# ---- Server ----
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", 5000))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

# ---- Simulation ----
DEFAULT_EXPERIMENT_ID = 1            # 3-class mode for alert testing
SIMULATION_INTERVAL_SECONDS = 2.0    # Seconds between health_update events
SIMULATION_SHUFFLE = False           # Cycle through CSV sequentially

# ---- Alert System ----
ALERT_RULES = {
    "unhealthy_prediction_threshold": 2,
    "consecutive_healthy_to_clear": 5,
    "min_alert_duration_seconds": 3.0,
    "telegram_cooldown_seconds": 30.0,
    "enable_telegram": True,
    "enable_console_alerts": True,
}

# ---- Telegram ----
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---- Feature Analysis ----
PCA_N_COMPONENTS = 3
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42

# ---- Label Mapping ----
LABEL_NAMES = {0: "Healthy", 1: "Sub-healthy", 2: "Unhealthy"}
LABEL_COLORS = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}   # green, yellow, red
LABEL_COLORS_HEX = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}

# ---- AI Agent Layer ----
AGENT_ENABLED = os.environ.get("AGENT_ENABLED", "True").lower() == "true"
AGENT_API_URL = os.environ.get("AGENT_API_URL", "")  # External FastAPI service
