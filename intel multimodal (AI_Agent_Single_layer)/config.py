"""
Central configuration for the Single AI Agent Layer.

All tunable parameters are loaded from environment variables with sensible
defaults matching the existing ``agent_config.py`` from the Dashboard layer.
"""

import os

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_layer",
)

# ---------------------------------------------------------------------------
# History & Trend parameters
# ---------------------------------------------------------------------------
HISTORY_WINDOW_SIZE: int = int(
    os.environ.get("HISTORY_WINDOW_SIZE", "20")
)
TREND_WINDOW_SIZE: int = int(
    os.environ.get("TREND_WINDOW_SIZE", "10")
)
DEGRADING_THRESHOLD: float = float(
    os.environ.get("DEGRADING_THRESHOLD", "0.3")
)
IMPROVING_THRESHOLD: float = float(
    os.environ.get("IMPROVING_THRESHOLD", "0.7")
)

# ---------------------------------------------------------------------------
# Advice history
# ---------------------------------------------------------------------------
ADVICE_HISTORY_MAX: int = int(
    os.environ.get("ADVICE_HISTORY_MAX", "50")
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
APP_TITLE: str = os.environ.get("APP_TITLE", "Single AI Agent Layer")
APP_VERSION: str = os.environ.get("APP_VERSION", "1.0.0")
API_V1_PREFIX: str = os.environ.get("API_V1_PREFIX", "/api/v1")
CORS_ORIGINS: list[str] = os.environ.get("CORS_ORIGINS", "*").split(",")

# ---------------------------------------------------------------------------
# Decision Rules  (evaluated in order — first match wins)
#
# Carried over verbatim from agent_config.py in the existing agent layer.
# These seed the decision_rules table on first startup.
# ---------------------------------------------------------------------------
DEFAULT_DECISION_RULES: list[dict] = [
    # ------------------------------------------------------------------
    # HIGH-severity rules
    # ------------------------------------------------------------------
    {
        "id": "rule_001",
        "name": "severe_degradation_influenza",
        "priority": 0,
        "condition": {
            "current_prediction": 2,
            "trend": "degrading",
            "hr_trend_min": 5.0,
        },
        "result": {
            "severity": "high",
            "possible_condition": "Possible Influenza / Severe Systemic Infection",
            "advice": (
                "Immediate medical consultation is recommended. "
                "Sustained unhealthy classification with rising heart rate (≥5 bpm) "
                "and a degrading health trend may indicate a systemic infection such as "
                "influenza. Monitor body temperature, hydration, and respiratory "
                "symptoms closely."
            ),
            "actions": ["notify_physician", "continuous_vitals_monitoring"],
        },
    },
    {
        "id": "rule_002",
        "name": "severe_respiratory_distress",
        "priority": 1,
        "condition": {
            "current_prediction": 2,
            "trend": "degrading",
            "spo2_trend_max": -3.0,
        },
        "result": {
            "severity": "high",
            "possible_condition": "Possible Respiratory Infection / Pneumonia",
            "advice": (
                "Urgent respiratory evaluation advised. "
                "Unhealthy classification combined with declining oxygen saturation "
                "(≥3% drop) and a degrading trend may indicate pneumonia, bronchitis, "
                "or COVID-19. Check SpO2 with a pulse oximeter immediately. "
                "Seek emergency care if SpO2 falls below 92%."
            ),
            "actions": ["notify_physician", "check_spo2", "respiratory_assessment"],
        },
    },
    {
        "id": "rule_002b",
        "name": "unhealthy_degrading_general",
        "priority": 2,
        "condition": {
            "current_prediction": 2,
            "trend": "degrading",
        },
        "result": {
            "severity": "high",
            "possible_condition": "Unhealthy State with Degrading Trend — Urgent Evaluation Recommended",
            "advice": (
                "The patient is currently classified as unhealthy with a degrading "
                "health trend, although specific vital-sign thresholds for influenza "
                "or respiratory distress are not yet met. This combination is clinically "
                "concerning and warrants prompt medical evaluation. Monitor all vital "
                "signs closely and consult a healthcare provider."
            ),
            "actions": ["notify_physician", "comprehensive_evaluation", "increase_monitoring"],
        },
    },
    {
        "id": "rule_003",
        "name": "persistent_unhealthy",
        "priority": 3,
        "condition": {
            "current_prediction": 2,
            "trend": "stable",
            "unhealthy_ratio_min": 0.5,
        },
        "result": {
            "severity": "high",
            "possible_condition": "Persistent Unhealthy State — Multiple Possible Causes",
            "advice": (
                "The patient has been in a persistent unhealthy state with elevated "
                "ratios of unhealthy predictions. This may indicate a chronic condition "
                "flare-up or unresolved acute illness. Comprehensive clinical evaluation "
                "is recommended including blood work, imaging, and vital sign assessment."
            ),
            "actions": ["notify_physician", "comprehensive_evaluation"],
        },
    },

    # ------------------------------------------------------------------
    # MEDIUM-severity rules
    # ------------------------------------------------------------------
    {
        "id": "rule_004",
        "name": "early_degradation_warning",
        "priority": 4,
        "condition": {
            "current_prediction": 1,
            "trend": "degrading",
        },
        "result": {
            "severity": "medium",
            "possible_condition": "Early Warning — Health Status Declining",
            "advice": (
                "Health indicators show early signs of decline from sub-healthy state "
                "with a degrading trend. Increased monitoring frequency is advised. "
                "Review recent lifestyle factors (sleep, nutrition, stress). "
                "If symptoms develop or vital signs worsen, consult a physician."
            ),
            "actions": ["increase_monitoring_frequency", "lifestyle_review"],
        },
    },
    {
        "id": "rule_005",
        "name": "subhealthy_elevated_hr",
        "priority": 5,
        "condition": {
            "current_prediction": 1,
            "trend": "stable",
            "hr_trend_min": 3.0,
        },
        "result": {
            "severity": "medium",
            "possible_condition": "Possible Stress Response / Early Cardiovascular Strain",
            "advice": (
                "Sub-healthy classification with mildly elevated heart rate trend "
                "(≥3 bpm rise). This may reflect stress, anxiety, dehydration, or "
                "early cardiovascular strain. Ensure adequate hydration and rest. "
                "Re-check vitals in 2-4 hours."
            ),
            "actions": ["monitor_hr", "hydration_check", "rest_recommendation"],
        },
    },
    {
        "id": "rule_006",
        "name": "unhealthy_isolated_spike",
        "priority": 6,
        "condition": {
            "current_prediction": 2,
            "trend": "improving",
        },
        "result": {
            "severity": "medium",
            "possible_condition": "Isolated Unhealthy Reading — Possible Transient Issue",
            "advice": (
                "An isolated unhealthy reading was detected, but the overall health "
                "trend is improving. This may be a transient anomaly or measurement "
                "noise. Continue monitoring; if unhealthy readings persist for more "
                "than 3 consecutive assessments, seek clinical evaluation."
            ),
            "actions": ["continue_monitoring", "verify_reading"],
        },
    },

    # ------------------------------------------------------------------
    # LOW-severity rules
    # ------------------------------------------------------------------
    {
        "id": "rule_007",
        "name": "healthy_recovery",
        "priority": 7,
        "condition": {
            "current_prediction": 0,
            "trend": "improving",
        },
        "result": {
            "severity": "low",
            "possible_condition": "Healthy Recovery — Continue Current Regimen",
            "advice": (
                "Health indicators show a clear improving trend with current healthy "
                "classification. The patient appears to be recovering well. Continue "
                "current treatment or wellness regimen. Maintain regular monitoring "
                "schedule."
            ),
            "actions": ["maintain_regimen", "routine_monitoring"],
        },
    },
    {
        "id": "rule_007b",
        "name": "subhealthy_recovery",
        "priority": 8,
        "condition": {
            "current_prediction": 1,
            "trend": "improving",
        },
        "result": {
            "severity": "low",
            "possible_condition": "Sub-Healthy Recovery — Cautious Optimism",
            "advice": (
                "The patient is currently classified as sub-healthy but the overall "
                "trend is improving. This suggests recovery is underway, though not "
                "yet complete. Continue monitoring, maintain rest and hydration, "
                "and follow existing treatment plans. Re-assess if symptoms persist "
                "beyond the expected recovery window."
            ),
            "actions": ["continue_monitoring", "lifestyle_review", "routine_monitoring"],
        },
    },
    {
        "id": "rule_007c",
        "name": "healthy_early_warning",
        "priority": 9,
        "condition": {
            "current_prediction": 0,
            "trend": "degrading",
        },
        "result": {
            "severity": "medium",
            "possible_condition": "Early Warning — Healthy but Trend Degrading",
            "advice": (
                "The patient is currently classified as healthy, but the overall "
                "health trend is degrading. This may represent the earliest detectable "
                "shift from wellness toward illness. Increase monitoring frequency, "
                "review recent exposures and lifestyle factors, and watch for "
                "development of symptoms. Early intervention may prevent progression "
                "to sub-healthy or unhealthy states."
            ),
            "actions": ["increase_monitoring_frequency", "lifestyle_review", "watch_for_symptoms"],
        },
    },
    {
        "id": "rule_008",
        "name": "stable_healthy",
        "priority": 10,
        "condition": {
            "current_prediction": 0,
            "trend": "stable",
        },
        "result": {
            "severity": "low",
            "possible_condition": "Stable Healthy State — Routine Monitoring",
            "advice": (
                "The patient is in a stable healthy state. No concerning patterns "
                "detected. Continue routine health monitoring and maintain healthy "
                "lifestyle habits including regular exercise, balanced nutrition, "
                "and adequate sleep."
            ),
            "actions": ["routine_monitoring", "lifestyle_maintenance"],
        },
    },
]

# ---------------------------------------------------------------------------
# Default advice — returned when no rule matches
# ---------------------------------------------------------------------------
DEFAULT_ADVICE: dict = {
    "severity": "low",
    "possible_condition": "No Specific Pattern Detected",
    "advice": (
        "Current health readings do not match any specific risk pattern. "
        "Continue regular monitoring. Consult a healthcare provider if "
        "you experience any concerning symptoms."
    ),
    "actions": ["routine_monitoring"],
}

# ---------------------------------------------------------------------------
# Severity colour mapping
# ---------------------------------------------------------------------------
SEVERITY_COLORS: dict[str, str] = {
    "high": "#ef4444",
    "medium": "#eab308",
    "low": "#22c55e",
}
