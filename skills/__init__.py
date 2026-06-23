"""
Skills modules for the Multi-AI Agent Layer.

Each skill is a self-contained class that can be called by the
AgentCoordinator or directly via the /api/v1/multi/skills endpoint.
"""

from skills.anomaly_detector import AnomalyDetector
from skills.advanced_trend_analyzer import AdvancedTrendAnalyzer
from skills.llm_advice_generator import LLMAdviceGenerator

__all__ = [
    "AnomalyDetector",
    "AdvancedTrendAnalyzer",
    "LLMAdviceGenerator",
]
