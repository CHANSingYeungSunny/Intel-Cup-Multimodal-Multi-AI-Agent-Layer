"""
DecisionEngine — evaluates configurable decision rules against the current
health context and generates structured natural-language advice.

Rules are defined in :mod:`agent_layer.agent_config` and evaluated in
priority order (first match wins).

╔═══════════════════════════════════════════════════════════════════════════╗
║ DEPRECATED — Replaced by ../../decision_engine.py (FastAPI service).     ║
║ See ../../README.md for migration guide.  Kept for backward compat.      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

from typing import Any, Optional, List, Dict

from agent_layer.agent_config import DECISION_RULES, DEFAULT_ADVICE


class DecisionEngine:
    """
    Rule-based decision engine for health-state advice generation.

    Parameters
    ----------
    rules : list[dict], optional
        Decision rules to evaluate.  Defaults to ``DECISION_RULES`` from config.
    """

    def __init__(self, rules: list[dict] = None):
        self._rules = rules if rules is not None else list(DECISION_RULES)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        trend_summary: dict,
        current_prediction: int,
        hr_slope: float = 0.0,
        spo2_slope: float = 0.0,
        rr_slope: float = 0.0,
    ) -> dict:
        """
        Evaluate all rules in priority order; return the **first** matching
        rule's result, or ``DEFAULT_ADVICE`` if none match.

        Parameters
        ----------
        trend_summary : dict
            Output of ``TrendAnalyzer.get_summary()``.
        current_prediction : int
            The most recent prediction (0=Healthy, 1=Sub-healthy, 2=Unhealthy).
        hr_slope : float
            Heart-rate slope from TrendAnalyzer (bpm/observation).
        spo2_slope : float
            SpO₂ slope from TrendAnalyzer (%/observation).
        rr_slope : float
            RR-interval slope from TrendAnalyzer (s/observation).

        Returns
        -------
        dict
            Advice dict with keys: ``matched_rule_id``, ``matched_rule_name``,
            ``severity``, ``possible_condition``, ``advice``, ``actions``,
            ``context``, ``timestamp``.
        """
        from datetime import datetime, timezone

        context = {
            "current_prediction": current_prediction,
            "trend": trend_summary.get("trend", "stable"),
            "unhealthy_ratio": trend_summary.get("unhealthy_ratio", 0.0),
            "healthy_ratio": trend_summary.get("healthy_ratio", 0.0),
            "hr_slope": round(hr_slope, 3),
            "spo2_slope": round(spo2_slope, 3),
            "rr_slope": round(rr_slope, 3),
        }

        for rule in self._rules:
            if self._rule_matches(rule, context):
                result = dict(rule["result"])  # shallow copy
                result["matched_rule_id"] = rule["id"]
                result["matched_rule_name"] = rule["name"]
                result["context"] = context
                result["timestamp"] = datetime.now(timezone.utc).isoformat()
                return result

        # No rule matched — return default
        default = dict(DEFAULT_ADVICE)
        default["matched_rule_id"] = None
        default["matched_rule_name"] = "default"
        default["context"] = context
        default["timestamp"] = datetime.now(timezone.utc).isoformat()
        return default

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------
    def get_all_rules(self) -> list[dict]:
        """Return metadata for all registered rules."""
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "condition": r["condition"],
                "result_severity": r["result"]["severity"],
                "result_condition": r["result"]["possible_condition"],
            }
            for r in self._rules
        ]

    def add_rule(self, rule_dict: dict):
        """Append a new rule at runtime (lower priority)."""
        self._rules.append(rule_dict)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by id.  Returns ``True`` if a rule was removed."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r["id"] != rule_id]
        return len(self._rules) < before

    def get_rule_count(self) -> int:
        return len(self._rules)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _rule_matches(rule: dict, context: dict) -> bool:
        """
        Check every key in ``rule["condition"]`` against *context*.

        Supported condition keys (all optional — omit to skip):

        * ``current_prediction`` — exact int match
        * ``trend`` — exact string match ("degrading" / "improving" / "stable")
        * ``hr_trend_min`` — context ``hr_slope`` ≥ this value
        * ``hr_trend_max`` — context ``hr_slope`` ≤ this value
        * ``spo2_trend_min`` — context ``spo2_slope`` ≥ this value
        * ``spo2_trend_max`` — context ``spo2_slope`` ≤ this value
        * ``rr_trend_min`` — context ``rr_slope`` ≥ this value
        * ``rr_trend_max`` — context ``rr_slope`` ≤ this value
        * ``unhealthy_ratio_min`` — context ``unhealthy_ratio`` ≥ this value
        * ``healthy_ratio_min`` — context ``healthy_ratio`` ≥ this value
        """
        cond = rule.get("condition", {})

        # Exact matches
        if "current_prediction" in cond:
            if context["current_prediction"] != cond["current_prediction"]:
                return False

        if "trend" in cond:
            if context["trend"] != cond["trend"]:
                return False

        # Numeric thresholds (min — value must be ≥ threshold)
        for key, ctx_key in [
            ("hr_trend_min", "hr_slope"),
            ("spo2_trend_min", "spo2_slope"),
            ("rr_trend_min", "rr_slope"),
            ("unhealthy_ratio_min", "unhealthy_ratio"),
            ("healthy_ratio_min", "healthy_ratio"),
        ]:
            if key in cond and context.get(ctx_key, 0) < cond[key]:
                return False

        # Numeric thresholds (max — value must be ≤ threshold)
        for key, ctx_key in [
            ("hr_trend_max", "hr_slope"),
            ("spo2_trend_max", "spo2_slope"),
            ("rr_trend_max", "rr_slope"),
        ]:
            if key in cond and context.get(ctx_key, 0) > cond[key]:
                return False

        return True
