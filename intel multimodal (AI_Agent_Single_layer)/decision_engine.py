"""
DecisionEngine — evaluates configurable decision rules against the current
health context and returns the first matching rule's result.

Evolved from the existing ``agent_layer/decision_engine.py`` with added
PostgreSQL-backed dynamic rule CRUD.  The hot-path ``evaluate()`` method
operates on in-memory rules only (no DB call per tick).

Rules are evaluated in priority order (lowest *priority* value first).
The first rule whose condition fully matches the current context wins.
If no rule matches, the configured ``DEFAULT_ADVICE`` is returned.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import DEFAULT_DECISION_RULES, DEFAULT_ADVICE
from models import DecisionRule

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Rule-based decision engine for health-state advice generation.

    Parameters
    ----------
    rules : list[dict], optional
        Initial set of in-memory rules.  If omitted, loads from the DB
        or falls back to ``config.DEFAULT_DECISION_RULES``.
    """

    def __init__(self, rules: Optional[list[dict]] = None):
        self._rules: list[dict] = rules if rules is not None else []

    # ------------------------------------------------------------------
    # Core evaluation (hot path — in-memory only, no DB call)
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
            Matched rule with *matched_rule_id* and *matched_rule_name* keys
            injected into the result dict, plus *context* and *timestamp*.
        """
        context = {
            "current_prediction": current_prediction,
            "trend": trend_summary.get("trend", "stable"),
            "unhealthy_ratio": trend_summary.get("unhealthy_ratio", 0.0),
            "healthy_ratio": trend_summary.get("healthy_ratio", 0.0),
            "hr_slope": round(hr_slope, 3),
            "spo2_slope": round(spo2_slope, 3),
            "rr_slope": round(rr_slope, 3),
        }

        # Iterate enabled rules in priority order
        sorted_rules = sorted(
            [r for r in self._rules if r.get("enabled", True)],
            key=lambda r: r.get("priority", 0),
        )

        for rule in sorted_rules:
            if self._rule_matches(rule, context):
                return {
                    "matched_rule_id": rule.get("id"),
                    "matched_rule_name": rule.get("name", "unknown"),
                    "context": context,
                    "timestamp": self._utc_now_iso(),
                    **rule.get("result", {}),
                }

        # No rule matched — return default
        default = dict(DEFAULT_ADVICE)
        return {
            "matched_rule_id": None,
            "matched_rule_name": "default",
            "context": context,
            "timestamp": self._utc_now_iso(),
            "severity": default.get("severity", "low"),
            "possible_condition": default.get("possible_condition", ""),
            "advice": default.get("advice", ""),
            "actions": default.get("actions", []),
        }

    # ------------------------------------------------------------------
    # Rule management (in-memory accessors — no DB needed)
    # ------------------------------------------------------------------

    def get_all_rules(self) -> list[dict]:
        """Return metadata for all registered rules."""
        return [
            {
                "rule_id": r["id"],
                "name": r["name"],
                "condition": r.get("condition", {}),
                "result_severity": r.get("result", {}).get("severity", "low"),
                "result_condition": r.get("result", {}).get("possible_condition", ""),
                "result_advice": r.get("result", {}).get("advice", ""),
                "result_actions": r.get("result", {}).get("actions", []),
                "priority": r.get("priority", 0),
                "enabled": r.get("enabled", True),
            }
            for r in self._rules
        ]

    def get_rule_count(self) -> int:
        """Return the number of enabled rules."""
        return sum(1 for r in self._rules if r.get("enabled", True))

    # ------------------------------------------------------------------
    # PostgreSQL-backed rule CRUD
    # ------------------------------------------------------------------

    async def load_rules_from_db(self, db: AsyncSession) -> int:
        """
        Load enabled rules from the ``decision_rules`` table into memory,
        ordered by *priority* ascending.

        If the table is empty, seeds it from ``config.DEFAULT_DECISION_RULES``
        and reloads.

        Returns the number of rules loaded.
        """
        stmt = (
            select(DecisionRule)
            .order_by(DecisionRule.priority.asc())
        )
        result = await db.execute(stmt)
        rows = list(result.scalars().all())

        if not rows:
            logger.info("No rules in DB — seeding from config.DEFAULT_DECISION_RULES")
            await self._seed_default_rules(db)
            result = await db.execute(stmt)
            rows = list(result.scalars().all())

        self._rules = [row.to_rule_dict() for row in rows]
        logger.info("Loaded %d rules from database", len(self._rules))
        return len(self._rules)

    async def add_rule(
        self,
        db: AsyncSession,
        rule_data: dict,
    ) -> dict:
        """
        Insert a new rule into PostgreSQL and reload the in-memory rule set.

        Parameters
        ----------
        db : AsyncSession
        rule_data : dict
            Must contain: ``rule_id``, ``name``, ``condition``,
            ``result_severity``, ``result_condition``, ``result_advice``,
            ``result_actions``, ``priority``.

        Returns
        -------
        dict
            The rule metadata for the newly created rule.
        """
        orm_rule = DecisionRule(
            rule_id=rule_data["rule_id"],
            name=rule_data["name"],
            condition=rule_data.get("condition", {}),
            result_severity=rule_data.get("result_severity", "low"),
            result_condition=rule_data.get("result_condition", ""),
            result_advice=rule_data.get("result_advice", ""),
            result_actions=rule_data.get("result_actions", []),
            priority=rule_data.get("priority", 0),
            enabled=True,
        )
        db.add(orm_rule)
        await db.commit()
        await self.load_rules_from_db(db)
        logger.info("Added rule %s", rule_data["rule_id"])
        return orm_rule.to_api_dict()

    async def remove_rule(self, db: AsyncSession, rule_id: str) -> bool:
        """
        Delete a rule from PostgreSQL by *rule_id* and reload in-memory rules.

        Returns ``True`` if a rule was deleted, ``False`` if not found.
        """
        stmt = select(DecisionRule).where(DecisionRule.rule_id == rule_id)
        result = await db.execute(stmt)
        orm_rule = result.scalar_one_or_none()

        if orm_rule is None:
            logger.warning("Rule %s not found for deletion", rule_id)
            return False

        await db.delete(orm_rule)
        await db.commit()
        await self.load_rules_from_db(db)
        logger.info("Removed rule %s", rule_id)
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _seed_default_rules(self, db: AsyncSession) -> None:
        """Insert ``DEFAULT_DECISION_RULES`` into the ``decision_rules`` table."""
        for rule in DEFAULT_DECISION_RULES:
            orm_rule = DecisionRule(
                rule_id=rule["id"],
                name=rule["name"],
                condition=rule.get("condition", {}),
                result_severity=rule["result"]["severity"],
                result_condition=rule["result"]["possible_condition"],
                result_advice=rule["result"]["advice"],
                result_actions=rule["result"]["actions"],
                priority=rule.get("priority", 0),
                enabled=True,
            )
            db.add(orm_rule)
        await db.commit()
        logger.info(
            "Seeded %d default rules into decision_rules table",
            len(DEFAULT_DECISION_RULES),
        )

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

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
