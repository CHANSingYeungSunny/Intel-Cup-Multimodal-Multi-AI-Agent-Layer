"""
AdviceGenerator — constructs structured natural-language advice dicts.

Separates advice assembly from rule matching (which lives in
:class:`DecisionEngine`).  This module is responsible for taking a matched
rule result (or the default fallback) and producing the final advice dict
that is returned to callers and persisted to the ``advice_log`` table.

Extension point: the optional ``enrich_with_llm()`` method provides a clear
hook for future LLM-based advice enhancement.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Optional

from config import DEFAULT_ADVICE

logger = logging.getLogger(__name__)


class AdviceGenerator:
    """
    Assembles full advice dicts from matched-rule results and evaluation context.

    This class is intentionally stateless — it is safe to use as a singleton
    stored on ``app.state``.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        matched_rule: dict,
        context: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Build a full advice dict from a matched decision rule.

        Accepts either:

        * The raw rule dict from config (with ``"id"``, ``"name"``, ``"result"``), or
        * The output of :meth:`DecisionEngine.evaluate` (which already has
          ``matched_rule_id``, ``matched_rule_name``, ``severity``, etc.).

        Parameters
        ----------
        matched_rule : dict
            Matched rule data.
        context : dict, optional
            Evaluation context.  Used when *matched_rule* does not already
            contain a ``"context"`` key.
        timestamp : str, optional
            ISO 8601 UTC timestamp.  Generated if not provided.

        Returns
        -------
        dict
            Full advice dict with keys: ``matched_rule_id``,
            ``matched_rule_name``, ``severity``, ``possible_condition``,
            ``advice``, ``actions``, ``context``, ``timestamp``.
        """
        # Support both raw rule dicts and evaluate() output
        if "result" in matched_rule and "severity" not in matched_rule:
            # Raw rule format: {'id': ..., 'name': ..., 'result': {...}}
            result = copy.deepcopy(matched_rule.get("result", {}))
            return {
                "matched_rule_id": matched_rule.get("id"),
                "matched_rule_name": matched_rule.get("name", "unknown"),
                "severity": result.get("severity", "low"),
                "possible_condition": result.get("possible_condition", ""),
                "advice": result.get("advice", ""),
                "actions": result.get("actions", []),
                "context": context or {},
                "timestamp": timestamp or self._utc_now_iso(),
            }
        else:
            # evaluate() output: {'matched_rule_id': ..., 'severity': ..., ...}
            return {
                "matched_rule_id": matched_rule.get("matched_rule_id"),
                "matched_rule_name": matched_rule.get("matched_rule_name", "unknown"),
                "severity": matched_rule.get("severity", "low"),
                "possible_condition": matched_rule.get("possible_condition", ""),
                "advice": matched_rule.get("advice", ""),
                "actions": matched_rule.get("actions", []),
                "context": context or matched_rule.get("context", {}),
                "timestamp": timestamp or matched_rule.get("timestamp") or self._utc_now_iso(),
            }

    def generate_default(
        self,
        context: dict,
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Build the fallback / default advice dict when no rule matches.

        Parameters
        ----------
        context : dict
            Evaluation context.
        timestamp : str, optional
            ISO 8601 UTC timestamp.

        Returns
        -------
        dict
            Advice dict using ``config.DEFAULT_ADVICE`` values.
        """
        default = copy.deepcopy(DEFAULT_ADVICE)
        return {
            "matched_rule_id": None,
            "matched_rule_name": "default",
            "severity": default.get("severity", "low"),
            "possible_condition": default.get("possible_condition", ""),
            "advice": default.get("advice", ""),
            "actions": default.get("actions", []),
            "context": context,
            "timestamp": timestamp or self._utc_now_iso(),
        }

    # ------------------------------------------------------------------
    # Future extension point
    # ------------------------------------------------------------------

    async def enrich_with_llm(self, advice_dict: dict) -> dict:
        """
        Placeholder for future LLM-based advice enrichment.

        When implemented, this method would call an external LLM API to
        rewrite or augment the advice text while preserving the structured
        fields (severity, actions, context).

        Parameters
        ----------
        advice_dict : dict
            The advice dict produced by :meth:`generate` or
            :meth:`generate_default`.

        Returns
        -------
        dict
            The enriched advice dict (currently a no-op passthrough).
        """
        # TODO: Integrate with an LLM API (e.g. Claude, GPT) to rephrase
        #       or supplement the advice text based on full clinical context.
        return advice_dict

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
