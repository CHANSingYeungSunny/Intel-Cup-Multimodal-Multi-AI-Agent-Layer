"""
LLM Advice Generator Skill — enriches health advice with an LLM backend.

Implements the ``enrich_with_llm()`` extension point that was a no-op in
the Single-layer AdviceGenerator.  Supports multiple backends:

- ``"none"``    — passthrough (no enrichment, preserves original text)
- ``"openai"``  — calls OpenAI Chat Completions API
- ``"claude"``  — calls Anthropic Messages API
- ``"local"``   — calls any OpenAI-compatible local endpoint (e.g. Ollama)

Usage::

    gen = LLMAdviceGenerator(backend="openai", api_key="sk-...")
    enriched = await gen.enrich(advice_dict, clinical_context)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMAdviceGenerator:
    """
    LLM-powered medical advice enrichment.

    Takes a structured advice dict from the rule-based pipeline and
    optionally rewrites / supplements the natural-language advice text
    using a large language model.

    Parameters
    ----------
    backend : str
        One of ``"none"``, ``"openai"``, ``"claude"``, ``"local"``.
    api_key : str or None
        API key for the LLM service (if required).
    model : str
        Model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-6"``).
    max_tokens : int
        Maximum tokens in the LLM response.
    temperature : float
        Sampling temperature (0–2).
    local_endpoint : str
        Base URL for local LLM endpoint (only used when backend="local").
    """

    SYSTEM_PROMPT = (
        "You are a clinical decision support assistant integrated into a "
        "multimodal health monitoring system. Your role is to enhance "
        "automatically generated health advice with additional clinical "
        "reasoning, context, and actionable recommendations.\n\n"
        "Guidelines:\n"
        "- Be concise — one short paragraph, 2–4 sentences.\n"
        "- Use plain, professional language suitable for a clinical dashboard.\n"
        "- Do NOT change the severity level or suggested actions.\n"
        "- Do NOT make definitive diagnoses — use phrases like 'may indicate', "
        "'is consistent with', 'consider evaluating for'.\n"
        "- Base your response on the provided vital signs, trends, and "
        "prediction data. Do not invent data."
    )

    def __init__(
        self,
        backend: str = "none",
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 512,
        temperature: float = 0.3,
        local_endpoint: str = "http://localhost:11434/v1/chat/completions",
    ):
        self._backend = backend.lower()
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._local_endpoint = local_endpoint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enrich(
        self,
        advice_dict: dict,
        clinical_context: Optional[dict] = None,
    ) -> dict:
        """
        Enrich a structured advice dict with LLM-generated text.

        Always preserves the original structured fields (severity,
        actions, matched_rule_id).  Only the ``advice`` text field
        may be rewritten / supplemented.

        Returns the (possibly enriched) advice dict.
        """
        if self._backend == "none" or not self._api_key:
            # Passthrough — no enrichment
            return advice_dict

        clinical_context = clinical_context or {}

        try:
            enriched_text = await self._call_llm(advice_dict, clinical_context)
            if enriched_text:
                advice_dict = dict(advice_dict)  # shallow copy
                advice_dict["advice"] = enriched_text.strip()
                advice_dict["llm_enriched"] = True
                advice_dict["llm_backend"] = self._backend
                advice_dict["llm_model"] = self._model
        except Exception as exc:
            logger.warning(
                "LLM enrichment failed (%s): %s — returning original advice",
                self._backend,
                exc,
            )
            advice_dict = dict(advice_dict)
            advice_dict["llm_enriched"] = False
            advice_dict["llm_error"] = str(exc)

        return advice_dict

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    async def _call_llm(self, advice_dict: dict, context: dict) -> Optional[str]:
        """Dispatch to the appropriate backend."""
        if self._backend == "openai":
            return await self._enrich_openai(advice_dict, context)
        elif self._backend == "claude":
            return await self._enrich_claude(advice_dict, context)
        elif self._backend == "local":
            return await self._enrich_local(advice_dict, context)
        else:
            logger.warning("Unknown LLM backend: %s", self._backend)
            return None

    def _build_user_prompt(self, advice_dict: dict, context: dict) -> str:
        """Build the user prompt from advice + clinical context."""
        parts = [
            "Current health advice:",
            f"  Severity: {advice_dict.get('severity', 'unknown')}",
            f"  Possible condition: {advice_dict.get('possible_condition', 'N/A')}",
            f"  Advice: {advice_dict.get('advice', 'N/A')}",
            f"  Suggested actions: {', '.join(advice_dict.get('actions', []))}",
        ]

        trend = context.get("trend", "unknown")
        parts.append(f"\nTrend: {trend}")

        for key, label in [
            ("hr_slope", "HR slope (bpm/tick)"),
            ("spo2_slope", "SpO2 slope (%/tick)"),
            ("rr_slope", "RR slope (s/tick)"),
        ]:
            if key in context:
                parts.append(f"  {label}: {context[key]:.3f}")

        if "unhealthy_ratio" in context:
            parts.append(
                f"  Unhealthy ratio: {context['unhealthy_ratio']:.2f}"
            )
        if "healthy_ratio" in context:
            parts.append(
                f"  Healthy ratio: {context['healthy_ratio']:.2f}"
            )
        if "current_prediction" in context:
            pred_names = {0: "Healthy", 1: "Sub-healthy", 2: "Unhealthy"}
            pred_name = pred_names.get(
                context["current_prediction"], str(context["current_prediction"])
            )
            parts.append(f"  Current prediction: {pred_name}")

        parts.append(
            "\nPlease provide additional clinical reasoning or "
            "recommendations to supplement this advice."
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    async def _enrich_openai(
        self, advice_dict: dict, context: dict
    ) -> Optional[str]:
        """Call the OpenAI Chat Completions API."""
        try:
            import openai  # noqa: F811
        except ImportError:
            logger.warning("openai package not installed — skipping enrichment")
            return None

        client = openai.AsyncOpenAI(api_key=self._api_key)
        user_prompt = self._build_user_prompt(advice_dict, context)

        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Anthropic Claude
    # ------------------------------------------------------------------

    async def _enrich_claude(
        self, advice_dict: dict, context: dict
    ) -> Optional[str]:
        """Call the Anthropic Messages API."""
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed — skipping enrichment")
            return None

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        user_prompt = self._build_user_prompt(advice_dict, context)

        response = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # response.content is a list of ContentBlock; extract text
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return None

    # ------------------------------------------------------------------
    # Local (OpenAI-compatible endpoint, e.g. Ollama)
    # ------------------------------------------------------------------

    async def _enrich_local(
        self, advice_dict: dict, context: dict
    ) -> Optional[str]:
        """Call a local OpenAI-compatible endpoint."""
        import httpx

        user_prompt = self._build_user_prompt(advice_dict, context)

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self._local_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
