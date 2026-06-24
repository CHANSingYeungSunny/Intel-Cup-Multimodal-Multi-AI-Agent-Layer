"""Unit tests for HealthAgent orchestrator."""

import pytest
from agent_orchestrator import HealthAgent
from config import DEFAULT_DECISION_RULES


@pytest.fixture
def agent():
    """Create a HealthAgent without DB persistence (pure in-memory)."""
    a = HealthAgent(db_session_factory=None)
    # Seed rules directly
    a.decision_engine._rules = list(DEFAULT_DECISION_RULES)
    return a


@pytest.mark.asyncio
async def test_process_tick_returns_advice(agent):
    """A valid tick returns a full advice dict."""
    advice = await agent.process_tick(
        prediction=2,
        subject_id="subject14",
        hr_sim=95.0,
        spo2_sim=94.0,
        rr_sim=0.72,
    )
    assert advice is not None
    assert "matched_rule_id" in advice
    assert "severity" in advice
    assert "advice" in advice
    assert "context" in advice
    assert "timestamp" in advice


@pytest.mark.asyncio
async def test_process_tick_deduplicates(agent):
    """Two identical ticks → second returns None."""
    payload = dict(prediction=0, subject_id="s1", hr_sim=72.0, spo2_sim=98.0, rr_sim=0.88)
    r1 = await agent.process_tick(**payload)
    assert r1 is not None

    r2 = await agent.process_tick(**payload)
    assert r2 is None  # deduplicated


@pytest.mark.asyncio
async def test_process_tick_with_feature_vector(agent):
    """Providing a feature_vector triggers proxy computation."""
    advice = await agent.process_tick(
        prediction=1,
        subject_id="s1",
        feature_vector=[0.1] * 256,
    )
    assert advice is not None


@pytest.mark.asyncio
async def test_get_current_advice(agent):
    """get_current_advice returns the latest, or None before any tick."""
    assert agent.get_current_advice() is None
    await agent.process_tick(prediction=2, subject_id="s1", hr_sim=100.0)
    assert agent.get_current_advice() is not None


@pytest.mark.asyncio
async def test_get_advice_history(agent):
    """get_advice_history returns recent entries."""
    await agent.process_tick(prediction=2, subject_id="s1", hr_sim=100.0)
    await agent.process_tick(prediction=0, subject_id="s1", hr_sim=72.0)
    history = agent.get_advice_history(n=10)
    assert len(history) >= 1


@pytest.mark.asyncio
async def test_get_trend_summary(agent):
    """get_trend_summary returns expected keys."""
    await agent.process_tick(prediction=1, subject_id="s1", hr_sim=75.0)
    summary = agent.get_trend_summary()
    assert "trend" in summary
    assert "hr_slope" in summary
    assert "history_size" in summary


def test_get_rules(agent):
    """get_rules returns all active rules."""
    rules = agent.get_rules()
    assert len(rules) > 0
    assert "rule_id" in rules[0]


def test_get_status(agent):
    """get_status returns valid status dict."""
    status = agent.get_status()
    assert status["enabled"] is True
    assert "rules_count" in status
    assert "trend" in status


@pytest.mark.asyncio
async def test_reset(agent):
    """reset() clears all state."""
    await agent.process_tick(prediction=2, subject_id="s1", hr_sim=100.0)
    assert agent.get_current_advice() is not None
    assert agent.get_advice_history()  # not empty

    agent.reset()
    assert agent.get_current_advice() is None
    assert agent.get_advice_history() == []
    assert agent.trend_analyzer.get_history_size() == 0


def test_decision_engine_property(agent):
    """The decision_engine property exposes the engine."""
    assert agent.decision_engine is not None
    assert agent.decision_engine.get_rule_count() > 0


def test_trend_analyzer_property(agent):
    """The trend_analyzer property exposes the analyzer."""
    assert agent.trend_analyzer is not None
