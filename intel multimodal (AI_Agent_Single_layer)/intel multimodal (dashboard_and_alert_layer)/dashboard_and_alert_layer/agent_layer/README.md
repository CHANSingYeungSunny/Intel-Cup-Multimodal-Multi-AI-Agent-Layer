# AI Agent Layer

> **⚠️ DEPRECATED** — This Flask-based agent has been replaced by the standalone
> FastAPI service at `../../` (Single AI Agent Layer). See `../../README.md` for
> the migration guide. This module is kept for backward compatibility and will be
> removed in v2.0.
>
> To use the new service with the Dashboard, start it separately and pass
> `--agent-api-url http://localhost:8000/api/v1` to `run.py`.

**Lightweight rule-based AI agent for the Intel Multimodal Health Monitoring System.**

The Agent Layer interprets Fusion Layer predictions in real time, detects health trends over time, applies configurable decision rules, and generates structured natural-language health advice. It integrates with the existing Dashboard & Alerts layer through both REST endpoints and SocketIO events.

---

## Architecture

```
HealthSimulator tick (every 2s)
  │
  └── HealthAgent.process_tick()
        │
        ├── TrendAnalyzer.add_observation()
        │     └── Rolling buffer of {prediction, hr_proxy, spo2_proxy}
        │     └── Trend classification: degrading | improving | stable
        │     └── Vital-sign slopes via linear regression
        │
        ├── DecisionEngine.evaluate()
        │     └── Iterates DECISION_RULES (first-match priority)
        │     └── Rule match → {severity, possible_condition, advice, actions}
        │     └── No match  → DEFAULT_ADVICE
        │
        └── Emit "agent_advice" SocketIO event (deduplicated)
```

## Module Structure

```
agent_layer/
  __init__.py                 # Package marker
  agent_config.py             # Configurable rules, thresholds, advice templates
  trend_analyzer.py           # TrendAnalyzer — rolling history buffer + trend detection
  decision_engine.py          # DecisionEngine — rule matching + advice generation
  health_agent.py             # HealthAgent — orchestrator (singleton)
  routes/
    __init__.py
    agent_routes.py           # Flask blueprint: REST API endpoints
  README.md                   # This file
```

---

## Quick Start

The agent starts automatically when launching the dashboard:

```bash
python run.py
```

Disable the agent with the `--no-agent` flag:

```bash
python run.py --no-agent
```

Or set the environment variable:

```bash
# Windows
set AGENT_ENABLED=False
python run.py

# Linux / macOS
AGENT_ENABLED=False python run.py
```

---

## REST API Endpoints

All endpoints are served under `/api/` when the agent is enabled.

| Endpoint | Method | Description |
|---|---|---|
| `/api/agent_advice` | GET | Latest advice + trend summary |
| `/api/agent_history?n=20` | GET | Recent advice history (default 20, max 50) |
| `/api/agent_rules` | GET | Active decision rules metadata |
| `/api/agent_status` | GET | Lightweight agent status |

### Example Response: `/api/agent_advice`

```json
{
  "latest_advice": {
    "matched_rule_id": "rule_001",
    "matched_rule_name": "severe_degradation_influenza",
    "severity": "high",
    "possible_condition": "Possible Influenza / Severe Systemic Infection",
    "advice": "Immediate medical consultation is recommended...",
    "actions": ["notify_physician", "continuous_vitals_monitoring"],
    "context": {
      "current_prediction": 2,
      "trend": "degrading",
      "unhealthy_ratio": 0.6,
      "healthy_ratio": 0.2,
      "hr_slope": 7.2,
      "spo2_slope": -1.5
    },
    "timestamp": "2026-06-23T14:30:00.000Z"
  },
  "trend_summary": {
    "trend": "degrading",
    "history_size": 20,
    "hr_slope": 7.2,
    "spo2_slope": -1.5
  },
  "active_rules_count": 8
}
```

### Example Response: `/api/agent_history?n=3`

```json
{
  "history": [
    {
      "matched_rule_id": "rule_001",
      "matched_rule_name": "severe_degradation_influenza",
      "severity": "high",
      "possible_condition": "Possible Influenza / Severe Systemic Infection",
      "advice": "Immediate medical consultation is recommended...",
      "actions": ["notify_physician", "continuous_vitals_monitoring"],
      "context": { "current_prediction": 2, "trend": "degrading", "hr_slope": 7.2, "spo2_slope": -1.5, "rr_slope": 0.01, "unhealthy_ratio": 0.6, "healthy_ratio": 0.2 },
      "timestamp": "2026-06-23T14:30:00.000Z"
    }
  ],
  "count": 1
}
```

### Example Response: `/api/agent_rules`

```json
{
  "rules": [
    {
      "id": "rule_001",
      "name": "severe_degradation_influenza",
      "condition": { "current_prediction": 2, "trend": "degrading", "hr_trend_min": 5.0 },
      "result_severity": "high",
      "result_condition": "Possible Influenza / Severe Systemic Infection"
    }
  ],
  "count": 11
}
```

### Example Response: `/api/agent_status`

```json
{
  "enabled": true,
  "rules_count": 11,
  "history_size": 15,
  "latest_severity": "medium",
  "latest_condition": "Early Warning — Health Status Declining",
  "trend": "degrading"
}
```

### Error Response (503 — agent disabled)

```json
{
  "enabled": false,
  "error": "AI Agent is not enabled"
}
```

---

## WebSocket Events

The agent emits two SocketIO events:

| Event | Payload | When |
|---|---|---|
| `agent_advice` | Full advice dict | Emitted on each simulator tick when advice changes |
| `agent_error` | `{"error": "..."}` | On agent processing errors |

The `request_agent_advice` client event triggers an immediate re-emit of the latest advice.

---

## Decision Rule Format

Rules are defined in `agent_config.py` as a list of dicts. Each rule has:

```python
{
    "id": "rule_001",                          # Unique identifier
    "name": "severe_degradation_influenza",    # Human-readable name
    "condition": {                             # Every present key must match for the rule to fire
        "current_prediction": 2,               # int: 0=Healthy, 1=Sub-healthy, 2=Unhealthy
        "trend": "degrading",                  # str: "degrading" | "improving" | "stable" (omit to skip)
        "hr_trend_min": 5.0,                   # float: HR slope ≥ N bpm/tick (omit to skip)
        "spo2_trend_max": -3.0,                # float: SpO₂ slope ≤ N %/tick (omit to skip)
        "unhealthy_ratio_min": 0.5,            # float: unhealthy fraction ≥ N (omit to skip)
    },
    "result": {
        "severity": "high",                    # "high" | "medium" | "low"
        "possible_condition": "Possible Flu",  # Displayed condition name
        "advice": "Detailed advice text...",   # Natural-language recommendation
        "actions": ["action_1", "action_2"],   # Suggested follow-up actions
    },
}
```

**Omit condition keys to skip them** — a rule with only `{"current_prediction": 2}` matches any unhealthy
reading regardless of trend or vitals. Do **not** set unused threshold keys to `None`; simply leave them
out of the condition dict entirely (values like `"hr_trend_max": None` would cause a runtime `TypeError`
when the engine tries to compare a float against `None`).

Rules are evaluated in list order. **The first matching rule wins.**

Available condition keys:

| Key | Comparison | Example |
|-----|-----------|---------|
| `current_prediction` | exact int match | `2` |
| `trend` | exact string match | `"degrading"` |
| `hr_trend_min` | `hr_slope` ≥ value | `5.0` |
| `hr_trend_max` | `hr_slope` ≤ value | `-3.0` |
| `spo2_trend_min` | `spo2_slope` ≥ value | `2.0` |
| `spo2_trend_max` | `spo2_slope` ≤ value | `-3.0` |
| `rr_trend_min` | `rr_slope` ≥ value | `0.05` |
| `rr_trend_max` | `rr_slope` ≤ value | `-0.02` |
| `unhealthy_ratio_min` | unhealthy fraction ≥ value | `0.5` |
| `healthy_ratio_min` | healthy fraction ≥ value | `0.7` |

The minimum number of observations required before a non-stable trend can be
declared is `max(2, TREND_WINDOW_SIZE // 2)` = **5** (configurable in
`agent_config.py`).

---

## Adding Custom Rules

Edit `agent_config.py` and add your rule to the `DECISION_RULES` list. No code changes are required.

**Example — adding a rule for bradycardia detection:**

```python
{
    "id": "custom_bradycardia",
    "name": "low_hr_warning",
    "condition": {
        "current_prediction": 1,       # Sub-healthy
        "hr_trend_max": -8.0,          # HR dropping sharply
    },
    "result": {
        "severity": "medium",
        "possible_condition": "Possible Bradycardia / Low Cardiac Output",
        "advice": "Heart rate is declining significantly...",
        "actions": ["check_hr", "notify_physician"],
    },
},
```

Place high-priority rules at the top of the list.

---

## Testing

Run all agent tests from the `dashboard_and_alert_layer/` directory:

```bash
python -m pytest tests_agent/ -v
```

Run alongside existing dashboard tests:

```bash
python -m pytest tests/ tests_agent/ -v
```

The test suite covers:
- `test_trend_analyzer.py` — history buffer, trend classification, vital-sign slopes (13 tests)
- `test_decision_engine.py` — rule matching, priority, defaults, dynamic rules (12 tests)
- `test_health_agent.py` — tick processing, deduplication, proxying, history management (13 tests)
- `test_agent_routes.py` — REST endpoint responses, status codes, error handling (7 tests)

---

## Integration Points

| Component | Integration | File |
|---|---|---|
| **HealthSimulator** | Calls `agent.process_tick()` on each tick | `dashboard/backend/health_simulator.py` |
| **Flask REST** | Blueprint registered at `/api/` | `agent_layer/routes/agent_routes.py` |
| **SocketIO** | Emits `agent_advice` and `agent_error` | `dashboard/backend/health_simulator.py` |
| **System Status** | Includes agent info in heartbeat | `dashboard/backend/app.py:_build_system_status()` |
| **React Frontend** | `AgentSuggestionsPanel` component | `dashboard/frontend/src/components/AgentSuggestionsPanel.jsx` |

---

## Configuration Reference

Settings in `agent_config.py`:

| Constant | Default | Description |
|---|---|---|
| `HISTORY_WINDOW_SIZE` | 20 | Max observations in rolling buffer |
| `TREND_WINDOW_SIZE` | 10 | Observations used for trend classification |
| `DEGRADING_THRESHOLD` | 0.3 | Unhealthy fraction → "degrading" |
| `IMPROVING_THRESHOLD` | 0.7 | Healthy fraction → "improving" |
| `DECISION_RULES` | 8 rules | Ordered list of decision rules |
| `DEFAULT_ADVICE` | — | Fallback when no rule matches |

Settings in `config.py` (top-level):

| Constant | Default | Description |
|---|---|---|
| `AGENT_ENABLED` | `True` | Master toggle (also set via `AGENT_ENABLED` env var) |
| `AGENT_HISTORY_WINDOW` | 20 | Passed to TrendAnalyzer |
| `AGENT_TREND_WINDOW` | 10 | Passed to TrendAnalyzer |

---

## Design Philosophy

- **Lightweight**: Pure Python, no ML inference, no new dependencies beyond `numpy`.
- **Configurable**: All rules in a single config file; no code changes needed for new rules.
- **Modular**: Agent logic is fully separated from the dashboard, alerts, and simulator.
- **Observable**: Every decision includes the matched rule ID and evaluation context.
- **Deduplicated**: SocketIO events only fire when advice changes, reducing frontend churn.
