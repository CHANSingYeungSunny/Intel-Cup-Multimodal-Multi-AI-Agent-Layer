# Single AI Agent Layer

**Production-ready, standalone FastAPI microservice for AI-driven health monitoring advice.**

Part of the Intel Multimodal Health Monitoring System. Ingests health observations from fused vision/audio/physiological predictions, detects trends, evaluates configurable decision rules, and emits structured natural-language health advice. Backed by PostgreSQL for persistent history, trends, and dynamic rule management.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      OVERALL SYSTEM PIPELINE                              │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐                       │
│  │  Vision  │  │  Audio   │  │  Physiological   │                       │
│  │  Layer   │  │  Layer   │  │  Layer            │                       │
│  │ (Swin-T) │  │  (AST)   │  │  (iTransformer)   │                       │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘                       │
│       │  768-dim     │  128-dim       │  128-dim                         │
│       └──────────────┼────────────────┘                                  │
│                      ▼                                                   │
│            ┌──────────────────┐                                          │
│            │   Fusion Layer   │                                          │
│            │  (Transformer)   │                                          │
│            │  256-dim CLS     │                                          │
│            └────────┬─────────┘                                          │
│                     │  predictions.csv                                   │
│                     ▼                                                    │
│  ┌──────────────────────────────────────────────┐                        │
│  │     Single AI Agent Layer  (THIS SERVICE)    │                        │
│  │                                              │                        │
│  │  ┌────────────────────────────────────────┐  │                        │
│  │  │         HealthAgent Orchestrator        │  │                        │
│  │  │                                        │  │                        │
│  │  │  ┌──────────────┐  ┌────────────────┐  │  │                        │
│  │  │  │TrendAnalyzer │  │ DecisionEngine │  │  │                        │
│  │  │  │              │  │                │  │  │                        │
│  │  │  │  Rolling buf │  │  10 cond keys  │  │  │                        │
│  │  │  │  Trend det.  │  │  Priority order│  │  │                        │
│  │  │  │  numpy slope │  │  Dynamic CRUD  │  │  │                        │
│  │  │  └──────┬───────┘  └───────┬────────┘  │  │                        │
│  │  │         └──────────┬───────┘           │  │                        │
│  │  │                    ▼                   │  │                        │
│  │  │          ┌──────────────────┐          │  │                        │
│  │  │          │ AdviceGenerator  │          │  │                        │
│  │  │          │  Template-based  │          │  │                        │
│  │  │          │  LLM ext. point  │          │  │                        │
│  │  │          └────────┬─────────┘          │  │                        │
│  │  └───────────────────┼────────────────────┘  │                        │
│  │                      │ REST API (FastAPI)    │                        │
│  └──────────────────────┼───────────────────────┘                        │
│                         │                                                │
│                         ▼                                                │
│  ┌──────────────────────────────────────────────┐                        │
│  │       Dashboard & Alert Layer                │                        │
│  │                                              │                        │
│  │  ┌────────────────┐  ┌────────────────────┐  │                        │
│  │  │ Flask Backend  │  │  React Frontend    │  │                        │
│  │  │ + SocketIO     │  │                    │  │                        │
│  │  │                │  │  Health Gauge      │  │                        │
│  │  │ HealthSimulator│  │  Disease Metrics   │  │                        │
│  │  │ AlertManager   │  │  Feature Viz       │  │                        │
│  │  │                │  │  Alert Panel       │  │                        │
│  │  │ agent_advice   │  │  AgentSuggestions  │  │                        │
│  │  │ SocketIO event │  │  Panel             │  │                        │
│  │  └────────────────┘  └────────────────────┘  │                        │
│  └──────────────────────────────────────────────┘                        │
│                                                                          │
│  ┌──────────────────────────────────────────────┐                        │
│  │            PostgreSQL Database                │                        │
│  │  ┌─────────────┐ ┌──────────┐ ┌───────────┐  │                        │
│  │  │ observations│ │advice_log│ │decision   │  │                        │
│  │  │             │ │          │ │_rules     │  │                        │
│  │  └─────────────┘ └──────────┘ └───────────┘  │                        │
│  │  ┌─────────────┐                              │                        │
│  │  │trend_       │                              │                        │
│  │  │snapshots    │                              │                        │
│  │  └─────────────┘                              │                        │
│  └──────────────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────────┘
```

### Internal Data Flow

```
POST /api/v1/tick
      │
      ├─ 1. Compute vital-sign proxies (HR / SpO₂ / RR from feature vector)
      │
      ├─ 2. TrendAnalyzer.add_observation()  ────► observations (PostgreSQL)
      │      └─ Rolling deque (max 20)
      │      └─ Trend: degrading / improving / stable
      │      └─ Slopes: numpy.polyfit linear regression
      │
      ├─ 3. DecisionEngine.evaluate()
      │      └─ In-memory rules (loaded from PostgreSQL at startup)
      │      └─ 10 condition keys, first-match-wins
      │      └─ Falls back to DEFAULT_ADVICE if no rule matches
      │
      ├─ 4. AdviceGenerator.generate()
      │      └─ Template-based advice assembly
      │      └─ Extension point: enrich_with_llm() for future LLM
      │
      ├─ 5. Deduplication check (skip if same rule_id as last tick)
      │
      └─ 6. Persist ───► advice_log + trend_snapshots (PostgreSQL)
```

---

## Project Structure

```
AI_Agent_Single_layer/
├── main.py                   # FastAPI app factory, lifespan, all 11 routes
├── agent_orchestrator.py     # HealthAgent — central orchestrator (NEW)
├── config.py                 # Env-var-driven configuration (11 rules, thresholds)
├── database.py               # SQLAlchemy async engine + session (lazy init)
├── models.py                 # ORM models (4 tables)
├── schemas.py                # Pydantic v2 request/response schemas
├── decision_engine.py        # Rule engine (10 condition keys, DB CRUD)
├── trend_analyzer.py         # Rolling buffer + trend detection + DB persistence
├── advice_generator.py       # Structured advice text assembly
├── requirements.txt          # Python dependencies
├── Dockerfile                # Multi-stage container build
├── docker-compose.yml        # PostgreSQL + app orchestration
├── pytest.ini                # Pytest configuration (async mode)
├── README.md                 # This file
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Async fixtures (SQLite for testing)
│   ├── test_advice_generator.py   # 14 tests
│   ├── test_trend_analyzer.py     # 22 tests
│   ├── test_decision_engine.py    # 18 tests
│   ├── test_agent_orchestrator.py # 11 tests
│   └── test_api.py                # 18 integration tests
└── intel multimodal (dashboard_and_alert_layer)/  ← existing dashboard
```

---

## REST API Reference

All endpoints under `/api/v1/`. OpenAPI docs at **http://localhost:8000/docs**.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/tick` | Process a health observation → returns advice or `null` (dedup) |
| `POST` | `/reset` | Clear all agent state (history, buffer, dedup key) |
| `GET` | `/advice/current` | Latest generated advice |
| `GET` | `/advice/history?n=20` | Recent advice history (1-100) |
| `GET` | `/trends/current` | Current trend summary |
| `GET` | `/trends/history?window=100` | Historical trend snapshots |
| `GET` | `/rules` | List all decision rules |
| `POST` | `/rules` | Create a new decision rule |
| `DELETE` | `/rules/{rule_id}` | Delete a decision rule |
| `GET` | `/status` | Agent status (for system heartbeat) |
| `GET` | `/health` | Health check + DB connectivity |

### Example: Process a health observation

```http
POST /api/v1/tick
Content-Type: application/json

{
  "prediction": 2,
  "subject_id": "subject14",
  "feature_vector": [0.123, -0.456, ...],
  "hr_sim": 95.0,
  "spo2_sim": 94.0
}
```

*Response:*

```json
{
  "matched_rule_id": "rule_003",
  "matched_rule_name": "persistent_unhealthy",
  "severity": "high",
  "possible_condition": "Persistent Unhealthy State — Multiple Possible Causes",
  "advice": "The patient has been in a persistent unhealthy state...",
  "actions": ["notify_physician", "comprehensive_evaluation"],
  "context": {
    "current_prediction": 2,
    "trend": "degrading",
    "unhealthy_ratio": 0.6,
    "healthy_ratio": 0.2,
    "hr_slope": 7.2,
    "spo2_slope": -1.5,
    "rr_slope": 0.01
  },
  "timestamp": "2026-06-23T14:30:00.000Z"
}
```

---

## Database Schema

| Table | Purpose | Key Columns |
|---|---|---|
| `observations` | Every health tick | subject_id, prediction, hr, spo2, rr, feature_vector, timestamp |
| `advice_log` | Every generated advice | matched_rule_id, severity, possible_condition, advice, actions, context, timestamp |
| `decision_rules` | Configurable rules | rule_id (UNIQUE), name, condition(JSONB), priority, enabled |
| `trend_snapshots` | Periodic trend state | trend, unhealthy_ratio, healthy_ratio, hr_slope, spo2_slope, rr_slope, timestamp |

---

## Quick Start

### 1. Install dependencies

```bash
cd "C:\Users\Asus\Desktop\intel multimodal (AI_Agent_Single_layer)"
pip install -r requirements.txt
```

### 2. Run everything (3 terminals)

```bash
# ═══════════════════════════════════════════════════════════════════
# Terminal 1 — Agent Service (FastAPI)
#   Start this FIRST from the project root.
#   → OpenAPI docs:  http://localhost:8000/docs
#   → Health check:  http://localhost:8000/api/v1/health
# ═══════════════════════════════════════════════════════════════════
cd "C:\Users\Asus\Desktop\intel multimodal (AI_Agent_Single_layer)"
python -m uvicorn main:create_app --factory --reload


# ═══════════════════════════════════════════════════════════════════
# Terminal 2 — Dashboard Backend (Flask + SocketIO)
#   Start this SECOND.  Points to the Agent service via AGENT_API_URL.
#   → API:           http://localhost:5000/api/health_state
# ═══════════════════════════════════════════════════════════════════
cd "C:\Users\Asus\Desktop\intel multimodal (AI_Agent_Single_layer)\intel multimodal (dashboard_and_alert_layer)\dashboard_and_alert_layer"
set AGENT_API_URL=http://localhost:8000/api/v1
python run.py


# ═══════════════════════════════════════════════════════════════════
# Terminal 3 — Dashboard Frontend (React)
#   Start this LAST.  Opens the Dashboard UI in your browser.
#   → Dashboard:     http://localhost:3000
# ═══════════════════════════════════════════════════════════════════
cd "C:\Users\Asus\Desktop\intel multimodal (AI_Agent_Single_layer)\intel multimodal (dashboard_and_alert_layer)\dashboard_and_alert_layer\dashboard\frontend"
npm install
npm start
```

> **Note for PowerShell users:** Replace `set AGENT_API_URL=...` with `$env:AGENT_API_URL="http://localhost:8000/api/v1"`

---

## How AI Agent Outputs Appear in the Dashboard UI

When the Dashboard is configured with `AGENT_API_URL`, the `HealthSimulator`
calls `POST /api/v1/tick` on each simulation cycle (every 2 seconds). The
returned advice dict is emitted to the frontend via SocketIO as an
`agent_advice` event.

### What the user sees

The **AgentSuggestionsPanel** component (in the Dashboard sidebar) displays:

| Element | Source | Example |
|---|---|---|
| **Severity badge** | `advice.severity` | 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW |
| **Matched rule** | `advice.matched_rule_name` | "persistent_unhealthy" |
| **Condition** | `advice.possible_condition` | "Persistent Unhealthy State" |
| **Advice text** | `advice.advice` | "The patient has been in a persistent..." |
| **Action chips** | `advice.actions` | "Notify Physician", "Comprehensive Evaluation" |
| **Trend indicator** | `advice.context.trend` | "degrading" / "improving" / "stable" |
| **Timestamp** | `advice.timestamp` | Formatted locale string |
| **Context details** | `advice.context` (toggle) | Prediction, HR slope, SpO₂ slope, ratios |

### Data flow through the system

```
Fusion Layer              Agent Service            Dashboard Backend       Dashboard Frontend
(predictions.csv)         (FastAPI :8000)          (Flask :5000)           (React :3000)
     │                         │                        │                      │
     │  HealthSimulator        │                        │                      │
     │  reads row ─────────────┤                        │                      │
     │                         │  POST /tick            │                      │
     │                         │◄───────────────────────│                      │
     │                         │                        │                      │
     │                         │  advice dict           │                      │
     │                         │────────────────────────┤                      │
     │                         │                        │  agent_advice event  │
     │                         │                        │─────────────────────►│
     │                         │                        │                      │
     │                         │                        │              AgentSuggestions
     │                         │                        │              Panel renders
     │                         │                        │              advice card
```

---

## Configuration Reference

### Agent Service (FastAPI)

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/agent_layer` | PostgreSQL connection |
| `HISTORY_WINDOW_SIZE` | `20` | Max observations in rolling buffer |
| `TREND_WINDOW_SIZE` | `10` | Observations for trend classification |
| `DEGRADING_THRESHOLD` | `0.3` | Unhealthy fraction → "degrading" |
| `IMPROVING_THRESHOLD` | `0.7` | Healthy fraction → "improving" |
| `ADVICE_HISTORY_MAX` | `50` | Max in-memory advice entries |

### Dashboard Integration

| Variable | Default | Description |
|---|---|---|
| `AGENT_API_URL` | `""` (empty = use built-in agent) | FastAPI service URL for external agent |
| `AGENT_ENABLED` | `True` | Enable the built-in Flask agent (ignored when `AGENT_API_URL` is set) |

---

## Decision Rules

Rules are evaluated in priority order (lowest `priority` first). The first rule whose condition fully matches the current context wins.

### Condition Keys

| Key | Comparison | Example |
|---|---|---|
| `current_prediction` | Exact int match (0/1/2) | `2` |
| `trend` | Exact string match | `"degrading"` |
| `hr_trend_min` | HR slope ≥ value | `5.0` |
| `hr_trend_max` | HR slope ≤ value | `-3.0` |
| `spo2_trend_min` | SpO₂ slope ≥ value | `2.0` |
| `spo2_trend_max` | SpO₂ slope ≤ value | `-3.0` |
| `rr_trend_min` | RR slope ≥ value | `0.05` |
| `rr_trend_max` | RR slope ≤ value | `-0.02` |
| `unhealthy_ratio_min` | Unhealthy fraction ≥ value | `0.5` |
| `healthy_ratio_min` | Healthy fraction ≥ value | `0.7` |

Omit keys to skip them. An empty condition `{}` matches everything (catch-all).

---

## Testing

```bash
# Run all tests (80 tests, SQLite-based — no PostgreSQL needed)
pytest tests/ -v

# Run specific test files
pytest tests/test_decision_engine.py -v
pytest tests/test_api.py -v
```

All integration tests use an in-file SQLite database — no external services required.

---

## Backward Compatibility

The existing Dashboard can run with either the built-in Flask agent or the new FastAPI service:

```bash
# Use the built-in agent (backward compatible)
python run.py

# Use the new standalone FastAPI service
AGENT_API_URL=http://localhost:8000/api/v1 python run.py
# or
python run.py --agent-api-url http://localhost:8000/api/v1
```

The advice dict format is identical between the old and new agents.

---

## Migration from Old agent_layer/

The old Flask-based agent at `dashboard_and_alert_layer/agent_layer/` is **deprecated**. All functionality has been migrated to the standalone FastAPI service at this root level.

| Old Module | New Module | Notes |
|---|---|---|
| `agent_layer/health_agent.py` | `agent_orchestrator.py` | HealthAgent orchestrator with async DB |
| `agent_layer/decision_engine.py` | `decision_engine.py` | Same logic + PostgreSQL rule CRUD |
| `agent_layer/trend_analyzer.py` | `trend_analyzer.py` | Same logic + DB persistence |
| `agent_layer/agent_config.py` | `config.py` | Same rules + env-var driven |
| `agent_layer/routes/agent_routes.py` | `main.py` | FastAPI REST (11 endpoints vs 4) |
| — | `advice_generator.py` | NEW: separated advice assembly |

To migrate: set `AGENT_API_URL=http://localhost:8000/api/v1` and start the FastAPI service alongside the Dashboard.

---

## Extensibility — Future Multi-AI Agent Layer

- **Modular HealthAgent** — can be subclassed or composed into a multi-agent orchestrator
- **PostgreSQL-backed rules** — new agent types register rule sets without code changes
- **`enrich_with_llm()`** — clear hook for LLM-based advice enhancement
- **TrendAnalyzer persistence** — historical observations enable cross-agent correlation
- **API versioning** — `/api/v1/` leaves room for `/api/v2/` with multi-agent endpoints
