# Multi AI Agent Layer

**Version 2.0.0** — Evolves the Single AI Agent Layer into a multi-agent system with MCP (Memory-Control-Planning) orchestration, specialized skills modules, and agent coordination.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Dashboard Layer (Flask + React)                  │
│  ┌─────────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │ AgentSuggestionsPanel│  │ PhysioTrendChart │  │ AlertStatusPanel   │  │
│  └─────────┬───────────┘  └────────┬─────────┘  └─────────┬──────────┘  │
│            │                       │                       │             │
│            └───────────────────────┼───────────────────────┘             │
│                                    │ REST + Socket.IO                    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Multi AI Agent Layer (FastAPI :8000)                 │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        MCP Server                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │   │
│  │  │ Memory Store  │  │  Controller  │  │        Planner           │ │   │
│  │  │              │  │              │  │                          │ │   │
│  │  │ • LRU Cache  │  │ • Register   │  │ • Task Decomposition    │ │   │
│  │  │ • TTL Expiry │  │ • Fan-out    │  │ • Dependency Resolution │ │   │
│  │  │ • PG Persist │  │ • Fan-in     │  │ • Workflow Sessions     │ │   │
│  │  │ • Knowledge  │  │ • Health     │  │ • Topological Sort      │ │   │
│  │  │   Sharing    │  │   Checks     │  │                          │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Agent Coordinator                              │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │   │
│  │  │ HealthAgent │  │  Skills    │  │  External  │                  │   │
│  │  │  (Single)   │  │  Engine    │  │  Agents    │                  │   │
│  │  └────────────┘  └────────────┘  └────────────┘                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                          Skills                                    │   │
│  │  ┌───────────────────┐ ┌──────────────────────┐ ┌──────────────┐ │   │
│  │  │ Anomaly Detector  │ │ Advanced Trend       │ │ LLM Advice   │ │   │
│  │  │                   │ │ Analyzer              │ │ Generator    │ │   │
│  │  │ • Rolling Z-Score │ │ • Multi-Scale Trends │ │ • OpenAI     │ │   │
│  │  │ • Persistence     │ │ • Linear+Exp         │ │ • Claude     │ │   │
│  │  │   Detection       │ │   Smoothing Forecast │ │ • Local LLM  │ │   │
│  │  └───────────────────┘ └──────────────────────┘ └──────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                Single AI Agent Layer (wrapped)                     │   │
│  │  HealthAgent → TrendAnalyzer → DecisionEngine → AdviceGenerator   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PostgreSQL 16                                    │
│  ┌──────────────────────┐  ┌───────────────────────────────────────┐   │
│  │  Single-layer tables  │  │  Multi-layer tables                    │   │
│  │  • observations       │  │  • agent_registry                     │   │
│  │  • advice_log         │  │  • anomaly_events                     │   │
│  │  • decision_rules     │  │  • skill_executions                   │   │
│  │  • trend_snapshots    │  │  • mcp_state                          │   │
│  │                        │  │  • multi_agent_sessions               │   │
│  └──────────────────────┘  └───────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Multi_AI_Agent_layer/
├── run.py                          # Launcher — python run.py
├── __init__.py                     # Package init, version 2.0.0
├── _ensure_imports.py              # Forces Multi modules into sys.modules
├── base.py                         # MultiBase (separate DeclarativeBase)
├── config.py                       # Re-exports Single config + Multi keys
├── database.py                     # MultiBase engine + Single DB re-exports
├── models.py                       # 5 new ORM models + Single model re-exports
├── schemas.py                      # Multi schemas + Single schema re-exports
├── main.py                         # FastAPI app factory (21 endpoints)
├── mcp_server.py                   # MCP orchestration (Memory/Control/Planning)
├── agent_coordinator.py            # Multi-agent fan-out/fan-in coordinator
├── skills/
│   ├── __init__.py
│   ├── anomaly_detector.py         # Rolling z-score anomaly detection
│   ├── advanced_trend_analyzer.py  # Multi-timescale trend + forecasting
│   └── llm_advice_generator.py     # Configurable LLM advice enrichment
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── PROPOSAL.md                     # English proposal (groupmate-ready)
├── PROPOSAL_CN.md                  # Chinese proposal (繁體中文)
├── README.md
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_mcp_server.py          # 26 tests
│   ├── test_skills.py              # 23 tests
│   ├── test_agent_coordinator.py   # 13 tests
│   └── test_api.py                 # 30 integration tests
└── intel multimodal (AI_Agent_Single_layer)/   ← Single AI Agent Layer (v1)
    ├── main.py                     # FastAPI app (create_app, 11 endpoints)
    ├── agent_orchestrator.py       # HealthAgent (TrendAnalyzer + DecisionEngine + AdviceGenerator)
    ├── config.py, database.py      # Single-layer configuration + DB
    ├── models.py, schemas.py       # 4 ORM tables + Pydantic schemas
    ├── tests/                      # 83 tests
    └── intel multimodal (dashboard_and_alert_layer)/   ← Dashboard + Fusion
        ├── dashboard_and_alert_layer/
        │   ├── run.py              # Dashboard launcher (Flask :5000)
        │   └── dashboard/
        │       ├── backend/        # Flask + SocketIO backend
        │       └── frontend/       # React frontend (:3000)
        └── intel multimodal (fusion layer)/
            ├── Fusion-Layer/       # MultimodalFusionEncoder (256-dim)
            ├── intel multimodal (vision layer)/   # Swin-Tiny rPPG (768-dim)
            ├── intel multimodal (audio layer)/    # AST (128-dim)
            └── intel multimodal (physiological layer)/ # iTransformer (128-dim)
```

---

## Single AI Agent Layer (v1) — Included in this repo

The **Single AI Agent Layer** is the foundation that the Multi layer builds upon.
It is included as a nested directory and provides:

| Component | Role |
|-----------|------|
| **HealthAgent** | Central orchestrator wrapping TrendAnalyzer + DecisionEngine + AdviceGenerator |
| **TrendAnalyzer** | Rolling buffer (20 obs), degrading/improving/stable classification, numpy.polyfit slopes |
| **DecisionEngine** | 11 priority-ordered clinical decision rules, first-match-wins, dynamic CRUD |
| **AdviceGenerator** | Template-based structured health advice (severity + condition + actions) |
| **11 REST endpoints** | `/api/v1/tick`, `/advice`, `/trends`, `/rules`, `/status`, `/health` |
| **PostgreSQL** | 4 tables: observations, advice_log, decision_rules, trend_snapshots |
| **83 tests** | Unit + integration (aiosqlite) |

The Multi layer **wraps** the Single layer — it imports `HealthAgent` and calls
`process_tick()` on every request, then adds anomaly detection, multi-scale
trends, LLM enrichment, and multi-agent coordination on top. All 11 Single-layer
endpoints are **preserved unchanged** under `/api/v1`.

> **GitHub:** [Intel-Cup-Multimodal-Single-AI-Agent-Layer](https://github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Single-AI-Agent-Layer)

---

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 16 (or use the Docker Compose file included)
- Node.js 18+ (for Dashboard frontend)

### 1. Install dependencies

```bash
cd "C:\Users\Asus\Desktop\intel multimodal (Multi_AI_Agent_layer)"
python -m pip install -r requirements.txt
```

> **Tip:** You can also use the Single AI Agent Layer's existing `.venv` — it already
> contains all required packages plus `openai` and `anthropic`.

### 2. Run everything (3 terminals)

```bash
# ═══════════════════════════════════════════════════════════════════
# Terminal 1 — Backend : Multi AI Agent Layer (FastAPI)
#   Start this FIRST from the project root.
#   → OpenAPI docs:  http://localhost:8000/docs
#   → Health check:  http://localhost:8000/api/v1/health
# ═══════════════════════════════════════════════════════════════════
cd "C:\Users\Asus\Desktop\intel multimodal (Multi_AI_Agent_layer)"
python run.py


# ═══════════════════════════════════════════════════════════════════
# Terminal 2 — Dashboard Backend (Flask + SocketIO)
#   Start this SECOND.  Points to the Multi Agent service.
#   → API:           http://localhost:5000/api/health_state
# ═══════════════════════════════════════════════════════════════════
cd "C:\Users\Asus\Desktop\intel multimodal (Multi_AI_Agent_layer)\intel multimodal (AI_Agent_Single_layer)\intel multimodal (dashboard_and_alert_layer)\dashboard_and_alert_layer"
set AGENT_API_URL=http://localhost:8000/api/v1
python run.py --no-agent


# ═══════════════════════════════════════════════════════════════════
# Terminal 3 — Dashboard Frontend (React)
#   Start this LAST.  Opens the Dashboard UI in your browser.
#   → Dashboard:     http://localhost:3000
# ═══════════════════════════════════════════════════════════════════
cd "C:\Users\Asus\Desktop\intel multimodal (Multi_AI_Agent_layer)\intel multimodal (AI_Agent_Single_layer)\intel multimodal (dashboard_and_alert_layer)\dashboard_and_alert_layer\dashboard\frontend"
npm install
npm start
```

> **Note for PowerShell users:** Replace `set AGENT_API_URL=...` with
> `$env:AGENT_API_URL="http://localhost:8000/api/v1"`

### Option: Docker Compose

```bash
cd "C:\Users\Asus\Desktop\intel multimodal (Multi_AI_Agent_layer)"
docker compose up --build
```

The backend starts on **http://localhost:8000** and PostgreSQL 16 is included.
OpenAPI docs at **http://localhost:8000/docs**.

---

## Full Integration Check

Once all three terminals are running, verify end-to-end connectivity:

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Send a test tick
curl -X POST http://localhost:8000/api/v1/tick \
  -H "Content-Type: application/json" \
  -d '{"prediction": 2, "subject_id": "test_subject", "hr_sim": 95, "spo2_sim": 94, "rr_sim": 0.72}'

# Get multi-agent aggregated advice
curl http://localhost:8000/api/v1/multi/advice

# Check anomalies
curl http://localhost:8000/api/v1/multi/anomalies

# List registered agents
curl http://localhost:8000/api/v1/multi/agents

# MCP status
curl http://localhost:8000/api/v1/mcp/status
```

---

## API Reference

### Single-Agent Endpoints (Backward Compatible)

All under **`/api/v1`** — identical to the Single AI Agent Layer:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tick` | Process a health observation |
| `POST` | `/reset` | Reset all agent state |
| `GET` | `/advice/current` | Latest generated advice |
| `GET` | `/advice/history?n=20` | Recent advice history |
| `GET` | `/trends/current` | Current trend summary |
| `GET` | `/trends/history?window=100` | Historical trend snapshots |
| `GET` | `/rules` | List all decision rules |
| `POST` | `/rules` | Create a new decision rule |
| `DELETE` | `/rules/{rule_id}` | Delete a decision rule |
| `GET` | `/status` | Lightweight agent status |
| `GET` | `/health` | Health check (+ DB ping) |

### Multi-Agent Endpoints

All under **`/api/v1/multi`**:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/multi/advice` | Aggregated advice from all agents |
| `GET` | `/multi/trends` | Multi-scale trends + forecasts |
| `GET` | `/multi/anomalies?n=20` | Recent anomaly events |
| `POST` | `/multi/skills` | Execute skills on demand |
| `GET` | `/multi/agents` | List registered agents |

### MCP Server Endpoints

All under **`/api/v1/mcp`**:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/mcp/status` | MCP server status |
| `POST` | `/mcp/agents` | Register a new agent |
| `DELETE` | `/mcp/agents/{agent_id}` | Deregister an agent |
| `POST` | `/mcp/workflow` | Start a multi-agent workflow |
| `GET` | `/mcp/workflow/{session_id}` | Get workflow status |

**Total: 21 endpoints.**

---

## Dashboard Integration

The Dashboard's **AgentSuggestionsPanel.jsx** displays AI agent advice from the backend. The Multi-AI Agent Layer is fully compatible:

1. **Single-agent advice** → displayed in the main advice card (severity badge, rule name, condition, advice text, action chips, trend indicator)
2. **Multi-agent advice** → available via `/api/v1/multi/advice` (aggregated contributions, consensus severity)
3. **Anomalies** → available via `/api/v1/multi/anomalies` (z-score events, persistence alerts)
4. **Multi-scale trends** → available via `/api/v1/multi/trends` (forecasts, cross-scale insights)

### What the Dashboard Displays

| Component | Data Source | Multi-Layer Enhancement |
|-----------|------------|------------------------|
| AgentSuggestionsPanel | `/api/v1/advice/current` + Socket.IO | LLM-enriched advice text |
| PhysioTrendChart | `/api/v1/trends/current` | Multi-scale forecasts |
| AlertStatusPanel | Socket.IO `alert_triggered` | Anomaly events supplement alerts |
| SystemStatusBar | `/api/v1/status` + Socket.IO | Agent count, MCP status |

---

## Configuration Reference

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_TITLE` | `Multi AI Agent Layer` | API title in OpenAPI docs |
| `APP_VERSION` | `2.0.0` | API version |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/agent_layer` | Single-layer DB |
| `MULTI_AGENT_DB_URL` | same as `DATABASE_URL` | Multi-layer DB |
| `API_V1_PREFIX` | `/api/v1` | Single-agent route prefix |
| `MULTI_API_PREFIX` | `/api/v1/multi` | Multi-agent route prefix |
| `MCP_API_PREFIX` | `/api/v1/mcp` | MCP route prefix |
| `CORS_ORIGINS` | `*` | CORS allowed origins |

### MCP Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_ENABLED` | `true` | Enable MCP server |
| `MCP_MEMORY_MAX_ENTRIES` | `1000` | LRU cache size |
| `MCP_DEFAULT_TTL_SECONDS` | `3600` | Default TTL for memory entries |

### Skills Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SKILLS_ENABLED` | `anomaly_detector,advanced_trend_analyzer,llm_advice_generator` | Active skills |
| `ANOMALY_DETECTOR_ZSCORE_THRESHOLD` | `2.5` | Z-score threshold |
| `ANOMALY_DETECTOR_WINDOW` | `30` | Rolling window size |
| `ADVANCED_TREND_WINDOWS` | `5,10,30,60` | Multi-scale windows |
| `FORECAST_HORIZON` | `5` | Forecast steps ahead |

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `none` | `none` / `openai` / `claude` / `local` |
| `LLM_API_KEY` | *(empty)* | API key for LLM service |
| `LLM_MODEL` | `gpt-4o` | Model identifier |
| `LLM_MAX_TOKENS` | `512` | Max response tokens |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_LOCAL_ENDPOINT` | `http://localhost:11434/v1/chat/completions` | Local LLM URL |

---

## Migration Guide: Single → Multi Agent

### What Changed

| Aspect | Single AI Agent Layer (v1.0.0) | Multi AI Agent Layer (v2.0.0) |
|--------|-------------------------------|-------------------------------|
| Architecture | One HealthAgent with 3 sub-components | HealthAgent + MCP + Skills + Coordinator |
| Endpoints | 11 under `/api/v1` | 11 (same) + 10 new |
| Database | 4 tables | 4 (same) + 5 new |
| Advice | Template-based only | Template + optional LLM enrichment |
| Trends | Single-window (10 obs) | Multi-scale (5/10/30/60) + forecasts |
| Anomalies | Not available | Rolling z-score + persistence detection |
| Agent Model | Single monolithic | Multi-agent with fan-out/fan-in |
| External Agents | Not supported | Register via MCP, fan-out via HTTP |
| Workflows | Not available | MCP Planner with task decomposition |

### Backward Compatibility

All existing `/api/v1/` endpoints accept and return the same JSON schemas:

- `POST /api/v1/tick` → same `TickRequest` → same `AdviceResponse`
- `GET /api/v1/advice/current` → same `AdviceResponse`
- `GET /api/v1/trends/current` → same `TrendResponse`
- All other endpoints unchanged

**No Dashboard changes are required.** Set `AGENT_API_URL=http://localhost:8000/api/v1` and the Dashboard works as before.

### Enabling LLM Enrichment

```bash
# OpenAI
export LLM_BACKEND="openai"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"

# Claude
export LLM_BACKEND="claude"
export LLM_API_KEY="sk-ant-..."
export LLM_MODEL="claude-sonnet-4-6-20250514"

# Local (Ollama)
export LLM_BACKEND="local"
export LLM_MODEL="llama3"
export LLM_LOCAL_ENDPOINT="http://localhost:11434/v1/chat/completions"
```

---

## Running Tests

```bash
# Activate the virtual environment
source .venv/Scripts/activate

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_mcp_server.py -v
pytest tests/test_skills.py -v
pytest tests/test_agent_coordinator.py -v
pytest tests/test_api.py -v

# Run with coverage
python -m pip install pytest-cov
pytest tests/ -v --cov=. --cov-report=term-missing
```

Tests use **aiosqlite** (in-memory SQLite) — no PostgreSQL required.

---

## Adding a New Skill

1. Create a new class in `skills/`:

```python
# skills/my_skill.py
class MySkill:
    def __init__(self, ...):
        ...
    def update(self, data):
        ...
```

2. Register it in `skills/__init__.py`

3. Add it to `SKILLS_ENABLED` env var

4. Add a handler in `agent_coordinator.py` `execute_skills()`

---

## Development

```bash
# Watch mode (auto-reload)
python run.py

# Run tests on change
pytest tests/ -v --looponfail
```

---

## License

Part of the Intel Cup Multimodal Health Monitoring System.
