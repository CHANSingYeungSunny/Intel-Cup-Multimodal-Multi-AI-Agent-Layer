# 多模態流感健康監測與早期預警系統

## Multimodal Influenza Health Monitoring & Early Warning System

**Intel Cup 2025 · Multi-AI Agent Layer v2.0.0**

A complete end-to-end AI pipeline: facial video + cough audio + physiological signals → AI fusion → real-time health classification → AI agent reasoning → web dashboard with alerts and clinical advice.

---

## Full System Architecture

```
                         INPUT DATA
    ┌─────────────────────┼──────────────────────────┐
    ▼                     ▼                          ▼
┌────────────┐    ┌──────────────┐    ┌─────────────────────┐
│  VISION    │    │    AUDIO     │    │   PHYSIOLOGICAL     │
│  Layer     │    │    Layer     │    │   Layer             │
│            │    │              │    │                     │
│ Swin-Tiny  │    │ AST (custom) │    │ iTransformer        │
│ 768-dim    │    │ 128-dim CLS  │    │ 128-dim pooled      │
│ UBFC rPPG  │    │ COUGHVID V3  │    │ BIDMC PPG+Resp      │
└─────┬──────┘    └──────┬───────┘    └──────────┬──────────┘
      │ features         │ features             │ features
      └──────────────────┼─────────────────────┘
                         ▼
             ┌──────────────────────────┐
             │      FUSION LAYER        │
             │                          │
             │ MultimodalFusionEncoder  │
             │ 4-layer Transformer      │
             │ 1024-dim → 256-dim CLS   │
             │ 3-class prediction       │
             └────────────┬─────────────┘
                          │ predictions.csv
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│              DASHBOARD & ALERT LAYER                             │
│                                                                  │
│  Flask Backend (:5000)               React Frontend (:3000)      │
│  ┌────────────────────────┐         ┌─────────────────────────┐ │
│  │ HealthSimulator        │  Socket │ 🩺 Health Gauge         │ │
│  │ (replays CSV, 2s tick) │◄───────►│ 📈 Physio Trends        │ │
│  │ AlertManager           │         │ 🧠 AI Agent Advice      │ │
│  │ (LED·Buzzer·Telegram)  │         │ ⚠️ Alert Status         │ │
│  └───────────┬────────────┘         │ 🔬 Feature Viz (PCA)    │ │
│              │ POST /api/v1/tick    └─────────────────────────┘ │
└──────────────┼──────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│               SINGLE AI AGENT LAYER (v1 · FastAPI)               │
│                                                                  │
│  HealthAgent: TrendAnalyzer → DecisionEngine → AdviceGenerator  │
│  11 REST endpoints · 4 PostgreSQL tables · 83 tests              │
└──────────────────────────────┬───────────────────────────────────┘
                               │ wrapped & extended by
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│               MULTI AI AGENT LAYER (v2 · FastAPI :8000)          │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────────────────────────┐  │
│  │   MCP Server    │  │          3 Skills                     │  │
│  │                 │  │                                      │  │
│  │ Memory·Control  │  │ 🔴 Anomaly Detector (z-score)       │  │
│  │ ·Planning       │  │ 📈 Adv. Trend Analyzer (multi-scale) │  │
│  │                 │  │ 🤖 LLM Advice Gen (GPT/Claude/Local) │  │
│  └─────────────────┘  └──────────────────────────────────────┘  │
│                                                                  │
│  Agent Coordinator: fan-out → parallel execution → fan-in       │
│  21 REST endpoints · 9 PostgreSQL tables · 92 tests             │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PostgreSQL 16                                │
│  Single tables (4) + Multi tables (5) = 9 tables total          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Modality Models

Three separate AI models analyze different types of health data. Each is trained on a public dataset.

### 📹 Vision Layer — Swin-Tiny for rPPG

| Property | Value |
|----------|-------|
| **Model** | Swin-Tiny (28M params) |
| **Input** | RGB facial video frames, 224×224 |
| **Task** | Remote photoplethysmography (rPPG) — heart rate from facial color changes |
| **Dataset** | UBFC 1+2 (42 subjects) |
| **Output** | 768-dim feature vector + 3-class logits |

### 🎤 Audio Layer — Audio Spectrogram Transformer

| Property | Value |
|----------|-------|
| **Model** | Custom lightweight AST (3-layer Transformer, d_model=128) |
| **Input** | Mel-spectrogram (128 mel bands × 192 time frames) |
| **Task** | Cough sound classification and respiratory pattern detection |
| **Dataset** | COUGHVID V3 (~2,800 samples) |
| **Output** | 128-dim CLS embedding + 3-class logits |

### 💓 Physiological Layer — iTransformer

| Property | Value |
|----------|-------|
| **Model** | iTransformerClassifier (attention across channels) |
| **Input** | 4-channel time series (1250 steps): PPG, ECG, HR, SpO₂ |
| **Task** | Cardiovascular and respiratory time-series modeling |
| **Dataset** | BIDMC PPG & Respiration (53 subjects) |
| **Output** | 128-dim pooled embedding + 3-class logits |

### 📦 Dataset Download Links

| Layer | Dataset | Link |
|-------|---------|------|
| 📹 Vision | UBFC 1+2 | [Google Drive](https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk) |
| 🎤 Audio | COUGHVID V3 | [Kaggle](https://www.kaggle.com/datasets/orvile/coughvid-v3/data) |
| 💓 Physiological | BIDMC PPG & Respiration | [PhysioNet](https://physionet.org/content/bidmc/1.0.0/) |

---

## Layer 2: Fusion Layer

| Property | Value |
|----------|-------|
| **Model** | MultimodalFusionEncoder |
| **Input** | Concatenated: Vision(768) + Audio(128) + Physio(128) = **1024-dim** |
| **Architecture** | 4-token Transformer: [CLS, Vision_proj, Audio_proj, Physio_proj], each 256-dim |
| **Transformer** | 4 layers, 8 heads, d_ff=512, GELU, pre-norm |
| **Output** | **256-dim CLS embedding** + 3-class prediction (Healthy / Sub-healthy / Unhealthy) |
| **Training** | 5 experiments, Focal Loss (γ=2.0), best accuracy 77.2% |
| **Output file** | `predictions.csv` — 92 samples with 256-dim feature vectors |

**Label-matched pairing**: Since the three datasets have no shared subjects, the fusion loader pairs samples from different datasets that share the same class label to create training triples.

---

## Layer 3: Dashboard & Alert Layer

### Backend (Flask + SocketIO, port 5000)

| Component | Role |
|-----------|------|
| **HealthSimulator** | Background thread replays `predictions.csv` at 2s intervals. Sends each row to the AI Agent API and broadcasts results via Socket.IO |
| **AlertManager** | Monitors for Unhealthy predictions. Triggers LED (simulated red blink), Buzzer (simulated beep), and Telegram Bot after debouncing |
| **DataStore** | Loads and caches fusion predictions + experiment results |
| **FeatureAnalyzer** | PCA/t-SNE dimensionality reduction on 256-dim embeddings |

### Frontend (React, port 3000)

| Component | Displays |
|-----------|----------|
| **Health Gauge** | Doughnut chart: Healthy / Sub-healthy / Unhealthy counts |
| **AI Agent Suggestions** | Severity badge, condition name, advice text, action chips, trend indicator |
| **Physio Trend Chart** | Multi-line chart: HR, SpO₂, RR interval over time |
| **Cough Waveform** | Respiratory pattern visualization |
| **Disease Classification** | Confusion matrix, per-class precision/recall/F1, accuracy |
| **Feature Visualization** | PCA / t-SNE scatter plot of 256-dim embeddings |
| **Alert Status Panel** | Alert log, LED/buzzer state, Telegram notification status |
| **Experiment Selector** | Dropdown to switch between 5 trained fusion experiments |

### Alerts

- **LED**: Red blinking on Unhealthy detection
- **Buzzer**: Continuous beep during alert state
- **Telegram Bot**: Push notification with subject ID and prediction

---

## Layer 4: Single AI Agent Layer (v1)

### Why it exists

The fusion layer gives a raw number (0/1/2). The AI Agent answers clinical questions:

- **Trend**: Is the patient getting worse, better, or stable?
- **Diagnosis**: Which of 11 clinical patterns does this match?
- **Action**: What should a clinician do?
- **History**: What was the state over time?

### HealthAgent Pipeline

```
POST /api/v1/tick  { prediction, subject_id, feature_vector (256-dim) }
  │
  ├─ 1. Compute vital-sign proxies from fusion embedding → HR, SpO₂, RR
  ├─ 2. TrendAnalyzer: rolling buffer (20 obs) → "degrading" / "improving" / "stable"
  ├─ 3. DecisionEngine: 11 priority rules → matched rule + severity
  ├─ 4. AdviceGenerator: structured advice dict
  ├─ 5. Deduplication: same rule → return null
  └─ 6. PostgreSQL persistence: observations, advice_log, trend_snapshots
```

### Components

| Component | Function |
|-----------|----------|
| **TrendAnalyzer** | Rolling deque (20 obs), trend classification via unhealthy/healthy ratios, numpy.polyfit vital-sign slopes |
| **DecisionEngine** | 11 priority-ordered rules with 10 condition keys, first-match-wins, dynamic CRUD via PostgreSQL |
| **AdviceGenerator** | Template-based advice with severity (high/medium/low), condition name, actions list |

### 11 Decision Rules (examples)

| Rule | Severity | Condition |
|------|----------|-----------|
| rule_001 | HIGH | prediction=2 + degrading + HR↑≥5 bpm → "Possible Influenza" |
| rule_002 | HIGH | prediction=2 + degrading + SpO₂↓≥3% → "Possible Pneumonia / COVID-19" |
| rule_003 | HIGH | prediction=2 + stable + unhealthy_ratio≥0.5 → "Persistent Unhealthy State" |
| rule_004 | MEDIUM | prediction=1 + degrading → "Early Warning — Health Declining" |
| rule_007 | LOW | prediction=0 + improving → "Healthy Recovery" |

### API (11 endpoints under `/api/v1`)

| Method | Path | Purpose |
|--------|------|---------|
| POST | /tick | Submit health observation → get advice |
| POST | /reset | Clear all agent state |
| GET | /advice/current | Latest advice |
| GET | /advice/history | Recent advice entries |
| GET | /trends/current | Current trend summary |
| GET | /trends/history | Historical trend snapshots |
| GET/POST/DELETE | /rules | CRUD decision rules |
| GET | /status | Agent heartbeat |
| GET | /health | Health check + DB ping |

---

## Layer 5: Multi AI Agent Layer (v2)

### What v2 adds on top of v1

| v1 Limitation | v2 Solution |
|---------------|-------------|
| Single trend window (10 obs) | **Advanced Trend Analyzer**: 4 windows (5/10/30/60) + 5-step forecast |
| No anomaly detection | **Anomaly Detector**: rolling z-score (σ=2.5) + persistence alerts |
| Fixed template advice | **LLM Advice Generator**: enrich with GPT-4/Claude clinical reasoning |
| Monolithic — can't extend | **MCP Server**: register external agents via API, fan-out in parallel |
| Single agent | **Agent Coordinator**: runs v1 agent + 3 skills + external agents, aggregates results |

### MCP Server (Memory · Control · Planning)

| Component | Function |
|-----------|----------|
| **Memory Store** | LRU cache (1000 entries) + TTL expiry + PostgreSQL persistence. Cross-agent knowledge sharing |
| **Controller** | Agent registry, fan-out to multiple agents in parallel, fan-in aggregation (majority/average/all) |
| **Planner** | Task decomposition into subtask DAGs, topological sort, workflow session management |

### Three Skills

| Skill | Algorithm | Output |
|-------|-----------|--------|
| **Anomaly Detector** | Rolling z-score per metric (window=30), |z|>2.5 warning, |z|>3.5 critical. Persistence detection | Anomaly events with severity, z-score, expected vs observed |
| **Advanced Trend Analyzer** | 4 rolling windows + linear regression + exponential smoothing forecast (5 steps) | Multi-scale trends, forecast values, cross-scale insight |
| **LLM Advice Generator** | Clinical prompt → OpenAI/Claude/Local LLM → enriched advice text. Preserves structured fields | Clinically reasoned advice paragraph |

### Agent Coordinator

Every tick runs the full pipeline in parallel:

```
process_tick_multi():
  1. v1 HealthAgent.process_tick()      → single-agent advice (always)
  2. AnomalyDetector.update()           → anomaly events
  3. AdvancedTrendAnalyzer.update()     → multi-scale trends + forecast
  4. LLMAdviceGenerator.enrich()        → enriched advice text (opt-in)
  5. MCP Controller.fan_out()           → dispatch to external agents
  6. MCP Controller.fan_in()            → aggregate: consensus severity
  7. Persist to DB                     → anomaly_events, skill_executions
```

### API (+10 new endpoints = 21 total)

**Multi-Agent (`/api/v1/multi`):**

| Method | Path | Purpose |
|--------|------|---------|
| GET | /multi/advice | Aggregated advice from all agents |
| GET | /multi/trends | Multi-scale trends + forecasts |
| GET | /multi/anomalies | Recent anomaly events |
| POST | /multi/skills | Execute skills on demand |
| GET | /multi/agents | Registered agent directory |

**MCP Server (`/api/v1/mcp`):**

| Method | Path | Purpose |
|--------|------|---------|
| GET | /mcp/status | MCP server status |
| POST | /mcp/agents | Register external agent |
| DELETE | /mcp/agents/{agent_id} | Deregister agent |
| POST | /mcp/workflow | Start planned workflow |
| GET | /mcp/workflow/{session_id} | Check workflow progress |

---

## How Layers Connect — End-to-End Data Flow

```
1. MODALITY MODELS produce per-modality predictions.csv:
   Vision:  607 rows × 768-dim features
   Audio:   ~2,800 rows × 128-dim features
   Physio:  ~53 rows × 128-dim features

2. FUSION LOADER pairs by label → 92 triples
   Concatenates to 1024-dim → MultimodalFusionEncoder → 256-dim CLS
   Output: predictions.csv (92 rows, 256-dim feature vectors)

3. DASHBOARD HealthSimulator reads predictions.csv row by row
   Sends POST /api/v1/tick → Multi AI Agent Layer

4. AI AGENT processes:
   a. Computes vitals from 256-dim embedding
   b. TrendAnalyzer classifies trend
   c. DecisionEngine matches rule
   d. AdviceGenerator builds structured advice
   e. Anomaly Detector checks for vital-sign spikes
   f. Trend Analyzer runs multi-scale + forecast
   g. LLM enriches advice (if configured)
   Returns: AdviceResponse JSON

5. DASHBOARD BACKEND receives response → Socket.IO "agent_advice" event

6. DASHBOARD FRONTEND renders AgentSuggestionsPanel:
   🔴 HIGH — rule_002 "severe_respiratory_distress"
   "Possible Respiratory Infection / Pneumonia"
   [Notify Physician] [Check SpO₂] [Respiratory Assessment]
   ▼ Context: trend=degrading, HR↑+5.2, SpO₂↓-3.1
```

---

## Single Agent vs Multi Agent

| | Single Agent (v1) | Multi Agent (v2) |
|---|---|---|
| **Trend analysis** | 1 window (10 obs) | 4 windows (5/10/30/60) + forecast |
| **Anomaly detection** | ❌ | ✅ z-score + persistence |
| **Advice quality** | Fixed templates | Templates + optional LLM enrichment |
| **External agents** | ❌ | ✅ MCP register + fan-out |
| **Workflow planning** | ❌ | ✅ DAG + topological sort |
| **API endpoints** | 11 | 21 (all v1 + 10 new) |
| **DB tables** | 4 | 9 (all v1 + 5 new) |
| **Tests** | 83 | 92 |
| **Backward compatible** | — | ✅ Same /tick response format |

---

## Quick Start

### One Command (Windows)

```powershell
.\start_all.ps1
```

Opens 3 PowerShell windows: Agent `:8000` · Dashboard `:5000` · Frontend `:3000`.

If PowerShell blocks the script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

First-time only (if no `node_modules`):
```powershell
cd "intel multimodal (AI_Agent_Single_layer)\intel multimodal (dashboard_and_alert_layer)\dashboard_and_alert_layer\dashboard\frontend"
npm install
```

### One Command (Linux)

```bash
./start_all.sh
```

### 🎮 Demo Mode

```powershell
$env:DEMO_MODE_ENABLED = "true"
.\start_all.ps1
```

### Verify it's working

```bash
curl http://localhost:8000/api/v1/health
# → {"status":"ok","version":"2.0.0"}
```

Open **http://localhost:3000**

---

## Configuration Reference

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_TITLE` | `Multi AI Agent Layer` | API title |
| `APP_VERSION` | `2.0.0` | API version |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Database URL |
| `CORS_ORIGINS` | `*` | CORS allowed origins |

### MCP

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_ENABLED` | `true` | Enable MCP server |
| `MCP_MEMORY_MAX_ENTRIES` | `1000` | LRU cache size |
| `MCP_DEFAULT_TTL_SECONDS` | `3600` | Memory TTL |

### Skills

| Variable | Default | Description |
|----------|---------|-------------|
| `SKILLS_ENABLED` | `anomaly_detector,advanced_trend_analyzer,llm_advice_generator` | Active skills |
| `ANOMALY_DETECTOR_ZSCORE_THRESHOLD` | `2.5` | Z-score threshold |
| `ANOMALY_DETECTOR_WINDOW` | `30` | Rolling window size |
| `ADVANCED_TREND_WINDOWS` | `5,10,30,60` | Multi-scale windows |
| `FORECAST_HORIZON` | `5` | Forecast steps |

### LLM (opt-in)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `none` | `none` / `openai` / `claude` / `local` |
| `LLM_API_KEY` | *(empty)* | API key |
| `LLM_MODEL` | `gpt-4o` | Model ID |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |

---

## Tests

```bash
cd Multi_AI_Agent_layer
pytest tests/ -v
# 92 passed — uses SQLite (no PostgreSQL needed)
```

---

## Team

| Member | Role |
|--------|------|
| **Justin** | Hardware integration, DK-2500 edge deployment, sensors |
| **Sunny** | Vision/Audio models, Fusion Transformer, Dashboard frontend, Multi AI Agent Layer |
| **Baileys** | Physiological modeling, data processing, edge deployment |

## GitHub Repositories

| Layer | URL |
|-------|-----|
| **Multi AI Agent (v2)** | [Intel-Cup-Multimodal-Multi-AI-Agent-Layer](https://github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Multi-AI-Agent-Layer) |
| **Single AI Agent (v1)** | [Intel-Cup-Multimodal-Single-AI-Agent-Layer](https://github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Single-AI-Agent-Layer) |
