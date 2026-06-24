# 多模態流感健康監測與早期預警系統

## Multimodal Influenza Health Monitoring & Early Warning System

**Intel Cup 2025 · Multi-AI Agent Layer v2.0.0**

---

## What This Project Does

This system takes **facial video, cough audio, and physiological signals** from a person, runs each through a dedicated AI model, fuses the results, and classifies them as **Healthy (0), Sub-healthy (1), or Unhealthy (2)**.

On top of this, a **Multi-AI Agent Layer** watches the data stream in real time — detecting trends, spotting anomalies, evaluating 11 medical decision rules, and generating natural-language health advice (optionally enriched by an LLM like GPT-4 or Claude). Everything is displayed on a **web dashboard** with live charts, AI recommendations, and alert indicators.

> **In short:** Sensor data → AI models classify health → AI Agent reasons about it → Dashboard shows what's happening and what to do.

---

## System Architecture

```
                       INPUT DATA
    ┌─────────────────────┼──────────────────────────┐
    ▼                     ▼                          ▼
┌────────────┐    ┌──────────────┐    ┌─────────────────────┐
│  VISION    │    │    AUDIO     │    │   PHYSIOLOGICAL     │
│            │    │              │    │                     │
│ Facial     │    │ Cough sounds │    │ 4-channel time      │
│ video      │    │ (webm)       │    │ series (1250 steps) │
│ (UBFC rPPG)│    │ (COUGHVID)   │    │ (BIDMC PPG+ECG)     │
│            │    │              │    │                     │
│ Swin-Tiny  │    │ AST (custom) │    │ iTransformer        │
│ 768-dim    │    │ 128-dim CLS  │    │ 128-dim pooled      │
└─────┬──────┘    └──────┬───────┘    └──────────┬──────────┘
      │ features         │ features             │ features
      └──────────────────┼─────────────────────┘
                         ▼
             ┌──────────────────────────┐
             │       FUSION LAYER       │
             │                          │
             │ MultimodalFusionEncoder  │
             │ 4-token Transformer      │
             │ (CLS + Vision + Audio    │
             │  + Physiological)        │
             │                          │
             │ d_model=256, 4 layers    │
             │ 8 heads, GELU            │
             │                          │
             │ Output: 256-dim CLS      │
             │    + 3-class prediction  │
             └────────────┬─────────────┘
                          │ predictions.csv (92 samples)
                          │
                          ▼
┌══════════════════════════════════════════════════════════════════┐
║              AI AGENT LAYER  (FastAPI :8000)                     ║
║                                                                  ║
║  ┌────────────────────────────────────────────────────────────┐ ║
║  │           SINGLE AI AGENT (v1 — always runs)               │ ║
║  │                                                            │ ║
║  │  POST /api/v1/tick  { prediction, subject_id,             │ ║
║  │                        feature_vector (256-dim) }          │ ║
║  │       │                                                    │ ║
║  │       ├─ Compute vital-sign proxies from fusion embedding  │ ║
║  │       ├─ TrendAnalyzer: rolling buffer (20 obs)           │ ║
║  │       │   → "degrading" / "improving" / "stable"          │ ║
║  │       │   → HR/SpO₂/RR slopes via numpy.polyfit           │ ║
║  │       ├─ DecisionEngine: 11 priority-ordered rules        │ ║
║  │       │   → matched rule + severity + condition            │ ║
║  │       └─ AdviceGenerator: structured advice dict           │ ║
║  │                                                            │ ║
║  │  Persists to PostgreSQL (4 tables)                        │ ║
║  └────────────────────────────────────────────────────────────┘ ║
║                           │                                      ║
║                           ▼                                      ║
║  ┌────────────────────────────────────────────────────────────┐ ║
║  │           MULTI-AI EXTENSIONS (v2 — adds on top)           │ ║
║  │                                                            │ ║
║  │  ┌──────────────────┐  ┌────────────────────────────────┐  │ ║
║  │  │   MCP SERVER     │  │        3 SKILLS                │  │ ║
║  │  │                  │  │                                │  │ ║
║  │  │ Memory: LRU + PG │  │ 🔴 Anomaly Detector            │  │ ║
║  │  │ Control: Fan-out │  │    Rolling z-score (σ=2.5)     │  │ ║
║  │  │   + Fan-in       │  │    + persistence detection     │  │ ║
║  │  │ Planning: DAG    │  │                                │  │ ║
║  │  │   workflows      │  │ 📈 Adv. Trend Analyzer         │  │ ║
║  │  └──────────────────┘  │    4 windows (5/10/30/60)      │  │ ║
║  │                        │    + linear+exp smoothing       │  │ ║
║  │  ┌──────────────────┐  │    forecast (5 steps ahead)     │  │ ║
║  │  │   COORDINATOR    │  │                                │  │ ║
║  │  │                  │  │ 🤖 LLM Advice Generator        │  │ ║
║  │  │ Runs v1 agent +  │  │    OpenAI / Claude / Local     │  │ ║
║  │  │ all 3 skills +   │  │    Clinical reasoning prompt   │  │ ║
║  │  │ external agents  │  │    Preserves structured fields │  │ ║
║  │  │ in parallel       │  │                                │  │ ║
║  │  └──────────────────┘  └────────────────────────────────┘  │ ║
║  │                                                            │ ║
║  │  21 REST endpoints · PostgreSQL: 9 tables total            │ ║
║  │  92 tests · Backward-compatible with v1                    │ ║
║  └────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════╝
                          │  HTTP REST + Socket.IO
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│               DASHBOARD & ALERT LAYER                            │
│                                                                  │
│  Flask Backend (:5000)               React Frontend (:3000)      │
│  ┌────────────────────────┐         ┌─────────────────────────┐ │
│  │ HealthSimulator        │         │ 🩺 Health Gauge          │ │
│  │ (replays predictions   │ Socket  │ 📈 Physio Trend Chart    │ │
│  │  .csv, 2s per tick)    │◄───────►│ 🫁 Cough Waveform        │ │
│  │                        │         │ 🧠 AI Agent Suggestions  │ │
│  │ AlertManager           │         │ ⚠️ Alert Status Panel    │ │
│  │ (LED·Buzzer·Telegram)  │         │ 🔬 Feature Viz (PCA)     │ │
│  └────────────────────────┘         └─────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Modality Models + Fusion

Three separate AI models, each trained on a different public dataset. There are **no shared subjects** across datasets — the fusion layer uses label-matched pairing to create training triples.

### Vision Layer — Swin-Tiny for rPPG

| Property | Value |
|----------|-------|
| **Model** | Swin-Tiny (28M params, `microsoft/swin-tiny-patch4-window7-224`) |
| **Input** | RGB facial video frames, 224×224 |
| **What it measures** | Remote photoplethysmography (rPPG) — heart rate from facial color changes |
| **Dataset** | UBFC rPPG (42 subjects, ~607 samples) |
| **Output** | 768-dim feature vector + 3-class logits |

### Audio Layer — Audio Spectrogram Transformer

| Property | Value |
|----------|-------|
| **Model** | Custom lightweight AST (3-layer Transformer, d_model=128) |
| **Input** | Mel-spectrogram (128 mel bands × 192 time frames) |
| **What it detects** | Cough sound patterns, respiratory audio features |
| **Dataset** | COUGHVID (~2,800 samples) |
| **Output** | 128-dim CLS embedding + 3-class logits |

### Physiological Layer — iTransformer

| Property | Value |
|----------|-------|
| **Model** | iTransformerClassifier (attention across channels, not timesteps) |
| **Input** | 4-channel time series (1250 steps): PPG, ECG, HR, SpO₂ derived signals |
| **What it models** | Cardiovascular and respiratory patterns |
| **Dataset** | BIDMC PPG & Respiration (53 subjects) |
| **Output** | 128-dim pooled embedding + 3-class logits |

### Fusion Layer — MultimodalFusionEncoder

| Property | Value |
|----------|-------|
| **Input** | Concatenated features: Vision(768) + Audio(128) + Physio(128) = **1024-dim** |
| **Architecture** | 4-token sequence: [CLS, Vision_proj, Audio_proj, Physio_proj], each 256-dim after Linear projection |
| **Transformer** | 4 layers, 8 heads, d_ff=512, GELU, pre-norm |
| **Output** | **256-dim CLS embedding** + 3-class prediction |
| **Training** | 5 experiments, Focal Loss (γ=2.0), best: Exp 2 (77.2% accuracy, 0.775 weighted F1) |
| **Output file** | `predictions.csv` — 92 samples, each with 256-dim feature vector |

---

## Layer 2: Single AI Agent (v1)

### Why it exists

The fusion layer outputs a raw number (0/1/2). That's not clinically useful alone. The AI Agent answers:

- **Trend**: Is the patient getting worse, better, or staying the same?
- **Diagnosis**: Which of 11 clinical patterns does this match?
- **Action**: What should a clinician do right now?
- **History**: What was the state 5, 10, 30 minutes ago?

### How it works — the `process_tick()` pipeline

```
POST /api/v1/tick
{
  "prediction": 2,
  "subject_id": "subject14",
  "feature_vector": [0.12, -0.45, 0.78, ...]  // 256-dim fusion embedding
}

                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: Vital-sign proxy computation                    │
│   Splits 256-dim vector into thirds, maps to:           │
│   HR  = 75 + mean(first_third)  × 10  →  95 bpm        │
│   SpO₂ = 97 − |mean(mid_third)| × 5  →  93%            │
│   RR  = 0.85 + mean(last_third) × 0.2 → 0.72 s         │
├─────────────────────────────────────────────────────────┤
│ Step 2: TrendAnalyzer                                   │
│   Rolling deque (max 20 obs)                            │
│   unhealthy_ratio ≥ 0.3 → "degrading"                   │
│   healthy_ratio ≥ 0.7   → "improving"                   │
│   Otherwise             → "stable"                      │
│   numpy.polyfit slopes: HR=+5.2, SpO₂=-3.1, RR=-0.04   │
├─────────────────────────────────────────────────────────┤
│ Step 3: DecisionEngine (11 rules, first-match-wins)     │
│   Rule matched: rule_002 "severe_respiratory_distress"  │
│   Condition: prediction=2 + degrading + SpO₂↓≥3%        │
│   Severity: HIGH                                        │
├─────────────────────────────────────────────────────────┤
│ Step 4: AdviceGenerator                                 │
│   Possible condition: "Respiratory Infection/Pneumonia" │
│   Advice: "Urgent respiratory evaluation advised..."    │
│   Actions: [notify_physician, check_spo2, ...]          │
├─────────────────────────────────────────────────────────┤
│ Step 5: Deduplication                                   │
│   Same rule as last tick? → return null (unchanged)     │
├─────────────────────────────────────────────────────────┤
│ Step 6: PostgreSQL persistence                           │
│   → observations, advice_log, trend_snapshots           │
└─────────────────────────────────────────────────────────┘
```

### API Response (what the Dashboard receives)

```json
{
  "matched_rule_id": "rule_002",
  "matched_rule_name": "severe_respiratory_distress",
  "severity": "high",
  "possible_condition": "Possible Respiratory Infection / Pneumonia",
  "advice": "Urgent respiratory evaluation advised. Unhealthy classification combined with declining oxygen saturation (≥3% drop) and a degrading trend may indicate pneumonia, bronchitis, or COVID-19. Check SpO₂ with a pulse oximeter immediately.",
  "actions": ["notify_physician", "check_spo2", "respiratory_assessment"],
  "context": {
    "trend": "degrading",
    "unhealthy_ratio": 0.45,
    "healthy_ratio": 0.10,
    "hr_slope": 5.2,
    "spo2_slope": -3.1,
    "rr_slope": -0.04
  },
  "timestamp": "2026-06-24T00:16:32.154Z"
}
```

---

## Layer 3: Multi AI Agent (v2)

### What v2 adds on top of v1

The Single Agent does rule-based advice well. But it has gaps:

| v1 Gap | v2 Solution |
|---------|-------------|
| Can't detect sudden vital-sign spikes | **Anomaly Detector**: z-score on rolling window, flags critical outliers |
| Only one trend window (10 obs) | **Advanced Trend Analyzer**: 4 windows (5/10/30/60) + 5-step forecast |
| Advice is fixed template text | **LLM Advice Generator**: enriches with GPT-4/Claude clinical reasoning |
| Monolithic — can't add new agents | **MCP Server**: register external agents via API, fan-out in parallel |
| No cross-agent coordination | **Agent Coordinator**: runs all components, aggregates results |

### MCP Server (Memory · Control · Planning)

| Component | What it does | Example |
|-----------|-------------|---------|
| **Memory Store** | Shared key-value store with TTL + PostgreSQL backing | `"patient_14:last_critical_hr" = 142` (expires 3600s) |
| **Controller** | Agent registry, fan-out requests to multiple agents in parallel, fan-in with aggregation (majority/average/all) | Send tick to HealthAgent + AnomalyDetector + TrendAnalyzer simultaneously |
| **Planner** | Breaks goals into subtask DAGs, topological sort, workflow sessions | "monitor health and detect anomalies" → 3 subtasks → execute in dependency order |

### Three Skills

**1. Anomaly Detector** — catches things the rule engine misses:
- Rolling z-score per metric (HR, SpO₂, RR, prediction), window=30
- |z| > 2.5 → warning, |z| > 3.5 → critical
- Persistence detection: 3+ consecutive unhealthy predictions when history was healthy
- Example: *"HR z-score = +3.8 (CRITICAL). Observed 142 bpm vs expected 82 bpm."*

**2. Advanced Trend Analyzer** — multi-scale view:
- 4 simultaneous rolling windows: 5, 10, 30, 60 observations
- Trend classification at each scale
- Linear regression + exponential smoothing forecast (5 steps ahead)
- Cross-scale insight: *"Short-term degrading against long-term improving backdrop — may be transient."*

**3. LLM Advice Generator** — clinical reasoning layer (optional, opt-in):
- System prompt constrains the LLM to act as a clinical decision support assistant
- Preserves structured fields (severity, actions, rule_id)
- Only enriches the `advice` text field
- Supports: OpenAI (gpt-4o), Anthropic Claude, local (Ollama)
- When not configured: passes through template advice unchanged (zero-cost)

### Agent Coordinator

Every tick runs this pipeline:

```
process_tick_multi():
  1. v1 HealthAgent.process_tick()         → single-agent advice (always)
  2. AnomalyDetector.update()              → anomaly events list
  3. AdvancedTrendAnalyzer.update()        → multi-scale trends + forecast
  4. LLMAdviceGenerator.enrich()           → enriched advice text (if configured)
  5. MCP Controller.fan_out()              → dispatch to external agents (HTTP)
  6. MCP Controller.fan_in()               → aggregate: consensus severity
  7. Persist to DB                         → anomaly_events + skill_executions
```

---

## Layer 4: Dashboard

### How data reaches the Dashboard

The **HealthSimulator** is a background thread that replays `predictions.csv` (92 fusion samples) at 2-second intervals. It does **not** use real sensors — this is a simulation replay for demo purposes. On each tick:

1. Reads a row from predictions.csv
2. Sends it to the AI Agent via `POST /api/v1/tick`
3. Receives the AdviceResponse
4. Emits `agent_advice` via Socket.IO to the frontend
5. Also emits `health_update` with the raw prediction data

### What the Dashboard shows

| Component | What you see | Data source |
|-----------|-------------|-------------|
| **Health Gauge** | Doughnut: Healthy / Sub-healthy / Unhealthy counts | REST `/api/health_state` |
| **AI Agent Suggestions** | Severity badge (🔴🟡🟢), condition name, advice text, action chips, trend indicator, collapsible context (vital slopes, ratios) | REST `/api/agent_advice` + Socket.IO `agent_advice` |
| **Physio Trend Chart** | Multi-line: HR, SpO₂, RR interval over time | REST `/api/physio_trend` |
| **Cough Waveform** | Respiratory pattern visualization | REST `/api/cough_curve` |
| **Disease Classification** | Confusion matrix, per-class precision/recall/F1, accuracy | REST `/api/disease_classification` |
| **Feature Visualization** | PCA/t-SNE scatter plot of 256-dim fusion embeddings | REST `/api/feature_viz` |
| **Alert Status** | Alert log, LED (red blink), Buzzer (beep), Telegram notification on Unhealthy | Socket.IO `alert_triggered` |
| **Experiment Selector** | Switch between 5 trained fusion experiments | REST `/api/experiments` |

---

## End-to-End: One Prediction's Journey

```
1. FUSION produces predictions.csv row:
   filename: "v:UBFC2/subject14/...|a:COUGHVID/uuid|p:bidmc19_..."
   prediction: 2 (Unhealthy)
   label: 2 (ground truth)
   feature_vector: "[0.12, -0.45, 0.78, ...]"  (256 floats as JSON string)

2. HEALTH SIMULATOR reads row, sends to Agent:
   POST http://localhost:8000/api/v1/tick
   { prediction: 2, subject_id: "subject14", feature_vector: [0.12, -0.45, ...] }

3. AI AGENT processes:
   a. Vitals from embedding: HR=95, SpO₂=93%, RR=0.72s
   b. Trend: "degrading" (HR↑, SpO₂↓)
   c. Rule matched: rule_002 → HIGH severity → "Possible Pneumonia"
   d. Anomaly Detector: HR z-score +3.8 → CRITICAL alert
   e. Trend Analyzer: short-term degrading, long-term stable
   f. LLM (if enabled): enriches advice with clinical context
   Returns: AdviceResponse JSON

4. FLASK BACKEND receives response → Socket.IO "agent_advice" event

5. REACT FRONTEND renders AgentSuggestionsPanel:
   🔴 HIGH — Rule: severe_respiratory_distress
   Possible Respiratory Infection / Pneumonia
   "Urgent respiratory evaluation advised..."
   [Notify Physician] [Check SpO₂] [Respiratory Assessment]
   ▼ Context: trend=degrading, HR↑5.2, SpO₂↓3.1
```

---

## Single Agent vs Multi Agent

| | Single Agent (v1) | Multi Agent (v2) |
|---|---|---|
| **Core function** | Rule-based health advice | Same + anomaly detection + multi-scale trends + LLM enrichment |
| **Trend windows** | 1 (10 obs) | 4 (5/10/30/60) + 5-step forecast |
| **Anomaly detection** | ❌ | ✅ z-score + persistence alerts |
| **Advice source** | Fixed templates | Templates + optional AI-enriched clinical reasoning |
| **External agents** | ❌ | ✅ Register via MCP, fan-out via HTTP |
| **Workflow planning** | ❌ | ✅ Task decomposition + dependency DAG |
| **API endpoints** | 11 | 21 (all v1 preserved + 10 new) |
| **DB tables** | 4 | 9 (all v1 preserved + 5 new) |
| **Tests** | 83 | 92 |
| **Dashboard compatibility** | ✅ | ✅ (backward compatible — same /tick response format) |

**The Multi Agent wraps the Single Agent.** It does everything v1 does, plus more. The Dashboard calls the same `/api/v1/tick` and gets the same response shape — no Dashboard changes needed.

---

## How to Run (3 Terminals)

```bash
# ═══════════════════════════════════════════════════════════════════
# Terminal 1 — AI Agent Backend (FastAPI :8000)
#   Docs: http://localhost:8000/docs
#   Health: http://localhost:8000/api/v1/health
# ═══════════════════════════════════════════════════════════════════
cd Multi_AI_Agent_layer
python run.py

# ═══════════════════════════════════════════════════════════════════
# Terminal 2 — Dashboard Backend (Flask + SocketIO :5000)
#   API: http://localhost:5000/api/health_state
# ═══════════════════════════════════════════════════════════════════
cd Multi_AI_Agent_layer\"intel multimodal (AI_Agent_Single_layer)"\"intel multimodal (dashboard_and_alert_layer)"\dashboard_and_alert_layer
$env:AGENT_API_URL="http://localhost:8000/api/v1"
python run.py --no-agent

# ═══════════════════════════════════════════════════════════════════
# Terminal 3 — Dashboard Frontend (React :3000)
# ═══════════════════════════════════════════════════════════════════
cd Multi_AI_Agent_layer\"intel multimodal (AI_Agent_Single_layer)"\"intel multimodal (dashboard_and_alert_layer)"\dashboard_and_alert_layer\dashboard\frontend
npm install    # first time only
npm start
```

Open **http://localhost:3000** — the Dashboard shows live health data streaming from the simulator, with AI Agent advice in the sidebar.

### Smoke test

```bash
curl http://localhost:8000/api/v1/health
# → {"status":"ok","version":"2.0.0","db_connected":false}

curl -X POST http://localhost:8000/api/v1/tick \
  -H "Content-Type: application/json" \
  -d '{"prediction":2,"subject_id":"demo","feature_vector":[0.12,-0.45,0.78]}'
# → structured advice JSON

curl http://localhost:8000/api/v1/multi/agents
# → ["health_agent","anomaly_detector","advanced_trend_analyzer","llm_advice_generator"]

curl http://localhost:8000/api/v1/mcp/status
# → {"memory_entries":0,"control_active":true,"planning_queue_size":0,...}
```

---

## API Reference (21 Endpoints)

### Single-Agent — `/api/v1`

| Method | Path | Purpose |
|--------|------|---------|
| POST | /tick | Submit health observation → get advice |
| POST | /reset | Clear all agent state |
| GET | /advice/current | Latest advice |
| GET | /advice/history?n=20 | Recent advice entries |
| GET | /trends/current | Current trend summary |
| GET | /trends/history?window=100 | Trend snapshots over time |
| GET | /rules | List all 11 decision rules |
| POST | /rules | Create custom rule |
| DELETE | /rules/{rule_id} | Delete a rule |
| GET | /status | Agent heartbeat |
| GET | /health | Health check + DB ping |

### Multi-Agent — `/api/v1/multi`

| Method | Path | Purpose |
|--------|------|---------|
| GET | /multi/advice | Aggregated advice from all agents |
| GET | /multi/trends | Multi-scale trends + forecasts |
| GET | /multi/anomalies?n=20 | Recent anomaly events |
| POST | /multi/skills | Execute skills on demand |
| GET | /multi/agents | Registered agent directory |

### MCP Server — `/api/v1/mcp`

| Method | Path | Purpose |
|--------|------|---------|
| GET | /mcp/status | MCP server status |
| POST | /mcp/agents | Register external agent |
| DELETE | /mcp/agents/{agent_id} | Deregister agent |
| POST | /mcp/workflow | Start planned workflow |
| GET | /mcp/workflow/{session_id} | Check workflow progress |

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

## Repositories

| Layer | GitHub |
|-------|--------|
| Single AI Agent (v1) | [github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Single-AI-Agent-Layer](https://github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Single-AI-Agent-Layer) |
| Multi AI Agent (v2) | [github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Multi-AI-Agent-Layer](https://github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Multi-AI-Agent-Layer) |
