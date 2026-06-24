# Intel Multimodal — Health Monitoring System

A multimodal AI pipeline for health state classification (Healthy / Sub-healthy / Unhealthy) from vision, audio, and physiological signals, with a real-time dashboard and alert system.

## Project Overview

```
┌──────────────────────────────────────────────────────────────┐
│                 MULTIMODAL FUSION PIPELINE                    │
│                                                              │
│  Vision Layer        Audio Layer        Physiological Layer  │
│  (Swin-Tiny)         (AST)              (iTransformer)       │
│  768-dim features    128-dim features   128-dim features     │
│       │                   │                    │             │
│       └───────────────────┼────────────────────┘             │
│                           ▼                                  │
│                   Fusion Layer                               │
│              (Transformer Encoder)                           │
│              256-dim embeddings                              │
│              3-class predictions                             │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              DASHBOARD & ALERTS LAYER                         │
│                                                              │
│  Flask + React Dashboard    │    Alert System                 │
│  • Health state gauge       │    • LED simulator              │
│  • Cough/respiration curves │    • Buzzer simulator           │
│  • Physiological trends     │    • Telegram bot               │
│  • Disease classification   │                                 │
│  • Feature visualization    │                                 │
└──────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
├── README.md                                    ← This file
│
├── intel multimodal (fusion layer)/             ← Fusion pipeline
│   ├── Fusion-Layer/                            ← Multimodal fusion model
│   ├── intel multimodal (vision layer)/         ← Vision (Swin-Tiny)
│   ├── intel multimodal (audio layer)/          ← Audio (AST)
│   └── intel multimodal (physiological layer)/  ← Physiological (iTransformer)
│
└── dashboard_and_alert_layer/                   ← Dashboard & Alerts
    ├── README.md                                ← Detailed docs
    ├── requirements.txt
    ├── config.py
    ├── run.py                                   ← Single entry point
    ├── fusion_layer/outputs/                    ← Fusion Layer output CSVs
    ├── dashboard/
    │   ├── backend/                             ← Flask + SocketIO API
    │   └── frontend/                            ← React + Chart.js UI
    ├── alerts/                                  ← LED, Buzzer, Telegram
    └── tests/                                   ← Unit tests (38 passing)
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- Git

### 1. Clone & install

```bash
git clone https://github.com/CHANSingYeungSunny/Intel-Cup-Multimodal-Dashboard-and-Alerts-Layer.git
cd "intel multimodal (dashboard_and_alert_layer)"

# Python dependencies
cd dashboard_and_alert_layer
pip install -r requirements.txt

# React dependencies
cd dashboard/frontend
npm install
cd ../..
```

### 2. Run the Dashboard

```bash
# Terminal 1 — Flask backend + alert system
python run.py

# Terminal 2 — React frontend (dev mode)
cd dashboard/frontend
npm start
```

- **Dashboard**: http://localhost:3000 (dev) or http://localhost:5000 (production)
- **API**: http://localhost:5000/api/health_state

### 3. Production build

```bash
cd dashboard/frontend
npm run build
cd ../..
python run.py
# Everything served from http://localhost:5000
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

## Key Results

| Layer | Model | Params | Best Accuracy |
|---|---|---|---|
| Vision | Swin-Tiny | 27.5M | 47.4% (pseudo-labels) |
| Audio | AST | ~1M | 74.9% |
| Physiological | iTransformer | 575K | 67.7% |
| **Fusion** | **Transformer** | **2.4M** | **77.2% (binary)** |

## Alert System

When the model predicts **Unhealthy** (class 2), the alert system activates:

- **LED**: Red blinking indicator (console ANSI + `alert_log.txt`)
- **Buzzer**: Beep patterns (console + `winsound.Beep()` on Windows)
- **Telegram**: Push notification (set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` env vars)

Without Telegram credentials, alerts run in dry-run mode (console output only).

## Configuration

All settings in `dashboard_and_alert_layer/config.py`:

| Setting | Default | Description |
|---|---|---|
| `FLASK_PORT` | 5000 | Server port |
| `DEFAULT_EXPERIMENT_ID` | 1 | 1 = 3-class (alerts), 2 = binary (best accuracy) |
| `SIMULATION_INTERVAL_SECONDS` | 2.0 | Seconds between prediction updates |
| `TELEGRAM_BOT_TOKEN` | `""` | Set via env var for real Telegram alerts |
| `TELEGRAM_CHAT_ID` | `""` | Set via env var for real Telegram alerts |

## Experiment Modes

| ID | Mode | Accuracy | Macro F1 | Use Case |
|---|---|---|---|---|
| 1 | 3-class | 71.7% | 78.3% | Default — alert testing (has Unhealthy samples) |
| **2** | **Binary** | **77.2%** | **77.0%** | **Best accuracy** |
| 3 | 3-class, bs=128 | 71.7% | 78.1% | Larger batch size |
| 4 | 3-class, freeze | 66.3% | 73.2% | Frozen encoder |
| 5 | 3-class, lr=1e-4 | 66.3% | 73.2% | Lower learning rate |

Switch experiments via the dashboard dropdown or `python run.py --experiment 2`.

## Note

Edge deployment (DK-2500 hardware) is not included — LED and buzzer are simulated. Raw modality data (waveforms) is not stored in the Fusion Layer outputs, so cough/physio curves are PCA-derived approximations for visualization purposes.
