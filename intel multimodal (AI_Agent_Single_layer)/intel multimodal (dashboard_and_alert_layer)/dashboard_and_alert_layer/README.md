# Dashboard & Alerts Layer — Multimodal Health Monitoring

Real-time visualization and alert system for the Multimodal Fusion Layer that classifies health states (Healthy / Sub-healthy / Unhealthy) from fused vision, audio, and physiological signals.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FUSION LAYER (existing)                  │
│  predictions.csv (92 samples, 256-dim features)             │
│  experiment_results_with_accuracy.csv (5 experiments, 33 cols)│
└─────────────────────┬───────────────────────────────────────┘
                      │ loaded at startup
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 DASHBOARD & ALERTS LAYER                     │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  Flask Backend   │  │  Alert System     │                │
│  │  (app.py)        │  │  (alert_manager)  │                │
│  │                  │  │                   │                │
│  │  REST API ───────┼──┤  LED Simulator    │                │
│  │  SocketIO ───────┼──┤  Buzzer Simulator │                │
│  │  HealthSimulator─┼──┤  Telegram Bot     │                │
│  └────────┬─────────┘  └──────────────────┘                │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  React Frontend  │                                       │
│  │  (Chart.js)      │                                       │
│  │                  │                                       │
│  │  Health Gauge    │                                       │
│  │  Cough Curves    │                                       │
│  │  Physio Trends   │                                       │
│  │  Disease Metrics │                                       │
│  │  Feature Viz     │                                       │
│  │  Alert Panel     │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
dashboard_and_alert_layer/
├── README.md                     ← This file
├── requirements.txt              ← Python dependencies
├── config.py                     ← Centralized configuration
├── run.py                        ← Single entry point
├── fusion_layer/
│   └── outputs/                  ← Fusion Layer outputs (input)
│       ├── predictions.csv       ← 92 test samples, 256-dim features
│       └── experiment_results_with_accuracy.csv
├── dashboard/
│   ├── backend/
│   │   ├── app.py                ← Flask + SocketIO server
│   │   ├── data_loader.py        ← CSV loading & caching
│   │   ├── feature_analyzer.py   ← PCA, t-SNE, signal generation
│   │   ├── health_simulator.py   ← Streaming prediction simulator
│   │   └── routes/               ← REST API endpoints
│   │       ├── health_routes.py
│   │       ├── signal_routes.py
│   │       ├── disease_routes.py
│   │       ├── feature_routes.py
│   │       └── experiment_routes.py
│   └── frontend/                 ← React application
│       ├── package.json
│       ├── public/index.html
│       └── src/
│           ├── App.js
│           ├── App.css
│           ├── components/       ← 9 React components
│           ├── hooks/            ← useSocket, useApi
│           └── utils/            ← API helpers, chart config
├── alerts/
│   ├── alert_manager.py          ← State machine orchestrator
│   ├── led_simulator.py          ← Console LED simulation
│   ├── buzzer_simulator.py       ← Console buzzer simulation
│   ├── telegram_bot.py           ← Telegram push notifications
│   └── alert_rules.py            ← Threshold configuration
└── tests/
    ├── test_data_loader.py
    ├── test_feature_analyzer.py
    ├── test_routes.py
    └── test_alerts.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health_state` | GET | Aggregated health state counts, percentages, experiment accuracy |
| `/api/cough_curve?subject=<id>` | GET | Simulated respiratory waveform (50 points over 2s) |
| `/api/physio_trend?subject=<id>` | GET | Multi-vital-sign trends (HR, SpO2, RR Interval) across windows |
| `/api/disease_classification` | GET | Per-class metrics, confusion matrix, all predictions |
| `/api/feature_viz?method=pca&components=2` | GET | PCA or t-SNE scatterplot coordinates |
| `/api/experiments` | GET | List of all 5 experiments with summary metrics |
| `/api/experiments/<id>` | GET | Full experiment data including training curves |
| `/api/health_history` | GET | All 92 predictions with filenames and features |

### WebSocket Events

**Server → Client:**
- `health_update` — New simulated prediction every 2s
- `alert_triggered` — Fires when Unhealthy state detected
- `alert_cleared` — Fires when system returns to healthy
- `system_status` — Periodic heartbeat (every 10s)
- `experiment_changed` — Confirms experiment switch

**Client → Server:**
- `set_experiment {experiment_id}` — Switch active experiment
- `set_simulation_speed {speed}` — Adjust streaming rate
- `pause_simulation` — Toggle pause/resume
- `request_alert_test` — Manually trigger test alert

## Alert System

### State Machine

```
NORMAL ──(prediction = Unhealthy)──▶ ALERTING
  ▲                                      │
  └──(5 consecutive healthy)─────────────┘
```

### Alert Channels

| Channel | Implementation | Behavior |
|---|---|---|
| **LED** | `led_simulator.py` | ANSI red background + blink in console; writes to `alerts/alert_log.txt` |
| **Buzzer** | `buzzer_simulator.py` | Console `BEEP` patterns; optional `winsound.Beep()` on Windows |
| **Telegram** | `telegram_bot.py` | Sends 🚨 alert message when configured; dry-run mode otherwise |

### Telegram Configuration

```bash
# Set these environment variables to enable real Telegram alerts:
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"

# Then start normally:
python run.py
```

Without these variables, Telegram operates in **dry-run mode** — messages are printed to the console and logged to `alerts/alert_log.txt`.

## Quick Start

### 1. Install Python dependencies

```bash
cd dashboard_and_alert_layer
pip install -r requirements.txt
```

### 2. Start the Flask backend + alert system

```bash
# Default: 3-class mode (Experiment 1) for alert testing
python run.py

# Use Experiment 2 (binary, 77.2% accuracy — best model):
python run.py --experiment 2

# Custom port and speed:
python run.py --port 8080 --speed 2.0

# Disable alerts:
python run.py --no-alerts
```

### 3. Start the React frontend (in a separate terminal)

```bash
cd dashboard_and_alert_layer/dashboard/frontend
npm install
npm start
```

The React dev server runs on `http://localhost:3000` and proxies API calls to `http://localhost:5000`.

### 4. Or build for production (Flask serves static files)

```bash
cd dashboard_and_alert_layer/dashboard/frontend
npm run build

# Then Flask serves everything at http://localhost:5000
cd ../..
python run.py
```

### 5. Run tests

```bash
cd dashboard_and_alert_layer
python -m pytest tests/ -v
```

## Input

- **`predictions.csv`**: 92 test samples from the Fusion Layer with columns:
  - `filename` — Composite ID: `v:{vision_id}|a:{audio_id}|p:{physio_id}`
  - `prediction` — Model output: 0=Healthy, 1=Sub-healthy, 2=Unhealthy
  - `label` — Ground truth
  - `feature_vector` — 256-dim CLS token embedding (JSON array)

- **`experiment_results_with_accuracy.csv`**: 5 experiment configurations with 33 columns including accuracy, macro/weighted F1, confusion matrix, and training curves.

## Output

1. **Web Dashboard** — Dark-themed React SPA with:
   - Doughnut gauge: current health state distribution
   - Line charts: simulated respiratory waveforms and multi-vital trends
   - Pie chart + confusion matrix: disease classification results
   - PCA/t-SNE scatterplot: feature space visualization
   - Alert log panel: real-time alert timeline with LED/buzzer indicators

2. **Alert System** — Triggers on Unhealthy prediction:
   - LED simulation: ANSI console colors
   - Buzzer simulation: console patterns (+ Windows beep)
   - Telegram bot: push notifications (when configured)

3. **Alert Log** — `alerts/alert_log.txt` records all alert events with timestamps

## Methodology

- **Flask + Flask-SocketIO**: Serves REST API and WebSocket for real-time streaming
- **HealthSimulator**: Background thread cycles through predictions.csv to simulate a real-time monitoring device
- **FeatureAnalyzer**: PCA reduces 256-dim features to 2D/3D for visualization; maps component scores to simulated physiological signals (respiratory rate, heart rate, SpO2)
- **Chart.js**: Declarative React charting with doughnut, line, scatter, and pie charts
- **AlertManager**: State machine with debounce (3s minimum alert, 5 consecutive healthy to clear, 30s Telegram cooldown)

### Note on Simulated Signals

Since raw modality data (vision, audio, physiological waveforms) is not included in the Fusion Layer outputs, the cough/respiration curves and physiological trends are **simulated approximations** derived from PCA of the 256-dim fusion embeddings. They are intended for dashboard visualization demonstration and should not be interpreted as clinical measurements.

## Experiment Comparison

| Exp | Mode | Accuracy | Macro F1 | Notes |
|---|---|---|---|---|
| 1 | 3-class | 71.7% | 78.3% | **Default for alerts** (has Unhealthy samples) |
| **2** | **Binary** | **77.2%** | **77.0%** | **Best accuracy** (Healthy vs Unhealthy) |
| 3 | 3-class, bs=128 | 71.7% | 78.1% | Larger batch size |
| 4 | 3-class, freeze | 66.3% | 73.2% | Frozen encoder layers |
| 5 | 3-class, lr=1e-4 | 66.3% | 73.2% | Lower learning rate |
