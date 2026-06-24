#!/usr/bin/env python
"""
Single entry point for the Dashboard & Alerts Layer.

Starts the Flask + SocketIO server, initializes the data store,
feature analyzer, alert manager, and health simulator.

Usage:
    python run.py                          # Start with defaults
    python run.py --port 8080             # Custom port
    python run.py --experiment 2          # Use binary experiment (best accuracy)
    python run.py --speed 0.5             # Half-speed simulation
    python run.py --no-alerts             # Disable alert system
"""
import argparse
import sys
import os

# eventlet is optional — falls back to threading mode
try:
    import eventlet
    eventlet.monkey_patch()
except ImportError:
    pass

# Ensure the package root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FLASK_HOST, FLASK_PORT, AGENT_ENABLED, AGENT_API_URL
from dashboard.backend.data_loader import store
from dashboard.backend.feature_analyzer import FeatureAnalyzer
from dashboard.backend.app import create_app, socketio, set_alert_manager, start_simulator
from alerts.alert_manager import AlertManager
from agent_layer.health_agent import HealthAgent
from agent_layer.routes.agent_routes import set_agent_instance


def main():
    parser = argparse.ArgumentParser(description="Multimodal Health Dashboard & Alerts")
    parser.add_argument("--port", type=int, default=FLASK_PORT,
                        help=f"Server port (default: {FLASK_PORT})")
    parser.add_argument("--host", type=str, default=FLASK_HOST,
                        help=f"Server host (default: {FLASK_HOST})")
    parser.add_argument("--experiment", type=int, default=1,
                        help="Default experiment ID (1=3-class for alerts, 2=binary best)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Simulation speed multiplier (default: 1.0)")
    parser.add_argument("--no-alerts", action="store_true",
                        help="Disable the alert system")
    parser.add_argument("--no-agent", action="store_true",
                        help="Disable the AI Agent Layer")
    parser.add_argument("--agent-api-url", type=str, default=AGENT_API_URL,
                        help="External FastAPI agent service URL (e.g. http://localhost:8000/api/v1)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Flask debug mode")
    args = parser.parse_args()

    print("=" * 60)
    print("  Multimodal Health Monitoring — Dashboard & Alerts Layer")
    print("=" * 60)

    total_steps = 6  # load + pca + alerts + agent + flask + simulator
    step = 0

    # ---- 1. Load data ----
    step += 1
    print(f"\n[{step}/{total_steps}] Loading Fusion Layer outputs...")
    store.load_all()
    store.set_active_experiment(args.experiment)
    exp = store.get_experiment(args.experiment)
    if exp:
        print(f"      Active experiment: Exp {args.experiment} — {exp.get('config_label', '')}")
        print(f"      Test accuracy: {float(exp.get('test_accuracy', 0)) * 100:.1f}%")

    # ---- 2. Fit feature analyzer ----
    step += 1
    print(f"\n[{step}/{total_steps}] Fitting feature analyzer (PCA)...")
    analyzer = FeatureAnalyzer(store.get_feature_matrix())
    analyzer.fit_pca()
    print(f"      Subjects: {len(store.get_unique_subjects())}")

    # ---- 3. Initialize alerts ----
    alert_manager = None
    step += 1
    if not args.no_alerts:
        print(f"\n[{step}/{total_steps}] Initializing alert system...")
        alert_manager = AlertManager()
        set_alert_manager(alert_manager)
        print("      LED: simulated (console + log)")
        print("      Buzzer: simulated (console + log)")
        print("      Telegram: " + ("configured" if alert_manager._telegram.is_configured() else "DRY-RUN mode"))
    else:
        print(f"\n[{step}/{total_steps}] Alert system DISABLED (--no-alerts)")

    # ---- 4. Initialize AI Agent ----
    agent = None
    agent_api_url = args.agent_api_url
    step += 1
    if agent_api_url:
        print(f"\n[{step}/{total_steps}] Using external AI Agent Layer at {agent_api_url}")
        print("      Agent advice will be fetched via HTTP POST /tick")
    elif AGENT_ENABLED and not args.no_agent:
        print(f"\n[{step}/{total_steps}] Initializing AI Agent Layer...")
        agent = HealthAgent(store)
        set_agent_instance(agent)
        print(f"      Decision rules: {agent.get_status()['rules_count']}")
        print(f"      History buffer: {agent.get_status().get('history_size', 0)} observations")
    else:
        print(f"\n[{step}/{total_steps}] AI Agent Layer DISABLED (--no-agent)")

    # ---- 5. Create Flask app ----
    step += 1
    print(f"\n[{step}/{total_steps}] Creating Flask + SocketIO application...")
    app = create_app()
    app.config["DEBUG"] = args.debug

    # ---- 6. Start simulator ----
    step += 1
    total_steps = 7 if agent_api_url else 6  # adjust step count
    print(f"\n[{step}/{total_steps}] Starting health data simulator...")
    sim = start_simulator(agent=agent, agent_api_url=agent_api_url)
    sim.set_speed(args.speed)
    print(f"      Interval: {2.0 / args.speed:.1f}s per sample")

    # ---- Ready ----
    print("\n" + "=" * 60)
    print(f"  Server starting at http://{args.host}:{args.port}")
    print(f"  Dashboard: http://localhost:{args.port}")
    print(f"  API:       http://localhost:{args.port}/api/health_state")
    print("=" * 60)
    print("\nPress Ctrl+C to stop.\n")

    if args.debug:
        socketio.run(app, host=args.host, port=args.port, debug=True)
    else:
        socketio.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
