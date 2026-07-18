"""
Flask + Flask-SocketIO application factory.

Serves the React frontend build and provides REST + WebSocket APIs.
"""
import os
import time
import threading
from datetime import datetime, timezone

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from config import FLASK_DEBUG
from dashboard.backend.data_loader import store
from dashboard.backend.routes.health_routes import health_bp
from dashboard.backend.routes.media_routes import media_bp
from dashboard.backend.routes.signal_routes import signal_bp
from dashboard.backend.routes.disease_routes import disease_bp
from dashboard.backend.routes.feature_routes import feature_bp
from dashboard.backend.routes.experiment_routes import experiment_bp
from agent_layer.routes.agent_routes import agent_bp
from dashboard.backend.health_simulator import HealthSimulator

socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")
simulator = None
_alert_manager = None
_agent_instance = None
_start_time = None


def create_app():
    """Create and configure the Flask application."""
    global _start_time
    _start_time = time.time()

    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "build"),
        static_url_path="",
    )
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "multimodal-health-dashboard")
    app.config["DEBUG"] = FLASK_DEBUG

    CORS(app)
    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    app.register_blueprint(health_bp)
    app.register_blueprint(media_bp)
    app.register_blueprint(signal_bp)
    app.register_blueprint(disease_bp)
    app.register_blueprint(feature_bp)
    app.register_blueprint(experiment_bp)
    app.register_blueprint(agent_bp)

    # Demo mode — only when explicitly enabled (env var)
    if os.environ.get("DEMO_MODE_ENABLED", "").lower() in ("1", "true", "yes"):
        from dashboard.backend.routes.demo_routes import demo_bp

        app.register_blueprint(demo_bp)
        print("[app] Demo Mode ENABLED — /api/demo/* routes registered")

    @socketio.on("connect")
    def on_connect():
        emit("system_status", _build_system_status())

    @socketio.on("disconnect")
    def on_disconnect():
        pass

    @socketio.on("set_experiment")
    def on_set_experiment(data):
        exp_id = int(data.get("experiment_id", 1))
        ok = store.set_active_experiment(exp_id)
        exp = store.get_experiment(exp_id) if ok else None
        emit("experiment_changed", {
            "experiment_id": exp_id,
            "success": ok,
            "label": exp.get("config_label", "") if exp else "",
            "mode": exp.get("label_mode", "") if exp else "",
        })

    @socketio.on("set_simulation_speed")
    def on_set_speed(data):
        speed = float(data.get("speed", 1.0))
        if simulator:
            simulator.set_speed(speed)
        emit("simulation_status", {"speed": speed})

    @socketio.on("pause_simulation")
    def on_pause():
        if simulator:
            new_state = not simulator.is_paused()
            simulator.set_paused(new_state)
            emit("simulation_status", {"paused": new_state})

    @socketio.on("request_alert_test")
    def on_alert_test():
        if _alert_manager:
            event = _alert_manager.trigger_test_alert()
            if event:
                emit("alert_triggered", event)

    @socketio.on("request_agent_advice")
    def on_request_agent_advice():
        if _agent_instance:
            advice = _agent_instance.get_current_advice()
            if advice:
                emit("agent_advice", advice)

    @app.route("/")
    def serve_index():
        build_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")
        if os.path.exists(os.path.join(build_dir, "index.html")):
            return send_from_directory(build_dir, "index.html")
        return (
            "<h1>Dashboard Backend Running</h1>"
            "<p>React frontend not built yet. Run <code>npm run build</code> in "
            "<code>dashboard/frontend/</code>.</p>"
            "<p>API endpoints available at:</p><ul>"
            "<li><a href='/api/health_state'>/api/health_state</a></li>"
            "<li><a href='/api/live_sensors'>/api/live_sensors</a></li>"
            "<li><a href='/api/live_summary'>/api/live_summary</a></li>"
            "<li><a href='/api/camera_snapshot'>/api/camera_snapshot</a></li>"
            "<li><a href='/api/microphone_level'>/api/microphone_level</a></li>"
            "<li><a href='/api/experiments'>/api/experiments</a></li>"
            "<li><a href='/api/disease_classification'>/api/disease_classification</a></li>"
            "<li><a href='/api/feature_viz'>/api/feature_viz</a></li>"
            "<li><a href='/api/cough_curve'>/api/cough_curve?subject=...</a></li>"
            "<li><a href='/api/physio_trend'>/api/physio_trend?subject=...</a></li>"
            "</ul>"
        )

    @app.route("/<path:path>")
    def serve_static(path):
        build_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")
        if os.path.exists(os.path.join(build_dir, path)):
            return send_from_directory(build_dir, path)
        return send_from_directory(build_dir, "index.html")

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def server_error(e):
        return {"error": "Internal server error"}, 500

    return app



def set_alert_manager(mgr):
    global _alert_manager
    _alert_manager = mgr



def set_agent_instance(agent):
    """Store the HealthAgent singleton for SocketIO handlers and simulator."""
    global _agent_instance
    _agent_instance = agent
    from agent_layer.routes.agent_routes import set_agent_instance as route_set_agent
    route_set_agent(agent)



def start_simulator(agent=None, agent_api_url=None):
    global simulator
    simulator = HealthSimulator(store, socketio, _alert_manager, agent=agent, agent_api_url=agent_api_url)
    simulator.start()
    threading.Thread(target=_emit_system_status, daemon=True).start()
    return simulator



def _emit_system_status():
    while True:
        time.sleep(10)
        try:
            socketio.emit("system_status", _build_system_status())
        except Exception:
            pass



def _build_system_status():
    uptime = time.time() - _start_time if _start_time else 0
    sim_stats = simulator.get_stats() if simulator else {}
    alert_status = _alert_manager.get_status() if _alert_manager else {}
    agent_status = _agent_instance.get_status() if _agent_instance else {}

    return {
        "uptime": round(uptime, 1),
        "predictions_processed": sim_stats.get("predictions_processed", 0),
        "alerts_triggered": sim_stats.get("alerts_triggered", 0),
        "simulation_running": sim_stats.get("running", False),
        "simulation_paused": sim_stats.get("paused", False),
        "alert_active": alert_status.get("active", False),
        "connection_ok": True,
        "agent_enabled": agent_status.get("enabled", False),
        "agent_condition": agent_status.get("latest_condition", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

