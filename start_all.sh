#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PC_IP="$(hostname -I | awk '{print $1}')"

if [ ! -d "$VENV_DIR" ] && [ -d "$ROOT_DIR/venv" ]; then
    VENV_DIR="$ROOT_DIR/venv"
fi

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

DASHBOARD_BACKEND_DIR="$ROOT_DIR/intel multimodal (AI_Agent_Single_layer)/intel multimodal (dashboard_and_alert_layer)/dashboard_and_alert_layer"
FRONTEND_DIR="$DASHBOARD_BACKEND_DIR/dashboard/frontend"

activate_venv() {
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    else
        echo "ERROR: venv not found at $VENV_DIR"
        exit 1
    fi
}

wait_for_port() {
    local port="$1"
    local name="$2"
    local log_file="$3"

    echo "Checking $name on port $port..."

    for i in {1..30}; do
        if ss -ltn | grep -q ":$port "; then
            echo "$name is running on port $port"
            return 0
        fi
        sleep 1
    done

    echo ""
    echo "ERROR: $name did not start on port $port"
    echo "Last 40 lines of log:"
    tail -40 "$log_file"
    echo ""
    cleanup
    exit 1
}

cleanup() {
    echo ""
    echo "Stopping Intel Multimodal services..."

    kill "$AGENT_PID" "$DASHBOARD_PID" "$FRONTEND_PID" 2>/dev/null || true

    # Extra cleanup for child processes
    pkill -f "python run.py --no-agent" 2>/dev/null || true
    pkill -f "npm start" 2>/dev/null || true
    pkill -f "react-scripts start" 2>/dev/null || true

    echo "Stopped."
}

trap cleanup SIGINT SIGTERM

echo "Starting Intel Multimodal system..."
echo "Root: $ROOT_DIR"
echo "Logs: $LOG_DIR"
echo ""

# Terminal 1 — Multi AI Agent Backend :8000
(
    cd "$ROOT_DIR" || exit 1
    activate_venv
    echo "Starting Multi AI Agent Backend..."
    python -u run.py
) > "$LOG_DIR/agent_backend.log" 2>&1 &

AGENT_PID=$!
wait_for_port 8000 "FastAPI Multi Agent Backend" "$LOG_DIR/agent_backend.log"

# Terminal 2 — Dashboard Backend :5000
(
    cd "$DASHBOARD_BACKEND_DIR" || exit 1
    activate_venv
    export AGENT_API_URL="http://localhost:8000/api/v1"
    echo "Starting Flask Dashboard Backend..."
    python -u run.py --no-agent
) > "$LOG_DIR/dashboard_backend.log" 2>&1 &

DASHBOARD_PID=$!
wait_for_port 5000 "Flask Dashboard Backend" "$LOG_DIR/dashboard_backend.log"

# Terminal 3 — React Frontend :3000
(
    cd "$FRONTEND_DIR" || exit 1
    echo "Starting React Dashboard Frontend..."

    if [ ! -d "node_modules" ]; then
        echo "node_modules not found. Running npm install..."
        npm install
    fi

    npm start
) > "$LOG_DIR/dashboard_frontend.log" 2>&1 &

FRONTEND_PID=$!
wait_for_port 3000 "React Frontend" "$LOG_DIR/dashboard_frontend.log"

echo ""
echo "All services are running."
echo ""
echo "FastAPI docs local:        http://localhost:8000/docs"
echo "Dashboard backend local:   http://localhost:5000"
echo "Dashboard frontend local:  http://localhost:3000"
echo ""
echo "Open from your PC/browser:"
echo "FastAPI docs:              http://$PC_IP:8000/docs"
echo "Dashboard backend:         http://$PC_IP:5000"
echo "Dashboard frontend:        http://$PC_IP:3000"
echo ""
echo "Logs:"
echo "  $LOG_DIR/agent_backend.log"
echo "  $LOG_DIR/dashboard_backend.log"
echo "  $LOG_DIR/dashboard_frontend.log"
echo ""
echo "Health test:"
echo "  curl http://localhost:5000/api/health_state"
echo "  curl http://localhost:3000/api/health_state"
echo ""
echo "Press Ctrl+C to stop all services."

wait
