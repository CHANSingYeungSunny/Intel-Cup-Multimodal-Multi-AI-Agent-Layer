#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PC_IP="$(hostname -I | awk '{print $1}')"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
FALLBACK_NODE="/home/user/.nvm/versions/node/v24.18.0/bin/node"
REACT_SCRIPTS_ENTRY="node_modules/react-scripts/bin/react-scripts.js"

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

load_nvm() {
    if [ -s "$NVM_DIR/nvm.sh" ]; then
        # Load nvm for noninteractive shells such as SSH sessions.
        . "$NVM_DIR/nvm.sh"
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
    pkill -f "$REACT_SCRIPTS_ENTRY start" 2>/dev/null || true

    echo "Stopped."
}

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
    load_nvm

    npm_cmd="$(command -v npm 2>/dev/null || true)"

    if [ ! -d "node_modules" ] && [ -n "$npm_cmd" ]; then
        echo "node_modules not found. Running npm install..."
        "$npm_cmd" install
    elif [ ! -d "node_modules" ]; then
        echo "ERROR: node_modules not found and npm is unavailable."
        exit 1
    fi

    if [ -n "$npm_cmd" ]; then
        "$npm_cmd" start
    elif [ -x "$FALLBACK_NODE" ] && [ -f "$REACT_SCRIPTS_ENTRY" ]; then
        echo "npm not found after loading NVM. Falling back to $FALLBACK_NODE"
        "$FALLBACK_NODE" "$REACT_SCRIPTS_ENTRY" start
    else
        echo "ERROR: npm not found and fallback React startup command is unavailable."
        exit 1
    fi
) > "$LOG_DIR/dashboard_frontend.log" 2>&1 &

FRONTEND_PID=$!
wait_for_port 3000 "React Frontend" "$LOG_DIR/dashboard_frontend.log"

echo ""
echo "Services started in background."
echo ""
echo "Open:"
echo "- Dashboard frontend: http://localhost:3000"
echo "- Dashboard backend:  http://localhost:5000"
echo "- Agent backend:      http://localhost:8000/docs"
echo ""
echo "Open from your PC/browser:"
echo "- Dashboard frontend: http://$PC_IP:3000"
echo "- Dashboard backend:  http://$PC_IP:5000"
echo "- Agent backend:      http://$PC_IP:8000/docs"
echo ""
echo "To stop:"
echo "- Run ./stop_all.sh"
echo ""
echo "Logs:"
echo "- logs/agent_backend.log"
echo "- logs/dashboard_backend.log"
echo "- logs/dashboard_frontend.log"

exit 0