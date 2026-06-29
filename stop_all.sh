#!/usr/bin/env bash

echo "Stopping Intel Multimodal services..."

# Kill anything using the three development ports
for PORT in 8000 5000 3000; do
    PID=$(lsof -ti tcp:$PORT)

    if [ -n "$PID" ]; then
        echo "Stopping process on port $PORT: PID $PID"
        kill $PID 2>/dev/null || true
        sleep 1

        # Force kill if still running
        PID=$(lsof -ti tcp:$PORT)
        if [ -n "$PID" ]; then
            echo "Force stopping process on port $PORT: PID $PID"
            kill -9 $PID 2>/dev/null || true
        fi
    else
        echo "No process found on port $PORT"
    fi
done

echo "Shutdown complete."
