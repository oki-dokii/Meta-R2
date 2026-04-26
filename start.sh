#!/bin/bash
set -e

echo "================================================"
echo "  LifeStack — Starting services"
echo "================================================"

# Start OpenEnv server in the background on port 8000.
# Redirect its logs with a prefix so they are visible in HF logs.
# If server.py crashes (e.g. openenv-core import fails), the Flask
# UI on port 7860 keeps the Space alive so HF never restart-loops.
python server.py 2>&1 | sed 's/^/[openenv] /' &
OPENENV_PID=$!
echo "[start.sh] OpenEnv server started (PID $OPENENV_PID) on port 8000"

# Give the OpenEnv server 3 seconds to bind its port before Flask starts.
sleep 3

# Start Flask UI in the foreground on port 7860.
# HuggingFace health-checks port 7860, so this MUST be the foreground process.
echo "[start.sh] Starting Flask UI on port 7860"
exec python app_flask.py
