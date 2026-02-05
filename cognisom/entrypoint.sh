#!/bin/bash
# Cognisom HDT Platform — Production entrypoint
# Starts Flask API (gunicorn) + Streamlit dashboard in parallel

set -e

echo "=========================================="
echo "  Cognisom HDT Platform"
echo "  eyentelligence inc."
echo "=========================================="

# Load .env if present
if [ -f /app/.env ]; then
    export $(grep -v '^#' /app/.env | xargs)
fi

# Ensure data directories exist
mkdir -p /app/data/auth \
         /app/data/scrna \
         /app/data/research_cache \
         /app/data/subscriptions \
         /app/data/flywheel \
         /app/data/feedback \
         /app/data/agent_interactions \
         /app/data/model_registry \
         /app/data/distilled_models \
         /app/data/calibration \
         /app/exports

# Start Flask API with gunicorn in background
echo "[1/2] Starting Flask API on port 5000..."
cd /app
gunicorn api.rest_server:app \
    --bind 0.0.0.0:5000 \
    --workers 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - &

FLASK_PID=$!

# Start Streamlit dashboard
echo "[2/2] Starting Streamlit dashboard on port 8501..."
streamlit run cognisom/dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none &

STREAMLIT_PID=$!

echo ""
echo "Services running:"
echo "  Flask API:   http://localhost:5000"
echo "  Dashboard:   http://localhost:8501"
echo ""

# Wait for either process to exit
wait -n $FLASK_PID $STREAMLIT_PID

# If one dies, kill the other and exit
echo "A service exited. Shutting down..."
kill $FLASK_PID $STREAMLIT_PID 2>/dev/null
wait
exit 1
