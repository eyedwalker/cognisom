#!/bin/bash
# ============================================================================
# Cognisom Brev Sync Script - Fast Updates via Git
# ============================================================================
# Usage: ./deploy/sync_brev.sh
# ============================================================================

set -e

BREV_HOST="eyentelligence"

echo "=========================================="
echo "Syncing Cognisom to Brev GPU Instance"
echo "=========================================="

# Step 1: Commit and push local changes
echo ""
echo "Step 1: Pushing local changes to GitHub..."
cd "$(dirname "$0")/.."

# Check if there are uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo "  Found uncommitted changes, committing..."
    git add -A
    git commit -m "Quick sync to Brev $(date +%Y-%m-%d_%H:%M)"
fi

git push origin main 2>/dev/null || echo "  Already up to date"

# Also push submodule
cd cognisom
if [[ -n $(git status -s) ]]; then
    git add -A
    git commit -m "Quick sync $(date +%Y-%m-%d_%H:%M)"
fi
git push origin main 2>/dev/null || echo "  Submodule already up to date"
cd ..

# Step 2: Pull on Brev and restart
echo ""
echo "Step 2: Pulling changes on Brev..."
brev shell $BREV_HOST << 'EOFCMD'
cd ~/cognisom

# Pull latest changes
git pull origin main 2>/dev/null || true
git submodule update --init --recursive 2>/dev/null || true

# Restart Streamlit
pkill -f streamlit 2>/dev/null || true
sleep 1

source ~/cognisom/venv/bin/activate
nohup streamlit run cognisom/dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    > /tmp/streamlit.log 2>&1 &

sleep 3
curl -s localhost:8501 > /dev/null && echo "✓ Dashboard restarted!" || echo "✗ Dashboard failed to start"
EOFCMD

echo ""
echo "=========================================="
echo "Sync complete!"
echo "Dashboard: https://dashboard-q1vv3fhh8.brevlab.com"
echo "=========================================="
