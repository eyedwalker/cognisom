#!/bin/bash
# Deploy Cognisom WebRTC Viewer
# ==============================
# Builds the WebRTC viewer and deploys it to nginx on the Cognisom server.
#
# Prerequisites:
#   - Node.js 18+ and npm 10+ on the build machine
#   - SSH access to cognisom-dedicated (52.32.247.131)
#
# Usage:
#   cd cognisom/omniverse/kit_extension/webrtc-viewer
#   ./deploy.sh               # Build and deploy
#   ./deploy.sh --build-only  # Just build locally

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER="ubuntu@52.32.247.131"
SSH_KEY="${HOME}/.ssh/wabah-key.pem"
REMOTE_PATH="/var/www/rtx-viewer"

echo "=== Cognisom WebRTC Viewer Build ==="

# Install dependencies
cd "$SCRIPT_DIR"
echo "Installing dependencies..."
npm install

# Build
echo "Building with Vite..."
npm run build

echo "Build complete: dist/"
ls -la dist/

if [[ "${1:-}" == "--build-only" ]]; then
    echo "Build-only mode, skipping deploy."
    exit 0
fi

# Deploy to server
echo "=== Deploying to ${SERVER} ==="

# Create remote directory
ssh -i "$SSH_KEY" "$SERVER" "sudo mkdir -p ${REMOTE_PATH} && sudo chown ubuntu:ubuntu ${REMOTE_PATH}"

# Upload built files
scp -i "$SSH_KEY" -r dist/* "${SERVER}:${REMOTE_PATH}/"

echo "Files uploaded to ${REMOTE_PATH}"

# Add nginx config for /rtx-viewer/
echo "Configuring nginx..."
ssh -i "$SSH_KEY" "$SERVER" 'cat > /tmp/rtx-viewer-nginx.conf << "NGINX"
    # WebRTC RTX Viewer (Cognisom)
    location /rtx-viewer/ {
        alias /var/www/rtx-viewer/;
        try_files $uri $uri/ /rtx-viewer/index.html;
    }
NGINX

# Check if the location block already exists
if ! grep -q "rtx-viewer" /etc/nginx/sites-available/default 2>/dev/null; then
    echo "Adding /rtx-viewer/ location to nginx config..."
    # Insert before the closing } of the server block
    sudo sed -i "/^}/i\\    include /tmp/rtx-viewer-nginx.conf;" /etc/nginx/sites-available/default
    sudo nginx -t && sudo systemctl reload nginx
    echo "Nginx reloaded with /rtx-viewer/ route"
else
    echo "Nginx config already has /rtx-viewer/ route"
    sudo cp /tmp/rtx-viewer-nginx.conf /etc/nginx/snippets/rtx-viewer.conf
    sudo nginx -t && sudo systemctl reload nginx
fi'

echo ""
echo "=== Deploy Complete ==="
echo "WebRTC viewer: https://cognisom.com/rtx-viewer/"
echo "MJPEG viewer:  https://cognisom.com/kit/streaming/client"
echo ""
echo "To connect to Kit WebRTC:"
echo "  https://cognisom.com/rtx-viewer/?server=52.32.247.131&signalingPort=49100"
