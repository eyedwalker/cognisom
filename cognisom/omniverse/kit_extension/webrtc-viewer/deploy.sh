#!/bin/bash
# Deploy Cognisom WebRTC Viewer
# ==============================
# Builds the WebRTC viewer and deploys it to nginx on the Cognisom server.
#
# Prerequisites:
#   - Node.js 18+ and npm 10+ on the build machine
#   - KIT_SERVER set to user@host for the server (there is NO default:
#     the old address 52.32.247.131 was released with the GPU box)
#
# Usage:
#   cd cognisom/omniverse/kit_extension/webrtc-viewer
#   ./deploy.sh               # Build and deploy
#   ./deploy.sh --build-only  # Just build locally

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER="${KIT_SERVER:-}"
HOST_ONLY="${SERVER##*@}"
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

# The Elastic IP this project used (52.32.247.131) was RELEASED when the
# GPU box was terminated, so that address may now belong to someone
# else's instance. Nothing here gets a default host: a deploy script
# that silently targets a released address uploads to a stranger.
if [[ -z "${KIT_SERVER:-}" ]]; then
    cat >&2 <<'USAGE'
KIT_SERVER is not set, and there is deliberately no default.

Set it to the GPU box you mean, as user@host:
    export KIT_SERVER=ubuntu@<the box's current address>

The previous box (i-0ac9eb88c1b046163, g6e.2xlarge, L40S) was terminated
and its Elastic IP released, so the old address is not safe to assume.
List what exists with:
    aws ec2 describe-instances --region us-west-2 \
        --query 'Reservations[].Instances[].{Id:InstanceId,Name:Tags[?Key==`Name`]|[0].Value,Type:InstanceType,State:State.Name,IP:PublicIpAddress}' \
        --output table
USAGE
    exit 2
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
echo "  https://cognisom.com/rtx-viewer/?server=${HOST_ONLY}&signalingPort=49100"
