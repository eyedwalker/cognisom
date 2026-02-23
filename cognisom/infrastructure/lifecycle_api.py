"""Flask API for GPU instance lifecycle management.

Runs on the apps-server (always-on gateway) and exposes endpoints to
start/stop the GPU instance. Also serves a landing page that lets users
wake up the GPU instance when it has been auto-stopped due to inactivity.

Endpoints:
    GET  /                         — Landing page (wake-up UI)
    POST /api/lifecycle/start      — Start GPU instance
    POST /api/lifecycle/stop       — Stop GPU instance
    GET  /api/lifecycle/status     — Get GPU instance state (public)
    POST /_cognisom/heartbeat      — Update heartbeat (called from GPU instance JS)

Environment variables:
    GPU_INSTANCE_ID        — EC2 instance ID to manage
    GPU_INSTANCE_URL       — URL of the GPU Streamlit app (default: https://cognisom.com)
    AWS_REGION             — AWS region (default: us-west-2)
    LIFECYCLE_API_PORT     — Port to listen on (default: 8502)
"""

from __future__ import annotations

import logging
import os
import time

from flask import Flask, jsonify, request

log = logging.getLogger(__name__)

app = Flask(__name__)

GPU_INSTANCE_URL = os.environ.get("GPU_INSTANCE_URL", "https://cognisom.com")


def _get_lifecycle_manager():
    """Lazy-load EC2LifecycleManager."""
    from .ec2_lifecycle import EC2LifecycleManager

    return EC2LifecycleManager(
        instance_id=os.environ.get("GPU_INSTANCE_ID", ""),
        region=os.environ.get("AWS_REGION", "us-west-2"),
    )


# ── Landing Page ──────────────────────────────────────────────────

LANDING_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cognisom</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0a; color: #e0e0e0;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh;
  }
  .card {
    background: #1a1a2e; border: 1px solid #2a2a4a;
    border-radius: 16px; padding: 48px; max-width: 480px;
    text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  h1 { font-size: 28px; margin-bottom: 8px; color: #fff; }
  .subtitle { color: #888; font-size: 14px; margin-bottom: 32px; }
  .status-dot {
    display: inline-block; width: 12px; height: 12px;
    border-radius: 50%; margin-right: 8px; vertical-align: middle;
  }
  .status-dot.running { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
  .status-dot.stopped { background: #666; }
  .status-dot.pending { background: #f59e0b; animation: pulse 1s infinite; }
  @keyframes pulse { 50% { opacity: 0.5; } }
  .status-line {
    font-size: 16px; margin-bottom: 24px; padding: 12px;
    background: #111; border-radius: 8px;
  }
  #start-btn {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: #fff; border: none; border-radius: 10px;
    padding: 14px 40px; font-size: 16px; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
  }
  #start-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(99,102,241,0.4); }
  #start-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  #start-btn.go-btn {
    background: linear-gradient(135deg, #22c55e, #16a34a);
  }
  .progress { margin-top: 16px; font-size: 13px; color: #888; min-height: 20px; }
  .cost-info { margin-top: 24px; font-size: 12px; color: #555; }
</style>
</head>
<body>
<div class="card">
  <h1>Cognisom</h1>
  <p class="subtitle">Bio-Digital Twin Platform</p>
  <div class="status-line">
    <span id="dot" class="status-dot stopped"></span>
    <span id="state-text">Checking...</span>
  </div>
  <button id="start-btn" onclick="handleClick()" disabled>Start Instance</button>
  <div id="progress" class="progress"></div>
  <div class="cost-info">GPU auto-stops after 15 min of inactivity</div>
</div>
<script>
const GPU_URL = '__GPU_URL__';
let currentState = 'unknown';

async function checkStatus() {
  try {
    const r = await fetch('/api/lifecycle/status');
    const d = await r.json();
    updateUI(d.state, d.ready, d.uptime_seconds);
  } catch(e) {
    updateUI('error', false, 0);
  }
}

function updateUI(state, ready, uptime) {
  const dot = document.getElementById('dot');
  const text = document.getElementById('state-text');
  const btn = document.getElementById('start-btn');
  currentState = state;

  dot.className = 'status-dot ' + (state === 'running' ? (ready ? 'running' : 'pending') : state === 'pending' ? 'pending' : 'stopped');

  if (state === 'running' && ready) {
    text.textContent = 'Running' + (uptime ? ' (' + formatUptime(uptime) + ')' : '');
    btn.textContent = 'Open Dashboard';
    btn.disabled = false;
    btn.className = 'go-btn';
    btn.id = 'start-btn';
  } else if (state === 'running' || state === 'pending') {
    text.textContent = state === 'pending' ? 'Starting...' : 'Starting services...';
    btn.textContent = 'Starting...';
    btn.disabled = true;
    btn.className = '';
    btn.id = 'start-btn';
  } else {
    text.textContent = state === 'stopped' ? 'Stopped' : state;
    btn.textContent = 'Start Instance';
    btn.disabled = false;
    btn.className = '';
    btn.id = 'start-btn';
  }
}

function formatUptime(s) {
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm';
  return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
}

async function handleClick() {
  if (currentState === 'running') {
    window.location.href = GPU_URL;
    return;
  }
  const btn = document.getElementById('start-btn');
  const prog = document.getElementById('progress');
  btn.disabled = true;
  btn.textContent = 'Starting...';
  prog.textContent = 'Sending start command...';

  try {
    const r = await fetch('/api/lifecycle/start', {method:'POST'});
    const d = await r.json();
    if (d.success) {
      prog.textContent = 'Instance starting. Waiting for services (~60-90s)...';
      pollUntilReady();
    } else {
      prog.textContent = 'Error: ' + d.message;
      btn.disabled = false;
      btn.textContent = 'Retry';
    }
  } catch(e) {
    prog.textContent = 'Network error: ' + e.message;
    btn.disabled = false;
    btn.textContent = 'Retry';
  }
}

function pollUntilReady() {
  const prog = document.getElementById('progress');
  let elapsed = 0;
  const interval = setInterval(async () => {
    elapsed += 5;
    try {
      const r = await fetch('/api/lifecycle/status');
      const d = await r.json();
      updateUI(d.state, d.ready, d.uptime_seconds);
      if (d.ready) {
        clearInterval(interval);
        prog.textContent = 'Ready! Redirecting...';
        setTimeout(() => { window.location.href = GPU_URL; }, 1000);
      } else {
        prog.textContent = 'Waiting for services... (' + elapsed + 's)';
      }
    } catch(e) {
      prog.textContent = 'Checking... (' + elapsed + 's)';
    }
  }, 5000);
}

checkStatus();
setInterval(checkStatus, 10000);
</script>
</body>
</html>""".replace("__GPU_URL__", "{{GPU_URL}}")


# ── Endpoints ──────────────────────────────────────────────────────


@app.route("/")
def landing():
    """Landing page with instance wake-up UI."""
    html = LANDING_PAGE.replace("{{GPU_URL}}", GPU_INSTANCE_URL)
    return html, 200, {"Content-Type": "text/html"}


@app.route("/api/lifecycle/status", methods=["GET"])
def lifecycle_status():
    """Get GPU instance status. Public endpoint."""
    mgr = _get_lifecycle_manager()
    status = mgr.get_status()
    return jsonify(status)


@app.route("/api/lifecycle/start", methods=["POST"])
def lifecycle_start():
    """Start the GPU instance."""
    mgr = _get_lifecycle_manager()
    ok, msg = mgr.start_instance()
    return jsonify({"success": ok, "message": msg}), 200 if ok else 500


@app.route("/api/lifecycle/stop", methods=["POST"])
def lifecycle_stop():
    """Stop the GPU instance."""
    mgr = _get_lifecycle_manager()
    ok, msg = mgr.stop_instance()
    return jsonify({"success": ok, "message": msg}), 200 if ok else 500


@app.route("/_cognisom/heartbeat", methods=["POST"])
def heartbeat():
    """Update activity heartbeat. Called by JS tracker in Streamlit."""
    from .inactivity import update_heartbeat

    update_heartbeat()
    return "", 204


@app.route("/api/lifecycle/health", methods=["GET"])
def health():
    """Health check for the lifecycle API itself."""
    return jsonify({"status": "ok", "timestamp": time.time()})


# ── Main ───────────────────────────────────────────────────────────


def main():
    """Run the lifecycle API server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    port = int(os.environ.get("LIFECYCLE_API_PORT", "8502"))
    log.info("Starting lifecycle API on port %d", port)
    log.info("GPU_INSTANCE_ID: %s", os.environ.get("GPU_INSTANCE_ID", "(not set)"))
    log.info("GPU_INSTANCE_URL: %s", GPU_INSTANCE_URL)

    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
