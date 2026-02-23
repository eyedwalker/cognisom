"""Inactivity monitor for auto-shutdown of the GPU instance.

Tracks user activity via heartbeat file and triggers instance self-stop
after a configurable idle timeout.

Environment variables:
    IDLE_TIMEOUT_MINUTES — Minutes of inactivity before shutdown (default: 15)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

HEARTBEAT_FILE = Path("/tmp/cognisom_heartbeat")
IDLE_TIMEOUT_MINUTES = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "15"))


def update_heartbeat():
    """Write current timestamp to the heartbeat file.

    Called on every Streamlit page load and by the JS activity tracker.
    """
    try:
        HEARTBEAT_FILE.write_text(str(time.time()))
    except OSError as e:
        log.warning("Failed to update heartbeat: %s", e)


def get_last_activity() -> float:
    """Read the last activity timestamp from the heartbeat file."""
    try:
        return float(HEARTBEAT_FILE.read_text().strip())
    except (OSError, ValueError):
        return time.time()


def inject_activity_tracker() -> str:
    """Return HTML/JS that tracks user activity and sends heartbeats.

    Embeds an invisible iframe-like component that:
    - Listens for mouse, keyboard, scroll, and touch events
    - Sends a Streamlit rerun signal via hidden button click to update heartbeat
    - Shows a warning banner 2 minutes before auto-logout
    """
    idle_ms = IDLE_TIMEOUT_MINUTES * 60 * 1000
    warn_ms = max(idle_ms - 120_000, 60_000)  # Warn 2 min before, min 1 min

    return f"""
    <div id="cognisom-activity-tracker"></div>
    <div id="cognisom-idle-warning" style="
        display:none; position:fixed; top:0; left:0; right:0; z-index:99999;
        background:linear-gradient(135deg, #d97706, #f59e0b); color:#000;
        padding:12px 20px; text-align:center; font-weight:700; font-size:14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    ">
        Session will end due to inactivity. Move mouse or press a key to stay active.
    </div>
    <script>
    (function() {{
        let lastActivity = Date.now();
        let heartbeatInterval = null;
        let warningShown = false;
        const IDLE_TIMEOUT = {idle_ms};
        const WARN_AT = {warn_ms};

        function onActivity() {{
            lastActivity = Date.now();
            if (warningShown) {{
                document.getElementById('cognisom-idle-warning').style.display = 'none';
                warningShown = false;
            }}
        }}

        // Track all user interaction events
        ['mousemove', 'mousedown', 'keydown', 'scroll', 'touchstart', 'click'].forEach(
            function(evt) {{ document.addEventListener(evt, onActivity, {{passive: true}}); }}
        );

        // Send heartbeat every 60 seconds if user was active
        heartbeatInterval = setInterval(function() {{
            const idle = Date.now() - lastActivity;

            // Server-side heartbeat is updated on every Streamlit page rerun.
            // JS only handles client-side idle warning display.

            if (idle >= WARN_AT && !warningShown) {{
                document.getElementById('cognisom-idle-warning').style.display = 'block';
                warningShown = true;
            }}
        }}, 60000);
    }})();
    </script>
    """


class InactivityMonitor:
    """Background daemon thread that watches for inactivity and triggers shutdown.

    Checks the heartbeat file periodically. If no activity for IDLE_TIMEOUT_MINUTES,
    calls EC2LifecycleManager.self_stop() to shut down the GPU instance.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, idle_timeout_minutes: int = IDLE_TIMEOUT_MINUTES):
        self.idle_timeout = idle_timeout_minutes * 60  # Convert to seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @classmethod
    def get_or_start(cls, idle_timeout_minutes: int = IDLE_TIMEOUT_MINUTES) -> "InactivityMonitor":
        """Get the singleton monitor, starting it if necessary."""
        with cls._lock:
            if cls._instance is None or not cls._instance.is_running():
                cls._instance = cls(idle_timeout_minutes=idle_timeout_minutes)
                cls._instance.start()
            return cls._instance

    def start(self):
        """Start the inactivity monitor daemon thread."""
        if self._thread and self._thread.is_alive():
            return

        # Initialize heartbeat
        update_heartbeat()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="cognisom-inactivity-monitor",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "Inactivity monitor started (timeout: %d minutes)",
            self.idle_timeout // 60,
        )

    def stop(self):
        """Stop the monitor thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _monitor_loop(self):
        """Main monitoring loop — checks heartbeat every 60 seconds."""
        while not self._stop_event.is_set():
            self._stop_event.wait(60)
            if self._stop_event.is_set():
                break

            last = get_last_activity()
            idle_seconds = time.time() - last

            if idle_seconds >= self.idle_timeout:
                log.warning(
                    "Idle for %.0f seconds (limit: %d). Initiating self-stop.",
                    idle_seconds,
                    self.idle_timeout,
                )
                self._trigger_shutdown()
                break

    def _trigger_shutdown(self):
        """Trigger instance self-stop."""
        try:
            from .ec2_lifecycle import EC2LifecycleManager

            mgr = EC2LifecycleManager()
            ok, msg = mgr.self_stop()
            if ok:
                log.info("Self-stop initiated: %s", msg)
            else:
                log.error("Self-stop failed: %s", msg)
        except Exception as e:
            log.error("Exception during self-stop: %s", e)
