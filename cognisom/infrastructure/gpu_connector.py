"""
GPU Connector
=============

Manage connection to the GPU instance (cognisom-dedicated) for RTX
rendering and Kit services. Used when the dashboard runs on a separate
CPU instance (apps-server) from the GPU.

Environment variables:
    GPU_HOST: IP or hostname of the GPU instance (default: host.docker.internal)
    GPU_INSTANCE_ID: EC2 instance ID of the GPU box (for start/stop)
    GPU_REGION: AWS region of the GPU instance (default: us-west-2)
    IS_GPU_INSTANCE: Set to "true" if running ON the GPU box itself

The defaults below are STALE and kept only so the names resolve. The
g6e.2xlarge this project rendered on until 2026-09 (L40S 48 GB,
i-0ac9eb88c1b046163, Elastic IP 52.32.247.131) has been terminated and
its address released, so ``GPU_HOST`` and ``GPU_INSTANCE_ID`` must both
be set to reach a real box. ``get_gpu_instance_state`` returns
``STATE_MISSING`` rather than "unknown" when the instance is gone, so
this shows up as a fact instead of a silence.
"""

import logging
import os
import time
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# GPU instance configuration
GPU_HOST = os.environ.get("GPU_HOST", "host.docker.internal")
GPU_INSTANCE_ID = os.environ.get("GPU_INSTANCE_ID", "i-0ac9eb88c1b046163")
GPU_REGION = os.environ.get("GPU_REGION", "us-west-2")
IS_GPU_INSTANCE = os.environ.get("IS_GPU_INSTANCE", "").lower() == "true"

# Kit ports
KIT_STREAMING_PORT = 8600
KIT_WEBRTC_PORT = 8211


def get_kit_server_url() -> str:
    """Get the Kit streaming server URL for server-side Python requests.

    Returns a URL like "http://<gpu host>:8600" or
    "http://host.docker.internal:8600"
    depending on deployment mode.
    """
    custom = os.environ.get("KIT_SERVER_URL", "")
    if custom:
        return custom.rstrip("/")
    return f"http://{GPU_HOST}:{KIT_STREAMING_PORT}"


def get_kit_browser_url() -> str:
    """Get the Kit URL for browser-side access (MJPEG iframe).

    When behind nginx, this returns "/kit" (proxied).
    When direct, returns the full URL.
    """
    # If running on the GPU instance itself, nginx proxies /kit/ → localhost:8600
    if IS_GPU_INSTANCE:
        return "/kit"

    # Running on separate instance — browser needs direct access
    # nginx on apps-server will proxy /kit/ → GPU_HOST:8600
    return "/kit"


def is_kit_available() -> bool:
    """Check if the Kit streaming server is reachable."""
    try:
        url = get_kit_server_url()
        resp = requests.get(f"{url}/status", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


#: Returned when EC2 says the configured instance does not exist. It is
#: NOT "unknown": a terminated box cannot be started, and reporting it as
#: unknown invites waiting for a machine that is never coming back. The
#: g6e.2xlarge this project used until 2026-09 (L40S, i-0ac9eb88c1b046163)
#: was terminated along with its Elastic IP, which is how this state was
#: found -- the lifecycle code reported "unknown" for months.
STATE_MISSING = "missing"


def get_gpu_instance_state() -> str:
    """Get the current state of the GPU instance.

    Returns "running", "stopped", "pending", "stopping", ``STATE_MISSING``
    when the configured instance does not exist, or "unknown" when the
    question could not be asked at all (no credentials, no boto3, no
    network).
    """
    if IS_GPU_INSTANCE:
        return "running"  # We are the GPU instance

    try:
        import boto3
        ec2 = boto3.client("ec2", region_name=GPU_REGION)
        resp = ec2.describe_instances(InstanceIds=[GPU_INSTANCE_ID])
        state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
        return state
    except Exception as e:
        if "InvalidInstanceID.NotFound" in str(e):
            logger.error(
                "GPU instance %s does not exist in %s. Set GPU_INSTANCE_ID "
                "to a live instance.", GPU_INSTANCE_ID, GPU_REGION)
            return STATE_MISSING
        logger.warning(f"Cannot check GPU instance state: {e}")
        return "unknown"


def start_gpu_instance() -> Tuple[bool, str]:
    """Start the GPU instance if it's stopped.

    Returns: (success, message)
    """
    if IS_GPU_INSTANCE:
        return True, "Already running (this is the GPU instance)"

    try:
        import boto3
        ec2 = boto3.client("ec2", region_name=GPU_REGION)

        # Check current state
        state = get_gpu_instance_state()
        if state == "running":
            return True, "GPU instance is already running"
        if state == "pending":
            return True, "GPU instance is starting up"
        if state == STATE_MISSING:
            return False, (
                f"GPU instance {GPU_INSTANCE_ID} does not exist in "
                f"{GPU_REGION}. It cannot be started. Set GPU_INSTANCE_ID "
                f"to a live GPU instance, or launch one.")
        if state not in ("stopped", "unknown"):
            return False, f"GPU instance is in state: {state}"

        # Start it
        ec2.start_instances(InstanceIds=[GPU_INSTANCE_ID])
        logger.info(f"Started GPU instance {GPU_INSTANCE_ID}")
        return True, "GPU instance starting (allow 2-3 minutes for Kit to initialize)"
    except Exception as e:
        logger.error(f"Failed to start GPU instance: {e}")
        return False, f"Failed to start GPU: {e}"


def wait_for_kit(timeout_seconds: int = 180, poll_interval: int = 10) -> bool:
    """Wait for Kit streaming server to become available.

    Args:
        timeout_seconds: Max time to wait.
        poll_interval: Seconds between checks.

    Returns: True if Kit became available, False if timeout.
    """
    start = time.time()
    while time.time() - start < timeout_seconds:
        if is_kit_available():
            return True
        time.sleep(poll_interval)
    return False
