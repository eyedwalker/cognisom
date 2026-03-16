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

    Returns URL like "http://52.32.247.131:8600" or "http://host.docker.internal:8600"
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


def get_gpu_instance_state() -> str:
    """Get the current state of the GPU instance.

    Returns: "running", "stopped", "pending", "stopping", or "unknown"
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
