"""EC2 Lifecycle Manager for on-demand GPU instance start/stop.

Manages the lifecycle of a GPU EC2 instance via boto3:
- Start instance when user logs in
- Stop instance after inactivity timeout
- Query instance state and readiness
- Self-stop from within the instance (IMDSv2)

Environment variables:
    GPU_INSTANCE_ID   — EC2 instance ID to manage
    AWS_REGION        — AWS region (default: us-west-2)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)


class EC2LifecycleManager:
    """Manages EC2 GPU instance start/stop lifecycle."""

    def __init__(
        self,
        instance_id: Optional[str] = None,
        region: str = "us-west-2",
    ):
        self.instance_id = instance_id or os.environ.get("GPU_INSTANCE_ID", "")
        self.region = region
        self._ec2 = boto3.client("ec2", region_name=self.region)

    def start_instance(self) -> Tuple[bool, str]:
        """Start the GPU instance.

        Returns (success, message).
        """
        if not self.instance_id:
            return False, "No GPU_INSTANCE_ID configured"

        try:
            current = self._get_state()
            if current in ("running", "pending"):
                return True, f"Instance already {current}"

            if current == "stopping":
                return False, "Instance is currently stopping, try again in a moment"

            self._ec2.start_instances(InstanceIds=[self.instance_id])
            log.info("Started GPU instance %s", self.instance_id)
            return True, "Instance starting"

        except ClientError as e:
            msg = f"Failed to start instance: {e.response['Error']['Code']}"
            log.error(msg)
            return False, msg

    def stop_instance(self) -> Tuple[bool, str]:
        """Stop the GPU instance.

        Returns (success, message).
        """
        if not self.instance_id:
            return False, "No GPU_INSTANCE_ID configured"

        try:
            current = self._get_state()
            if current in ("stopped", "terminated"):
                return True, f"Instance already {current}"

            self._ec2.stop_instances(InstanceIds=[self.instance_id])
            log.info("Stopped GPU instance %s", self.instance_id)
            return True, "Instance stopping"

        except ClientError as e:
            msg = f"Failed to stop instance: {e.response['Error']['Code']}"
            log.error(msg)
            return False, msg

    def get_status(self) -> Dict:
        """Get instance status including state, readiness, and uptime.

        Returns dict with keys: state, ready, public_ip, uptime_seconds, instance_id.
        """
        if not self.instance_id:
            return {
                "state": "unknown",
                "ready": False,
                "public_ip": None,
                "uptime_seconds": 0,
                "instance_id": "",
            }

        try:
            resp = self._ec2.describe_instances(InstanceIds=[self.instance_id])
            inst = resp["Reservations"][0]["Instances"][0]

            state = inst["State"]["Name"]
            public_ip = inst.get("PublicIpAddress")
            launch_time = inst.get("LaunchTime")

            uptime = 0
            if launch_time and state == "running":
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                uptime = int((now - launch_time).total_seconds())

            # Check if Streamlit is actually reachable (instance running + health)
            ready = False
            if state == "running" and public_ip:
                ready = self._check_streamlit_health(public_ip)

            return {
                "state": state,
                "ready": ready,
                "public_ip": public_ip,
                "uptime_seconds": uptime,
                "instance_id": self.instance_id,
            }

        except ClientError as e:
            log.error("Failed to describe instance: %s", e)
            return {
                "state": "error",
                "ready": False,
                "public_ip": None,
                "uptime_seconds": 0,
                "instance_id": self.instance_id,
            }

    def wait_until_ready(self, timeout: int = 300, poll_interval: int = 5) -> bool:
        """Wait until the instance is running and Streamlit is reachable.

        Args:
            timeout: Maximum wait time in seconds.
            poll_interval: Seconds between status checks.

        Returns True if instance became ready within timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.get_status()
            if status["ready"]:
                return True
            if status["state"] in ("stopped", "terminated", "shutting-down"):
                return False
            time.sleep(poll_interval)
        return False

    def _get_state(self) -> str:
        """Get the current instance state string."""
        try:
            resp = self._ec2.describe_instances(InstanceIds=[self.instance_id])
            return resp["Reservations"][0]["Instances"][0]["State"]["Name"]
        except (ClientError, KeyError, IndexError):
            return "unknown"

    def _check_streamlit_health(self, ip: str, port: int = 8501, timeout: int = 3) -> bool:
        """Check if Streamlit is responding on the instance."""
        import urllib.request
        import urllib.error

        try:
            url = f"http://{ip}:{port}/_stcore/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    # ── Self-stop (called from within the GPU instance) ────────────

    @staticmethod
    def get_own_instance_id() -> str:
        """Get this instance's ID via IMDSv2 metadata."""
        import urllib.request
        import urllib.error

        try:
            # Step 1: Get IMDSv2 token
            token_req = urllib.request.Request(
                "http://169.254.169.254/latest/api/token",
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            )
            with urllib.request.urlopen(token_req, timeout=2) as resp:
                token = resp.read().decode()

            # Step 2: Get instance ID
            id_req = urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/instance-id",
                headers={"X-aws-ec2-metadata-token": token},
            )
            with urllib.request.urlopen(id_req, timeout=2) as resp:
                return resp.read().decode().strip()

        except (urllib.error.URLError, OSError, TimeoutError):
            log.warning("Failed to get instance ID from IMDS (not on EC2?)")
            return ""

    @staticmethod
    def get_own_region() -> str:
        """Get this instance's region via IMDSv2 metadata."""
        import urllib.request
        import urllib.error

        try:
            token_req = urllib.request.Request(
                "http://169.254.169.254/latest/api/token",
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            )
            with urllib.request.urlopen(token_req, timeout=2) as resp:
                token = resp.read().decode()

            az_req = urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/placement/availability-zone",
                headers={"X-aws-ec2-metadata-token": token},
            )
            with urllib.request.urlopen(az_req, timeout=2) as resp:
                az = resp.read().decode().strip()
                return az[:-1]  # us-west-2a -> us-west-2

        except (urllib.error.URLError, OSError, TimeoutError):
            return os.environ.get("AWS_REGION", "us-west-2")

    def self_stop(self) -> Tuple[bool, str]:
        """Stop this instance (called from within the GPU instance).

        Auto-detects instance ID via IMDSv2 if not set.
        """
        instance_id = self.instance_id or self.get_own_instance_id()
        if not instance_id:
            return False, "Cannot determine own instance ID"

        region = self.get_own_region()
        ec2 = boto3.client("ec2", region_name=region)

        try:
            ec2.stop_instances(InstanceIds=[instance_id])
            log.info("Self-stop initiated for instance %s", instance_id)
            return True, f"Self-stop initiated for {instance_id}"
        except ClientError as e:
            msg = f"Self-stop failed: {e.response['Error']['Code']}"
            log.error(msg)
            return False, msg
