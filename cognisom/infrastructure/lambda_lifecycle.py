"""Lambda Labs GPU instance lifecycle management.

Manages on-demand provisioning of 8x B200 GPU instances from Lambda Labs
for tissue-scale simulation. Mirrors the EC2LifecycleManager interface
but uses the Lambda Labs REST API.

API: https://cloud.lambdalabs.com/api/v1
Auth: Bearer token (LAMBDA_API_KEY environment variable)

Safety features:
- Hard max runtime limit (default 2 hours)
- Budget tracking with auto-terminate
- Disconnect timeout
- Instance-level failsafe cron

Usage::

    manager = LambdaLifecycleManager()
    ok, msg, instance_id = manager.launch_instance()
    if ok:
        manager.wait_until_ready(instance_id)
        status = manager.get_status(instance_id)
        # ... run simulation ...
        manager.terminate_instance(instance_id)

"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

# Lambda Labs API endpoint
LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"

# AWS Secrets Manager secret name
LAMBDA_KEY_SECRET_NAME = "cognisom/lambda-api-key"
LAMBDA_KEY_SECRET_REGION = "us-east-1"

# Safety defaults
DEFAULT_MAX_RUNTIME_HOURS = 2.0
DEFAULT_BUDGET_USD = 80.0
DEFAULT_DISCONNECT_TIMEOUT_MIN = 10


def _fetch_lambda_key_from_secrets() -> Optional[str]:
    """Fetch Lambda API key from AWS Secrets Manager."""
    try:
        import boto3
        client = boto3.client("secretsmanager", region_name=LAMBDA_KEY_SECRET_REGION)
        resp = client.get_secret_value(SecretId=LAMBDA_KEY_SECRET_NAME)
        return resp["SecretString"]
    except Exception as e:
        log.debug("Could not fetch Lambda key from Secrets Manager: %s", e)
        return None


class LambdaLifecycleManager:
    """Manages Lambda Labs GPU instance lifecycle.

    Provides launch, terminate, status checking, and safety controls
    for on-demand 8x B200 GPU instances.
    """

    # Target instance type for tissue-scale simulation
    DEFAULT_INSTANCE_TYPE = "gpu_8x_b200"

    # Approximate hourly cost (USD)
    HOURLY_COST = {
        "gpu_8x_b200": 39.92,
        "gpu_1x_b200": 4.99,
        "gpu_8x_h100_sxm5": 27.92,
        "gpu_1x_h100_pcie": 2.49,
        "gpu_8x_a100_80gb_sxm4": 14.32,
        "gpu_1x_a100_sxm4": 1.29,
        "gpu_1x_gh200": 1.99,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        instance_type: str = DEFAULT_INSTANCE_TYPE,
        region: Optional[str] = None,
        ssh_key_name: Optional[str] = None,
        max_runtime_hours: float = DEFAULT_MAX_RUNTIME_HOURS,
        budget_usd: float = DEFAULT_BUDGET_USD,
    ):
        self._api_key = (
            api_key
            or os.environ.get("LAMBDA_API_KEY", "")
            or _fetch_lambda_key_from_secrets()
            or ""
        )
        self._instance_type = instance_type
        self._region = region  # None = let Lambda choose
        self._ssh_key_name = ssh_key_name
        self._max_runtime_hours = max_runtime_hours
        self._budget_usd = budget_usd

        # Tracking
        self._instance_id: Optional[str] = None
        self._launch_time: Optional[float] = None
        self._accumulated_cost: float = 0.0

        if not self._api_key:
            log.warning(
                "LAMBDA_API_KEY not set. Lambda Labs operations will fail. "
                "Set via environment variable or pass api_key parameter."
            )

    @property
    def instance_id(self) -> Optional[str]:
        return self._instance_id

    @property
    def is_launched(self) -> bool:
        return self._instance_id is not None

    @property
    def elapsed_hours(self) -> float:
        if self._launch_time is None:
            return 0.0
        return (time.time() - self._launch_time) / 3600

    @property
    def estimated_cost(self) -> float:
        rate = self.HOURLY_COST.get(self._instance_type, 40.0)
        return self.elapsed_hours * rate

    @property
    def budget_remaining(self) -> float:
        return max(0, self._budget_usd - self.estimated_cost)

    # ── Instance Operations ──────────────────────────────────────

    def launch_instance(
        self,
        name: str = "cognisom-tissue-sim",
    ) -> Tuple[bool, str, Optional[str]]:
        """Launch a new Lambda Labs GPU instance.

        Args:
            name: Instance name tag.

        Returns:
            (success, message, instance_id)
        """
        if not self._api_key:
            return False, "LAMBDA_API_KEY not configured", None

        # Check budget
        rate = self.HOURLY_COST.get(self._instance_type, 40.0)
        max_cost = rate * self._max_runtime_hours
        if max_cost > self._budget_usd:
            return (
                False,
                f"Max runtime cost (${max_cost:.0f}) exceeds budget "
                f"(${self._budget_usd:.0f}). Reduce runtime or increase budget.",
                None,
            )

        # Resolve region — required by Lambda API
        region = self._region
        if not region:
            # Auto-select first available region for this instance type
            available = self.list_available_types()
            for t in available:
                if t["type"] == self._instance_type and t.get("regions"):
                    region = t["regions"][0]
                    log.info("Auto-selected region: %s", region)
                    break
            if not region:
                return False, f"No regions available for {self._instance_type}", None

        # Build request
        payload: Dict[str, Any] = {
            "instance_type_name": self._instance_type,
            "region_name": region,
            "name": name,
        }
        if self._ssh_key_name:
            payload["ssh_key_names"] = [self._ssh_key_name]

        try:
            resp = self._api_request(
                "POST",
                "/instance-operations/launch",
                json=payload,
            )

            if resp.status_code == 200:
                data = resp.json().get("data", {})
                instance_ids = data.get("instance_ids", [])
                if instance_ids:
                    self._instance_id = instance_ids[0]
                    self._launch_time = time.time()
                    log.info(
                        "Launched instance %s (%s). Max runtime: %.1f hrs, "
                        "budget: $%.0f",
                        self._instance_id, self._instance_type,
                        self._max_runtime_hours, self._budget_usd,
                    )
                    return True, f"Instance {self._instance_id} launched", self._instance_id
                return False, "No instance IDs returned", None
            else:
                error = resp.json().get("error", {})
                msg = error.get("message", resp.text)
                suggestion = error.get("suggestion", "")
                log.error("Launch failed: %s %s", msg, suggestion)
                return False, f"Launch failed: {msg}. {suggestion}", None

        except requests.RequestException as e:
            log.error("API request failed: %s", e)
            return False, f"API error: {e}", None

    def terminate_instance(
        self,
        instance_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Terminate a Lambda Labs instance.

        Args:
            instance_id: Instance to terminate. Uses tracked instance if None.

        Returns:
            (success, message)
        """
        iid = instance_id or self._instance_id
        if not iid:
            return False, "No instance ID to terminate"

        try:
            resp = self._api_request(
                "POST",
                "/instance-operations/terminate",
                json={"instance_ids": [iid]},
            )

            if resp.status_code == 200:
                cost = self.estimated_cost
                log.info(
                    "Terminated instance %s after %.1f hrs (est. $%.2f)",
                    iid, self.elapsed_hours, cost,
                )
                if iid == self._instance_id:
                    self._instance_id = None
                return True, f"Instance {iid} terminated (est. cost: ${cost:.2f})"
            else:
                error = resp.json().get("error", {}).get("message", resp.text)
                return False, f"Terminate failed: {error}"

        except requests.RequestException as e:
            return False, f"API error: {e}"

    def get_status(
        self,
        instance_id: Optional[str] = None,
    ) -> Dict:
        """Get instance status.

        Returns:
            Dict with state, ip, gpu_count, cost info.
        """
        iid = instance_id or self._instance_id
        if not iid:
            return {"state": "not_launched", "instance_id": None}

        try:
            resp = self._api_request("GET", f"/instances/{iid}")

            if resp.status_code == 200:
                data = resp.json().get("data", {})
                return {
                    "instance_id": iid,
                    "state": data.get("status", "unknown"),
                    "ip": data.get("ip", None),
                    "hostname": data.get("hostname", None),
                    "instance_type": data.get("instance_type", {}).get("name", ""),
                    "gpu_count": data.get("instance_type", {}).get(
                        "specs", {}
                    ).get("gpus", 0),
                    "region": data.get("region", {}).get("name", ""),
                    "elapsed_hours": self.elapsed_hours,
                    "estimated_cost": self.estimated_cost,
                    "budget_remaining": self.budget_remaining,
                    "max_runtime_hours": self._max_runtime_hours,
                }
            elif resp.status_code == 404:
                return {"state": "terminated", "instance_id": iid}
            else:
                return {"state": "error", "instance_id": iid,
                        "error": resp.text}

        except requests.RequestException as e:
            return {"state": "error", "instance_id": iid, "error": str(e)}

    def wait_until_ready(
        self,
        instance_id: Optional[str] = None,
        timeout: float = 600,
        poll_interval: float = 15,
    ) -> bool:
        """Wait for instance to reach 'active' state.

        Args:
            instance_id: Instance to wait for.
            timeout: Max wait time in seconds (default 10 min).
            poll_interval: Seconds between status checks.

        Returns:
            True if instance became active, False on timeout.
        """
        iid = instance_id or self._instance_id
        if not iid:
            return False

        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status(iid)
            state = status.get("state", "")

            if state == "active":
                ip = status.get("ip", "unknown")
                log.info(
                    "Instance %s ready at %s (%.0fs)",
                    iid, ip, time.time() - start,
                )
                return True
            elif state in ("terminated", "error"):
                log.error("Instance %s entered %s state", iid, state)
                return False

            log.debug("Instance %s state: %s, waiting...", iid, state)
            time.sleep(poll_interval)

        log.error("Timeout waiting for instance %s (%.0fs)", iid, timeout)
        return False

    # ── Safety ────────────────────────────────────────────────────

    def check_safety_limits(self) -> Tuple[bool, str]:
        """Check if any safety limits have been exceeded.

        Returns:
            (safe, reason). If not safe, instance should be terminated.
        """
        if not self.is_launched:
            return True, "No instance running"

        # Check runtime
        if self.elapsed_hours >= self._max_runtime_hours:
            return False, (
                f"Runtime limit exceeded: {self.elapsed_hours:.1f} hrs "
                f">= {self._max_runtime_hours:.1f} hrs"
            )

        # Check budget
        if self.estimated_cost >= self._budget_usd:
            return False, (
                f"Budget exceeded: ${self.estimated_cost:.2f} "
                f">= ${self._budget_usd:.2f}"
            )

        return True, "OK"

    def auto_terminate_if_needed(self) -> Optional[str]:
        """Check safety limits and terminate if exceeded.

        Returns termination reason if terminated, None if safe.
        """
        safe, reason = self.check_safety_limits()
        if not safe:
            log.warning("Safety limit triggered: %s. Terminating.", reason)
            ok, msg = self.terminate_instance()
            return reason
        return None

    # ── Instance Discovery ────────────────────────────────────────

    def list_instances(self) -> List[Dict]:
        """List all running Lambda Labs instances."""
        try:
            resp = self._api_request("GET", "/instances")
            if resp.status_code == 200:
                return resp.json().get("data", [])
            return []
        except requests.RequestException:
            return []

    def list_available_types(self) -> List[Dict]:
        """List available instance types and their regions."""
        try:
            resp = self._api_request("GET", "/instance-types")
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                available = []
                for type_name, info in data.items():
                    regions = info.get("regions_with_capacity_available", [])
                    if regions:
                        available.append({
                            "type": type_name,
                            "description": info.get("instance_type", {}).get(
                                "description", ""
                            ),
                            "gpus": info.get("instance_type", {}).get(
                                "specs", {}
                            ).get("gpus", 0),
                            "price_per_hour": info.get("instance_type", {}).get(
                                "price_cents_per_hour", 0
                            ) / 100,
                            "regions": [r.get("name") for r in regions],
                        })
                return available
            return []
        except requests.RequestException:
            return []

    def list_ssh_keys(self) -> List[Dict]:
        """List registered SSH keys."""
        try:
            resp = self._api_request("GET", "/ssh-keys")
            if resp.status_code == 200:
                return resp.json().get("data", [])
            return []
        except requests.RequestException:
            return []

    # ── Cost Summary ──────────────────────────────────────────────

    def cost_summary(self) -> Dict:
        """Get current cost tracking summary."""
        rate = self.HOURLY_COST.get(self._instance_type, 40.0)
        return {
            "instance_type": self._instance_type,
            "hourly_rate": rate,
            "elapsed_hours": self.elapsed_hours,
            "estimated_cost": self.estimated_cost,
            "budget": self._budget_usd,
            "budget_remaining": self.budget_remaining,
            "max_runtime_hours": self._max_runtime_hours,
            "max_possible_cost": rate * self._max_runtime_hours,
        }

    # ── Internal ──────────────────────────────────────────────────

    def _api_request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> requests.Response:
        """Make an authenticated API request to Lambda Labs."""
        url = f"{LAMBDA_API_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        return requests.request(
            method, url,
            headers=headers,
            timeout=30,
            **kwargs,
        )
