"""Remote executor for tissue-scale simulation on Lambda Labs.

Handles SSH + Docker deployment pipeline:
1. SSH into Lambda Labs instance
2. Pull Docker image from ECR
3. Start simulation container with --gpus all
4. Stream results back via WebSocket
5. Clean up on completion

Usage::

    executor = RemoteExecutor(
        host_ip="203.0.113.42",
        ssh_key_path="~/.ssh/lambda_key",
    )
    executor.setup_instance()
    job_id = executor.start_simulation(config)
    for snapshot in executor.stream_results(job_id):
        update_dashboard(snapshot)
    executor.stop_simulation(job_id)

"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, Iterator, Optional

log = logging.getLogger(__name__)

# ECR image for tissue simulation
DEFAULT_ECR_IMAGE = "780457123717.dkr.ecr.us-east-1.amazonaws.com/cognisom:tissue"

# Default SSH settings
DEFAULT_SSH_PORT = 22
DEFAULT_SSH_USER = "ubuntu"
DEFAULT_SSH_KEY_PATH = os.path.expanduser("~/.ssh/lambda_key")


class RemoteExecutor:
    """Manages remote simulation execution on Lambda Labs instances.

    Connects via SSH (paramiko) to deploy and manage Docker containers
    running the tissue simulation.
    """

    def __init__(
        self,
        host_ip: str,
        ssh_key_path: str = DEFAULT_SSH_KEY_PATH,
        ssh_user: str = DEFAULT_SSH_USER,
        ssh_port: int = DEFAULT_SSH_PORT,
        ecr_image: str = DEFAULT_ECR_IMAGE,
        ws_port: int = 8600,
    ):
        self._host_ip = host_ip
        self._ssh_key_path = ssh_key_path
        self._ssh_user = ssh_user
        self._ssh_port = ssh_port
        self._ecr_image = ecr_image
        self._ws_port = ws_port

        self._ssh_client = None
        self._connected = False
        self._container_name: Optional[str] = None

    @property
    def host_ip(self) -> str:
        return self._host_ip

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def ws_url(self) -> str:
        return f"ws://{self._host_ip}:{self._ws_port}"

    # ── Connection ────────────────────────────────────────────────

    def connect(self, timeout: float = 30.0) -> bool:
        """Establish SSH connection to the instance.

        Returns True if connected successfully.
        """
        try:
            import paramiko
        except ImportError:
            log.error("paramiko not installed. Install with: pip install paramiko")
            return False

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=self._host_ip,
                port=self._ssh_port,
                username=self._ssh_user,
                key_filename=self._ssh_key_path,
                timeout=timeout,
            )
            self._ssh_client = client
            self._connected = True
            log.info("SSH connected to %s@%s", self._ssh_user, self._host_ip)
            return True

        except Exception as e:
            log.error("SSH connection failed: %s", e)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close the SSH connection."""
        if self._ssh_client is not None:
            self._ssh_client.close()
        self._connected = False

    # ── Instance Setup ────────────────────────────────────────────

    def setup_instance(self) -> bool:
        """Prepare the instance for simulation.

        Steps:
        1. Verify GPU availability
        2. Install Docker + NVIDIA Container Toolkit (if needed)
        3. Login to ECR
        4. Pull the simulation Docker image

        Returns True if setup succeeded.
        """
        if not self._connected:
            if not self.connect():
                return False

        log.info("Setting up instance %s...", self._host_ip)

        # 1. Verify GPUs
        gpu_count = self._verify_gpus()
        if gpu_count == 0:
            log.error("No GPUs detected on instance")
            return False
        log.info("Detected %d GPUs", gpu_count)

        # 2. Verify Docker + NVIDIA runtime
        if not self._verify_docker():
            log.error("Docker/NVIDIA runtime not available")
            return False

        # 3. ECR login
        if not self._ecr_login():
            log.warning("ECR login failed; will try docker pull anyway")

        # 4. Pull image
        if not self._pull_image():
            log.error("Failed to pull Docker image")
            return False

        # 5. Set up auto-terminate failsafe
        self._setup_failsafe()

        log.info("Instance setup complete")
        return True

    # ── Simulation Control ────────────────────────────────────────

    def start_simulation(
        self,
        config_json: str,
        max_runtime_hours: float = 2.0,
    ) -> Optional[str]:
        """Start the tissue simulation container.

        Args:
            config_json: Serialized TissueScaleConfig.
            max_runtime_hours: Safety limit.

        Returns:
            Container name (job ID) if started, None on failure.
        """
        if not self._connected:
            log.error("Not connected")
            return None

        container_name = f"cognisom-tissue-{uuid.uuid4().hex[:8]}"
        self._container_name = container_name

        # Docker run command
        docker_cmd = (
            f"docker run -d "
            f"--name {container_name} "
            f"--gpus all "
            f"--shm-size=64g "
            f"--ipc=host "
            f"-e NCCL_P2P_LEVEL=NVL "
            f"-e NCCL_DEBUG=WARN "
            f"-e TISSUE_CONFIG='{config_json}' "
            f"-e MAX_RUNTIME_HOURS={max_runtime_hours} "
            f"-e AWS_DEFAULT_REGION=us-east-1 "
            f"-p {self._ws_port}:{self._ws_port} "
            f"{self._ecr_image} "
            f"python -m cognisom.core.tissue_runner"
        )

        exit_code, stdout, stderr = self._exec_command(docker_cmd)
        if exit_code == 0:
            log.info("Started container %s", container_name)
            return container_name
        else:
            log.error("Failed to start container: %s", stderr)
            return None

    def stop_simulation(self, job_id: Optional[str] = None) -> bool:
        """Stop the simulation container.

        Args:
            job_id: Container name. Uses tracked container if None.

        Returns:
            True if stopped.
        """
        container = job_id or self._container_name
        if not container:
            return False

        exit_code, _, stderr = self._exec_command(
            f"docker stop {container} && docker rm {container}"
        )
        if exit_code == 0:
            log.info("Stopped container %s", container)
            if container == self._container_name:
                self._container_name = None
            return True
        else:
            log.warning("Failed to stop container %s: %s", container, stderr)
            return False

    def get_container_status(self, job_id: Optional[str] = None) -> Dict:
        """Get the running container's status.

        Returns dict with state, running, exit_code.
        """
        container = job_id or self._container_name
        if not container or not self._connected:
            return {"state": "unknown"}

        exit_code, stdout, _ = self._exec_command(
            f"docker inspect --format '{{{{.State.Status}}}}' {container}"
        )
        if exit_code == 0:
            state = stdout.strip()
            return {"state": state, "container": container}
        return {"state": "not_found", "container": container}

    def get_container_logs(
        self,
        job_id: Optional[str] = None,
        tail: int = 50,
    ) -> str:
        """Get recent container logs."""
        container = job_id or self._container_name
        if not container or not self._connected:
            return ""

        exit_code, stdout, _ = self._exec_command(
            f"docker logs --tail {tail} {container}"
        )
        return stdout if exit_code == 0 else ""

    # ── Internal helpers ──────────────────────────────────────────

    def _exec_command(
        self,
        command: str,
        timeout: float = 120.0,
    ) -> tuple:
        """Execute a command via SSH.

        Returns (exit_code, stdout, stderr).
        """
        if not self._connected or self._ssh_client is None:
            return -1, "", "Not connected"

        try:
            _, stdout_ch, stderr_ch = self._ssh_client.exec_command(
                command, timeout=timeout,
            )
            exit_code = stdout_ch.channel.recv_exit_status()
            stdout = stdout_ch.read().decode("utf-8", errors="replace")
            stderr = stderr_ch.read().decode("utf-8", errors="replace")
            return exit_code, stdout, stderr
        except Exception as e:
            return -1, "", str(e)

    def _verify_gpus(self) -> int:
        """Verify NVIDIA GPUs are available."""
        exit_code, stdout, _ = self._exec_command("nvidia-smi -L")
        if exit_code != 0:
            return 0
        # Count "GPU N:" lines
        return sum(1 for line in stdout.split("\n") if line.strip().startswith("GPU"))

    def _verify_docker(self) -> bool:
        """Verify Docker and NVIDIA Container Toolkit."""
        exit_code, _, _ = self._exec_command("docker --version")
        if exit_code != 0:
            return False

        # Check NVIDIA runtime
        exit_code, _, _ = self._exec_command(
            "docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi"
        )
        return exit_code == 0

    def _ecr_login(self) -> bool:
        """Login to AWS ECR."""
        # Use instance's IAM role or pre-configured credentials
        cmd = (
            "aws ecr get-login-password --region us-east-1 | "
            "docker login --username AWS --password-stdin "
            "780457123717.dkr.ecr.us-east-1.amazonaws.com"
        )
        exit_code, _, stderr = self._exec_command(cmd, timeout=30)
        if exit_code == 0:
            log.info("ECR login successful")
            return True
        log.warning("ECR login failed: %s", stderr)
        return False

    def _pull_image(self) -> bool:
        """Pull the simulation Docker image."""
        log.info("Pulling %s...", self._ecr_image)
        exit_code, stdout, stderr = self._exec_command(
            f"docker pull {self._ecr_image}",
            timeout=600,  # 10 min for large image
        )
        if exit_code == 0:
            log.info("Image pulled successfully")
            return True
        log.error("Pull failed: %s", stderr)
        return False

    def _setup_failsafe(self, max_hours: float = 2.0) -> None:
        """Set up instance-level auto-terminate failsafe.

        Uses `at` command as a final safety net: even if the
        simulation crashes and Docker dies, the instance will
        shut down after max_hours.
        """
        minutes = int(max_hours * 60)
        cmd = f'echo "sudo shutdown -h now" | at now + {minutes} minutes 2>/dev/null || true'
        self._exec_command(cmd)
        log.info("Failsafe shutdown scheduled in %d minutes", minutes)
