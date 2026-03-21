"""
Reference Genome Setup
======================

One-time provisioning of GRCh38 reference genome and known-sites files
on the cognisom-dedicated GPU instance.

Required for Parabricks variant calling. Downloads from:
  - Broad Institute public references (s3://broad-references)
  - GATK resource bundle

Files are stored at /opt/cognisom/ref/ on the GPU instance.

Usage via SSM:
    from cognisom.infrastructure.reference_setup import ReferenceSetup
    setup = ReferenceSetup()
    setup.provision_via_ssm()
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Cognisom-dedicated instance
GPU_INSTANCE_ID = "i-0ac9eb88c1b046163"
GPU_REGION = "us-west-2"

# Reference data directory on GPU instance
REF_DIR = "/opt/cognisom/ref"

# GRCh38 reference files from Broad Institute public bucket
REFERENCE_FILES = {
    "fasta": {
        "source": "s3://broad-references/hg38/v0/Homo_sapiens_assembly38.fasta",
        "dest": f"{REF_DIR}/Homo_sapiens_assembly38.fasta",
        "size_gb": 3.0,
        "required": True,
    },
    "fasta_index": {
        "source": "s3://broad-references/hg38/v0/Homo_sapiens_assembly38.fasta.fai",
        "dest": f"{REF_DIR}/Homo_sapiens_assembly38.fasta.fai",
        "size_gb": 0.001,
        "required": True,
    },
    "dict": {
        "source": "s3://broad-references/hg38/v0/Homo_sapiens_assembly38.dict",
        "dest": f"{REF_DIR}/Homo_sapiens_assembly38.dict",
        "size_gb": 0.001,
        "required": True,
    },
    "known_indels": {
        "source": "s3://broad-references/hg38/v0/Homo_sapiens_assembly38.known_indels.vcf.gz",
        "dest": f"{REF_DIR}/Homo_sapiens_assembly38.known_indels.vcf.gz",
        "size_gb": 0.012,
        "required": False,
    },
    "known_indels_index": {
        "source": "s3://broad-references/hg38/v0/Homo_sapiens_assembly38.known_indels.vcf.gz.tbi",
        "dest": f"{REF_DIR}/Homo_sapiens_assembly38.known_indels.vcf.gz.tbi",
        "size_gb": 0.001,
        "required": False,
    },
    "dbsnp": {
        "source": "s3://broad-references/hg38/v0/Homo_sapiens_assembly38.dbsnp138.vcf",
        "dest": f"{REF_DIR}/Homo_sapiens_assembly38.dbsnp138.vcf",
        "size_gb": 2.7,
        "required": False,
    },
}

# Parabricks Docker image
PARABRICKS_IMAGE = "nvcr.io/nvidia/clara/clara-parabricks:4.3.0-1"


class ReferenceSetup:
    """Provision reference genome on GPU instance via SSM."""

    def __init__(self, instance_id: str = GPU_INSTANCE_ID, region: str = GPU_REGION):
        self.instance_id = instance_id
        self.region = region

    def _send_ssm_command(self, commands: List[str], timeout: int = 600) -> str:
        """Send shell commands via SSM and return command ID."""
        import boto3

        ssm = boto3.client("ssm", region_name=self.region)
        resp = ssm.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": commands},
            TimeoutSeconds=timeout,
        )
        return resp["Command"]["CommandId"]

    def _wait_ssm_command(self, command_id: str, timeout: int = 600) -> Dict:
        """Wait for SSM command to complete and return output."""
        import boto3
        import time

        ssm = boto3.client("ssm", region_name=self.region)
        for _ in range(timeout // 5):
            time.sleep(5)
            try:
                resp = ssm.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=self.instance_id,
                )
                if resp["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
                    return {
                        "status": resp["Status"],
                        "stdout": resp.get("StandardOutputContent", ""),
                        "stderr": resp.get("StandardErrorContent", ""),
                    }
            except Exception:
                continue
        return {"status": "Timeout", "stdout": "", "stderr": "SSM polling timeout"}

    def check_references(self) -> Dict[str, bool]:
        """Check which reference files are present on the GPU instance."""
        checks = " && ".join(
            f'[ -f {info["dest"]} ] && echo "{name}:YES" || echo "{name}:NO"'
            for name, info in REFERENCE_FILES.items()
        )
        cmd_id = self._send_ssm_command([
            f"mkdir -p {REF_DIR}",
            checks,
            f"nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'NO GPU'",
            f"docker images --format '{{{{.Repository}}}}:{{{{.Tag}}}}' | grep parabricks || echo 'NO PARABRICKS IMAGE'",
        ], timeout=30)
        result = self._wait_ssm_command(cmd_id, timeout=60)

        status = {}
        for line in result.get("stdout", "").split("\n"):
            if ":" in line and ("YES" in line or "NO" in line):
                name, present = line.strip().split(":")
                status[name] = present == "YES"

        # Check for GPU and Parabricks
        stdout = result.get("stdout", "")
        status["gpu_available"] = "NO GPU" not in stdout
        status["parabricks_image"] = "NO PARABRICKS IMAGE" not in stdout

        return status

    def provision_references(self, required_only: bool = True) -> str:
        """Download reference files to GPU instance. Returns SSM command ID."""
        commands = [f"mkdir -p {REF_DIR}", "echo '=== Starting reference download ==='"]

        for name, info in REFERENCE_FILES.items():
            if required_only and not info["required"]:
                continue
            commands.append(
                f'if [ ! -f {info["dest"]} ]; then '
                f'echo "Downloading {name} ({info["size_gb"]} GB)..."; '
                f'aws s3 cp {info["source"]} {info["dest"]} --no-sign-request; '
                f'else echo "{name} already exists"; fi'
            )

        commands.append("echo '=== Reference download complete ==='")
        commands.append(f"ls -lh {REF_DIR}/")

        return self._send_ssm_command(commands, timeout=1800)  # 30 min timeout

    def pull_parabricks_image(self, ngc_api_key: Optional[str] = None) -> str:
        """Pull Parabricks Docker image from NGC."""
        if ngc_api_key is None:
            # Try to get from AWS Secrets Manager
            try:
                import boto3
                sm = boto3.client("secretsmanager", region_name=self.region)
                secret = sm.get_secret_value(SecretId="cognisom/ngc-api-key")
                ngc_api_key = secret["SecretString"]
            except Exception:
                ngc_api_key = ""

        commands = [
            f'echo "{ngc_api_key}" | docker login nvcr.io --username \'$oauthtoken\' --password-stdin'
            if ngc_api_key else "echo 'No NGC key — attempting anonymous pull'",
            f"docker pull {PARABRICKS_IMAGE}",
            f"docker images | grep parabricks",
            "echo '=== Parabricks image ready ==='",
        ]

        return self._send_ssm_command(commands, timeout=600)

    def stage_fastq_from_s3(self, s3_prefix: str, local_dir: str = "/opt/cognisom/fastq") -> str:
        """Copy FASTQ files from S3 to GPU instance local storage."""
        commands = [
            f"mkdir -p {local_dir}",
            f"aws s3 cp {s3_prefix} {local_dir}/ --recursive",
            f"ls -lh {local_dir}/",
            "echo '=== FASTQ staging complete ==='",
        ]
        return self._send_ssm_command(commands, timeout=1800)

    def provision_via_ssm(self, required_only: bool = True) -> Dict:
        """Full provisioning: references + Parabricks image.

        Returns dict with command IDs for monitoring.
        """
        logger.info("Starting reference genome provisioning on %s", self.instance_id)

        ref_cmd = self.provision_references(required_only=required_only)
        pb_cmd = self.pull_parabricks_image()

        return {
            "reference_command_id": ref_cmd,
            "parabricks_command_id": pb_cmd,
            "instance_id": self.instance_id,
        }
