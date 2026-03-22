"""
Parabricks Pipeline Runner
============================

Execute NVIDIA Parabricks genomics pipelines on the GPU instance.
Supports germline variant calling (DeepVariant), somatic variant
calling (Mutect2), and RNA-seq fusion detection (STAR-Fusion).

The runner:
1. Ensures GPU instance is running
2. Creates a job directory on the GPU box
3. Launches Parabricks Docker container via SSM
4. Monitors job progress
5. Retrieves result VCF when complete

Usage:
    runner = ParabricksRunner()
    job_id = runner.run_germline("/path/to/R1.fastq.gz", "/path/to/R2.fastq.gz", "SAMPLE-001")
    status = runner.get_job_status(job_id)
    if status["state"] == "completed":
        vcf_text = runner.get_result_vcf(job_id)
"""

import json
import logging
import os
import time
import uuid
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Configuration
GPU_INSTANCE_ID = os.environ.get("GPU_INSTANCE_ID", "i-0ac9eb88c1b046163")
GPU_REGION = os.environ.get("GPU_REGION", "us-west-2")
PARABRICKS_IMAGE = "nvcr.io/nvidia/clara/clara-parabricks:4.7.0-1"
REF_DIR = "/opt/cognisom/ref"
JOBS_DIR = "/opt/cognisom/jobs"
REF_FASTA = f"{REF_DIR}/Homo_sapiens_assembly38.fasta"
KNOWN_SITES = f"{REF_DIR}/Homo_sapiens_assembly38.known_indels.vcf.gz"


class ParabricksRunner:
    """Execute Parabricks pipelines on the GPU instance."""

    def __init__(self):
        try:
            import boto3
            self.ssm = boto3.client("ssm", region_name=GPU_REGION)
            self.ec2 = boto3.client("ec2", region_name=GPU_REGION)
        except Exception as e:
            logger.error("AWS clients failed: %s", e)
            self.ssm = None
            self.ec2 = None

    def _ensure_gpu_running(self) -> bool:
        """Start GPU instance if stopped."""
        try:
            resp = self.ec2.describe_instances(InstanceIds=[GPU_INSTANCE_ID])
            state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
            if state == "running":
                return True
            if state == "stopped":
                self.ec2.start_instances(InstanceIds=[GPU_INSTANCE_ID])
                waiter = self.ec2.get_waiter("instance_status_ok")
                waiter.wait(InstanceIds=[GPU_INSTANCE_ID])
                return True
            return False
        except Exception as e:
            logger.error("GPU instance check failed: %s", e)
            return False

    def _run_ssm(self, commands: list, timeout: int = 3600) -> Tuple[bool, str]:
        """Run commands on GPU instance via SSM."""
        try:
            resp = self.ssm.send_command(
                InstanceIds=[GPU_INSTANCE_ID],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": commands},
                TimeoutSeconds=timeout,
            )
            cmd_id = resp["Command"]["CommandId"]
            # Wait for completion
            for _ in range(timeout // 10):
                time.sleep(10)
                try:
                    inv = self.ssm.get_command_invocation(
                        CommandId=cmd_id, InstanceId=GPU_INSTANCE_ID)
                    if inv["Status"] in ("Success", "Failed", "TimedOut", "Cancelled"):
                        return inv["Status"] == "Success", inv.get("StandardOutputContent", "")
                except Exception:
                    continue
            return False, "Timeout waiting for SSM command"
        except Exception as e:
            return False, str(e)

    def check_parabricks_ready(self) -> Dict:
        """Check if Parabricks and reference genome are installed."""
        ok, output = self._run_ssm([
            "export HOME=/root",
            "docker images --format '{{.Repository}}:{{.Tag}}' | grep parabricks || echo NO_PARABRICKS",
            "ls /opt/cognisom/ref/Homo_sapiens_assembly38.fasta 2>/dev/null && echo REF_OK || echo NO_REF",
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo NO_GPU",
        ], timeout=30)

        return {
            "parabricks_installed": "parabricks" in output and "NO_PARABRICKS" not in output,
            "reference_genome": "REF_OK" in output,
            "gpu_available": "NO_GPU" not in output,
            "gpu_info": output.split("\n")[-2] if "NO_GPU" not in output else "",
            "raw_output": output,
        }

    def run_germline(self, fastq_r1: str, fastq_r2: str,
                     sample_id: str = "") -> str:
        """Run germline variant calling: fq2bam + DeepVariant.

        Args:
            fastq_r1: Path to R1 FASTQ on GPU box or S3 URI
            fastq_r2: Path to R2 FASTQ on GPU box or S3 URI
            sample_id: Sample identifier

        Returns:
            Job ID for tracking
        """
        job_id = f"germline_{sample_id or 'sample'}_{uuid.uuid4().hex[:8]}"
        job_dir = f"{JOBS_DIR}/{job_id}"

        commands = [
            "export HOME=/root",
            f"mkdir -p {job_dir}",
            f'echo \'{{"state":"running","pipeline":"germline","sample":"{sample_id}","start_time":"{time.strftime("%Y-%m-%d %H:%M:%S")}"}}\' > {job_dir}/status.json',
            # Run fq2bam (alignment + sorting + duplicate marking + BQSR)
            f"docker run --rm --gpus all "
            f"-v {REF_DIR}:/ref "
            f"-v {job_dir}:/workdir "
            f"-v $(dirname {fastq_r1}):/input "
            f"{PARABRICKS_IMAGE} "
            f"pbrun fq2bam "
            f"--ref /ref/Homo_sapiens_assembly38.fasta "
            f"--in-fq /input/$(basename {fastq_r1}) /input/$(basename {fastq_r2}) "
            f"--knownSites /ref/Homo_sapiens_assembly38.known_indels.vcf.gz "
            f"--out-bam /workdir/aligned.bam "
            f"2>&1 | tee {job_dir}/fq2bam.log",
            # Run DeepVariant
            f"docker run --rm --gpus all "
            f"-v {REF_DIR}:/ref "
            f"-v {job_dir}:/workdir "
            f"{PARABRICKS_IMAGE} "
            f"pbrun deepvariant "
            f"--ref /ref/Homo_sapiens_assembly38.fasta "
            f"--in-bam /workdir/aligned.bam "
            f"--out-variants /workdir/output.vcf "
            f"2>&1 | tee {job_dir}/deepvariant.log",
            f'echo \'{{"state":"completed","pipeline":"germline","output_vcf":"{job_dir}/output.vcf"}}\' > {job_dir}/status.json',
        ]

        self._ensure_gpu_running()
        self._run_ssm(commands, timeout=7200)  # 2 hour timeout
        return job_id

    def run_somatic(self, tumor_r1: str, tumor_r2: str,
                    normal_r1: str, normal_r2: str,
                    sample_id: str = "") -> str:
        """Run somatic variant calling: fq2bam (x2) + Mutect2.

        Args:
            tumor_r1, tumor_r2: Tumor FASTQ files
            normal_r1, normal_r2: Normal FASTQ files
            sample_id: Sample identifier

        Returns:
            Job ID
        """
        job_id = f"somatic_{sample_id or 'sample'}_{uuid.uuid4().hex[:8]}"
        job_dir = f"{JOBS_DIR}/{job_id}"

        commands = [
            "export HOME=/root",
            f"mkdir -p {job_dir}",
            f'echo \'{{"state":"running","pipeline":"somatic","sample":"{sample_id}"}}\' > {job_dir}/status.json',
            # Align tumor
            f"docker run --rm --gpus all "
            f"-v {REF_DIR}:/ref -v {job_dir}:/workdir "
            f"-v $(dirname {tumor_r1}):/input_tumor "
            f"{PARABRICKS_IMAGE} "
            f"pbrun fq2bam "
            f"--ref /ref/Homo_sapiens_assembly38.fasta "
            f"--in-fq /input_tumor/$(basename {tumor_r1}) /input_tumor/$(basename {tumor_r2}) "
            f"--knownSites /ref/Homo_sapiens_assembly38.known_indels.vcf.gz "
            f"--out-bam /workdir/tumor.bam "
            f"2>&1 | tee {job_dir}/tumor_fq2bam.log",
            # Align normal
            f"docker run --rm --gpus all "
            f"-v {REF_DIR}:/ref -v {job_dir}:/workdir "
            f"-v $(dirname {normal_r1}):/input_normal "
            f"{PARABRICKS_IMAGE} "
            f"pbrun fq2bam "
            f"--ref /ref/Homo_sapiens_assembly38.fasta "
            f"--in-fq /input_normal/$(basename {normal_r1}) /input_normal/$(basename {normal_r2}) "
            f"--knownSites /ref/Homo_sapiens_assembly38.known_indels.vcf.gz "
            f"--out-bam /workdir/normal.bam "
            f"2>&1 | tee {job_dir}/normal_fq2bam.log",
            # Run Mutect2
            f"docker run --rm --gpus all "
            f"-v {REF_DIR}:/ref -v {job_dir}:/workdir "
            f"{PARABRICKS_IMAGE} "
            f"pbrun mutectcaller "
            f"--ref /ref/Homo_sapiens_assembly38.fasta "
            f"--tumor-name tumor --normal-name normal "
            f"--in-tumor-bam /workdir/tumor.bam "
            f"--in-normal-bam /workdir/normal.bam "
            f"--out-vcf /workdir/output.vcf "
            f"2>&1 | tee {job_dir}/mutect2.log",
            f'echo \'{{"state":"completed","pipeline":"somatic","output_vcf":"{job_dir}/output.vcf"}}\' > {job_dir}/status.json',
        ]

        self._ensure_gpu_running()
        self._run_ssm(commands, timeout=14400)  # 4 hour timeout
        return job_id

    def run_rnaseq_fusion(self, fastq_r1: str, fastq_r2: str,
                          sample_id: str = "") -> str:
        """Run RNA-seq alignment + STAR-Fusion for gene fusion detection.

        Returns:
            Job ID
        """
        job_id = f"rnaseq_{sample_id or 'sample'}_{uuid.uuid4().hex[:8]}"
        job_dir = f"{JOBS_DIR}/{job_id}"

        commands = [
            "export HOME=/root",
            f"mkdir -p {job_dir}",
            f'echo \'{{"state":"running","pipeline":"rnaseq","sample":"{sample_id}"}}\' > {job_dir}/status.json',
            # RNA-seq alignment
            f"docker run --rm --gpus all "
            f"-v {REF_DIR}:/ref -v {job_dir}:/workdir "
            f"-v $(dirname {fastq_r1}):/input "
            f"{PARABRICKS_IMAGE} "
            f"pbrun rna_fq2bam "
            f"--ref /ref/Homo_sapiens_assembly38.fasta "
            f"--in-fq /input/$(basename {fastq_r1}) /input/$(basename {fastq_r2}) "
            f"--out-bam /workdir/rna_aligned.bam "
            f"--genome-lib-dir /ref "
            f"2>&1 | tee {job_dir}/rna_fq2bam.log",
            f'echo \'{{"state":"completed","pipeline":"rnaseq","output_bam":"{job_dir}/rna_aligned.bam"}}\' > {job_dir}/status.json',
        ]

        self._ensure_gpu_running()
        self._run_ssm(commands, timeout=7200)
        return job_id

    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a running Parabricks job."""
        ok, output = self._run_ssm([
            f"cat {JOBS_DIR}/{job_id}/status.json 2>/dev/null || "
            f'echo \'{{"state":"not_found"}}\'',
        ], timeout=30)

        try:
            return json.loads(output.strip())
        except Exception:
            return {"state": "unknown", "raw": output}

    def get_result_vcf(self, job_id: str) -> Optional[str]:
        """Download result VCF text from GPU box."""
        ok, output = self._run_ssm([
            f"cat {JOBS_DIR}/{job_id}/output.vcf 2>/dev/null | head -10000",
        ], timeout=60)

        if ok and output and output.startswith("##"):
            return output
        return None

    def list_jobs(self) -> list:
        """List all Parabricks jobs on GPU box."""
        ok, output = self._run_ssm([
            f"ls -d {JOBS_DIR}/*/ 2>/dev/null | while read d; do "
            f"echo $(basename $d) $(cat $d/status.json 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get('state','?'), d.get('pipeline','?'))\" 2>/dev/null); done",
        ], timeout=30)

        jobs = []
        for line in output.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 3:
                jobs.append({
                    "job_id": parts[0],
                    "state": parts[1],
                    "pipeline": parts[2],
                })
        return jobs
