"""
AWS HealthOmics Integration
==============================

Runs genomics workflows via AWS HealthOmics Ready2Run pipelines.
Alternative to self-managed GPU Parabricks — fully serverless,
pay-per-run, zero infrastructure management.

Available workflows:
- NVIDIA Parabricks FQ2BAM (alignment): ~$8.84, ~1:39
- NVIDIA Parabricks DeepVariant (germline): cost varies by coverage
- NVIDIA Parabricks Mutect2 (somatic): cost varies
- GATK-BP fq2vcf (complete germline): ~$10.00, ~12:30
- AlphaFold structure prediction: ~$6-9, ~7-11 min
- ESMFold structure prediction: ~$0.25, ~0:15

Usage:
    omics = HealthOmicsRunner()
    run_id = omics.run_germline_pipeline(
        fastq_r1="s3://cognisom-genomics/fastq/sample/R1.fastq.gz",
        fastq_r2="s3://cognisom-genomics/fastq/sample/R2.fastq.gz",
        sample_id="NA12878",
    )
    status = omics.get_run_status(run_id)
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REGION = os.environ.get("GPU_REGION", "us-west-2")
SEQUENCE_STORE_ID = os.environ.get("OMICS_SEQUENCE_STORE", "4595412546")
OUTPUT_BUCKET = os.environ.get("COGNISOM_GENOMICS_BUCKET", "cognisom-genomics")

# Ready2Run workflow IDs (from aws omics list-workflows)
WORKFLOWS = {
    # Parabricks alignment
    "parabricks_fq2bam_30x": "4974161",
    "parabricks_fq2bam_50x": "8211545",
    "parabricks_fq2bam_5x": "2647398",
    # Parabricks germline
    "parabricks_deepvariant_30x": "7330987",
    "parabricks_deepvariant_50x": "3585800",
    "parabricks_deepvariant_5x": "1993486",
    "parabricks_haplotypecaller_30x": "3021525",
    # Parabricks somatic
    "parabricks_mutect2_50x": "9701407",
    # GATK Best Practices
    "gatk_fq2bam": "3768383",
    "gatk_germline_bam2vcf_30x": "5454617",
    "gatk_germline_fq2vcf_30x": "9500764",
    "gatk_somatic_wes_bam2vcf": "5562080",
    # Structure prediction
    "alphafold_600": "4885129",
    "alphafold_1200": "6094971",
    "esmfold_800": "1830181",
    # Third party
    "ultima_deepvariant_40x": "1617262",
}

# IAM role for HealthOmics runs
OMICS_ROLE = os.environ.get(
    "OMICS_SERVICE_ROLE",
    "arn:aws:iam::780457123717:role/OmicsServiceRole"
)


class HealthOmicsRunner:
    """Run genomics workflows via AWS HealthOmics."""

    def __init__(self):
        try:
            import boto3
            self.omics = boto3.client("omics", region_name=REGION)
            self.s3 = boto3.client("s3", region_name=REGION)
        except Exception as e:
            logger.error("HealthOmics client failed: %s", e)
            self.omics = None

    def run_germline_pipeline(self,
                               fastq_r1: str,
                               fastq_r2: str,
                               sample_id: str,
                               reference_uri: str = None,
                               use_parabricks: bool = True,
                               ) -> Optional[str]:
        """Run germline variant calling pipeline.

        Two options:
        1. Parabricks (GPU, ~$8.84, ~2 min for alignment + ~varies for calling)
        2. GATK Best Practices (CPU, ~$10, ~12 min)

        Args:
            fastq_r1: S3 URI for R1 FASTQ
            fastq_r2: S3 URI for R2 FASTQ
            sample_id: Sample identifier
            reference_uri: S3 URI for reference genome (uses default if None)
            use_parabricks: If True, use Parabricks GPU; else use GATK CPU

        Returns:
            Run ID for tracking, or None on failure
        """
        if use_parabricks:
            workflow_id = WORKFLOWS["parabricks_fq2bam_30x"]
            workflow_name = "Parabricks FQ2BAM 30x"
        else:
            workflow_id = WORKFLOWS["gatk_germline_fq2vcf_30x"]
            workflow_name = "GATK Germline fq2vcf 30x"

        if reference_uri is None:
            reference_uri = f"s3://{OUTPUT_BUCKET}/ref/GRCh38/"

        output_uri = f"s3://{OUTPUT_BUCKET}/results/{sample_id}/"

        try:
            resp = self.omics.start_run(
                workflowId=workflow_id,
                workflowType="READY2RUN",
                name=f"cognisom-{sample_id}-{workflow_name.replace(' ', '-')}",
                roleArn=OMICS_ROLE,
                outputUri=output_uri,
                parameters={
                    "sample_name": sample_id,
                    "fastq_1": fastq_r1,
                    "fastq_2": fastq_r2,
                },
                tags={
                    "project": "cognisom",
                    "sample": sample_id,
                    "pipeline": workflow_name,
                },
            )
            run_id = resp["id"]
            logger.info("Started HealthOmics run %s: %s for %s",
                        run_id, workflow_name, sample_id)
            return run_id

        except Exception as e:
            logger.error("HealthOmics run failed: %s", e)
            return None

    def run_somatic_pipeline(self,
                              tumor_bam: str,
                              normal_bam: str,
                              sample_id: str) -> Optional[str]:
        """Run somatic variant calling (Mutect2)."""
        workflow_id = WORKFLOWS["parabricks_mutect2_50x"]
        output_uri = f"s3://{OUTPUT_BUCKET}/results/{sample_id}/"

        try:
            resp = self.omics.start_run(
                workflowId=workflow_id,
                workflowType="READY2RUN",
                name=f"cognisom-{sample_id}-mutect2",
                roleArn=OMICS_ROLE,
                outputUri=output_uri,
                parameters={
                    "sample_name": sample_id,
                    "tumor_bam": tumor_bam,
                    "normal_bam": normal_bam,
                },
                tags={"project": "cognisom", "sample": sample_id},
            )
            return resp["id"]
        except Exception as e:
            logger.error("Somatic pipeline failed: %s", e)
            return None

    def run_alphafold(self, sequence: str, protein_name: str) -> Optional[str]:
        """Run AlphaFold structure prediction."""
        if len(sequence) <= 600:
            workflow_id = WORKFLOWS["alphafold_600"]
        else:
            workflow_id = WORKFLOWS["alphafold_1200"]

        output_uri = f"s3://{OUTPUT_BUCKET}/results/alphafold/{protein_name}/"

        try:
            resp = self.omics.start_run(
                workflowId=workflow_id,
                workflowType="READY2RUN",
                name=f"cognisom-alphafold-{protein_name}",
                roleArn=OMICS_ROLE,
                outputUri=output_uri,
                parameters={"fasta": sequence},
                tags={"project": "cognisom", "protein": protein_name},
            )
            return resp["id"]
        except Exception as e:
            logger.error("AlphaFold run failed: %s", e)
            return None

    def run_esmfold(self, sequence: str, protein_name: str) -> Optional[str]:
        """Run ESMFold structure prediction ($0.25, 15 sec)."""
        workflow_id = WORKFLOWS["esmfold_800"]
        output_uri = f"s3://{OUTPUT_BUCKET}/results/esmfold/{protein_name}/"

        try:
            resp = self.omics.start_run(
                workflowId=workflow_id,
                workflowType="READY2RUN",
                name=f"cognisom-esmfold-{protein_name}",
                roleArn=OMICS_ROLE,
                outputUri=output_uri,
                parameters={"fasta": sequence},
                tags={"project": "cognisom", "protein": protein_name},
            )
            return resp["id"]
        except Exception as e:
            logger.error("ESMFold run failed: %s", e)
            return None

    def get_run_status(self, run_id: str) -> Dict:
        """Get status of a HealthOmics run."""
        try:
            resp = self.omics.get_run(id=run_id)
            return {
                "id": resp.get("id"),
                "status": resp.get("status"),
                "name": resp.get("name"),
                "start_time": str(resp.get("startTime", "")),
                "stop_time": str(resp.get("stopTime", "")),
                "output_uri": resp.get("outputUri", ""),
                "log_level": resp.get("logLevel", ""),
            }
        except Exception as e:
            return {"id": run_id, "status": "error", "error": str(e)}

    def wait_for_completion(self, run_id: str,
                             timeout_minutes: int = 30,
                             poll_interval: int = 30) -> Dict:
        """Wait for a run to complete."""
        for _ in range(timeout_minutes * 60 // poll_interval):
            status = self.get_run_status(run_id)
            state = status.get("status", "")
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                return status
            time.sleep(poll_interval)
        return {"id": run_id, "status": "timeout"}

    def list_runs(self, max_results: int = 20) -> List[Dict]:
        """List recent HealthOmics runs."""
        try:
            resp = self.omics.list_runs(maxResults=max_results)
            return [
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "status": r.get("status"),
                    "start_time": str(r.get("startTime", "")),
                }
                for r in resp.get("items", [])
            ]
        except Exception as e:
            logger.error("List runs failed: %s", e)
            return []

    def get_available_workflows(self) -> List[Dict]:
        """List all available Ready2Run workflows."""
        try:
            resp = self.omics.list_workflows(type="READY2RUN")
            return [
                {
                    "id": w.get("id"),
                    "name": w.get("name"),
                    "status": w.get("status"),
                    "type": w.get("type"),
                }
                for w in resp.get("items", [])
            ]
        except Exception as e:
            logger.error("List workflows failed: %s", e)
            return []
