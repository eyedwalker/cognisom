"""
S3 Genomics Data Manager
==========================

Manages genomic data flow through the S3 bucket per the DMP:
  s3://cognisom-genomics/
  ├── fastq/       → Raw reads (auto-archive to Glacier 90d)
  ├── bam/         → Aligned reads
  ├── vcf/         → Variant calls
  ├── results/     → Pipeline outputs
  ├── ref/         → Reference genomes
  ├── benchmark/   → GIAB, SEQC2 truth sets
  ├── truth-sets/  → Validation data
  ├── clinical/    → De-identified clinical data

All data encrypted at rest (AES-256). Transfer acceleration enabled.
Lifecycle rules auto-tier old data to IA → Glacier.
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

BUCKET = os.environ.get("COGNISOM_GENOMICS_BUCKET", "cognisom-genomics")
REGION = os.environ.get("GPU_REGION", "us-west-2")


class S3GenomicsManager:
    """Manage genomic data in S3 per the DMP architecture."""

    def __init__(self):
        try:
            import boto3
            self.s3 = boto3.client("s3", region_name=REGION)
        except Exception:
            self.s3 = None
            logger.warning("S3 client not available")

    def upload_fastq(self, local_path: str, sample_id: str) -> str:
        """Upload FASTQ to the landing zone."""
        filename = os.path.basename(local_path)
        key = f"fastq/{sample_id}/{filename}"
        self.s3.upload_file(local_path, BUCKET, key)
        logger.info("Uploaded FASTQ: s3://%s/%s", BUCKET, key)
        return f"s3://{BUCKET}/{key}"

    def upload_vcf(self, local_path: str, sample_id: str) -> str:
        """Upload VCF to the results store."""
        filename = os.path.basename(local_path)
        key = f"vcf/{sample_id}/{filename}"
        self.s3.upload_file(local_path, BUCKET, key)
        logger.info("Uploaded VCF: s3://%s/%s", BUCKET, key)
        return f"s3://{BUCKET}/{key}"

    def upload_result(self, local_path: str, sample_id: str,
                      result_type: str = "report") -> str:
        """Upload pipeline result (report, figures, etc.)."""
        filename = os.path.basename(local_path)
        key = f"results/{sample_id}/{result_type}/{filename}"
        self.s3.upload_file(local_path, BUCKET, key)
        return f"s3://{BUCKET}/{key}"

    def download_to_gpu(self, s3_uri: str, local_path: str) -> str:
        """Download from S3 to GPU instance local storage."""
        bucket, key = self._parse_uri(s3_uri)
        self.s3.download_file(bucket, key, local_path)
        return local_path

    def generate_presigned_upload_url(self, sample_id: str,
                                       filename: str,
                                       expiry_seconds: int = 3600) -> str:
        """Generate pre-signed URL for direct upload from browser."""
        key = f"fastq/{sample_id}/{filename}"
        url = self.s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=expiry_seconds,
        )
        return url

    def generate_presigned_download_url(self, key: str,
                                         expiry_seconds: int = 3600) -> str:
        """Generate pre-signed URL for download."""
        url = self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=expiry_seconds,
        )
        return url

    def list_samples(self) -> Dict[str, Dict]:
        """List all samples in the bucket."""
        samples = {}
        for prefix in ["fastq/", "vcf/", "results/"]:
            try:
                resp = self.s3.list_objects_v2(
                    Bucket=BUCKET, Prefix=prefix, Delimiter="/")
                for cp in resp.get("CommonPrefixes", []):
                    sample_id = cp["Prefix"].split("/")[1]
                    samples.setdefault(sample_id, {"data_types": []})
                    samples[sample_id]["data_types"].append(prefix.rstrip("/"))
            except Exception:
                pass
        return samples

    def get_bucket_stats(self) -> Dict:
        """Get storage statistics."""
        stats = {"bucket": BUCKET, "folders": {}}
        for prefix in ["fastq/", "bam/", "vcf/", "results/", "ref/",
                        "benchmark/", "truth-sets/", "clinical/"]:
            try:
                resp = self.s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
                count = resp.get("KeyCount", 0)
                total_size = sum(o.get("Size", 0) for o in resp.get("Contents", []))
                stats["folders"][prefix.rstrip("/")] = {
                    "objects": count,
                    "size_mb": round(total_size / 1024 / 1024, 1),
                }
            except Exception:
                stats["folders"][prefix.rstrip("/")] = {"objects": 0, "size_mb": 0}
        return stats

    @staticmethod
    def _parse_uri(s3_uri: str):
        """Parse s3://bucket/key into (bucket, key)."""
        path = s3_uri.replace("s3://", "")
        parts = path.split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""
