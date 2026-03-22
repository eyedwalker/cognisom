"""
GIAB Benchmark Runner
========================

Runs the Genome in a Bottle (GIAB) NA12878 benchmark to measure
variant calling accuracy. Compares Parabricks DeepVariant output
against the GIAB high-confidence truth set.

Two execution paths tested:
1. Self-managed: L40S GPU + Parabricks Docker
2. HealthOmics: Ready2Run Parabricks workflow

Metrics calculated:
- True Positives (TP): variants in both truth and calls
- False Positives (FP): variants only in calls
- False Negatives (FN): variants only in truth
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (Precision * Recall) / (Precision + Recall)

Citation: Zook et al. Nature Biotechnology 2019 (GIAB v4.2.1)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """GIAB benchmark result for one pipeline run."""
    pipeline: str  # "self_managed_l40s" or "healthomics"
    sample: str = "NA12878"
    reference: str = "GRCh38"

    # Timing
    alignment_seconds: float = 0.0
    variant_calling_seconds: float = 0.0
    total_seconds: float = 0.0

    # Cost
    cost_usd: float = 0.0

    # Variant counts
    total_calls: int = 0
    pass_calls: int = 0
    snps: int = 0
    indels: int = 0

    # Accuracy (from GIAB comparison)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # SNP-specific
    snp_tp: int = 0
    snp_fp: int = 0
    snp_fn: int = 0
    snp_precision: float = 0.0
    snp_recall: float = 0.0
    snp_f1: float = 0.0

    # Indel-specific
    indel_tp: int = 0
    indel_fp: int = 0
    indel_fn: int = 0
    indel_precision: float = 0.0
    indel_recall: float = 0.0
    indel_f1: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "pipeline": self.pipeline,
            "sample": self.sample,
            "reference": self.reference,
            "timing": {
                "alignment_seconds": self.alignment_seconds,
                "variant_calling_seconds": self.variant_calling_seconds,
                "total_seconds": self.total_seconds,
            },
            "cost_usd": self.cost_usd,
            "variants": {
                "total_calls": self.total_calls,
                "pass_calls": self.pass_calls,
                "snps": self.snps,
                "indels": self.indels,
            },
            "accuracy": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
            },
            "snp_accuracy": {
                "precision": round(self.snp_precision, 4),
                "recall": round(self.snp_recall, 4),
                "f1": round(self.snp_f1, 4),
            },
            "indel_accuracy": {
                "precision": round(self.indel_precision, 4),
                "recall": round(self.indel_recall, 4),
                "f1": round(self.indel_f1, 4),
            },
            "citation": "Zook et al. Nature Biotechnology 2019; GIAB v4.2.1",
        }


def calculate_metrics(tp: int, fp: int, fn: int):
    """Calculate precision, recall, F1 from confusion matrix."""
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.0001, precision + recall)
    return precision, recall, f1


def parse_isec_results(comparison_dir: str) -> Dict:
    """Parse bcftools isec output files.

    Files:
        0000.vcf: in truth only (false negatives)
        0001.vcf: in calls only (false positives)
        0002.vcf: in both from truth perspective (true positives)
        0003.vcf: in both from calls perspective (true positives)
    """
    counts = {}
    for fname in ["0000.vcf", "0001.vcf", "0002.vcf", "0003.vcf"]:
        filepath = os.path.join(comparison_dir, fname)
        if os.path.exists(filepath):
            with open(filepath) as f:
                count = sum(1 for line in f if not line.startswith("#"))
            counts[fname] = count
        else:
            counts[fname] = 0

    tp = counts.get("0002.vcf", 0)
    fp = counts.get("0001.vcf", 0)
    fn = counts.get("0000.vcf", 0)

    return {"tp": tp, "fp": fp, "fn": fn}


class GIABBenchmarkRunner:
    """Run and manage GIAB benchmarks on GPU instance via SSM.

    Orchestrates the full pipeline:
    1. Parabricks fq2bam (alignment)
    2. Parabricks DeepVariant (variant calling)
    3. bcftools isec comparison against GIAB truth set
    4. Metrics calculation and reporting
    """

    # Paths on GPU instance
    REF = "/opt/cognisom/ref/Homo_sapiens_assembly38.fasta"
    KNOWN_SITES = "/opt/cognisom/ref/Homo_sapiens_assembly38.known_indels.vcf.gz"
    TRUTH_VCF = "/opt/cognisom/benchmark/giab/truth.vcf.gz"
    TRUTH_BED = "/opt/cognisom/benchmark/giab/truth.bed"
    FASTQ_DIR = "/opt/cognisom/benchmark/giab/wgs"
    OUTPUT_DIR = "/opt/cognisom/jobs/giab_benchmark"
    COMPARISON_DIR = "/opt/cognisom/benchmark/giab/comparison"
    PARABRICKS_IMAGE = "nvcr.io/nvidia/clara/clara-parabricks:4.7.0-1"

    # L40S cost: $1.836/hr on-demand (g6e.2xlarge)
    GPU_COST_PER_HOUR = 1.836

    def __init__(self, instance_id: str = None, region: str = "us-west-2"):
        self.instance_id = instance_id or os.environ.get(
            "GPU_INSTANCE_ID", "i-0ac9eb88c1b046163"
        )
        self.region = region
        try:
            import boto3
            self.ssm = boto3.client("ssm", region_name=region)
        except Exception:
            self.ssm = None
            logger.warning("SSM client not available")

    def _run_ssm(self, commands: list, timeout: int = 7200) -> Optional[str]:
        """Execute commands on GPU via SSM, return stdout."""
        if not self.ssm:
            return None
        try:
            resp = self.ssm.send_command(
                InstanceIds=[self.instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": commands},
                TimeoutSeconds=timeout,
            )
            cmd_id = resp["Command"]["CommandId"]
            # Wait for completion
            import boto3
            waiter = self.ssm.get_waiter("command_executed")
            try:
                waiter.wait(
                    CommandId=cmd_id,
                    InstanceId=self.instance_id,
                    WaiterConfig={"Delay": 30, "MaxAttempts": timeout // 30},
                )
            except Exception:
                pass  # Waiter may timeout, check status manually

            result = self.ssm.get_command_invocation(
                CommandId=cmd_id, InstanceId=self.instance_id
            )
            if result["Status"] == "Success":
                return result["StandardOutputContent"]
            else:
                logger.error("SSM command failed: %s", result.get("StandardErrorContent", ""))
                return None
        except Exception as e:
            logger.error("SSM execution error: %s", e)
            return None

    def check_prerequisites(self) -> Dict:
        """Verify all required files exist on GPU box."""
        output = self._run_ssm([
            f'echo "ref_exists=$(test -f {self.REF} && echo true || echo false)"',
            f'echo "truth_exists=$(test -f {self.TRUTH_VCF} && echo true || echo false)"',
            f'echo "bed_exists=$(test -f {self.TRUTH_BED} && echo true || echo false)"',
            f'echo "r1_exists=$(test -f {self.FASTQ_DIR}/NA12878_30x_R1.fastq.gz && echo true || echo false)"',
            f'echo "r2_exists=$(test -f {self.FASTQ_DIR}/NA12878_30x_R2.fastq.gz && echo true || echo false)"',
            'echo "docker_exists=$(docker images --format "{{.Repository}}:{{.Tag}}" | '
            f'grep -c parabricks || echo 0)"',
        ])
        if not output:
            return {"ready": False, "error": "Cannot reach GPU instance"}

        checks = {}
        for line in output.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                checks[k] = v.strip()

        ready = all(
            checks.get(k) == "true"
            for k in ["ref_exists", "truth_exists", "bed_exists", "r1_exists", "r2_exists"]
        )
        return {"ready": ready, "checks": checks}

    def run_comparison(self, vcf_path: str = None) -> Optional[Dict]:
        """Run bcftools isec comparison against GIAB truth set.

        Args:
            vcf_path: Path to VCF on GPU box. Defaults to benchmark output.

        Returns:
            Dict with tp, fp, fn, precision, recall, f1 for overall/SNP/indel
        """
        if vcf_path is None:
            vcf_path = f"{self.OUTPUT_DIR}/NA12878_deepvariant.vcf"

        script = f"""#!/bin/bash
set -e
echo "=== GIAB Benchmark Comparison ==="
echo "VCF: {vcf_path}"
echo "Truth: {self.TRUTH_VCF}"
echo "BED: {self.TRUTH_BED}"

# Install bcftools if needed
which bcftools > /dev/null 2>&1 || apt-get install -y bcftools > /dev/null 2>&1

# Compress and index the call VCF if needed
if [ ! -f "{vcf_path}.gz" ] && [[ "{vcf_path}" != *.gz ]]; then
    bgzip -c {vcf_path} > {vcf_path}.gz
    tabix -p vcf {vcf_path}.gz
fi

CALL_VCF="{vcf_path}"
if [[ "{vcf_path}" != *.gz ]]; then
    CALL_VCF="{vcf_path}.gz"
fi

# Filter to PASS variants in high-confidence regions
mkdir -p {self.COMPARISON_DIR}
rm -rf {self.COMPARISON_DIR}/*

echo "Filtering to PASS variants in high-confidence regions..."
bcftools view -f PASS -R {self.TRUTH_BED} $CALL_VCF -Oz -o {self.COMPARISON_DIR}/calls_filtered.vcf.gz
tabix -p vcf {self.COMPARISON_DIR}/calls_filtered.vcf.gz

# Run bcftools isec
echo "Running bcftools isec..."
bcftools isec -p {self.COMPARISON_DIR}/isec_all \\
    {self.TRUTH_VCF} {self.COMPARISON_DIR}/calls_filtered.vcf.gz \\
    -R {self.TRUTH_BED}

# Count overall
echo "=== OVERALL ==="
FN=$(grep -cv "^#" {self.COMPARISON_DIR}/isec_all/0000.vcf 2>/dev/null || echo 0)
FP=$(grep -cv "^#" {self.COMPARISON_DIR}/isec_all/0001.vcf 2>/dev/null || echo 0)
TP=$(grep -cv "^#" {self.COMPARISON_DIR}/isec_all/0002.vcf 2>/dev/null || echo 0)
echo "TP=$TP FP=$FP FN=$FN"

# SNP-specific comparison
echo "=== SNP ==="
bcftools view -v snps {self.TRUTH_VCF} -R {self.TRUTH_BED} -Oz -o {self.COMPARISON_DIR}/truth_snps.vcf.gz
tabix -p vcf {self.COMPARISON_DIR}/truth_snps.vcf.gz
bcftools view -v snps {self.COMPARISON_DIR}/calls_filtered.vcf.gz -Oz -o {self.COMPARISON_DIR}/calls_snps.vcf.gz
tabix -p vcf {self.COMPARISON_DIR}/calls_snps.vcf.gz
bcftools isec -p {self.COMPARISON_DIR}/isec_snps \\
    {self.COMPARISON_DIR}/truth_snps.vcf.gz {self.COMPARISON_DIR}/calls_snps.vcf.gz \\
    -R {self.TRUTH_BED}
SNP_FN=$(grep -cv "^#" {self.COMPARISON_DIR}/isec_snps/0000.vcf 2>/dev/null || echo 0)
SNP_FP=$(grep -cv "^#" {self.COMPARISON_DIR}/isec_snps/0001.vcf 2>/dev/null || echo 0)
SNP_TP=$(grep -cv "^#" {self.COMPARISON_DIR}/isec_snps/0002.vcf 2>/dev/null || echo 0)
echo "SNP_TP=$SNP_TP SNP_FP=$SNP_FP SNP_FN=$SNP_FN"

# Indel-specific comparison
echo "=== INDEL ==="
bcftools view -v indels {self.TRUTH_VCF} -R {self.TRUTH_BED} -Oz -o {self.COMPARISON_DIR}/truth_indels.vcf.gz
tabix -p vcf {self.COMPARISON_DIR}/truth_indels.vcf.gz
bcftools view -v indels {self.COMPARISON_DIR}/calls_filtered.vcf.gz -Oz -o {self.COMPARISON_DIR}/calls_indels.vcf.gz
tabix -p vcf {self.COMPARISON_DIR}/calls_indels.vcf.gz
bcftools isec -p {self.COMPARISON_DIR}/isec_indels \\
    {self.COMPARISON_DIR}/truth_indels.vcf.gz {self.COMPARISON_DIR}/calls_indels.vcf.gz \\
    -R {self.TRUTH_BED}
INDEL_FN=$(grep -cv "^#" {self.COMPARISON_DIR}/isec_indels/0000.vcf 2>/dev/null || echo 0)
INDEL_FP=$(grep -cv "^#" {self.COMPARISON_DIR}/isec_indels/0001.vcf 2>/dev/null || echo 0)
INDEL_TP=$(grep -cv "^#" {self.COMPARISON_DIR}/isec_indels/0002.vcf 2>/dev/null || echo 0)
echo "INDEL_TP=$INDEL_TP INDEL_FP=$INDEL_FP INDEL_FN=$INDEL_FN"

# VCF stats
echo "=== VCF_STATS ==="
TOTAL=$(grep -cv "^#" $CALL_VCF 2>/dev/null || echo 0)
PASS_COUNT=$(bcftools view -f PASS $CALL_VCF | grep -cv "^#" 2>/dev/null || echo 0)
TOTAL_SNPS=$(bcftools view -v snps -f PASS $CALL_VCF | grep -cv "^#" 2>/dev/null || echo 0)
TOTAL_INDELS=$(bcftools view -v indels -f PASS $CALL_VCF | grep -cv "^#" 2>/dev/null || echo 0)
echo "TOTAL=$TOTAL PASS=$PASS_COUNT SNPS=$TOTAL_SNPS INDELS=$TOTAL_INDELS"

echo "=== DONE ==="
"""
        output = self._run_ssm([script], timeout=3600)
        if not output:
            return None

        return self._parse_comparison_output(output)

    def _parse_comparison_output(self, output: str) -> Dict:
        """Parse the benchmark comparison script output."""
        result = {}
        section = None

        for line in output.strip().split("\n"):
            line = line.strip()
            if line == "=== OVERALL ===":
                section = "overall"
            elif line == "=== SNP ===":
                section = "snp"
            elif line == "=== INDEL ===":
                section = "indel"
            elif line == "=== VCF_STATS ===":
                section = "vcf_stats"
            elif section and "=" in line and not line.startswith("==="):
                for pair in line.split():
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        try:
                            result[f"{section}_{k.lower()}"] = int(v)
                        except ValueError:
                            result[f"{section}_{k.lower()}"] = v

        return result

    def build_benchmark_result(self, raw: Dict,
                                alignment_seconds: float = 0,
                                calling_seconds: float = 0,
                                pipeline: str = "self_managed_l40s") -> BenchmarkResult:
        """Convert raw comparison output to BenchmarkResult."""
        br = BenchmarkResult(pipeline=pipeline)

        # Timing
        br.alignment_seconds = alignment_seconds
        br.variant_calling_seconds = calling_seconds
        br.total_seconds = alignment_seconds + calling_seconds
        br.cost_usd = round(br.total_seconds / 3600 * self.GPU_COST_PER_HOUR, 2)

        # VCF stats
        br.total_calls = raw.get("vcf_stats_total", 0)
        br.pass_calls = raw.get("vcf_stats_pass", 0)
        br.snps = raw.get("vcf_stats_snps", 0)
        br.indels = raw.get("vcf_stats_indels", 0)

        # Overall accuracy
        br.true_positives = raw.get("overall_tp", 0)
        br.false_positives = raw.get("overall_fp", 0)
        br.false_negatives = raw.get("overall_fn", 0)
        br.precision, br.recall, br.f1 = calculate_metrics(
            br.true_positives, br.false_positives, br.false_negatives
        )

        # SNP accuracy
        br.snp_tp = raw.get("snp_snp_tp", 0)
        br.snp_fp = raw.get("snp_snp_fp", 0)
        br.snp_fn = raw.get("snp_snp_fn", 0)
        br.snp_precision, br.snp_recall, br.snp_f1 = calculate_metrics(
            br.snp_tp, br.snp_fp, br.snp_fn
        )

        # Indel accuracy
        br.indel_tp = raw.get("indel_indel_tp", 0)
        br.indel_fp = raw.get("indel_indel_fp", 0)
        br.indel_fn = raw.get("indel_indel_fn", 0)
        br.indel_precision, br.indel_recall, br.indel_f1 = calculate_metrics(
            br.indel_tp, br.indel_fp, br.indel_fn
        )

        return br

    def save_result(self, result: BenchmarkResult, path: str = None) -> str:
        """Save benchmark result to JSON."""
        if path is None:
            path = os.path.join("data", "benchmark_results.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Load existing results
        results = []
        if os.path.exists(path):
            with open(path) as f:
                results = json.load(f)

        results.append(result.to_dict())
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Saved benchmark result to %s", path)
        return path
