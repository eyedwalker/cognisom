"""
Pipeline Comparison: Self-Managed L40S vs HealthOmics
=======================================================

Runs the same NA12878 sample through both execution paths and
compares cost, speed, and accuracy.

Self-managed: L40S GPU + Parabricks Docker (cognisom-dedicated)
HealthOmics: AWS Ready2Run Parabricks workflow (serverless)

Both outputs compared against GIAB v4.2.1 truth set.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineComparison:
    """Head-to-head comparison of two pipeline execution paths."""
    sample: str = "NA12878"
    reference: str = "GRCh38"
    coverage: str = "30x"

    # Self-managed results
    sm_alignment_seconds: float = 0.0
    sm_calling_seconds: float = 0.0
    sm_total_seconds: float = 0.0
    sm_cost_usd: float = 0.0
    sm_precision: float = 0.0
    sm_recall: float = 0.0
    sm_f1: float = 0.0
    sm_snp_f1: float = 0.0
    sm_indel_f1: float = 0.0
    sm_total_calls: int = 0
    sm_pass_calls: int = 0

    # HealthOmics results
    ho_alignment_seconds: float = 0.0
    ho_calling_seconds: float = 0.0
    ho_total_seconds: float = 0.0
    ho_cost_usd: float = 0.0
    ho_precision: float = 0.0
    ho_recall: float = 0.0
    ho_f1: float = 0.0
    ho_snp_f1: float = 0.0
    ho_indel_f1: float = 0.0
    ho_total_calls: int = 0
    ho_pass_calls: int = 0

    # Winner determination
    speed_winner: str = ""
    cost_winner: str = ""
    accuracy_winner: str = ""

    def determine_winners(self):
        """Determine which pipeline wins on each metric."""
        if self.sm_total_seconds > 0 and self.ho_total_seconds > 0:
            self.speed_winner = "Self-Managed" if self.sm_total_seconds < self.ho_total_seconds else "HealthOmics"
        if self.sm_cost_usd > 0 and self.ho_cost_usd > 0:
            self.cost_winner = "Self-Managed" if self.sm_cost_usd < self.ho_cost_usd else "HealthOmics"
        if self.sm_f1 > 0 and self.ho_f1 > 0:
            self.accuracy_winner = "Self-Managed" if self.sm_f1 > self.ho_f1 else "HealthOmics"

    def to_dict(self) -> Dict:
        self.determine_winners()
        return {
            "sample": self.sample,
            "reference": self.reference,
            "coverage": self.coverage,
            "self_managed": {
                "pipeline": "L40S + Parabricks Docker",
                "gpu": "NVIDIA L40S 48GB (g6e.2xlarge)",
                "timing": {
                    "alignment_seconds": self.sm_alignment_seconds,
                    "calling_seconds": self.sm_calling_seconds,
                    "total_seconds": self.sm_total_seconds,
                    "total_minutes": round(self.sm_total_seconds / 60, 1),
                },
                "cost_usd": self.sm_cost_usd,
                "accuracy": {
                    "precision": round(self.sm_precision, 4),
                    "recall": round(self.sm_recall, 4),
                    "f1": round(self.sm_f1, 4),
                    "snp_f1": round(self.sm_snp_f1, 4),
                    "indel_f1": round(self.sm_indel_f1, 4),
                },
                "variants": {
                    "total_calls": self.sm_total_calls,
                    "pass_calls": self.sm_pass_calls,
                },
            },
            "healthomics": {
                "pipeline": "AWS HealthOmics Ready2Run Parabricks",
                "gpu": "Managed (serverless)",
                "timing": {
                    "alignment_seconds": self.ho_alignment_seconds,
                    "calling_seconds": self.ho_calling_seconds,
                    "total_seconds": self.ho_total_seconds,
                    "total_minutes": round(self.ho_total_seconds / 60, 1),
                },
                "cost_usd": self.ho_cost_usd,
                "accuracy": {
                    "precision": round(self.ho_precision, 4),
                    "recall": round(self.ho_recall, 4),
                    "f1": round(self.ho_f1, 4),
                    "snp_f1": round(self.ho_snp_f1, 4),
                    "indel_f1": round(self.ho_indel_f1, 4),
                },
                "variants": {
                    "total_calls": self.ho_total_calls,
                    "pass_calls": self.ho_pass_calls,
                },
            },
            "winners": {
                "speed": self.speed_winner,
                "cost": self.cost_winner,
                "accuracy": self.accuracy_winner,
            },
            "citations": [
                "Zook et al. Nature Biotechnology 2019; GIAB v4.2.1",
                "Poplin et al. Nature Biotechnology 2018; DeepVariant",
            ],
        }


class PipelineComparisonRunner:
    """Orchestrate head-to-head pipeline comparison."""

    FASTQ_R1_S3 = "s3://cognisom-genomics/fastq/NA12878/NA12878_30x_R1.fastq.gz"
    FASTQ_R2_S3 = "s3://cognisom-genomics/fastq/NA12878/NA12878_30x_R2.fastq.gz"

    def __init__(self):
        self.comparison = PipelineComparison()

    def run_self_managed(self, benchmark_result: Dict = None) -> None:
        """Load self-managed benchmark results."""
        if benchmark_result:
            acc = benchmark_result.get("accuracy", {})
            snp = benchmark_result.get("snp_accuracy", {})
            indel = benchmark_result.get("indel_accuracy", {})
            timing = benchmark_result.get("timing", {})
            variants = benchmark_result.get("variants", {})

            self.comparison.sm_alignment_seconds = timing.get("alignment_seconds", 0)
            self.comparison.sm_calling_seconds = timing.get("variant_calling_seconds", 0)
            self.comparison.sm_total_seconds = timing.get("total_seconds", 0)
            self.comparison.sm_cost_usd = benchmark_result.get("cost_usd", 0)
            self.comparison.sm_precision = acc.get("precision", 0)
            self.comparison.sm_recall = acc.get("recall", 0)
            self.comparison.sm_f1 = acc.get("f1", 0)
            self.comparison.sm_snp_f1 = snp.get("f1", 0)
            self.comparison.sm_indel_f1 = indel.get("f1", 0)
            self.comparison.sm_total_calls = variants.get("total_calls", 0)
            self.comparison.sm_pass_calls = variants.get("pass_calls", 0)

    def run_healthomics(self) -> Optional[str]:
        """Launch HealthOmics pipeline and return run ID."""
        from cognisom.infrastructure.healthomics import HealthOmicsRunner

        runner = HealthOmicsRunner()
        run_id = runner.run_germline_pipeline(
            fastq_r1=self.FASTQ_R1_S3,
            fastq_r2=self.FASTQ_R2_S3,
            sample_id="NA12878",
            use_parabricks=True,
        )
        return run_id

    def load_healthomics_result(self, cost_usd: float = 8.84,
                                  total_seconds: float = 99.0,
                                  benchmark_result: Dict = None) -> None:
        """Load HealthOmics results (manual or from benchmark)."""
        self.comparison.ho_cost_usd = cost_usd
        self.comparison.ho_total_seconds = total_seconds

        if benchmark_result:
            acc = benchmark_result.get("accuracy", {})
            snp = benchmark_result.get("snp_accuracy", {})
            indel = benchmark_result.get("indel_accuracy", {})
            self.comparison.ho_precision = acc.get("precision", 0)
            self.comparison.ho_recall = acc.get("recall", 0)
            self.comparison.ho_f1 = acc.get("f1", 0)
            self.comparison.ho_snp_f1 = snp.get("f1", 0)
            self.comparison.ho_indel_f1 = indel.get("f1", 0)

    def get_comparison(self) -> PipelineComparison:
        return self.comparison

    def save(self, path: str = None) -> str:
        if path is None:
            path = os.path.join(
                os.environ.get("COGNISOM_DATA_DIR", "data"),
                "pipeline_comparison.json",
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.comparison.to_dict(), f, indent=2)
        return path
