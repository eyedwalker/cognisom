"""
Open-Source Genomic Data Acquisition
======================================

Tiered strategy for acquiring real, full-size sequencing data to validate
the MAD Agent pipeline end-to-end:

Phase 1: Open-access processed data (mutation-level, available today)
Phase 2: Open-access raw sequencing (FASTQ/BAM, public benchmarks)
Phase 3: Controlled-access raw data (dbGaP, requires PI sponsorship)

Each phase includes download helpers, validation, and S3 upload.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

S3_BUCKET = os.environ.get("COGNISOM_GENOMICS_BUCKET", "cognisom-genomics")
LOCAL_DATA_DIR = "/opt/cognisom/benchmark"


# ─────────────────────────────────────────────────────────────────────────
# Phase 1: Open-Access Processed Data (available NOW)
# ─────────────────────────────────────────────────────────────────────────

OPEN_PROCESSED_SOURCES = [
    {
        "name": "SU2C/PCF mCRPC 2019",
        "description": "429 mCRPC patients, 64,566 mutations, treatment outcomes",
        "data_type": "MAF (pre-called mutations)",
        "size_gb": 0.098,
        "patients": 429,
        "download_url": "https://datahub.assets.cbioportal.org/prad_su2c_2019.tar.gz",
        "has_treatment_data": True,
        "has_survival_data": True,
        "has_hla": False,
        "has_raw_reads": False,
        "cancer_type": "prostate (mCRPC)",
        "citation": "Abida et al. PNAS 2019; 116:11428",
        "status": "integrated",  # Already in cognisom
    },
    {
        "name": "TCGA-PRAD PanCancer",
        "description": "494 prostate cancer patients via cBioPortal API",
        "data_type": "MAF + clinical",
        "size_gb": 0.050,
        "patients": 494,
        "download_url": "https://www.cbioportal.org/api/",
        "has_treatment_data": False,
        "has_survival_data": True,
        "has_hla": False,
        "has_raw_reads": False,
        "cancer_type": "prostate (primary)",
        "citation": "Cancer Genome Atlas Research Network, Cell 2015",
        "status": "integrated",
    },
    {
        "name": "AACR Project GENIE v15",
        "description": "200,000+ patients, 700+ institutions, real-world clinical genomics",
        "data_type": "MAF + clinical",
        "size_gb": 2.5,
        "patients": 200000,
        "download_url": "https://www.synapse.org/#!Synapse:syn7222066",
        "has_treatment_data": True,
        "has_survival_data": True,
        "has_hla": False,
        "has_raw_reads": False,
        "cancer_type": "pan-cancer",
        "citation": "de Bruijn et al. Cancer Res 2023; PMID: 37668528",
        "status": "available",  # Not yet downloaded
    },
    {
        "name": "GDC Open Somatic Mutations",
        "description": "Masked somatic mutations for 11,000+ TCGA patients, 33 cancer types",
        "data_type": "MAF (open-access tier)",
        "size_gb": 5.0,
        "patients": 11000,
        "download_url": "https://portal.gdc.cancer.gov/",
        "has_treatment_data": False,
        "has_survival_data": True,
        "has_hla": False,
        "has_raw_reads": False,
        "cancer_type": "pan-cancer (33 types)",
        "citation": "Grossman et al. NEJM 2016; 375:1109",
        "status": "available",
    },
]


# ─────────────────────────────────────────────────────────────────────────
# Phase 2: Open-Access RAW Sequencing (FASTQ/BAM)
# ─────────────────────────────────────────────────────────────────────────

OPEN_RAW_SOURCES = [
    {
        "name": "GIAB NA12878 (HG001) 30x WGS",
        "description": "Gold-standard germline reference. Full 30x WGS FASTQ.",
        "data_type": "FASTQ (paired-end)",
        "size_gb": 90.0,
        "samples": 1,
        "matched_normal": False,
        "download": "ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/",
        "s3_location": "s3://cognisom-genomics/fastq/NA12878/",
        "truth_vcf": "GIAB v4.2.1 high-confidence calls",
        "use_case": "Germline variant calling benchmark",
        "cancer_type": "N/A (germline only)",
        "status": "on_s3",  # Already uploaded
    },
    {
        "name": "SEQC2 Somatic Reference Samples",
        "description": (
            "NCI SEQC2 consortium: matched tumor-normal pairs from "
            "well-characterized cell lines. Gold standard for somatic calling. "
            "Includes: HCC1395 (tumor) + HCC1395BL (matched normal)."
        ),
        "data_type": "FASTQ (paired-end, tumor + normal)",
        "size_gb": 200.0,  # ~100 GB per sample
        "samples": 2,  # tumor + normal
        "matched_normal": True,
        "download": (
            "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/seqc/Somatic/"
        ),
        "truth_vcf": "SEQC2 consortium truth set (3 callers consensus)",
        "use_case": "Somatic variant calling + MAD Agent full pipeline test",
        "cancer_type": "Breast cancer cell line (HCC1395)",
        "status": "available",
        "fastq_urls": {
            "tumor_r1": "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/seqc/Somatic/WGS/NIST/HCC1395_R1.fastq.gz",
            "tumor_r2": "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/seqc/Somatic/WGS/NIST/HCC1395_R2.fastq.gz",
            "normal_r1": "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/seqc/Somatic/WGS/NIST/HCC1395BL_R1.fastq.gz",
            "normal_r2": "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/seqc/Somatic/WGS/NIST/HCC1395BL_R2.fastq.gz",
        },
    },
    {
        "name": "1000 Genomes Project — High Coverage",
        "description": "3,202 samples at 30x WGS. Diverse populations. Open access CRAM/BAM.",
        "data_type": "CRAM (aligned reads)",
        "size_gb": 40.0,  # per sample
        "samples": 3202,
        "matched_normal": False,
        "download": (
            "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
            "1000G_2504_high_coverage/data/"
        ),
        "truth_vcf": "1000 Genomes phased call set",
        "use_case": "Population variant filtering (gnomAD alternative), HLA diversity",
        "cancer_type": "N/A (germline only)",
        "status": "available",
    },
    {
        "name": "Platinum Genomes (Illumina)",
        "description": "NA12878 family trio + extended pedigree. 50x WGS. Truth set from pedigree consistency.",
        "data_type": "FASTQ + BAM",
        "size_gb": 150.0,
        "samples": 6,
        "matched_normal": False,
        "download": "https://www.illumina.com/platinumgenomes.html",
        "truth_vcf": "Pedigree-validated calls",
        "use_case": "High-confidence germline benchmark (complementary to GIAB)",
        "cancer_type": "N/A (germline)",
        "status": "available",
    },
]


# ─────────────────────────────────────────────────────────────────────────
# Phase 3: Controlled-Access RAW Data (requires dbGaP/EGA)
# ─────────────────────────────────────────────────────────────────────────

CONTROLLED_SOURCES = [
    {
        "name": "TCGA Prostate (PRAD) — Raw BAM",
        "description": "494 patients with WGS/WES BAMs (tumor + normal matched pairs)",
        "data_type": "BAM (aligned, needs reversion to FASTQ for Parabricks)",
        "size_gb_per_patient": 120.0,
        "patients": 494,
        "matched_normal": True,
        "access": "dbGaP",
        "dbgap_study": "phs000178",
        "requires_pi": True,
        "approval_time": "4-8 weeks",
        "has_hla": True,  # Can be extracted from normal BAM
        "has_treatment_data": False,
        "cancer_type": "prostate",
    },
    {
        "name": "TCGA Pan-Cancer (all 33 types) — Raw BAM",
        "description": "11,000+ patients, tumor + normal WGS/WES",
        "data_type": "BAM",
        "size_gb_per_patient": 120.0,
        "patients": 11000,
        "matched_normal": True,
        "access": "dbGaP",
        "dbgap_study": "phs000178",
        "requires_pi": True,
        "approval_time": "4-8 weeks",
        "has_hla": True,
        "has_treatment_data": False,
        "cancer_type": "pan-cancer (33 types)",
    },
    {
        "name": "CheckMate-067 Melanoma (Anti-PD-1 + Anti-CTLA-4)",
        "description": "Ipilimumab vs nivolumab vs combo in melanoma. WES + RNA-seq with treatment response.",
        "data_type": "BAM (WES + RNA)",
        "size_gb_per_patient": 20.0,
        "patients": 945,
        "matched_normal": True,
        "access": "EGA (European Genome-phenome Archive)",
        "ega_study": "EGAS00001003436",
        "requires_pi": True,
        "approval_time": "6-12 weeks",
        "has_hla": True,
        "has_treatment_data": True,
        "cancer_type": "melanoma",
        "why_important": (
            "Gold standard for immunotherapy response prediction. "
            "Has actual treatment outcomes (response/progression/survival) "
            "linked to genomic data. Perfect for validating MAD Agent."
        ),
    },
    {
        "name": "IMvigor210 Bladder (Anti-PD-L1)",
        "description": "Atezolizumab in metastatic urothelial cancer. WES + RNA-seq + treatment outcomes.",
        "data_type": "BAM (WES + RNA)",
        "size_gb_per_patient": 15.0,
        "patients": 348,
        "matched_normal": True,
        "access": "EGA",
        "ega_study": "EGAS00001002556",
        "requires_pi": True,
        "approval_time": "6-12 weeks",
        "has_hla": True,
        "has_treatment_data": True,
        "cancer_type": "bladder (urothelial)",
        "why_important": "Published immunotherapy biomarker study with full genomic + outcome data.",
    },
]


# ─────────────────────────────────────────────────────────────────────────
# Required Tools for Real Pipeline
# ─────────────────────────────────────────────────────────────────────────

REQUIRED_TOOLS = {
    "variant_calling": {
        "name": "NVIDIA Parabricks 4.3",
        "docker": "nvcr.io/nvidia/clara/clara-parabricks:4.3.0-1",
        "gpu_required": True,
        "min_gpu_ram": "24 GB (A10), 40 GB (A100) recommended",
        "status": "integrated",
        "modal_compatible": True,
    },
    "hla_typing": {
        "name": "OptiType 1.3.5",
        "install": "conda install -c bioconda optitype",
        "docker": "fred2/optitype:1.3.5",
        "gpu_required": False,
        "input": "Normal (germline) BAM or FASTQ",
        "output": "6 HLA-I alleles (A, B, C × 2) at 4-digit resolution",
        "accuracy": ">97% concordance with serological typing",
        "status": "not_integrated",
        "modal_compatible": True,
    },
    "neoantigen_binding": {
        "name": "NetMHCpan 4.1",
        "install": "Download from https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/",
        "docker": None,  # Academic license required
        "gpu_required": False,
        "input": "Peptide sequences (8-14 AA) + HLA alleles",
        "output": "Predicted IC50 binding affinity (nM) + %rank",
        "accuracy": "AUC >0.95 for most alleles",
        "status": "not_integrated",
        "modal_compatible": True,
        "alternative": "MHCflurry 2.0 (pip install mhcflurry, open-source, similar accuracy)",
    },
    "bam_to_fastq": {
        "name": "samtools 1.19",
        "install": "apt install samtools OR conda install -c bioconda samtools",
        "command": "samtools fastq -1 R1.fq.gz -2 R2.fq.gz -0 /dev/null -s /dev/null input.bam",
        "gpu_required": False,
        "status": "available_on_gpu_box",
        "modal_compatible": True,
    },
    "population_filter": {
        "name": "gnomAD v4 (population allele frequencies)",
        "download": "https://gnomad.broadinstitute.org/downloads",
        "size_gb": 460.0,  # Full sites VCF
        "format": "VCF with AF field",
        "use": "Filter germline variants from somatic calls (AF > 0.01 = likely germline)",
        "status": "not_integrated",
    },
}


# ─────────────────────────────────────────────────────────────────────────
# Download & Setup Helpers
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class DownloadTask:
    """A data file to download for the pipeline."""
    name: str
    url: str
    local_path: str
    s3_path: str
    size_gb: float
    status: str = "pending"
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "local_path": self.local_path,
            "s3_path": self.s3_path,
            "size_gb": self.size_gb,
            "status": self.status,
        }


def get_seqc2_download_tasks() -> List[DownloadTask]:
    """Get download tasks for SEQC2 matched tumor-normal benchmark.

    This is the highest-priority download: real matched tumor-normal
    FASTQ files with a truth set for somatic variant validation.
    """
    seqc2 = next(s for s in OPEN_RAW_SOURCES if s["name"].startswith("SEQC2"))
    fastqs = seqc2["fastq_urls"]

    return [
        DownloadTask(
            name="SEQC2 HCC1395 Tumor R1",
            url=fastqs["tumor_r1"],
            local_path=f"{LOCAL_DATA_DIR}/seqc2/HCC1395_R1.fastq.gz",
            s3_path=f"s3://{S3_BUCKET}/fastq/SEQC2-HCC1395/tumor_R1.fastq.gz",
            size_gb=50.0,
        ),
        DownloadTask(
            name="SEQC2 HCC1395 Tumor R2",
            url=fastqs["tumor_r2"],
            local_path=f"{LOCAL_DATA_DIR}/seqc2/HCC1395_R2.fastq.gz",
            s3_path=f"s3://{S3_BUCKET}/fastq/SEQC2-HCC1395/tumor_R2.fastq.gz",
            size_gb=50.0,
        ),
        DownloadTask(
            name="SEQC2 HCC1395BL Normal R1",
            url=fastqs["normal_r1"],
            local_path=f"{LOCAL_DATA_DIR}/seqc2/HCC1395BL_R1.fastq.gz",
            s3_path=f"s3://{S3_BUCKET}/fastq/SEQC2-HCC1395/normal_R1.fastq.gz",
            size_gb=50.0,
        ),
        DownloadTask(
            name="SEQC2 HCC1395BL Normal R2",
            url=fastqs["normal_r2"],
            local_path=f"{LOCAL_DATA_DIR}/seqc2/HCC1395BL_R2.fastq.gz",
            s3_path=f"s3://{S3_BUCKET}/fastq/SEQC2-HCC1395/normal_R2.fastq.gz",
            size_gb=50.0,
        ),
    ]


def get_acquisition_plan() -> Dict[str, Any]:
    """Return the complete data acquisition plan with status."""
    return {
        "phase_1_processed": {
            "description": "Pre-called mutations (available NOW, no approval needed)",
            "sources": OPEN_PROCESSED_SOURCES,
            "total_patients": sum(s["patients"] for s in OPEN_PROCESSED_SOURCES),
            "validates": "Decision logic, biomarker concordance, treatment ranking",
        },
        "phase_2_raw": {
            "description": "Full-size FASTQ/BAM (open access, ~200 GB per matched pair)",
            "sources": OPEN_RAW_SOURCES,
            "priority_download": "SEQC2 HCC1395 tumor-normal (gold standard somatic benchmark)",
            "validates": "Variant calling, HLA typing, full Parabricks pipeline",
        },
        "phase_3_controlled": {
            "description": "Real patient data with treatment outcomes (requires dbGaP/EGA access)",
            "sources": CONTROLLED_SOURCES,
            "validates": "Clinical relevance, treatment response prediction accuracy",
            "required_for": "FDA credibility, publication, clinical utility evidence",
        },
        "tools_needed": REQUIRED_TOOLS,
        "compute_estimate": {
            "seqc2_matched_pair": {
                "parabricks_alignment": "~50 min (2 × 25 min on A100)",
                "mutect2_somatic": "~45 min on A100",
                "total_gpu_time": "~95 min",
                "modal_cost": "~$5.70 (A100 @ $3.60/hr)",
            },
            "tcga_per_patient": {
                "bam_to_fastq": "~30 min (samtools revert)",
                "parabricks_somatic": "~95 min on A100",
                "hla_typing": "~5 min (OptiType, CPU)",
                "neoantigen_prediction": "~2 min (NetMHCpan, CPU)",
                "total": "~2.5 hrs per patient",
                "modal_cost": "~$9 per patient",
            },
        },
    }


def print_acquisition_summary():
    """Print a human-readable summary of the acquisition plan."""
    plan = get_acquisition_plan()

    print("=" * 70)
    print("COGNISOM DATA ACQUISITION PLAN")
    print("=" * 70)

    for phase_key, phase in plan.items():
        if phase_key == "tools_needed" or phase_key == "compute_estimate":
            continue
        print(f"\n{'─' * 70}")
        print(f"  {phase_key.upper()}: {phase['description']}")
        print(f"  Validates: {phase['validates']}")
        if "sources" in phase:
            for src in phase["sources"]:
                status_icon = {"integrated": "✅", "on_s3": "📦", "available": "⬜"}
                icon = status_icon.get(src.get("status", ""), "⬜")
                name = src["name"]
                size = src.get("size_gb", src.get("size_gb_per_patient", "?"))
                print(f"    {icon} {name} ({size} GB)")

    print(f"\n{'─' * 70}")
    print("  REQUIRED TOOLS:")
    for key, tool in plan["tools_needed"].items():
        status_icon = {"integrated": "✅", "not_integrated": "❌",
                       "available_on_gpu_box": "📦"}
        icon = status_icon.get(tool["status"], "⬜")
        print(f"    {icon} {tool['name']} — GPU: {'Yes' if tool.get('gpu_required') else 'No'}")

    print(f"\n{'─' * 70}")
    print("  COMPUTE ESTIMATES:")
    for task, est in plan["compute_estimate"].items():
        print(f"    {task}:")
        for k, v in est.items():
            print(f"      {k}: {v}")


if __name__ == "__main__":
    print_acquisition_summary()
