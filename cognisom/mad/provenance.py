"""
Data Provenance Tracking
========================

Records the exact versions of all data sources, models, and databases
used in a MAD Agent decision. Required for FDA 7-Step Credibility
Framework (Step 6: Document Results).
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AnnotationSource:
    """A specific database or reference used during analysis."""

    name: str
    """e.g. 'ClinVar', 'OncoKB', 'COSMIC', 'IEDB'."""

    version: str
    """Version string, e.g. '2026-03', '4.8', 'v98'."""

    date_accessed: str
    """ISO date when this source was used."""

    url: str = ""
    """Reference URL for the source."""

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "date_accessed": self.date_accessed,
            "url": self.url,
        }


@dataclass
class DataProvenance:
    """Complete provenance record for a MAD Agent decision."""

    # Input data
    input_vcf_hash: str = ""
    """SHA256 of the input VCF data."""

    input_type: str = "vcf"
    """'vcf', 'fastq', 'synthetic'."""

    # Reference genome
    reference_genome: str = "GRCh38"
    reference_genome_version: str = "GRCh38.p14"

    # Annotation sources
    annotation_sources: List[AnnotationSource] = field(default_factory=list)

    # Platform versions
    entity_library_version: str = "285-entities-v1"
    pipeline_version: str = "cognisom-mad-v0.1.0"
    treatment_profiles_source: str = "entity_library + hardcoded_v1"

    # Cognisom module versions
    module_versions: Dict[str, str] = field(default_factory=lambda: {
        "treatment_simulator": "v1.0",
        "neoantigen_predictor": "pwm-v1.0",
        "hla_typer": "v1.0",
        "cell_state_classifier": "marker-heuristic-v1.0",
        "patient_profile": "v1.0",
        "digital_twin_config": "v1.0",
    })

    # Environment
    platform_info: str = field(
        default_factory=lambda: f"{platform.system()} {platform.release()}"
    )
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_vcf_hash": self.input_vcf_hash,
            "input_type": self.input_type,
            "reference_genome": self.reference_genome,
            "reference_genome_version": self.reference_genome_version,
            "annotation_sources": [s.to_dict() for s in self.annotation_sources],
            "entity_library_version": self.entity_library_version,
            "pipeline_version": self.pipeline_version,
            "treatment_profiles_source": self.treatment_profiles_source,
            "module_versions": self.module_versions,
            "platform_info": self.platform_info,
            "timestamp": self.timestamp,
        }

    @classmethod
    def create_default(cls, vcf_text: str = "") -> "DataProvenance":
        """Create a provenance record with default annotation sources."""
        vcf_hash = hashlib.sha256(vcf_text.encode()).hexdigest() if vcf_text else ""
        today = datetime.utcnow().strftime("%Y-%m-%d")

        return cls(
            input_vcf_hash=vcf_hash,
            annotation_sources=[
                AnnotationSource(
                    name="Cognisom Driver Database",
                    version="14-genes-v1",
                    date_accessed=today,
                    url="cognisom/genomics/variant_annotator.py",
                ),
                AnnotationSource(
                    name="IEDB Reference Panel",
                    version="20-peptides-v1",
                    date_accessed=today,
                    url="https://www.iedb.org",
                ),
                AnnotationSource(
                    name="HLA Population Frequencies",
                    version="caucasian-10-alleles-v1",
                    date_accessed=today,
                    url="cognisom/genomics/hla_typer.py",
                ),
                AnnotationSource(
                    name="NCCN Prostate Cancer Guidelines",
                    version="v2.2024",
                    date_accessed=today,
                    url="https://www.nccn.org",
                ),
                AnnotationSource(
                    name="Entity Library",
                    version="285-entities",
                    date_accessed=today,
                    url="/opt/cognisom/data/library/entities.db",
                ),
            ],
        )
