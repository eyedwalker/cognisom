"""
Patient Profile
===============

Aggregate patient genomic data into a unified profile that drives
the molecular digital twin simulation.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .vcf_parser import Variant, VCFParser
from .variant_annotator import VariantAnnotator, PROSTATE_CANCER_DRIVERS
from .gene_protein_mapper import GeneProteinMapper, ProteinInfo

logger = logging.getLogger(__name__)


@dataclass
class PatientProfile:
    """Complete patient genomic profile for the molecular digital twin.

    Aggregates variant data, affected proteins, cancer drivers,
    and computed biomarkers (TMB, MSI) into a single object that
    can be used by downstream phases (Cell2Sentence, treatment simulator).
    """
    # Identity
    patient_id: str = "anonymous"
    cancer_type: str = "prostate"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Variant data
    variants: List[Variant] = field(default_factory=list)
    coding_variants: List[Variant] = field(default_factory=list)
    cancer_driver_mutations: List[Variant] = field(default_factory=list)

    # Affected proteins
    affected_genes: List[str] = field(default_factory=list)
    affected_proteins: Dict[str, ProteinInfo] = field(default_factory=dict)

    # Biomarkers
    tumor_mutational_burden: float = 0.0  # Variants per megabase
    msi_status: str = "unknown"  # MSI-H, MSS, unknown
    hla_alleles: Optional[List[str]] = None

    # Summary
    variant_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_tmb_high(self) -> bool:
        """TMB ≥ 10 mutations/Mb is considered high."""
        return self.tumor_mutational_burden >= 10.0

    @property
    def has_dna_repair_defect(self) -> bool:
        """Check if patient has mutations in DNA repair genes."""
        repair_genes = {"BRCA1", "BRCA2", "ATM", "CDK12", "PALB2", "CHEK2"}
        return bool(repair_genes & set(self.affected_genes))

    @property
    def has_ar_mutation(self) -> bool:
        return "AR" in self.affected_genes

    @property
    def has_pten_loss(self) -> bool:
        return "PTEN" in self.affected_genes

    @property
    def has_tp53_mutation(self) -> bool:
        return "TP53" in self.affected_genes

    @property
    def parp_inhibitor_candidate(self) -> bool:
        """PARP inhibitors indicated for BRCA1/2, ATM, CDK12 mutations."""
        return self.has_dna_repair_defect

    @property
    def immunotherapy_candidate(self) -> bool:
        """Checkpoint inhibitors indicated for TMB-high or MSI-H."""
        return self.is_tmb_high or self.msi_status == "MSI-H"

    def get_driver_details(self) -> List[Dict]:
        """Get detailed info about each cancer driver mutation."""
        details = []
        for v in self.cancer_driver_mutations:
            gene_info = PROSTATE_CANCER_DRIVERS.get(v.gene, {})
            details.append({
                "gene": v.gene,
                "mutation": v.protein_change or f"{v.ref}>{v.alt}",
                "location": v.location_str,
                "consequence": v.consequence,
                "impact": v.impact,
                "role": gene_info.get("role", "unknown"),
                "full_name": gene_info.get("full_name", v.gene),
                "description": gene_info.get("description", ""),
                "clinical_significance": (
                    v.clinvar_significance or
                    gene_info.get("clinical_significance", "")
                ),
                "therapies": gene_info.get("therapies", []),
            })
        return details

    def get_therapy_recommendations(self) -> List[Dict]:
        """Generate therapy recommendations based on genomic profile."""
        recommendations = []

        if self.parp_inhibitor_candidate:
            repair_genes = [g for g in self.affected_genes
                          if g in {"BRCA1", "BRCA2", "ATM", "CDK12", "PALB2"}]
            recommendations.append({
                "therapy_class": "PARP Inhibitor",
                "drugs": ["olaparib", "rucaparib", "niraparib", "talazoparib"],
                "rationale": f"DNA repair defect in {', '.join(repair_genes)} — "
                           f"synthetic lethality with PARP inhibition",
                "evidence_level": "FDA-approved (olaparib for BRCA1/2 in mCRPC)",
                "confidence": "high" if "BRCA2" in repair_genes else "moderate",
            })

        if self.immunotherapy_candidate:
            reasons = []
            if self.is_tmb_high:
                reasons.append(f"TMB-high ({self.tumor_mutational_burden:.1f}/Mb)")
            if self.msi_status == "MSI-H":
                reasons.append("MSI-high")
            recommendations.append({
                "therapy_class": "Checkpoint Inhibitor",
                "drugs": ["pembrolizumab", "nivolumab", "ipilimumab"],
                "rationale": f"Immunotherapy responsive: {', '.join(reasons)}",
                "evidence_level": "FDA-approved (pembrolizumab for TMB-H/MSI-H)",
                "confidence": "high",
            })

        if self.has_ar_mutation:
            recommendations.append({
                "therapy_class": "AR-Targeted Therapy",
                "drugs": ["enzalutamide", "abiraterone", "darolutamide"],
                "rationale": "AR mutation detected — monitor for treatment resistance",
                "evidence_level": "Standard of care for CRPC",
                "confidence": "high",
            })

        if self.has_pten_loss:
            recommendations.append({
                "therapy_class": "PI3K/AKT Inhibitor",
                "drugs": ["ipatasertib", "alpelisib"],
                "rationale": "PTEN loss — PI3K/AKT pathway activation",
                "evidence_level": "Clinical trials (ipatasertib + abiraterone)",
                "confidence": "moderate",
            })

        return recommendations

    def to_dict(self) -> Dict:
        """Serialize to JSON-safe dict."""
        return {
            "patient_id": self.patient_id,
            "cancer_type": self.cancer_type,
            "created_at": self.created_at,
            "n_variants": len(self.variants),
            "n_coding": len(self.coding_variants),
            "n_drivers": len(self.cancer_driver_mutations),
            "affected_genes": self.affected_genes,
            "tumor_mutational_burden": self.tumor_mutational_burden,
            "msi_status": self.msi_status,
            "tmb_high": self.is_tmb_high,
            "dna_repair_defect": self.has_dna_repair_defect,
            "parp_candidate": self.parp_inhibitor_candidate,
            "immunotherapy_candidate": self.immunotherapy_candidate,
            "driver_details": self.get_driver_details(),
            "therapy_recommendations": self.get_therapy_recommendations(),
            "variant_summary": self.variant_summary,
        }

    def save(self, path: str):
        """Save profile to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved patient profile to {path}")

    @classmethod
    def from_dict(cls, data: Dict) -> "PatientProfile":
        """Load from serialized dict (without variant objects)."""
        profile = cls(
            patient_id=data.get("patient_id", "anonymous"),
            cancer_type=data.get("cancer_type", "prostate"),
            created_at=data.get("created_at", ""),
            affected_genes=data.get("affected_genes", []),
            tumor_mutational_burden=data.get("tumor_mutational_burden", 0.0),
            msi_status=data.get("msi_status", "unknown"),
            variant_summary=data.get("variant_summary", {}),
        )
        return profile


class PatientProfileBuilder:
    """Build a PatientProfile from raw VCF data.

    Orchestrates the full pipeline:
    VCF text/file → parse → annotate → map proteins → build profile.

    Example:
        builder = PatientProfileBuilder()
        profile = builder.from_vcf_file("patient.vcf", patient_id="MAYO-001")
        print(f"Drivers: {profile.affected_genes}")
        print(f"TMB: {profile.tumor_mutational_burden:.1f}")

        for rec in profile.get_therapy_recommendations():
            print(f"  {rec['therapy_class']}: {rec['drugs']}")
    """

    def __init__(self, cancer_type: str = "prostate"):
        self.parser = VCFParser()
        self.annotator = VariantAnnotator(cancer_type=cancer_type)
        self.mapper = GeneProteinMapper()
        self.cancer_type = cancer_type

    def from_vcf_file(self, vcf_path: str,
                      patient_id: str = "anonymous") -> PatientProfile:
        """Build profile from a VCF file."""
        variants = self.parser.parse_file(vcf_path)
        return self._build_profile(variants, patient_id)

    def from_vcf_text(self, vcf_text: str,
                      patient_id: str = "anonymous") -> PatientProfile:
        """Build profile from VCF text content."""
        variants = self.parser.parse_text(vcf_text)
        return self._build_profile(variants, patient_id)

    def _build_profile(self, variants: List[Variant],
                       patient_id: str) -> PatientProfile:
        """Build complete profile from parsed variants."""
        # Annotate variants
        self.annotator.annotate(variants)

        # Filter
        coding = [v for v in variants if v.is_coding]
        drivers = [v for v in variants if v.is_cancer_driver]

        # Affected genes
        affected_genes = sorted(set(
            v.gene for v in coding if v.gene
        ))

        # Map to proteins
        driver_genes = sorted(set(v.gene for v in drivers if v.gene))
        proteins = self.mapper.get_proteins_for_genes(driver_genes)

        # Compute biomarkers
        tmb = self.annotator.compute_tmb(variants)
        msi = self.annotator.classify_msi(variants)

        # Summary stats
        summary = self.parser.variant_summary(variants)

        profile = PatientProfile(
            patient_id=patient_id,
            cancer_type=self.cancer_type,
            variants=variants,
            coding_variants=coding,
            cancer_driver_mutations=drivers,
            affected_genes=affected_genes,
            affected_proteins=proteins,
            tumor_mutational_burden=tmb,
            msi_status=msi,
            variant_summary=summary,
        )

        logger.info(
            f"Built profile for {patient_id}: "
            f"{len(variants)} variants, {len(coding)} coding, "
            f"{len(drivers)} drivers, TMB={tmb:.1f}"
        )
        return profile
