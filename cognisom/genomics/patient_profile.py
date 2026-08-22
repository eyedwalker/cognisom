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
from .vep_annotator import VEPAnnotator
from .variant_annotator import VariantAnnotator, PROSTATE_CANCER_DRIVERS
from .gene_protein_mapper import GeneProteinMapper, ProteinInfo
from .hla_typer import HLATyper
from .neoantigen_predictor import NeoantigenPredictor, Neoantigen

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
    # False when the callset carries no functional annotation, in which
    # case `tumor_mutational_burden` is 0.0 because nothing could be
    # counted -- not because the tumor is quiet.
    tmb_is_estimable: bool = True
    msi_status: str = "unknown"  # MSI-H, MSS, unknown
    hla_alleles: Optional[List[str]] = None

    # Neoantigen predictions
    predicted_neoantigens: List[Neoantigen] = field(default_factory=list)

    # ── Provenance of the tumor/normal comparison ──────────────────
    # Whether the callset actually had a matched normal to compare
    # against. Without one, a somatic mutation and a rare inherited
    # variant are indistinguishable, and neither TMB nor any neoantigen
    # derived from this profile can be called tumor-specific.
    has_matched_normal: bool = False
    tumor_sample: Optional[str] = None
    normal_sample: Optional[str] = None
    somatic_status_counts: Dict[str, int] = field(default_factory=dict)
    germline_variants_excluded: int = 0

    # How the HLA alleles above were obtained. When this is not a
    # patient-specific method the alleles are a population stand-in, and
    # every "HLA-restricted" claim derived from them is unsupported.
    hla_typing_method: str = "not-run"
    hla_is_patient_specific: bool = False

    #: What the external annotation stage did, empty if it did not run.
    #: A raw callset with no VEP stage cannot yield protein changes, and
    #: this is how a reader tells that apart from a quiet tumour.
    vep_annotation: Dict[str, Any] = field(default_factory=dict)

    # Why variants produced no neoantigens, and which scorer ran.
    # An empty neoantigen list is otherwise ambiguous between "this
    # tumor presents nothing" and "we could not evaluate it".
    neoantigen_diagnostics: Dict[str, Any] = field(default_factory=dict)
    binding_method: str = "unknown"

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

    @property
    def neoantigen_vaccine_candidate(self) -> bool:
        """Patient has strong neoantigen targets for personalized vaccine."""
        return len(self.vaccine_neoantigens) >= 3

    @property
    def vaccine_neoantigens(self) -> List[Neoantigen]:
        """Neoantigens selected for vaccine inclusion."""
        return [n for n in self.predicted_neoantigens if n.include_in_vaccine]

    @property
    def strong_binder_count(self) -> int:
        """Number of strong MHC binders (< 50 nM)."""
        return sum(1 for n in self.predicted_neoantigens if n.is_strong_binder)

    @property
    def weak_binder_count(self) -> int:
        """Number of weak MHC binders (< 500 nM)."""
        return sum(1 for n in self.predicted_neoantigens if n.is_weak_binder)

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

        if self.neoantigen_vaccine_candidate:
            n_targets = len(self.vaccine_neoantigens)
            genes = sorted(set(n.source_gene for n in self.vaccine_neoantigens))
            recommendations.append({
                "therapy_class": "Personalized Neoantigen Vaccine",
                "drugs": ["mRNA neoantigen vaccine"],
                "rationale": (
                    f"{n_targets} predicted neoantigens from "
                    f"{', '.join(genes)} — strong MHC-I binding "
                    f"predicted for patient HLA alleles"
                ),
                "evidence_level": "Clinical trials (mRNA-4157/V940 + pembrolizumab)",
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
            "tmb_is_estimable": self.tmb_is_estimable,
            "msi_status": self.msi_status,
            "tmb_high": self.is_tmb_high,
            "dna_repair_defect": self.has_dna_repair_defect,
            "parp_candidate": self.parp_inhibitor_candidate,
            "immunotherapy_candidate": self.immunotherapy_candidate,
            "hla_alleles": self.hla_alleles,
            "neoantigen_vaccine_candidate": self.neoantigen_vaccine_candidate,
            "n_predicted_neoantigens": len(self.predicted_neoantigens),
            "n_vaccine_candidates": len(self.vaccine_neoantigens),
            "n_strong_binders": self.strong_binder_count,
            "predicted_neoantigens": [n.to_dict() for n in self.predicted_neoantigens[:20]],
            "driver_details": self.get_driver_details(),
            "therapy_recommendations": self.get_therapy_recommendations(),
            "variant_summary": self.variant_summary,
            # Provenance. These say what the numbers above are actually
            # based on: whether a germline comparison happened at all,
            # what it excluded, which scorer produced the affinities, and
            # what the pipeline could not evaluate.
            "has_matched_normal": self.has_matched_normal,
            "tumor_sample": self.tumor_sample,
            "normal_sample": self.normal_sample,
            "somatic_status_counts": self.somatic_status_counts,
            "germline_variants_excluded": self.germline_variants_excluded,
            "binding_method": self.binding_method,
            "neoantigen_diagnostics": self.neoantigen_diagnostics,
            "hla_typing_method": self.hla_typing_method,
            "hla_is_patient_specific": self.hla_is_patient_specific,
            "vep_annotation": self.vep_annotation,
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

    def __init__(self, cancer_type: str = "prostate",
                 vep_annotator: Optional["VEPAnnotator"] = None):
        """
        Args:
            cancer_type: Drives the driver-gene panel.
            vep_annotator: Optional VEP stage, run before the built-in
                annotator. Required for raw callsets: a protein
                consequence cannot be derived from a coordinate alone, so
                without it a VCF carrying no ANN/CSQ/AA_CHANGE yields no
                protein changes and therefore no neoantigens. VEP fills
                them from the MANE Select transcript, whose numbering
                matches the bundled reference proteome.
        """
        self.parser = VCFParser()
        self.annotator = VariantAnnotator(cancer_type=cancer_type)
        self.vep_annotator = vep_annotator
        self.mapper = GeneProteinMapper()
        self.hla_typer = HLATyper()
        self.neoantigen_predictor = NeoantigenPredictor()
        self.cancer_type = cancer_type
        self.vep_stats: Dict[str, Any] = {}

    def from_vcf_file(self, vcf_path: str,
                      patient_id: str = "anonymous",
                      normal_bam_path: Optional[str] = None,
                      normal_fastq: Optional[Any] = None) -> PatientProfile:
        """Build profile from a VCF file.

        Args:
            vcf_path: Path to the VCF. A matched tumor/normal callset is
                preferred; a tumor-only callset is accepted but cannot
                support somatic claims.
            patient_id: Patient identifier.
            normal_bam_path: Optional germline/normal BAM. When given, HLA
                alleles are typed from it with OptiType instead of falling
                back to a population profile.
            normal_fastq: Optional germline/normal reads -- a path, or an
                (R1, R2) pair. FASTQ is OptiType's native input, so a BAM
                is not required; the BAM path merely pre-filters chr6.
                Use the NORMAL sample: HLA type is the patient's genotype,
                and typing off tumour reads can drop an allele lost to
                HLA LOH, silently removing every neoantigen restricted
                to it.
        """
        variants = self.parser.parse_file(vcf_path)
        return self._build_profile(variants, patient_id, normal_bam_path,
                                   normal_fastq)

    def from_vcf_text(self, vcf_text: str,
                      patient_id: str = "anonymous",
                      normal_bam_path: Optional[str] = None,
                      normal_fastq: Optional[Any] = None) -> PatientProfile:
        """Build profile from VCF text content."""
        variants = self.parser.parse_text(vcf_text)
        return self._build_profile(variants, patient_id, normal_bam_path,
                                   normal_fastq)

    def _build_profile(self, variants: List[Variant],
                       patient_id: str,
                       normal_bam_path: Optional[str] = None,
                       normal_fastq: Optional[Any] = None) -> PatientProfile:
        """Build complete profile from parsed variants."""
        # External annotation first, so the built-in annotator sees real
        # consequences rather than an unannotated callset. Failures are
        # not swallowed: an un-annotated run and a genuinely quiet tumour
        # must not produce the same output.
        self.vep_stats = {}
        if self.vep_annotator is not None:
            self.vep_annotator.annotate(variants)
            self.vep_stats = self.vep_annotator.stats.to_dict()
            logger.info("VEP annotation: %s", self.vep_stats)

        # Annotate variants
        self.annotator.annotate(variants)

        # ── Tumor vs germline ──────────────────────────────────────
        # When the callset carries a matched normal, everything the
        # normal also carries is inherited, not tumor biology. Counting
        # it inflates TMB and, worse, feeds self antigens into vaccine
        # design. When there is no normal the question is unanswerable,
        # so the full set is carried forward and the profile records
        # that the comparison did not happen.
        has_matched_normal = self.parser.is_somatic_vcf
        somatic_status_counts: Dict[str, int] = {}
        for v in variants:
            somatic_status_counts[v.somatic_status] = (
                somatic_status_counts.get(v.somatic_status, 0) + 1
            )

        germline_excluded = 0
        if has_matched_normal:
            tumor_variants = self.parser.filter_somatic(variants)
            germline_excluded = len(variants) - len(tumor_variants)
            logger.info(
                "Matched normal present (tumor=%s, normal=%s): %d of %d "
                "variants are somatic; %d excluded as germline, artifact "
                "or low-evidence.",
                self.parser.tumor_sample, self.parser.normal_sample,
                len(tumor_variants), len(variants), germline_excluded,
            )
        else:
            tumor_variants = variants
            logger.warning(
                "No matched normal in this callset. Somatic and inherited "
                "variants cannot be separated, so TMB and every neoantigen "
                "below are tumor-ONLY estimates, not somatic calls."
            )

        # Filter
        coding = [v for v in tumor_variants if v.is_coding]
        drivers = [v for v in tumor_variants if v.is_cancer_driver]

        # Affected genes
        affected_genes = sorted(set(
            v.gene for v in coding if v.gene
        ))

        # Map to proteins
        driver_genes = sorted(set(v.gene for v in drivers if v.gene))
        proteins = self.mapper.get_proteins_for_genes(driver_genes)

        # Compute biomarkers over the tumor-specific set, not everything:
        # TMB is defined as somatic mutations per megabase.
        tmb = self.annotator.compute_tmb(tumor_variants)
        msi = self.annotator.classify_msi(tumor_variants)

        # TMB counts coding mutations, so it is only meaningful when
        # something in the callset says which mutations are coding. On a
        # raw VCF nothing does, and the result is 0.0 -- indistinguishable
        # from a genuinely quiet tumor, and read downstream as
        # "TMB-low, not an immunotherapy candidate".
        tmb_is_estimable = self.annotator.variants_with_consequence > 0
        if not tmb_is_estimable:
            logger.warning(
                "TMB is not estimable for %s: no variant in this callset "
                "carries a functional annotation, so no coding mutations "
                "can be counted. The reported TMB of %.1f means 'unknown', "
                "not 'low'. Annotate the VCF with VEP or SnpEff first.",
                patient_id, tmb,
            )

        # HLA typing. Prefer real typing from a normal BAM when one is
        # supplied: OptiType is a complete implementation in this repo but
        # `type_from_bam` had no callers anywhere, so the gold-standard
        # path was unreachable and every patient got a population profile.
        if normal_bam_path:
            hla_alleles = self.hla_typer.type_from_bam(
                normal_bam_path, sample_id=patient_id
            )
        elif normal_fastq:
            r1, r2 = (normal_fastq if isinstance(normal_fastq, (tuple, list))
                      else (normal_fastq, None))
            hla_alleles = self.hla_typer.type_from_fastq(
                r1, r2, sample_id=patient_id
            )
        else:
            hla_alleles = self.hla_typer.type_from_variants(
                variants, patient_id=patient_id
            )

        # Neoantigen prediction
        neoantigens = []
        if hla_alleles and drivers:
            try:
                neoantigens = self.neoantigen_predictor.predict(
                    cancer_mutations=drivers,
                    affected_proteins=proteins,
                    hla_alleles=hla_alleles,
                )
            except Exception as e:
                logger.warning(f"Neoantigen prediction failed: {e}")

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
            tmb_is_estimable=tmb_is_estimable,
            msi_status=msi,
            hla_alleles=hla_alleles,
            predicted_neoantigens=neoantigens,
            variant_summary=summary,
            has_matched_normal=has_matched_normal,
            tumor_sample=self.parser.tumor_sample,
            normal_sample=self.parser.normal_sample,
            somatic_status_counts=somatic_status_counts,
            germline_variants_excluded=germline_excluded,
            neoantigen_diagnostics=self.neoantigen_predictor.diagnostics.to_dict(),
            binding_method=self.neoantigen_predictor.binding_method,
            hla_typing_method=self.hla_typer.typing_method,
            hla_is_patient_specific=self.hla_typer.is_patient_specific,
            vep_annotation=self.vep_stats,
        )

        n_vaccine = len(profile.vaccine_neoantigens)
        logger.info(
            f"Built profile for {patient_id}: "
            f"{len(variants)} variants, {len(coding)} coding, "
            f"{len(drivers)} drivers, TMB={tmb:.1f}, "
            f"HLA={len(hla_alleles)} alleles, "
            f"{len(neoantigens)} neoantigens ({n_vaccine} vaccine candidates)"
        )
        return profile
