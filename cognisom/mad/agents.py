"""
MAD Specialist Agents
=====================

Three domain-specific agents that independently analyze patient data
and produce treatment rankings with traceable evidence. Each wraps
existing Cognisom modules — no new ML models, fully deterministic.

GenomicsAgent:  Variant-level analysis → biomarker-driven ranking
ImmuneAgent:    Immune microenvironment → checkpoint/vaccine ranking
ClinicalAgent:  Treatment simulation → efficacy/safety ranking
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .evidence import (
    EvidenceItem,
    FeatureContribution,
    TreatmentRanking,
    CHECKPOINT_TRIALS,
    PARP_TRIALS,
    VACCINE_TRIALS,
    GUIDELINE_EVIDENCE,
)
from .errors import MADError, MADErrorCode

logger = logging.getLogger(__name__)


@dataclass
class AgentOpinion:
    """Structured output from a specialist agent."""

    agent_name: str
    """Which agent produced this: 'genomics', 'immune', 'clinical'."""

    treatment_rankings: List[TreatmentRanking]
    """Treatments ranked best to worst, with evidence."""

    confidence: float
    """Overall confidence in this opinion (0-1)."""

    evidence_items: List[EvidenceItem]
    """All evidence considered by this agent."""

    dissenting_notes: List[str]
    """Areas of uncertainty or concern."""

    warnings: List[str] = field(default_factory=list)
    """Non-fatal warnings (e.g. MADErrorCode flags)."""

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    model_versions: Dict[str, str] = field(default_factory=dict)
    """Versions of models/databases used."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "treatment_rankings": [r.to_dict() for r in self.treatment_rankings],
            "confidence": round(self.confidence, 4),
            "evidence_items": [e.to_dict() for e in self.evidence_items],
            "dissenting_notes": self.dissenting_notes,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
            "model_versions": self.model_versions,
        }

    @property
    def top_treatment(self) -> Optional[str]:
        if self.treatment_rankings:
            return self.treatment_rankings[0].treatment_key
        return None


class MADAgent(ABC):
    """Base class for MAD specialist agents."""

    name: str = "base"

    @abstractmethod
    def analyze(self, **kwargs) -> AgentOpinion:
        """Analyze patient data and produce an opinion."""


class GenomicsAgent(MADAgent):
    """Analyzes patient genomic profile for treatment actionability.

    Wraps: PatientProfile, VariantAnnotator, GeneProteinMapper
    Focus: Driver mutations, biomarker status (TMB, MSI, HRD), actionability
    """

    name = "genomics"

    # Evidence level mapping for biomarker-drug associations
    ACTIONABILITY_MAP = {
        ("TMB_high", "checkpoint"): {
            "level": "1A",
            "description": "FDA-approved: pembrolizumab for TMB >=10 mut/Mb",
        },
        ("MSI_H", "checkpoint"): {
            "level": "1A",
            "description": "FDA-approved: pembrolizumab for MSI-H/dMMR",
        },
        ("HRD", "PARP"): {
            "level": "1A",
            "description": "FDA-approved: olaparib for BRCA1/2 mCRPC (PROfound)",
        },
        ("AR_mutation", "AR_antagonist"): {
            "level": "2A",
            "description": "NCCN: enzalutamide, monitor for resistance mutations",
        },
        ("neoantigen_high", "vaccine"): {
            "level": "2B",
            "description": "Clinical trials: mRNA-4157/V940 (KEYNOTE-942)",
        },
    }

    def analyze(
        self,
        profile: Any,  # PatientProfile
        twin: Any,      # DigitalTwinConfig
        **kwargs,
    ) -> AgentOpinion:
        """Produce genomics-driven treatment ranking."""
        evidence = []
        rankings = []
        dissenting = []
        warnings = []
        confidence = 0.7  # Base confidence for genomic analysis

        # --- Assess biomarker status ---
        tmb = getattr(profile, "tumor_mutational_burden", 0.0)
        msi = getattr(profile, "msi_status", "unknown")
        has_hrd = getattr(profile, "has_dna_repair_defect", False)
        has_ar = getattr(profile, "has_ar_mutation", False)
        has_pten = getattr(profile, "has_pten_loss", False)
        affected_genes = getattr(profile, "affected_genes", [])
        n_drivers = len(getattr(profile, "cancer_driver_mutations", []))
        n_variants = len(getattr(profile, "coding_variants", []))
        vaccine_candidates = getattr(twin, "vaccine_candidate_count", 0)
        is_vaccine_eligible = getattr(twin, "neoantigen_vaccine_candidate", False)

        # Warn on low variant count
        if n_variants < 30:
            warnings.append(MADErrorCode.INSUFFICIENT_VARIANTS.value)
            confidence -= 0.15

        # --- OncoKB enrichment (if available) ---
        oncokb_annotations = {}
        try:
            from ..genomics.oncokb_client import OncoKBClient
            oncokb = OncoKBClient()
            drivers = getattr(profile, "cancer_driver_mutations", [])
            for v in drivers:
                gene = getattr(v, "gene", "")
                pchange = getattr(v, "protein_change", "")
                if gene and pchange:
                    ann = oncokb.annotate_mutation(gene, pchange)
                    oncokb_annotations[gene] = ann
                    if ann.is_actionable:
                        evidence.append(EvidenceItem(
                            source_type="guideline",
                            source_name=f"OncoKB: {gene} {pchange}",
                            source_id=f"oncokb:{gene}:{pchange}",
                            claim=f"{gene} {pchange}: {ann.oncogenic} "
                                  f"(Level {ann.highest_sensitive_level}). "
                                  f"Drugs: {', '.join(ann.drugs)}",
                            strength=_oncokb_level_to_strength(
                                ann.highest_sensitive_level
                            ),
                            supporting_data={
                                "oncogenic": ann.oncogenic,
                                "mutation_effect": ann.mutation_effect,
                                "level": ann.highest_sensitive_level,
                                "drugs": ann.drugs,
                            },
                        ))
                        confidence += 0.02  # Small boost per actionable variant
        except Exception as e:
            logger.debug("OncoKB enrichment skipped: %s", e)

        # --- FDA Pharmacogenomic Biomarker enrichment ---
        try:
            from ..genomics.evidence_sources import check_fda_biomarkers
            for gene in affected_genes:
                fda_entries = check_fda_biomarkers(gene)
                for entry in fda_entries:
                    evidence.append(EvidenceItem(
                        source_type="guideline",
                        source_name=f"FDA Biomarker: {gene}",
                        source_id=f"fda-pgx:{gene}:{entry['drug']}",
                        claim=f"{entry['drug']}: {entry['use']}",
                        strength="1A" if entry["section"] == "Indications" else "2A",
                        supporting_data={"section": entry["section"]},
                    ))
        except Exception as e:
            logger.debug("FDA biomarker enrichment skipped: %s", e)

        # --- Score each treatment class from genomic perspective ---
        treatment_scores: Dict[str, float] = {}
        treatment_evidence: Dict[str, List[EvidenceItem]] = {}
        treatment_contributions: Dict[str, List[FeatureContribution]] = {}

        # 1. Checkpoint inhibitors
        ck_score = 0.3  # Base
        ck_contribs = []
        ck_evidence = []

        if tmb >= 10.0:
            delta = 0.35
            ck_score += delta
            ck_contribs.append(FeatureContribution(
                "TMB_high", delta, "positive",
                f"TMB {tmb:.1f} mut/Mb >= 10 threshold",
            ))
            ck_evidence.append(CHECKPOINT_TRIALS["keynote_158"])
            ck_evidence.append(GUIDELINE_EVIDENCE["nccn_biomarker_tmb"])

        if msi == "MSI-H":
            delta = 0.30
            ck_score += delta
            ck_contribs.append(FeatureContribution(
                "MSI_H", delta, "positive",
                "Microsatellite instability-high — FDA-approved for pembrolizumab",
            ))

        if tmb < 5.0 and msi != "MSI-H":
            delta = -0.20
            ck_score += delta
            ck_contribs.append(FeatureContribution(
                "TMB_low", delta, "negative",
                f"TMB {tmb:.1f} mut/Mb — limited neoantigen load",
            ))

        ck_score = max(0.0, min(1.0, ck_score))
        treatment_scores["pembrolizumab"] = ck_score
        treatment_evidence["pembrolizumab"] = ck_evidence
        treatment_contributions["pembrolizumab"] = ck_contribs

        # 2. PARP inhibitors
        parp_score = 0.2
        parp_contribs = []
        parp_evidence = []

        if has_hrd:
            # Identify specific repair genes
            repair_genes = {"BRCA1", "BRCA2", "ATM", "CDK12", "PALB2", "CHEK2"}
            mutated_repair = repair_genes & set(affected_genes)
            delta = 0.55
            parp_score += delta
            parp_contribs.append(FeatureContribution(
                "HRD_positive", delta, "positive",
                f"DNA repair defect: {', '.join(mutated_repair)}",
            ))
            parp_evidence.append(PARP_TRIALS["profound"])
            if "BRCA1" in mutated_repair or "BRCA2" in mutated_repair:
                parp_evidence.append(PARP_TRIALS["triton2"])
                confidence += 0.05  # BRCA = strongest evidence
        else:
            delta = -0.15
            parp_score += delta
            parp_contribs.append(FeatureContribution(
                "HRD_negative", delta, "negative",
                "No DNA repair gene mutations — PARP unlikely effective",
            ))

        parp_score = max(0.0, min(1.0, parp_score))
        treatment_scores["olaparib"] = parp_score
        treatment_evidence["olaparib"] = parp_evidence
        treatment_contributions["olaparib"] = parp_contribs

        # 3. AR antagonist
        ar_score = 0.4
        ar_contribs = []

        if has_ar:
            delta = -0.15
            ar_score += delta
            ar_contribs.append(FeatureContribution(
                "AR_mutation", delta, "negative",
                "AR mutation detected — potential resistance to AR antagonists",
            ))
            dissenting.append(
                "AR mutation may confer enzalutamide resistance; "
                "monitor for T877A and other known resistance variants"
            )
        else:
            delta = 0.20
            ar_score += delta
            ar_contribs.append(FeatureContribution(
                "AR_wildtype", delta, "positive",
                "Wild-type AR — standard of care responsiveness expected",
            ))

        ar_score = max(0.0, min(1.0, ar_score))
        treatment_scores["enzalutamide"] = ar_score
        treatment_evidence["enzalutamide"] = []
        treatment_contributions["enzalutamide"] = ar_contribs

        # 4. Neoantigen vaccine
        vax_score = 0.2
        vax_contribs = []
        vax_evidence = []

        if is_vaccine_eligible and vaccine_candidates >= 3:
            delta = min(0.40, vaccine_candidates * 0.04)
            vax_score += delta
            vax_contribs.append(FeatureContribution(
                "neoantigen_count", delta, "positive",
                f"{vaccine_candidates} vaccine-quality neoantigens identified",
            ))
            vax_evidence.append(VACCINE_TRIALS["keynote_942"])
        else:
            delta = -0.10
            vax_score += delta
            vax_contribs.append(FeatureContribution(
                "neoantigen_insufficient", delta, "negative",
                f"Only {vaccine_candidates} vaccine candidates — below threshold",
            ))

        vax_score = max(0.0, min(1.0, vax_score))
        treatment_scores["neoantigen_vaccine"] = vax_score
        treatment_evidence["neoantigen_vaccine"] = vax_evidence
        treatment_contributions["neoantigen_vaccine"] = vax_contribs

        # 5. Combination: PARP + checkpoint
        combo_score = (treatment_scores["olaparib"] * 0.6 +
                       treatment_scores["pembrolizumab"] * 0.4)
        if has_hrd and tmb >= 10.0:
            combo_score = min(1.0, combo_score + 0.15)
        treatment_scores["olaparib_pembro_combo"] = combo_score
        treatment_evidence["olaparib_pembro_combo"] = (
            treatment_evidence["olaparib"] + treatment_evidence["pembrolizumab"]
        )
        treatment_contributions["olaparib_pembro_combo"] = (
            treatment_contributions["olaparib"] + treatment_contributions["pembrolizumab"]
        )

        # 6. Combination: vaccine + checkpoint
        vp_score = (treatment_scores["neoantigen_vaccine"] * 0.5 +
                    treatment_scores["pembrolizumab"] * 0.5)
        if is_vaccine_eligible and tmb >= 10.0:
            vp_score = min(1.0, vp_score + 0.10)
        treatment_scores["neoantigen_vaccine_pembro"] = vp_score
        treatment_evidence["neoantigen_vaccine_pembro"] = (
            treatment_evidence["neoantigen_vaccine"] + treatment_evidence["pembrolizumab"]
        )
        treatment_contributions["neoantigen_vaccine_pembro"] = (
            treatment_contributions["neoantigen_vaccine"] + treatment_contributions["pembrolizumab"]
        )

        # --- Build ranked list ---
        sorted_treatments = sorted(treatment_scores.items(), key=lambda x: -x[1])
        for rank_idx, (key, score) in enumerate(sorted_treatments, 1):
            # Get display name from treatment profiles
            display_names = {
                "pembrolizumab": "Pembrolizumab (Keytruda)",
                "olaparib": "Olaparib (Lynparza)",
                "enzalutamide": "Enzalutamide (Xtandi)",
                "neoantigen_vaccine": "Neoantigen mRNA Vaccine",
                "olaparib_pembro_combo": "Olaparib + Pembrolizumab",
                "neoantigen_vaccine_pembro": "Neoantigen Vaccine + Pembrolizumab",
            }
            rankings.append(TreatmentRanking(
                treatment_key=key,
                treatment_name=display_names.get(key, key),
                score=score,
                rank=rank_idx,
                evidence=treatment_evidence.get(key, []),
                contributions=treatment_contributions.get(key, []),
            ))

        # Collect all evidence
        for ev_list in treatment_evidence.values():
            evidence.extend(ev_list)
        evidence.append(GUIDELINE_EVIDENCE["nccn_prostate"])

        # Conflicting biomarkers warning
        if msi == "MSI-H" and tmb < 5.0:
            warnings.append(MADErrorCode.CONFLICTING_BIOMARKERS.value)
            dissenting.append("MSI-H with low TMB is unusual — verify MSI status")

        if not any((has_hrd, tmb >= 10.0, msi == "MSI-H", is_vaccine_eligible)):
            if n_drivers == 0:
                warnings.append(MADErrorCode.NO_ACTIONABLE_TARGETS.value)
                dissenting.append("No strong biomarker-drug associations found")
                confidence -= 0.10

        return AgentOpinion(
            agent_name=self.name,
            treatment_rankings=rankings,
            confidence=max(0.0, min(1.0, confidence)),
            evidence_items=_deduplicate_evidence(evidence),
            dissenting_notes=dissenting,
            warnings=warnings,
            model_versions={
                "variant_annotator": "cognisom-v1",
                "gene_protein_mapper": "cognisom-v1",
                "nccn_guidelines": "v2.2024",
            },
        )


class ImmuneAgent(MADAgent):
    """Analyzes immune microenvironment for treatment selection.

    Wraps: CellStateClassifier, NeoantigenPredictor, HLATyper, DigitalTwinConfig
    Focus: Immune landscape, T-cell exhaustion, neoantigen quality, HLA coverage
    """

    name = "immune"

    def analyze(
        self,
        twin: Any,      # DigitalTwinConfig
        profile: Any,   # PatientProfile
        classification: Any = None,  # Optional ImmuneClassification
        **kwargs,
    ) -> AgentOpinion:
        """Produce immune-landscape-driven treatment ranking."""
        evidence = []
        rankings = []
        dissenting = []
        warnings = []
        confidence = 0.65

        # --- Extract immune parameters ---
        immune_score = getattr(twin, "immune_score", "unknown")
        exhaustion = getattr(twin, "t_cell_exhaustion_fraction", 0.0)
        mean_exhaustion = getattr(twin, "mean_exhaustion", 0.0)
        treg_fraction = getattr(twin, "treg_fraction", 0.0)
        m2_fraction = getattr(twin, "m2_macrophage_fraction", 0.0)
        immune_fraction = getattr(twin, "immune_cell_fraction", 0.0)
        pd_l1 = getattr(twin, "pd_l1_expression", 0.5)
        mhc1_down = getattr(twin, "mhc1_downregulation", 0.0)
        vaccine_candidates = getattr(twin, "vaccine_candidate_count", 0)
        strong_binders = getattr(twin, "strong_binder_count", 0)
        hla_alleles = getattr(twin, "hla_alleles", [])
        is_vaccine_eligible = getattr(twin, "neoantigen_vaccine_candidate", False)

        if not hla_alleles:
            warnings.append(MADErrorCode.NO_HLA_DATA.value)
            confidence -= 0.15

        if immune_score == "unknown":
            warnings.append(MADErrorCode.LOW_CONFIDENCE_IMMUNE.value)
            confidence -= 0.20

        # --- Score treatments from immune perspective ---
        treatment_scores: Dict[str, float] = {}
        treatment_contribs: Dict[str, List[FeatureContribution]] = {}
        treatment_evidence: Dict[str, List[EvidenceItem]] = {}

        # 1. Anti-PD-1 (pembrolizumab)
        pd1_score = 0.35
        pd1_contribs = []

        if immune_score == "hot":
            delta = 0.25
            pd1_score += delta
            pd1_contribs.append(FeatureContribution(
                "hot_tumor", delta, "positive",
                f"Hot immune microenvironment ({immune_fraction:.0%} immune cells)",
            ))
            evidence.append(EvidenceItem(
                source_type="biomarker",
                source_name="Immune Score: Hot",
                source_id="cognisom-immune-v1",
                claim="Hot tumors with immune infiltration respond to PD-1 blockade",
                strength="2A",
            ))
        elif immune_score == "cold":
            delta = -0.30
            pd1_score += delta
            pd1_contribs.append(FeatureContribution(
                "cold_tumor", delta, "negative",
                "Cold tumor — minimal immune infiltration, PD-1 blockade unlikely effective alone",
            ))

        # Exhaustion: need exhausted T-cells to reactivate
        if 0.2 < exhaustion < 0.8:
            delta = 0.15
            pd1_score += delta
            pd1_contribs.append(FeatureContribution(
                "reversible_exhaustion", delta, "positive",
                f"{exhaustion:.0%} T-cells exhausted — reactivation potential with PD-1 blockade",
            ))
        elif exhaustion >= 0.8:
            delta = -0.10
            pd1_score += delta
            pd1_contribs.append(FeatureContribution(
                "terminal_exhaustion", delta, "negative",
                f"{exhaustion:.0%} exhaustion — may be terminally exhausted, limited reactivation",
            ))
            dissenting.append(
                "High T-cell exhaustion (>80%) may indicate terminal exhaustion; "
                "consider anti-CTLA-4 combination to prime new T-cells"
            )

        # PD-L1 expression
        if pd_l1 > 0.5:
            delta = 0.10
            pd1_score += delta
            pd1_contribs.append(FeatureContribution(
                "PD_L1_high", delta, "positive",
                f"PD-L1 expression {pd_l1:.0%} — target for anti-PD-1",
            ))

        pd1_score = max(0.0, min(1.0, pd1_score))
        treatment_scores["pembrolizumab"] = pd1_score
        treatment_contribs["pembrolizumab"] = pd1_contribs
        treatment_evidence["pembrolizumab"] = [CHECKPOINT_TRIALS["keynote_199"]]

        # 2. Anti-CTLA-4 combo (pembro + ipi)
        combo_score = pd1_score * 0.6  # Inherit PD-1 base
        combo_contribs = list(pd1_contribs)  # Copy PD-1 contributions

        if treg_fraction > 0.2:
            delta = 0.25
            combo_score += delta
            combo_contribs.append(FeatureContribution(
                "high_treg", delta, "positive",
                f"Treg ratio {treg_fraction:.0%} — anti-CTLA-4 depletes Tregs",
            ))

        if immune_score == "suppressed":
            delta = 0.20
            combo_score += delta
            combo_contribs.append(FeatureContribution(
                "suppressed_tumor", delta, "positive",
                "Suppressed microenvironment — dual checkpoint may overcome immunosuppression",
            ))

        combo_score = max(0.0, min(1.0, combo_score))
        treatment_scores["pembro_ipi_combo"] = combo_score
        treatment_contribs["pembro_ipi_combo"] = combo_contribs
        treatment_evidence["pembro_ipi_combo"] = [CHECKPOINT_TRIALS["checkmate_650"]]

        # 3. Neoantigen vaccine
        vax_score = 0.25
        vax_contribs = []

        if is_vaccine_eligible and vaccine_candidates >= 3:
            delta = min(0.35, vaccine_candidates * 0.035)
            vax_score += delta
            vax_contribs.append(FeatureContribution(
                "vaccine_targets", delta, "positive",
                f"{vaccine_candidates} vaccine targets ({strong_binders} strong binders)",
            ))

            # HLA diversity bonus
            unique_hla = len(set(hla_alleles))
            if unique_hla >= 5:
                delta = 0.10
                vax_score += delta
                vax_contribs.append(FeatureContribution(
                    "hla_diversity", delta, "positive",
                    f"{unique_hla} unique HLA alleles — broad epitope presentation",
                ))
        else:
            delta = -0.15
            vax_score += delta
            vax_contribs.append(FeatureContribution(
                "low_vaccine_targets", delta, "negative",
                f"Only {vaccine_candidates} vaccine candidates",
            ))

        # MHC-I downregulation hurts vaccines
        if mhc1_down > 0.3:
            delta = -mhc1_down * 0.25
            vax_score += delta
            vax_contribs.append(FeatureContribution(
                "mhc1_downregulation", delta, "negative",
                f"MHC-I downregulation {mhc1_down:.0%} — reduced antigen presentation",
            ))
            dissenting.append(
                f"MHC-I downregulation ({mhc1_down:.0%}) may limit vaccine efficacy; "
                "consider IFN-gamma-inducing combinations"
            )

        # Immune microenvironment for vaccine
        if immune_score == "hot":
            delta = 0.15
            vax_score += delta
            vax_contribs.append(FeatureContribution(
                "hot_for_vaccine", delta, "positive",
                "Hot microenvironment supports vaccine-primed T-cell trafficking",
            ))
        elif immune_score == "cold":
            delta = -0.20
            vax_score += delta
            vax_contribs.append(FeatureContribution(
                "cold_for_vaccine", delta, "negative",
                "Cold tumor — vaccine-primed T-cells may not infiltrate",
            ))

        vax_score = max(0.0, min(1.0, vax_score))
        treatment_scores["neoantigen_vaccine"] = vax_score
        treatment_contribs["neoantigen_vaccine"] = vax_contribs
        treatment_evidence["neoantigen_vaccine"] = [VACCINE_TRIALS["keynote_942"]]

        # 4. Vaccine + pembro combo
        vp_score = (vax_score * 0.5 + pd1_score * 0.5)
        if is_vaccine_eligible and immune_score != "cold":
            vp_score = min(1.0, vp_score + 0.10)
        treatment_scores["neoantigen_vaccine_pembro"] = max(0.0, min(1.0, vp_score))
        treatment_contribs["neoantigen_vaccine_pembro"] = (
            treatment_contribs.get("neoantigen_vaccine", []) +
            treatment_contribs.get("pembrolizumab", [])
        )
        treatment_evidence["neoantigen_vaccine_pembro"] = (
            treatment_evidence.get("neoantigen_vaccine", []) +
            treatment_evidence.get("pembrolizumab", [])
        )

        # 5. PARP and AR — immune agent has limited input
        treatment_scores["olaparib"] = 0.3  # Neutral from immune perspective
        treatment_contribs["olaparib"] = [FeatureContribution(
            "immune_neutral_parp", 0.0, "positive",
            "PARP mechanism is independent of immune microenvironment",
        )]
        treatment_evidence["olaparib"] = []

        treatment_scores["enzalutamide"] = 0.3
        treatment_contribs["enzalutamide"] = [FeatureContribution(
            "immune_neutral_ar", 0.0, "positive",
            "AR antagonist mechanism is independent of immune microenvironment",
        )]
        treatment_evidence["enzalutamide"] = []

        # PARP + checkpoint combo gets immune boost
        opc_score = (treatment_scores["olaparib"] * 0.4 + pd1_score * 0.6)
        treatment_scores["olaparib_pembro_combo"] = max(0.0, min(1.0, opc_score))
        treatment_contribs["olaparib_pembro_combo"] = (
            treatment_contribs["olaparib"] + treatment_contribs["pembrolizumab"]
        )
        treatment_evidence["olaparib_pembro_combo"] = (
            treatment_evidence["olaparib"] + treatment_evidence["pembrolizumab"]
        )

        # --- Build ranked list ---
        sorted_treatments = sorted(treatment_scores.items(), key=lambda x: -x[1])
        display_names = {
            "pembrolizumab": "Pembrolizumab (Keytruda)",
            "pembro_ipi_combo": "Pembrolizumab + Ipilimumab",
            "olaparib": "Olaparib (Lynparza)",
            "enzalutamide": "Enzalutamide (Xtandi)",
            "neoantigen_vaccine": "Neoantigen mRNA Vaccine",
            "olaparib_pembro_combo": "Olaparib + Pembrolizumab",
            "neoantigen_vaccine_pembro": "Neoantigen Vaccine + Pembrolizumab",
        }

        for rank_idx, (key, score) in enumerate(sorted_treatments, 1):
            rankings.append(TreatmentRanking(
                treatment_key=key,
                treatment_name=display_names.get(key, key),
                score=score,
                rank=rank_idx,
                evidence=treatment_evidence.get(key, []),
                contributions=treatment_contribs.get(key, []),
            ))

        # Collect evidence
        for ev_list in treatment_evidence.values():
            evidence.extend(ev_list)

        return AgentOpinion(
            agent_name=self.name,
            treatment_rankings=rankings,
            confidence=max(0.0, min(1.0, confidence)),
            evidence_items=_deduplicate_evidence(evidence),
            dissenting_notes=dissenting,
            warnings=warnings,
            model_versions={
                "cell_state_classifier": "cognisom-v1",
                "hla_typer": "cognisom-v1",
                "neoantigen_predictor": "pwm-v1",
                "iedb_concordance": "75%",
            },
        )


class ClinicalAgent(MADAgent):
    """Analyzes treatment simulation results and clinical evidence.

    Wraps: TreatmentSimulator results, clinical trial data
    Focus: Simulated efficacy, safety (irAE), response classification, PFS
    """

    name = "clinical"

    def analyze(
        self,
        twin: Any,           # DigitalTwinConfig
        treatment_results: List[Any],  # List[TreatmentResult]
        **kwargs,
    ) -> AgentOpinion:
        """Produce simulation-driven treatment ranking."""
        evidence = []
        rankings = []
        dissenting = []
        warnings = []
        confidence = 0.70

        if not treatment_results:
            warnings.append(MADErrorCode.SIMULATION_FAILURE.value)
            return AgentOpinion(
                agent_name=self.name,
                treatment_rankings=[],
                confidence=0.0,
                evidence_items=[],
                dissenting_notes=["No treatment simulation results available"],
                warnings=warnings,
                model_versions={"treatment_simulator": "cognisom-v1"},
            )

        # --- Score each simulated treatment ---
        for result in treatment_results:
            contribs = []
            ev = []

            # Base score from best response
            best_response = getattr(result, "best_response", 1.0)
            response_score = max(0.0, 1.0 - best_response)  # Lower volume = better

            contribs.append(FeatureContribution(
                "tumor_response", response_score * 0.5, "positive",
                f"Best response: {(1-best_response)*100:.0f}% tumor reduction",
            ))

            # PFS contribution
            pfs = getattr(result, "progression_free_days", 0)
            pfs_score = min(0.25, pfs / 720.0)  # Max contribution at 2 years
            contribs.append(FeatureContribution(
                "progression_free_survival", pfs_score, "positive",
                f"PFS: {pfs} days ({pfs/30:.1f} months)",
            ))

            # Safety penalty (irAE risk)
            irae = getattr(result, "immune_related_adverse_events", 0.0)
            safety_penalty = -irae * 0.15
            contribs.append(FeatureContribution(
                "irae_risk", safety_penalty, "negative",
                f"irAE risk: {irae:.0%}",
            ))
            if irae > 0.4:
                dissenting.append(
                    f"{result.treatment_name}: High irAE risk ({irae:.0%}) — "
                    "requires careful monitoring"
                )

            # T-cell reactivation
            reactivation = getattr(result, "t_cell_reactivation_fraction", 0.0)
            react_score = reactivation * 0.10
            contribs.append(FeatureContribution(
                "t_cell_reactivation", react_score, "positive",
                f"T-cell reactivation: {reactivation:.0%}",
            ))

            # Composite score
            composite = response_score * 0.50 + pfs_score + react_score + safety_penalty
            composite = max(0.0, min(1.0, composite))

            # Response category evidence
            response_cat = getattr(result, "response_category", "unknown")
            ev.append(EvidenceItem(
                source_type="simulation",
                source_name=f"Cognisom Simulation: {result.treatment_name}",
                source_id="cognisom-sim-v1",
                claim=f"Simulated {response_cat} response "
                      f"({(1-best_response)*100:.0f}% reduction, PFS {pfs}d)",
                strength="simulation",
                supporting_data={
                    "best_response": round(best_response, 3),
                    "pfs_days": pfs,
                    "response_category": response_cat,
                    "irae_risk": round(irae, 3),
                },
            ))

            # Add trial evidence for known treatments
            tkey = getattr(result, "treatment_name", "").lower()
            if "pembrolizumab" in tkey or "keytruda" in tkey:
                ev.append(CHECKPOINT_TRIALS["keynote_199"])
            if "olaparib" in tkey or "lynparza" in tkey:
                ev.append(PARP_TRIALS["profound"])
            if "vaccine" in tkey:
                ev.append(VACCINE_TRIALS["keynote_942"])
            if "nivolumab" in tkey or "ipilimumab" in tkey:
                ev.append(CHECKPOINT_TRIALS["checkmate_650"])

            # Map result back to treatment key
            treatment_key = _result_to_key(result)

            rankings.append(TreatmentRanking(
                treatment_key=treatment_key,
                treatment_name=getattr(result, "treatment_name", treatment_key),
                score=composite,
                rank=0,  # Will be set after sorting
                evidence=ev,
                contributions=contribs,
            ))
            evidence.extend(ev)

        # Sort and assign ranks
        rankings.sort(key=lambda r: -r.score)
        for idx, ranking in enumerate(rankings, 1):
            ranking.rank = idx

        # Check for simulation failures
        good_responses = [r for r in rankings if r.score > 0.3]
        if not good_responses:
            dissenting.append("All simulated treatments show limited efficacy")
            confidence -= 0.15

        return AgentOpinion(
            agent_name=self.name,
            treatment_rankings=rankings,
            confidence=max(0.0, min(1.0, confidence)),
            evidence_items=_deduplicate_evidence(evidence),
            dissenting_notes=dissenting,
            warnings=warnings,
            model_versions={
                "treatment_simulator": "cognisom-v1",
                "tumor_dynamics": "stochastic-v1",
                "entity_library": "285-entities",
            },
        )


# --- Helpers ---

def _deduplicate_evidence(items: List[EvidenceItem]) -> List[EvidenceItem]:
    """Remove duplicate evidence items by source_id."""
    seen = set()
    unique = []
    for item in items:
        key = (item.source_type, item.source_id)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _result_to_key(result) -> str:
    """Map a TreatmentResult back to a treatment key."""
    name = getattr(result, "treatment_name", "").lower()
    key_map = {
        "pembrolizumab": "pembrolizumab",
        "keytruda": "pembrolizumab",
        "nivolumab": "nivolumab",
        "opdivo": "nivolumab",
        "ipilimumab": "ipilimumab",
        "yervoy": "ipilimumab",
        "olaparib": "olaparib",
        "lynparza": "olaparib",
        "enzalutamide": "enzalutamide",
        "xtandi": "enzalutamide",
        "vaccine + pembrolizumab": "neoantigen_vaccine_pembro",
        "neoantigen vaccine + pembrolizumab": "neoantigen_vaccine_pembro",
        "neoantigen mrna vaccine": "neoantigen_vaccine",
        "olaparib + pembrolizumab": "olaparib_pembro_combo",
        "pembrolizumab + ipilimumab": "pembro_ipi_combo",
    }
    for pattern, key in key_map.items():
        if pattern in name:
            return key
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def _oncokb_level_to_strength(level: str) -> str:
    """Convert OncoKB evidence level to evidence strength."""
    mapping = {
        "LEVEL_1": "1A",
        "LEVEL_2": "2A",
        "LEVEL_3A": "2B",
        "LEVEL_3B": "3",
        "LEVEL_4": "3",
        "LEVEL_R1": "2A",
        "LEVEL_R2": "2B",
    }
    return mapping.get(level, "3")
