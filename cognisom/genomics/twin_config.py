"""
Digital Twin Configuration
==========================

Merges PatientProfile (Phase 1) + Cell State Classification (Phase 2)
into a unified DigitalTwinConfig that drives personalized immune
simulation and treatment prediction.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .patient_profile import PatientProfile
from .cell_state_classifier import ImmuneClassification

logger = logging.getLogger(__name__)


@dataclass
class DigitalTwinConfig:
    """Personalized digital twin configuration combining all data sources.

    This is the central data structure that feeds Phase 3 treatment
    simulation and Phase 4 visualization.
    """
    # Source data
    patient: Optional[PatientProfile] = None
    immune_classification: Optional[ImmuneClassification] = None

    # Genomic biomarkers
    tumor_mutational_burden: float = 0.0
    microsatellite_instability: str = "unknown"  # MSI-H, MSS
    has_dna_repair_defect: bool = False
    has_ar_mutation: bool = False
    has_pten_loss: bool = False

    # Immune landscape (derived from C2S)
    immune_score: str = "unknown"  # hot, cold, excluded, suppressed
    t_cell_exhaustion_fraction: float = 0.0
    mean_exhaustion: float = 0.0
    m2_macrophage_fraction: float = 0.0
    treg_fraction: float = 0.0
    immune_cell_fraction: float = 0.0

    # Derived simulation parameters
    mhc1_downregulation: float = 0.0  # 0=normal, 1=complete loss
    pd_l1_expression: float = 0.5  # PD-L1 on tumor cells (0-1)
    neoantigen_count: int = 0

    # Neoantigen vaccine data
    hla_alleles: List[str] = field(default_factory=list)
    predicted_neoantigen_count: int = 0
    strong_binder_count: int = 0
    vaccine_candidate_count: int = 0
    neoantigen_vaccine_candidate: bool = False

    @classmethod
    def from_profile_and_classification(
        cls,
        profile: PatientProfile,
        classification: ImmuneClassification,
    ) -> "DigitalTwinConfig":
        """Build config from Phase 1 + Phase 2 outputs."""
        config = cls(
            patient=profile,
            immune_classification=classification,
            # Genomic
            tumor_mutational_burden=profile.tumor_mutational_burden,
            microsatellite_instability=profile.msi_status,
            has_dna_repair_defect=profile.has_dna_repair_defect,
            has_ar_mutation=profile.has_ar_mutation,
            has_pten_loss=profile.has_pten_loss,
            # Immune
            immune_score=classification.immune_score,
            t_cell_exhaustion_fraction=classification.exhausted_fraction,
            mean_exhaustion=classification.mean_exhaustion,
            m2_macrophage_fraction=classification.m2_fraction,
            treg_fraction=classification.composition.treg_ratio,
            immune_cell_fraction=classification.composition.immune_fraction,
        )

        # Neoantigen count: use actual predictions if available
        if profile.predicted_neoantigens:
            config.neoantigen_count = sum(
                1 for n in profile.predicted_neoantigens if n.is_weak_binder
            )
            config.predicted_neoantigen_count = len(profile.predicted_neoantigens)
            config.strong_binder_count = profile.strong_binder_count
            config.vaccine_candidate_count = len(profile.vaccine_neoantigens)
            config.neoantigen_vaccine_candidate = profile.neoantigen_vaccine_candidate
        else:
            # Fallback: estimate from TMB
            config.neoantigen_count = int(config.tumor_mutational_burden * 1.5)

        # HLA alleles
        config.hla_alleles = profile.hla_alleles or []

        # MHC-I downregulation based on genomic markers (entity-driven)
        try:
            from .parameter_resolver import SimulationParameterResolver
            from cognisom.library.store import EntityStore
            _resolver = SimulationParameterResolver(EntityStore())
            gene_effects = _resolver.get_gene_effects_for_mutations(
                profile.affected_genes
            )
            config.mhc1_downregulation = gene_effects.get("mhc1_downregulation", 0.0)
        except Exception:
            # Fallback to hardcoded values
            if profile.has_pten_loss:
                config.mhc1_downregulation += 0.3
            if profile.has_tp53_mutation:
                config.mhc1_downregulation += 0.2
            config.mhc1_downregulation = min(1.0, config.mhc1_downregulation)

        # PD-L1 expression correlates with immune infiltration
        if classification.immune_score == "hot":
            config.pd_l1_expression = 0.7
        elif classification.immune_score == "suppressed":
            config.pd_l1_expression = 0.5
        else:
            config.pd_l1_expression = 0.2

        logger.info(
            f"Built DigitalTwinConfig: TMB={config.tumor_mutational_burden:.1f}, "
            f"immune={config.immune_score}, exhaustion={config.mean_exhaustion:.2f}"
        )
        return config

    @classmethod
    def from_profile_only(cls, profile: PatientProfile) -> "DigitalTwinConfig":
        """Build config from genomic profile only (no C2S data)."""
        neo_count = (
            sum(1 for n in profile.predicted_neoantigens if n.is_weak_binder)
            if profile.predicted_neoantigens
            else int(profile.tumor_mutational_burden * 1.5)
        )
        # Calculate MHC-I downregulation from entity library
        mhc1_down = 0.0
        try:
            from .parameter_resolver import SimulationParameterResolver
            from cognisom.library.store import EntityStore
            _resolver = SimulationParameterResolver(EntityStore())
            gene_effects = _resolver.get_gene_effects_for_mutations(
                profile.affected_genes
            )
            mhc1_down = gene_effects.get("mhc1_downregulation", 0.0)
        except Exception:
            if profile.has_pten_loss:
                mhc1_down += 0.3
            if profile.has_tp53_mutation:
                mhc1_down += 0.2
            mhc1_down = min(1.0, mhc1_down)

        return cls(
            patient=profile,
            tumor_mutational_burden=profile.tumor_mutational_burden,
            microsatellite_instability=profile.msi_status,
            has_dna_repair_defect=profile.has_dna_repair_defect,
            has_ar_mutation=profile.has_ar_mutation,
            has_pten_loss=profile.has_pten_loss,
            immune_score="hot" if profile.is_tmb_high else "cold",
            neoantigen_count=neo_count,
            mhc1_downregulation=mhc1_down,
            hla_alleles=profile.hla_alleles or [],
            predicted_neoantigen_count=len(profile.predicted_neoantigens),
            strong_binder_count=profile.strong_binder_count,
            vaccine_candidate_count=len(profile.vaccine_neoantigens),
            neoantigen_vaccine_candidate=profile.neoantigen_vaccine_candidate,
        )

    def to_simulation_params(self) -> Dict[str, Any]:
        """Convert to parameters for the simulation engine.

        Returns dict compatible with cognisom's SimulationEngine.
        """
        return {
            "cellular": {
                "mhc1_downregulation": self.mhc1_downregulation,
                "pd_l1_expression": self.pd_l1_expression,
                "cancer_proliferation_boost": 1.2 if self.has_pten_loss else 1.0,
            },
            "immune": {
                "n_t_cells": max(5, int(30 * self.immune_cell_fraction / 0.15)),
                "n_nk_cells": max(3, int(10 * self.immune_cell_fraction / 0.15)),
                "n_macrophages": max(3, int(15 * self.immune_cell_fraction / 0.15)),
                "t_cell_initial_exhaustion": self.mean_exhaustion,
                "treg_fraction": self.treg_fraction,
                "m2_fraction": self.m2_macrophage_fraction,
                "neoantigen_count": self.neoantigen_count,
            },
            "spatial": {
                "pd_l1_field_strength": self.pd_l1_expression,
            },
        }

    def to_dict(self) -> Dict:
        """Serialize to JSON-safe dict."""
        return {
            "tumor_mutational_burden": self.tumor_mutational_burden,
            "microsatellite_instability": self.microsatellite_instability,
            "has_dna_repair_defect": self.has_dna_repair_defect,
            "has_ar_mutation": self.has_ar_mutation,
            "has_pten_loss": self.has_pten_loss,
            "immune_score": self.immune_score,
            "t_cell_exhaustion_fraction": self.t_cell_exhaustion_fraction,
            "mean_exhaustion": self.mean_exhaustion,
            "m2_macrophage_fraction": self.m2_macrophage_fraction,
            "treg_fraction": self.treg_fraction,
            "immune_cell_fraction": self.immune_cell_fraction,
            "mhc1_downregulation": self.mhc1_downregulation,
            "pd_l1_expression": self.pd_l1_expression,
            "neoantigen_count": self.neoantigen_count,
            "hla_alleles": self.hla_alleles,
            "predicted_neoantigen_count": self.predicted_neoantigen_count,
            "strong_binder_count": self.strong_binder_count,
            "vaccine_candidate_count": self.vaccine_candidate_count,
            "neoantigen_vaccine_candidate": self.neoantigen_vaccine_candidate,
        }
