"""
Treatment Simulator
===================

Simulate immunotherapy and targeted therapy response using the
personalized digital twin configuration.

Models checkpoint inhibitor effects (PD-1, CTLA-4 blockade),
PARP inhibitors, and combination therapies based on the patient's
genomic profile and immune landscape.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .twin_config import DigitalTwinConfig

logger = logging.getLogger(__name__)


@dataclass
class TreatmentResult:
    """Result of a simulated treatment."""
    treatment_name: str
    treatment_class: str
    duration_days: int

    # Tumor response
    tumor_response_curve: List[float]  # Relative tumor volume over time (1.0=baseline)
    best_response: float  # Minimum relative volume
    time_to_best_response_days: int
    progression_free_days: int  # Days until tumor grows past baseline

    # Immune effects
    t_cell_reactivation_fraction: float  # Fraction of exhausted T-cells reactivated
    final_exhaustion: float  # Post-treatment exhaustion score
    immune_related_adverse_events: float  # 0-1 risk score

    # Classification
    response_category: str  # CR, PR, SD, PD (RECIST-like)
    confidence: float

    # Details
    mechanism: str
    rationale: str

    def summary(self) -> Dict:
        return {
            "treatment": self.treatment_name,
            "class": self.treatment_class,
            "response": self.response_category,
            "best_response_pct": f"{(1 - self.best_response) * 100:.0f}% reduction",
            "pfs_days": self.progression_free_days,
            "confidence": round(self.confidence, 2),
            "irae_risk": round(self.immune_related_adverse_events, 2),
        }


# Immunotherapy drug profiles
TREATMENT_PROFILES = {
    "pembrolizumab": {
        "name": "Pembrolizumab (Keytruda)",
        "class": "anti-PD-1",
        "target": "PD-1",
        "mechanism": "Blocks PD-1/PD-L1 interaction, reactivating exhausted T-cells",
        "exhaustion_reversal": 0.55,  # Fraction of exhausted T-cells potentially reactivated
        "treg_effect": 0.0,  # No direct Treg effect
        "irae_base_risk": 0.15,
        "effect_onset_days": 14,
        "requires_tmb_high": False,
        "requires_msi_h": False,
        "best_for": ["hot", "suppressed"],
    },
    "nivolumab": {
        "name": "Nivolumab (Opdivo)",
        "class": "anti-PD-1",
        "target": "PD-1",
        "mechanism": "Blocks PD-1/PD-L1 interaction, reactivating exhausted T-cells",
        "exhaustion_reversal": 0.50,
        "treg_effect": 0.0,
        "irae_base_risk": 0.12,
        "effect_onset_days": 14,
        "requires_tmb_high": False,
        "requires_msi_h": False,
        "best_for": ["hot", "suppressed"],
    },
    "ipilimumab": {
        "name": "Ipilimumab (Yervoy)",
        "class": "anti-CTLA-4",
        "target": "CTLA-4",
        "mechanism": "Depletes Tregs and enhances T-cell priming in lymph nodes",
        "exhaustion_reversal": 0.20,
        "treg_effect": -0.40,  # Depletes 40% of Tregs
        "irae_base_risk": 0.35,  # Higher toxicity
        "effect_onset_days": 21,
        "requires_tmb_high": False,
        "requires_msi_h": False,
        "best_for": ["suppressed", "cold"],
    },
    "pembro_ipi_combo": {
        "name": "Pembrolizumab + Ipilimumab",
        "class": "combination checkpoint",
        "target": "PD-1 + CTLA-4",
        "mechanism": "Dual checkpoint blockade: reactivate T-cells + deplete Tregs",
        "exhaustion_reversal": 0.70,
        "treg_effect": -0.35,
        "irae_base_risk": 0.45,
        "effect_onset_days": 14,
        "requires_tmb_high": False,
        "requires_msi_h": False,
        "best_for": ["suppressed", "cold", "excluded"],
    },
    "olaparib": {
        "name": "Olaparib (Lynparza)",
        "class": "PARP inhibitor",
        "target": "PARP1/2",
        "mechanism": "Synthetic lethality in DNA repair-deficient cells (BRCA1/2, ATM)",
        "exhaustion_reversal": 0.0,
        "treg_effect": 0.0,
        "irae_base_risk": 0.05,
        "effect_onset_days": 28,
        "requires_dna_repair_defect": True,
        "best_for": ["any"],
    },
    "enzalutamide": {
        "name": "Enzalutamide (Xtandi)",
        "class": "AR antagonist",
        "target": "AR",
        "mechanism": "Competitively inhibits androgen receptor signaling",
        "exhaustion_reversal": 0.0,
        "treg_effect": 0.0,
        "irae_base_risk": 0.05,
        "effect_onset_days": 30,
        "requires_ar_sensitivity": True,
        "best_for": ["any"],
    },
    "olaparib_pembro_combo": {
        "name": "Olaparib + Pembrolizumab",
        "class": "PARP + checkpoint",
        "target": "PARP + PD-1",
        "mechanism": "PARP-induced DNA damage increases neoantigens + checkpoint blockade",
        "exhaustion_reversal": 0.50,
        "treg_effect": 0.0,
        "irae_base_risk": 0.25,
        "effect_onset_days": 21,
        "requires_dna_repair_defect": True,
        "best_for": ["any_with_hrd"],
    },
    "neoantigen_vaccine": {
        "name": "Personalized mRNA Neoantigen Vaccine",
        "class": "neoantigen vaccine",
        "target": "Tumor neoantigens",
        "mechanism": (
            "mRNA vaccine encoding patient-specific tumor neoantigens. "
            "LNP-encapsulated mRNA is translated by APCs, processed, and "
            "presented on MHC-I to prime neoantigen-specific CD8+ T-cells"
        ),
        "exhaustion_reversal": 0.10,  # Modest direct effect on exhaustion
        "treg_effect": 0.0,
        "irae_base_risk": 0.08,  # Low toxicity profile
        "effect_onset_days": 28,  # 4 weeks for T-cell priming
        "requires_neoantigens": True,
        "best_for": ["hot", "suppressed", "excluded"],
    },
    "neoantigen_vaccine_pembro": {
        "name": "Neoantigen Vaccine + Pembrolizumab",
        "class": "vaccine + checkpoint",
        "target": "Neoantigens + PD-1",
        "mechanism": (
            "Personalized mRNA neoantigen vaccine primes tumor-specific T-cells, "
            "while pembrolizumab removes PD-1/PD-L1 checkpoint brake — "
            "synergistic anti-tumor immunity (mRNA-4157/V940 paradigm)"
        ),
        "exhaustion_reversal": 0.60,
        "treg_effect": 0.0,
        "irae_base_risk": 0.20,
        "effect_onset_days": 21,
        "requires_neoantigens": True,
        "best_for": ["hot", "suppressed", "excluded", "cold"],
    },
}


class TreatmentSimulator:
    """Simulate treatment response using the digital twin.

    Models tumor dynamics under different therapies based on the
    patient's genomic profile and immune microenvironment.

    Supports two parameter sources:
    1. Entity library (preferred): Drug entities with physics_params
       drive the simulation. Add a drug → automatically available.
    2. Hardcoded fallback: TREATMENT_PROFILES dict for backward compatibility.

    Example:
        twin = DigitalTwinConfig.from_profile_and_classification(profile, classification)

        # Entity-driven (preferred):
        from cognisom.library.store import EntityStore
        sim = TreatmentSimulator(store=EntityStore())

        # Or legacy (hardcoded):
        sim = TreatmentSimulator()

        result = sim.simulate("pembrolizumab", twin, duration_days=180)
    """

    def __init__(self, store=None):
        """Initialize with optional EntityStore for entity-driven parameters.

        Args:
            store: EntityStore instance. If provided, drug treatment profiles
                   are loaded from the entity library instead of TREATMENT_PROFILES.
        """
        self._entity_profiles = None
        if store is not None:
            try:
                from .parameter_resolver import SimulationParameterResolver
                resolver = SimulationParameterResolver(store)
                self._entity_profiles = resolver.get_all_treatment_profiles()
                if self._entity_profiles:
                    logger.info(
                        "Loaded %d treatment profiles from entity library",
                        len(self._entity_profiles),
                    )
            except Exception as e:
                logger.warning("Entity profile loading failed, using hardcoded: %s", e)

    @property
    def _profiles(self) -> dict:
        """Get active treatment profiles (entity-driven or hardcoded)."""
        if self._entity_profiles:
            # Merge: entity profiles + hardcoded (entity takes precedence)
            merged = dict(TREATMENT_PROFILES)
            merged.update(self._entity_profiles)
            return merged
        return TREATMENT_PROFILES

    def get_available_treatments(self) -> list:
        """List all available treatment keys."""
        return sorted(self._profiles.keys())

    def simulate(self, treatment_key: str,
                 twin: DigitalTwinConfig,
                 duration_days: int = 180) -> TreatmentResult:
        """Simulate a treatment and predict response.

        Args:
            treatment_key: Key into treatment profiles (entity or hardcoded).
            twin: Personalized DigitalTwinConfig.
            duration_days: Simulation duration in days.

        Returns:
            TreatmentResult with tumor response curve and predictions.
        """
        profile = self._profiles.get(treatment_key)
        if not profile:
            raise ValueError(f"Unknown treatment: {treatment_key}. "
                           f"Available: {list(self._profiles.keys())}")

        # Calculate base effectiveness
        effectiveness = self._calculate_effectiveness(profile, twin)

        # Simulate tumor dynamics
        response_curve = self._simulate_tumor_dynamics(
            effectiveness, profile, twin, duration_days
        )

        # Find best response and PFS
        best_response = min(response_curve)
        best_day = response_curve.index(best_response)

        # PFS: first day tumor volume > baseline after initial response
        pfs = duration_days
        for i in range(best_day, len(response_curve)):
            if response_curve[i] > 1.0:
                pfs = i
                break

        # Classify response (RECIST-like)
        response_category = self._classify_response(best_response)

        # T-cell reactivation
        reversal = profile.get("exhaustion_reversal", 0) * effectiveness
        final_exhaustion = max(0, twin.mean_exhaustion * (1 - reversal))

        # irAE risk
        irae = profile.get("irae_base_risk", 0.1) * (1 + twin.immune_cell_fraction)

        return TreatmentResult(
            treatment_name=profile["name"],
            treatment_class=profile["class"],
            duration_days=duration_days,
            tumor_response_curve=response_curve,
            best_response=best_response,
            time_to_best_response_days=best_day,
            progression_free_days=pfs,
            t_cell_reactivation_fraction=reversal,
            final_exhaustion=final_exhaustion,
            immune_related_adverse_events=min(1.0, irae),
            response_category=response_category,
            confidence=self._estimate_confidence(profile, twin),
            mechanism=profile["mechanism"],
            rationale=self._generate_rationale(profile, twin, effectiveness),
        )

    def compare_treatments(self, treatment_keys: List[str],
                           twin: DigitalTwinConfig,
                           duration_days: int = 180) -> List[TreatmentResult]:
        """Compare multiple treatments.

        Returns list of TreatmentResult sorted by best response.
        """
        results = []
        for key in treatment_keys:
            result = self.simulate(key, twin, duration_days)
            results.append(result)

        results.sort(key=lambda r: r.best_response)
        return results

    def get_recommended_treatments(self, twin: DigitalTwinConfig) -> List[str]:
        """Recommend treatments based on the digital twin profile."""
        recommended = []

        # Checkpoint inhibitors for hot/suppressed tumors or TMB-high
        if (twin.immune_score in ("hot", "suppressed") or
                twin.tumor_mutational_burden >= 10.0 or
                twin.microsatellite_instability == "MSI-H"):
            recommended.append("pembrolizumab")
            if twin.treg_fraction > 0.2:
                recommended.append("pembro_ipi_combo")

        # PARP for DNA repair defects
        if twin.has_dna_repair_defect:
            recommended.append("olaparib")
            if twin.immune_score != "cold":
                recommended.append("olaparib_pembro_combo")

        # AR therapy if AR mutation
        if twin.has_ar_mutation:
            recommended.append("enzalutamide")

        # Neoantigen vaccine if sufficient targets
        if twin.neoantigen_vaccine_candidate:
            recommended.append("neoantigen_vaccine")
            # Vaccine + checkpoint combo is the strongest approach
            if twin.immune_score != "cold":
                recommended.append("neoantigen_vaccine_pembro")

        # Combo for cold tumors
        if twin.immune_score == "cold" and "pembro_ipi_combo" not in recommended:
            recommended.append("pembro_ipi_combo")

        return recommended or ["pembrolizumab"]  # Default

    def _calculate_effectiveness(self, profile: Dict,
                                 twin: DigitalTwinConfig) -> float:
        """Calculate treatment effectiveness (0-1) based on twin parameters."""
        eff = 0.5  # Baseline

        treatment_class = profile.get("class", "")

        if "PD-1" in treatment_class or "anti-PD" in treatment_class:
            # Anti-PD-1 effectiveness depends on:
            # - PD-L1 expression (higher → more responsive)
            eff += 0.2 * twin.pd_l1_expression
            # - Exhausted T-cells to reactivate (need some, but not all)
            if 0.2 < twin.t_cell_exhaustion_fraction < 0.8:
                eff += 0.15
            # - TMB (more neoantigens → more T-cell targets)
            eff += min(0.2, twin.tumor_mutational_burden / 50.0)
            # - Immune score
            if twin.immune_score == "hot":
                eff += 0.15
            elif twin.immune_score == "cold":
                eff -= 0.25

        if "CTLA-4" in treatment_class:
            # Anti-CTLA-4 works by depleting Tregs
            eff += 0.2 * twin.treg_fraction  # More Tregs → more effect
            eff += 0.1  # Priming enhancement

        if "PARP" in treatment_class:
            if twin.has_dna_repair_defect:
                eff += 0.35  # Major boost for HRD
            else:
                eff -= 0.3  # Not effective without HRD

        if "AR" in treatment_class:
            if twin.has_ar_mutation:
                eff -= 0.1  # Resistance mutations reduce effectiveness
            else:
                eff += 0.2  # Wild-type AR responds well

        # Neoantigen vaccine effectiveness
        if "vaccine" in treatment_class:
            # Effectiveness scales with number of vaccine-quality neoantigens
            if twin.vaccine_candidate_count >= 10:
                eff += 0.30
            elif twin.vaccine_candidate_count >= 5:
                eff += 0.20
            elif twin.vaccine_candidate_count >= 3:
                eff += 0.10
            else:
                eff -= 0.20  # Too few targets

            # Strong binders contribute more
            if twin.strong_binder_count >= 5:
                eff += 0.15

            # Immune microenvironment matters
            if twin.immune_score == "hot":
                eff += 0.15
            elif twin.immune_score == "cold":
                eff -= 0.15  # Hard to activate cold tumors

            # MHC-I downregulation reduces vaccine efficacy
            eff -= twin.mhc1_downregulation * 0.3

        # MSI-H boosts checkpoint response
        if twin.microsatellite_instability == "MSI-H":
            if "PD-1" in treatment_class or "checkpoint" in treatment_class:
                eff += 0.2

        return max(0.0, min(1.0, eff))

    def _simulate_tumor_dynamics(self, effectiveness: float,
                                 profile: Dict,
                                 twin: DigitalTwinConfig,
                                 duration_days: int) -> List[float]:
        """Simulate tumor volume changes over time.

        Returns list of relative tumor volumes (1.0 = baseline).
        """
        np.random.seed(hash(profile.get("name", "")) % 2**31)

        onset = profile.get("effect_onset_days", 14)
        volumes = [1.0]
        volume = 1.0

        # Base tumor growth rate (per day)
        growth_rate = 0.005  # ~0.5% per day baseline growth
        if twin.has_pten_loss:
            growth_rate *= 1.3
        if twin.has_ar_mutation:
            growth_rate *= 1.1

        # Treatment kill rate (per day, after onset)
        max_kill_rate = effectiveness * 0.025  # Up to 2.5% per day

        for day in range(1, duration_days + 1):
            # Ramp up treatment effect
            if day < onset:
                treatment_factor = 0
            else:
                ramp = min(1.0, (day - onset) / 21.0)  # 3-week ramp
                treatment_factor = max_kill_rate * ramp

            # Net growth = growth - treatment kill
            net_rate = growth_rate - treatment_factor

            # Add stochastic noise
            noise = np.random.normal(0, 0.002)

            volume *= (1.0 + net_rate + noise)
            volume = max(0.01, volume)  # Floor at 1% (not full eradication)

            # Resistance development over time (gradual)
            if day > 90:
                resistance = (day - 90) / 500.0  # Slow resistance
                volume *= (1.0 + resistance * 0.001)

            volumes.append(round(volume, 4))

        return volumes

    @staticmethod
    def _classify_response(best_response: float) -> str:
        """Classify response using RECIST-like criteria."""
        reduction = 1.0 - best_response
        if reduction >= 0.95:
            return "CR"  # Complete response
        elif reduction >= 0.30:
            return "PR"  # Partial response
        elif best_response <= 1.20:
            return "SD"  # Stable disease
        else:
            return "PD"  # Progressive disease

    def _estimate_confidence(self, profile: Dict,
                             twin: DigitalTwinConfig) -> float:
        """Estimate prediction confidence based on data completeness."""
        confidence = 0.5

        # More data → higher confidence
        if twin.patient:
            confidence += 0.1
        if twin.immune_classification:
            confidence += 0.15
        if twin.tumor_mutational_burden > 0:
            confidence += 0.1
        if twin.microsatellite_instability != "unknown":
            confidence += 0.05

        # Strong biomarker matches boost confidence
        if profile.get("requires_dna_repair_defect") and twin.has_dna_repair_defect:
            confidence += 0.1

        return min(0.95, confidence)

    def _generate_rationale(self, profile: Dict,
                            twin: DigitalTwinConfig,
                            effectiveness: float) -> str:
        """Generate human-readable rationale for the prediction."""
        parts = [f"{profile['name']}: {profile['mechanism']}."]

        if effectiveness > 0.7:
            parts.append("Strong predicted response based on:")
        elif effectiveness > 0.4:
            parts.append("Moderate predicted response based on:")
        else:
            parts.append("Limited predicted response:")

        if twin.immune_score == "hot":
            parts.append("- Hot tumor microenvironment with immune infiltration")
        elif twin.immune_score == "cold":
            parts.append("- Cold tumor — limited immune infiltration may reduce response")

        if twin.tumor_mutational_burden >= 10:
            parts.append(f"- High TMB ({twin.tumor_mutational_burden:.1f}/Mb) — more neoantigens")

        if twin.t_cell_exhaustion_fraction > 0.3:
            parts.append(f"- {twin.t_cell_exhaustion_fraction:.0%} exhausted T-cells available for reactivation")

        if twin.has_dna_repair_defect and "PARP" in profile.get("class", ""):
            parts.append("- DNA repair deficiency — synthetic lethality with PARP inhibition")

        if "vaccine" in profile.get("class", ""):
            parts.append(f"- {twin.vaccine_candidate_count} neoantigen vaccine targets identified")
            if twin.strong_binder_count > 0:
                parts.append(f"- {twin.strong_binder_count} strong MHC-I binders (<50 nM)")
            if twin.hla_alleles:
                parts.append(f"- HLA typing: {len(twin.hla_alleles)} alleles characterized")

        return " ".join(parts)
