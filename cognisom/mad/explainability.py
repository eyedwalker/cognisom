"""
Explainability Decomposition
=============================

Wraps TreatmentSimulator._calculate_effectiveness() to capture
which biomarkers contributed how much to each treatment score.

This is deterministic decomposition (not SHAP/LIME) because the
treatment logic is rule-based, not ML. Each if-branch becomes a
named FeatureContribution. This is more auditable for FDA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .evidence import FeatureContribution

logger = logging.getLogger(__name__)


@dataclass
class EffectivenessExplanation:
    """Decomposition of a treatment effectiveness score."""

    treatment_key: str
    treatment_class: str
    total_effectiveness: float
    baseline: float = 0.5
    contributions: List[FeatureContribution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "treatment_key": self.treatment_key,
            "treatment_class": self.treatment_class,
            "total_effectiveness": round(self.total_effectiveness, 4),
            "baseline": self.baseline,
            "contributions": [c.to_dict() for c in self.contributions],
            "sum_contributions": round(
                sum(c.delta for c in self.contributions), 4
            ),
        }

    @property
    def top_positive(self) -> List[FeatureContribution]:
        return sorted(
            [c for c in self.contributions if c.delta > 0],
            key=lambda c: -c.delta,
        )

    @property
    def top_negative(self) -> List[FeatureContribution]:
        return sorted(
            [c for c in self.contributions if c.delta < 0],
            key=lambda c: c.delta,
        )


def explain_effectiveness(
    profile: Dict[str, Any],
    twin: Any,  # DigitalTwinConfig
) -> EffectivenessExplanation:
    """Decompose _calculate_effectiveness into named contributions.

    Mirrors the logic in TreatmentSimulator._calculate_effectiveness()
    but captures each branch as a FeatureContribution.

    Args:
        profile: Treatment profile dict (from TREATMENT_PROFILES).
        twin: DigitalTwinConfig for the patient.

    Returns:
        EffectivenessExplanation with all named contributions.
    """
    baseline = 0.5
    contributions = []
    treatment_class = profile.get("class", "")
    treatment_key = profile.get("name", "unknown")

    # --- Anti-PD-1 branch ---
    if "PD-1" in treatment_class or "anti-PD" in treatment_class:
        # PD-L1 expression
        pd_l1 = getattr(twin, "pd_l1_expression", 0.5)
        delta = 0.2 * pd_l1
        contributions.append(FeatureContribution(
            "pd_l1_expression", round(delta, 4), "positive",
            f"PD-L1 expression {pd_l1:.0%} contributes to anti-PD-1 response",
        ))

        # Exhaustion window
        exhaustion = getattr(twin, "t_cell_exhaustion_fraction", 0.0)
        if 0.2 < exhaustion < 0.8:
            contributions.append(FeatureContribution(
                "exhaustion_window", 0.15, "positive",
                f"T-cell exhaustion {exhaustion:.0%} in reactivation window (20-80%)",
            ))

        # TMB
        tmb = getattr(twin, "tumor_mutational_burden", 0.0)
        tmb_delta = min(0.2, tmb / 50.0)
        contributions.append(FeatureContribution(
            "tmb_neoantigen_load", round(tmb_delta, 4), "positive",
            f"TMB {tmb:.1f} mut/Mb — neoantigen load",
        ))

        # Immune score
        immune_score = getattr(twin, "immune_score", "unknown")
        if immune_score == "hot":
            contributions.append(FeatureContribution(
                "hot_tumor", 0.15, "positive",
                "Hot tumor microenvironment — immune infiltration present",
            ))
        elif immune_score == "cold":
            contributions.append(FeatureContribution(
                "cold_tumor", -0.25, "negative",
                "Cold tumor — minimal immune infiltration",
            ))

    # --- Anti-CTLA-4 branch ---
    if "CTLA-4" in treatment_class:
        treg = getattr(twin, "treg_fraction", 0.0)
        delta = 0.2 * treg
        contributions.append(FeatureContribution(
            "treg_depletion_target", round(delta, 4), "positive",
            f"Treg fraction {treg:.0%} — target for anti-CTLA-4 depletion",
        ))
        contributions.append(FeatureContribution(
            "priming_enhancement", 0.10, "positive",
            "Anti-CTLA-4 enhances T-cell priming in lymph nodes",
        ))

    # --- PARP branch ---
    if "PARP" in treatment_class:
        has_hrd = getattr(twin, "has_dna_repair_defect", False)
        if has_hrd:
            contributions.append(FeatureContribution(
                "hrd_synthetic_lethality", 0.35, "positive",
                "DNA repair defect — synthetic lethality with PARP inhibition",
            ))
        else:
            contributions.append(FeatureContribution(
                "no_hrd", -0.30, "negative",
                "No DNA repair defect — PARP inhibitor not effective",
            ))

    # --- AR antagonist branch ---
    if "AR" in treatment_class and "PARP" not in treatment_class:
        has_ar = getattr(twin, "has_ar_mutation", False)
        if has_ar:
            contributions.append(FeatureContribution(
                "ar_resistance_mutation", -0.10, "negative",
                "AR mutation — potential resistance to AR antagonists",
            ))
        else:
            contributions.append(FeatureContribution(
                "ar_wildtype", 0.20, "positive",
                "Wild-type AR — expected responsiveness",
            ))

    # --- Vaccine branch ---
    if "vaccine" in treatment_class:
        vac_count = getattr(twin, "vaccine_candidate_count", 0)
        if vac_count >= 10:
            contributions.append(FeatureContribution(
                "neoantigen_high", 0.30, "positive",
                f"{vac_count} vaccine-quality neoantigens — strong candidate",
            ))
        elif vac_count >= 5:
            contributions.append(FeatureContribution(
                "neoantigen_moderate", 0.20, "positive",
                f"{vac_count} vaccine candidates — moderate target pool",
            ))
        elif vac_count >= 3:
            contributions.append(FeatureContribution(
                "neoantigen_low", 0.10, "positive",
                f"{vac_count} vaccine candidates — minimal target pool",
            ))
        else:
            contributions.append(FeatureContribution(
                "neoantigen_insufficient", -0.20, "negative",
                f"Only {vac_count} vaccine candidates — below threshold",
            ))

        strong = getattr(twin, "strong_binder_count", 0)
        if strong >= 5:
            contributions.append(FeatureContribution(
                "strong_binders", 0.15, "positive",
                f"{strong} strong MHC binders (<50 nM)",
            ))

        immune_score = getattr(twin, "immune_score", "unknown")
        if immune_score == "hot":
            contributions.append(FeatureContribution(
                "hot_for_vaccine", 0.15, "positive",
                "Hot microenvironment supports vaccine-induced T-cell response",
            ))
        elif immune_score == "cold":
            contributions.append(FeatureContribution(
                "cold_for_vaccine", -0.15, "negative",
                "Cold tumor — vaccine-primed T-cells may not infiltrate",
            ))

        mhc1_down = getattr(twin, "mhc1_downregulation", 0.0)
        if mhc1_down > 0:
            delta = -mhc1_down * 0.3
            contributions.append(FeatureContribution(
                "mhc1_downregulation", round(delta, 4), "negative",
                f"MHC-I downregulation {mhc1_down:.0%} reduces antigen presentation",
            ))

    # --- MSI-H boost for checkpoint ---
    msi = getattr(twin, "microsatellite_instability", "unknown")
    if msi == "MSI-H":
        if "PD-1" in treatment_class or "checkpoint" in treatment_class:
            contributions.append(FeatureContribution(
                "msi_h_checkpoint_boost", 0.20, "positive",
                "MSI-H — FDA-approved biomarker for checkpoint inhibitor response",
            ))

    # Calculate total
    total = baseline + sum(c.delta for c in contributions)
    total = max(0.0, min(1.0, total))

    return EffectivenessExplanation(
        treatment_key=treatment_key,
        treatment_class=treatment_class,
        total_effectiveness=total,
        baseline=baseline,
        contributions=contributions,
    )
