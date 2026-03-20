"""
Evidence tracking for MAD Agent decisions.

Every recommendation is backed by traceable EvidenceItems linking to
clinical trials (NCT numbers), biomarker data, NCCN guidelines, or
simulation results. This traceability is required for FDA 7-Step
Credibility Framework compliance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    """A single piece of traceable evidence supporting a recommendation."""

    source_type: str
    """One of: biomarker, clinical_trial, guideline, simulation, literature, validation."""

    source_name: str
    """Human-readable source name, e.g. 'KEYNOTE-199', 'NCCN Prostate v2.2024'."""

    source_id: str
    """Machine-readable ID: DOI, NCT number, guideline version, or internal ref."""

    claim: str
    """What this evidence supports, e.g. 'Pembrolizumab effective in TMB-H mCRPC'."""

    strength: str
    """Evidence level: '1A' (RCT), '2A' (guideline), '2B' (retrospective),
    '3' (case series), 'simulation' (in-silico)."""

    supporting_data: Dict[str, Any] = field(default_factory=dict)
    """Relevant numeric data, e.g. {'ORR': 0.29, 'median_PFS_months': 4.0}."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_name": self.source_name,
            "source_id": self.source_id,
            "claim": self.claim,
            "strength": self.strength,
            "supporting_data": self.supporting_data,
        }


@dataclass
class FeatureContribution:
    """A named biomarker's contribution to a treatment effectiveness score.

    Used for explainability decomposition — shows exactly which patient
    features drove the AI's recommendation up or down.
    """

    feature_name: str
    """e.g. 'TMB_high', 'cold_tumor', 'BRCA2_mutation'."""

    delta: float
    """Change in effectiveness score attributable to this feature."""

    direction: str
    """'positive' or 'negative' — whether this feature helps or hurts."""

    description: str = ""
    """Optional human-readable explanation."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "delta": round(self.delta, 4),
            "direction": self.direction,
            "description": self.description,
        }


@dataclass
class TreatmentRanking:
    """A treatment option with its score and supporting evidence."""

    treatment_key: str
    """Internal key, e.g. 'pembrolizumab', 'olaparib_pembro_combo'."""

    treatment_name: str
    """Display name, e.g. 'Pembrolizumab (Keytruda)'."""

    score: float
    """Composite effectiveness score (0-1)."""

    rank: int
    """Position in this agent's ranking (1 = best)."""

    evidence: List[EvidenceItem] = field(default_factory=list)
    """Evidence items supporting this ranking."""

    contributions: List[FeatureContribution] = field(default_factory=list)
    """Biomarker contributions to the score (explainability)."""

    contraindications: List[str] = field(default_factory=list)
    """Known contraindication flags."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "treatment_key": self.treatment_key,
            "treatment_name": self.treatment_name,
            "score": round(self.score, 4),
            "rank": self.rank,
            "evidence": [e.to_dict() for e in self.evidence],
            "contributions": [c.to_dict() for c in self.contributions],
            "contraindications": self.contraindications,
        }


# --- Clinical trial evidence constants ---

CHECKPOINT_TRIALS: Dict[str, EvidenceItem] = {
    "keynote_199": EvidenceItem(
        source_type="clinical_trial",
        source_name="KEYNOTE-199",
        source_id="NCT02787005",
        claim="Pembrolizumab monotherapy in mCRPC: ORR 3-5%, durable responses in PD-L1+ subset",
        strength="2B",
        supporting_data={"ORR_cohort1": 0.05, "ORR_cohort2": 0.03, "ORR_cohort3": 0.05},
    ),
    "keynote_158": EvidenceItem(
        source_type="clinical_trial",
        source_name="KEYNOTE-158",
        source_id="NCT02628067",
        claim="Pembrolizumab in TMB-H (>=10 mut/Mb) solid tumors: ORR 29%",
        strength="2A",
        supporting_data={"ORR_TMB_high": 0.29, "median_PFS_months": 4.1},
    ),
    "checkmate_650": EvidenceItem(
        source_type="clinical_trial",
        source_name="CheckMate-650",
        source_id="NCT02985957",
        claim="Nivolumab + ipilimumab in mCRPC: ORR 25% in post-chemo, 10% in pre-chemo",
        strength="2B",
        supporting_data={"ORR_post_chemo": 0.25, "ORR_pre_chemo": 0.10},
    ),
}

PARP_TRIALS: Dict[str, EvidenceItem] = {
    "profound": EvidenceItem(
        source_type="clinical_trial",
        source_name="PROfound",
        source_id="NCT02987543",
        claim="Olaparib in HRR-mutated mCRPC: rPFS 7.4 vs 3.6 months (HR 0.34)",
        strength="1A",
        supporting_data={"rPFS_olaparib": 7.4, "rPFS_control": 3.6, "HR": 0.34},
    ),
    "triton2": EvidenceItem(
        source_type="clinical_trial",
        source_name="TRITON2",
        source_id="NCT02952534",
        claim="Rucaparib in BRCA-mutated mCRPC: ORR 43.5%",
        strength="2B",
        supporting_data={"ORR_BRCA": 0.435},
    ),
}

VACCINE_TRIALS: Dict[str, EvidenceItem] = {
    "keynote_942": EvidenceItem(
        source_type="clinical_trial",
        source_name="KEYNOTE-942 (V940)",
        source_id="NCT03897881",
        claim="mRNA-4157/V940 + pembrolizumab in melanoma: 44% reduction in recurrence/death",
        strength="2A",
        supporting_data={"HR_RFS": 0.56, "reduction_pct": 44},
    ),
}

GUIDELINE_EVIDENCE: Dict[str, EvidenceItem] = {
    "nccn_prostate": EvidenceItem(
        source_type="guideline",
        source_name="NCCN Prostate Cancer v2.2024",
        source_id="NCCN-PROST-v2.2024",
        claim="Pembrolizumab for MSI-H/dMMR or TMB-H mCRPC; olaparib for BRCA1/2",
        strength="2A",
        supporting_data={},
    ),
    "nccn_biomarker_tmb": EvidenceItem(
        source_type="guideline",
        source_name="NCCN Biomarkers Compendium",
        source_id="NCCN-BIOMARKERS-2024",
        claim="TMB >=10 mut/Mb: pembrolizumab (tissue-agnostic FDA approval)",
        strength="1A",
        supporting_data={"fda_approval_year": 2020, "threshold_mut_per_mb": 10},
    ),
}
