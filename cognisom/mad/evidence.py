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
    """'positive', 'negative', or 'neutral'.

    'neutral' means the feature was considered and found absent -- it
    contributed nothing because it was never measured. That is reported
    rather than omitted so a decomposition shows what was unavailable
    instead of silently dropping it.
    """

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
        # ORR was the primary endpoint in cohorts 1 and 2 only -- RECIST-
        # measurable, PD-L1-positive and PD-L1-negative respectively. Cohort 3
        # enrolled bone-predominant disease, for which RECIST ORR is not the
        # endpoint, and no cohort-3 ORR could be verified against the
        # publication; the 0.05 recorded here was unsourced. It is replaced by
        # the disease control rates, which are published for all three
        # cohorts. Antonarakis et al., J Clin Oncol 2020;38:395-405.
        claim=("Pembrolizumab monotherapy in mCRPC: ORR 5% (cohort 1, PD-L1+) "
               "and 3% (cohort 2, PD-L1-) by central RECIST 1.1 review; "
               "cohort 3 is bone-predominant and has no RECIST ORR"),
        strength="2B",
        supporting_data={
            "ORR_cohort1": 0.05,
            "ORR_cohort2": 0.03,
            "DCR_cohort1": 0.10,
            "DCR_cohort2": 0.09,
            "DCR_cohort3": 0.22,
            "median_OS_months": {"cohort1": 9.5, "cohort2": 7.9, "cohort3": 14.1},
        },
    ),
    "keynote_158": EvidenceItem(
        source_type="clinical_trial",
        source_name="KEYNOTE-158",
        source_id="NCT02628067",
        claim="Pembrolizumab in TMB-H (>=10 mut/Mb) solid tumors: ORR 29%",
        strength="2A",
        # ORR 29% (95% CI 21-39) is the FDA-approval analysis over the 102
        # TMB-high patients of 755 with evaluable TMB. Median PFS in that
        # group is 2.1 months (95% CI 2.1-3.7) -- the same median as the
        # TMB-low group, with the separation showing in the 12- and
        # 24-month rates rather than the median. This read 4.1, which is
        # not a published KEYNOTE-158 figure.
        supporting_data={"ORR_TMB_high": 0.29, "median_PFS_months": 2.1},
    ),
    "checkmate_650": EvidenceItem(
        source_type="clinical_trial",
        source_name="CheckMate-650",
        source_id="NCT02985957",
        # The cohorts were assigned backwards here. Sharma et al. report ORR
        # 25% in the PRE-chemotherapy cohort (cohort 1, chemo-naive) and 10%
        # in the POST-chemotherapy cohort (cohort 2), with median rPFS 5.5 vs
        # 3.8 months and median OS 19.0 vs 15.2. Stating it the other way round
        # inverts the clinical message: it implied the combination works better
        # after chemotherapy has failed, when the published result is the
        # opposite.
        claim="Nivolumab + ipilimumab in mCRPC: ORR 25% pre-chemo, 10% post-chemo",
        strength="2B",
        supporting_data={
            "ORR_pre_chemo": 0.25,
            "ORR_post_chemo": 0.10,
            "median_rPFS_months_pre_chemo": 5.5,
            "median_rPFS_months_post_chemo": 3.8,
        },
    ),
}

PARP_TRIALS: Dict[str, EvidenceItem] = {
    "profound": EvidenceItem(
        source_type="clinical_trial",
        source_name="PROfound",
        source_id="NCT02987543",
        claim=("Olaparib in BRCA1/BRCA2/ATM-altered mCRPC (cohort A): rPFS "
               "7.4 vs 3.6 months (HR 0.34, 95% CI 0.25-0.47), confirmed ORR "
               "33% vs 2%"),
        strength="1A",
        # rPFS by blinded independent central review; unrounded 7.39 vs 3.55.
        # The control-arm median is what pins this model's one calibrated
        # parameter; see CALIBRATED_GROWTH_RATE_PER_DAY in
        # cognisom/genomics/treatment_simulator.py.
        supporting_data={
            "rPFS_olaparib": 7.4,
            "rPFS_control": 3.6,
            "HR": 0.34,
            "HR_95CI": [0.25, 0.47],
            "ORR_olaparib": 0.333,
            "ORR_control": 0.023,
        },
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
        # "44% reduction" is 1 - 0.561 and is a point estimate whose
        # interval crosses 1: the randomised phase 2b result did not reach
        # significance at the conventional threshold. Reporting the reduction
        # alone presented it as though it had. Weber et al., Lancet 2024.
        claim=("mRNA-4157/V940 + pembrolizumab in resected stage IIIB-IV "
               "melanoma: RFS HR 0.561 (95% CI 0.309-1.017, two-sided "
               "p=0.053) — not significant at 0.05"),
        strength="2A",
        supporting_data={
            "HR_RFS": 0.561,
            "HR_RFS_95CI": [0.309, 1.017],
            "p_two_sided": 0.053,
            "significant_at_0_05": False,
            "RFS_18mo_combination": 0.79,
            "RFS_18mo_pembrolizumab": 0.62,
            "setting": "adjuvant, resected — no measurable disease, so no ORR",
        },
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
