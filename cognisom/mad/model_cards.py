"""
Model Cards for MAD Agent Components
======================================

Structured metadata for each AI/algorithmic component, following
the FDA 7-Step Credibility Framework (Step 4: Credibility Plan).

Each model card documents: purpose, training data equivalent,
validation metrics, intended population, and known limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ModelCard:
    """FDA-aligned model documentation card."""

    name: str
    version: str
    purpose: str
    model_type: str
    """'rule-based', 'heuristic', 'position-weight-matrix', 'simulation'."""

    intended_population: str
    """Patient population the model was designed for."""

    # Data basis
    training_data_equivalent: str
    """What informed the model parameters (not ML training data)."""

    reference_trials: List[str]
    """NCT numbers or trial names that informed parameters."""

    # Validation
    validation_dataset: str
    validation_metrics: Dict[str, Any]
    validation_citation: str

    # Performance
    processing_speed: str

    # Limitations
    known_limitations: List[str]
    failure_modes: List[str]

    # Regulatory
    regulatory_classification: str
    """e.g. 'Non-Device CDS', 'Research Use Only'."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "purpose": self.purpose,
            "model_type": self.model_type,
            "intended_population": self.intended_population,
            "training_data_equivalent": self.training_data_equivalent,
            "reference_trials": self.reference_trials,
            "validation_dataset": self.validation_dataset,
            "validation_metrics": self.validation_metrics,
            "validation_citation": self.validation_citation,
            "processing_speed": self.processing_speed,
            "known_limitations": self.known_limitations,
            "failure_modes": self.failure_modes,
            "regulatory_classification": self.regulatory_classification,
        }


# --- Pre-defined model cards for Cognisom components ---

TREATMENT_SIMULATOR_CARD = ModelCard(
    name="Treatment Simulator",
    version="v1.0",
    purpose="Simulate immunotherapy and targeted therapy response using patient digital twin",
    model_type="rule-based + stochastic simulation",
    intended_population="Metastatic castration-resistant prostate cancer (mCRPC)",
    training_data_equivalent=(
        "Parameters derived from published clinical trial endpoints: "
        "KEYNOTE-199 (NCT02787005), KEYNOTE-158 (NCT02628067), "
        "CheckMate-650 (NCT02985957), PROfound (NCT02987543), "
        "TRITON2 (NCT02952534), KEYNOTE-942 (NCT03897881). "
        "Entity library physics_params for drug-specific parameters."
    ),
    reference_trials=[
        "NCT02787005", "NCT02628067", "NCT02985957",
        "NCT02987543", "NCT02952534", "NCT03897881",
    ],
    validation_dataset="SU2C/PCF Dream Team mCRPC 2019 (429 patients)",
    validation_metrics={
        "cohort_size": 429,
        "processing_speed_per_patient": "0.17s",
        "parp_candidate_concordance": "tracked",
        "tmb_correlation": "tracked",
    },
    validation_citation="Robinson et al., Cell 2019",
    processing_speed="0.17 seconds per patient",
    known_limitations=[
        "Based on prostate cancer; not validated for other tumor types",
        "Stochastic simulation — results vary with random seed",
        "Does not model drug-drug interactions or pharmacokinetics",
        "Does not account for prior treatment history or resistance evolution",
        "9 treatment regimens only; does not cover all FDA-approved options",
    ],
    failure_modes=[
        "Low variant count (<30) may produce unreliable TMB estimates",
        "Cold tumors with no actionable biomarkers yield low-confidence results",
        "Combination therapy modeling assumes additive effects (may miss synergies)",
    ],
    regulatory_classification="Research Use Only — Non-Device CDS target",
)

NEOANTIGEN_PREDICTOR_CARD = ModelCard(
    name="Neoantigen Predictor",
    version="pwm-v1.0",
    purpose="Predict peptide-MHC binding affinity for neoantigen vaccine candidate selection",
    model_type="position-weight-matrix",
    intended_population="Patients with somatic missense mutations in cancer driver genes",
    training_data_equivalent=(
        "Position-weight matrix parameters derived from published MHC binding data. "
        "Anchor residue preferences from NetMHCpan 4.1 and MHCflurry publications. "
        "Hydrophobicity scoring from Kyte-Doolittle scale."
    ),
    reference_trials=["NCT03897881"],
    validation_dataset="IEDB Reference Panel (20 peptides, Vita et al. NAR 2019)",
    validation_metrics={
        "concordance_rate": 0.75,
        "ic50_correlation": 0.63,
        "panel_size": 20,
        "alleles_tested": ["HLA-A*02:01", "HLA-A*24:02", "HLA-A*03:01", "HLA-B*07:02"],
    },
    validation_citation="Vita et al., Nucleic Acids Research 2019 (PMID: 30357391)",
    processing_speed="<1 second per patient",
    known_limitations=[
        "75% concordance with IEDB — not clinical-grade accuracy",
        "Position-weight matrix less accurate than deep learning (NetMHCpan 4.1)",
        "Limited HLA allele coverage (10 alleles per locus, Caucasian frequencies)",
        "Does not model peptide processing (proteasomal cleavage, TAP transport)",
        "Agretopicity scoring is simplified (ratio-based, not structural)",
    ],
    failure_modes=[
        "Rare HLA alleles not in population frequency table",
        "Non-standard amino acids in peptide sequences",
        "Very short peptides (<8 AA) or very long peptides (>11 AA)",
    ],
    regulatory_classification="Research Use Only",
)

CELL_STATE_CLASSIFIER_CARD = ModelCard(
    name="Cell State Classifier",
    version="marker-heuristic-v1.0",
    purpose="Classify tumor immune microenvironment from cell type composition",
    model_type="heuristic (threshold-based)",
    intended_population="Solid tumors with scRNA-seq or marker-based immune profiling",
    training_data_equivalent=(
        "Thresholds derived from published tumor immunology literature. "
        "Hot/cold classification: Galon & Bruni, Nat Rev Drug Discovery 2019. "
        "Exhaustion scoring: Wherry & Kurachi, Nat Rev Immunol 2015."
    ),
    reference_trials=[],
    validation_dataset="Synthetic prostate TME profiles",
    validation_metrics={
        "classification_categories": ["hot", "cold", "excluded", "suppressed"],
        "exhaustion_threshold": 0.6,
        "immune_fraction_hot": ">15%",
        "immune_fraction_cold": "<5%",
    },
    validation_citation="Galon & Bruni, Nat Rev Drug Discovery 2019",
    processing_speed="<0.1 seconds per patient",
    known_limitations=[
        "Threshold-based, not trained on patient data",
        "Binary classification (hot/cold) oversimplifies immune heterogeneity",
        "Synthetic validation only — not tested on real scRNA-seq cohorts",
        "Does not account for spatial immune distribution",
        "M1/M2 macrophage polarization is a simplification",
    ],
    failure_modes=[
        "Very small cell counts may produce unreliable fractions",
        "Missing cell types default to 0 (may misclassify)",
    ],
    regulatory_classification="Research Use Only",
)

MAD_BOARD_CARD = ModelCard(
    name="MAD Board Moderator",
    version="v0.1.0",
    purpose="Synthesize multi-agent treatment opinions into consensus recommendation",
    model_type="rule-based (weighted voting)",
    intended_population="mCRPC patients analyzed by the Cognisom pipeline",
    training_data_equivalent=(
        "Consensus logic designed by clinical oncology domain experts. "
        "Equal agent weighting by default. Agreement assessment based on "
        "top-3 treatment overlap across agents."
    ),
    reference_trials=[],
    validation_dataset="Pending: SU2C mCRPC 429-patient retrospective study",
    validation_metrics={
        "consensus_levels": ["unanimous", "majority", "split"],
        "n_agents": 3,
        "voting_scheme": "weighted average with confidence scaling",
    },
    validation_citation="Pending",
    processing_speed="<0.5 seconds per patient",
    known_limitations=[
        "Equal weighting may not reflect clinical priority of different data types",
        "No learning from outcomes — weights are static",
        "Split decisions require clinician judgment — no automatic tiebreaker",
        "Limited to 9 treatment options currently modeled",
    ],
    failure_modes=[
        "All agents return low confidence (<0.3) — board confidence unreliable",
        "Novel mutations not in driver database — genomics agent uninformed",
    ],
    regulatory_classification="Research Use Only — Non-Device CDS target",
)


def get_all_model_cards() -> Dict[str, ModelCard]:
    """Return all model cards indexed by component name."""
    return {
        "treatment_simulator": TREATMENT_SIMULATOR_CARD,
        "neoantigen_predictor": NEOANTIGEN_PREDICTOR_CARD,
        "cell_state_classifier": CELL_STATE_CLASSIFIER_CARD,
        "mad_board": MAD_BOARD_CARD,
    }
