"""
Board Moderator — Molecular Tumor Board Consensus
==================================================

Synthesizes opinions from GenomicsAgent, ImmuneAgent, and ClinicalAgent
into a unified BoardDecision with agreement tracking, conflict resolution,
and audit-ready evidence chains.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .agents import AgentOpinion, GenomicsAgent, ImmuneAgent, ClinicalAgent
from .evidence import EvidenceItem, TreatmentRanking
from .errors import MADError, MADErrorCode

logger = logging.getLogger(__name__)


@dataclass
class BoardDecision:
    """Final consensus decision from the MAD Molecular Tumor Board."""

    session_id: str
    """Unique ID for audit trail."""

    patient_id: str
    timestamp: str

    # Consensus
    recommended_treatment: str
    """Top consensus treatment key."""

    recommended_treatment_name: str
    """Display name of top treatment."""

    alternative_treatments: List[str]
    """Other viable treatments, ranked."""

    consensus_level: str
    """'unanimous', 'majority', 'split'."""

    confidence: float
    """Overall board confidence (0-1)."""

    # Per-agent breakdown
    agent_opinions: List[AgentOpinion]
    agreement_matrix: Dict[str, Dict[str, bool]]
    """treatment_key → {agent_name: True/False} — whether it's in agent's top 3."""

    # Synthesis
    unified_rationale: str
    """Natural language summary of the board decision."""

    evidence_chain: List[EvidenceItem]
    """All evidence, deduplicated across agents."""

    dissenting_views: List[str]
    """All dissenting notes from all agents."""

    warnings: List[str]
    """All warnings from all agents."""

    # FDA compliance
    context_of_use: str = (
        "Cognisom MAD Agent provides molecular analysis and treatment "
        "comparison evidence to support the clinician's independent review. "
        "It does not replace clinical judgment. All recommendations are "
        "accompanied by source evidence that the clinician should "
        "independently verify. FOR RESEARCH USE ONLY."
    )

    matching_trials: List[Dict[str, Any]] = field(default_factory=list)
    """Recruiting clinical trials matching this patient's biomarkers."""

    limitations: List[str] = field(default_factory=lambda: [
        "Retrospectively validated on prostate cancer cohorts (SU2C mCRPC, TCGA-PRAD)",
        "Treatment effectiveness estimates are simulation-based, not prospective",
        "Neoantigen binding uses MHCflurry when available, PWM fallback otherwise",
        "Immune microenvironment classification uses marker-based heuristics",
        "Does not account for prior treatment history, comorbidities, or patient preference",
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "recommended_treatment": self.recommended_treatment,
            "recommended_treatment_name": self.recommended_treatment_name,
            "alternative_treatments": self.alternative_treatments,
            "consensus_level": self.consensus_level,
            "confidence": round(self.confidence, 4),
            "agent_opinions": [a.to_dict() for a in self.agent_opinions],
            "agreement_matrix": self.agreement_matrix,
            "unified_rationale": self.unified_rationale,
            "evidence_chain": [e.to_dict() for e in self.evidence_chain],
            "dissenting_views": self.dissenting_views,
            "warnings": self.warnings,
            "matching_trials": self.matching_trials,
            "context_of_use": self.context_of_use,
            "limitations": self.limitations,
        }

    def content_hash(self) -> str:
        """SHA256 of the decision content for audit integrity."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


class BoardModerator:
    """Convenes the MAD Molecular Tumor Board and synthesizes a consensus.

    Takes opinions from 3 specialist agents, computes agreement,
    and produces a BoardDecision with traceable evidence.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        top_n: int = 3,
    ):
        """
        Args:
            weights: Agent weights for voting. Default: equal weights.
            top_n: Number of top treatments per agent to consider for agreement.
        """
        self.weights = weights or {
            "genomics": 1.0,
            "immune": 1.0,
            "clinical": 1.0,
        }
        self.top_n = top_n

    def convene(
        self,
        patient_id: str,
        genomics_opinion: AgentOpinion,
        immune_opinion: AgentOpinion,
        clinical_opinion: AgentOpinion,
    ) -> BoardDecision:
        """Synthesize three agent opinions into a board decision."""
        session_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        opinions = [genomics_opinion, immune_opinion, clinical_opinion]

        # --- Compute weighted consensus scores ---
        consensus_scores: Dict[str, float] = {}
        for opinion in opinions:
            weight = self.weights.get(opinion.agent_name, 1.0)
            agent_confidence = opinion.confidence
            for ranking in opinion.treatment_rankings:
                key = ranking.treatment_key
                weighted_score = ranking.score * weight * agent_confidence
                if key not in consensus_scores:
                    consensus_scores[key] = 0.0
                consensus_scores[key] += weighted_score

        # Normalize by total weight
        total_weight = sum(
            self.weights.get(o.agent_name, 1.0) * o.confidence
            for o in opinions
        )
        if total_weight > 0:
            for key in consensus_scores:
                consensus_scores[key] /= total_weight

        # Sort by consensus score
        ranked = sorted(consensus_scores.items(), key=lambda x: -x[1])

        # --- Compute agreement matrix ---
        agreement_matrix = self._compute_agreement(opinions)

        # --- Determine consensus level ---
        top_treatment = ranked[0][0] if ranked else "pembrolizumab"
        consensus_level = self._assess_consensus(
            top_treatment, opinions
        )

        # --- Build alternative treatments ---
        alternatives = [key for key, _ in ranked[1:4] if consensus_scores.get(key, 0) > 0.2]

        # --- Aggregate evidence ---
        all_evidence = []
        for opinion in opinions:
            all_evidence.extend(opinion.evidence_items)
        evidence_chain = _deduplicate_evidence(all_evidence)

        # --- Aggregate dissenting views ---
        dissenting = []
        for opinion in opinions:
            for note in opinion.dissenting_notes:
                prefixed = f"[{opinion.agent_name}] {note}"
                if prefixed not in dissenting:
                    dissenting.append(prefixed)

        # --- Aggregate warnings ---
        all_warnings = []
        for opinion in opinions:
            all_warnings.extend(opinion.warnings)
        unique_warnings = list(dict.fromkeys(all_warnings))

        # --- Check for agent disagreement ---
        if consensus_level == "split":
            unique_warnings.append(MADErrorCode.AGENT_DISAGREEMENT.value)

        # --- Compute overall confidence ---
        confidence = self._compute_confidence(opinions, consensus_level, consensus_scores)

        # --- Generate unified rationale ---
        top_name = self._get_display_name(top_treatment)
        rationale = self._generate_rationale(
            top_treatment, top_name, ranked, opinions,
            agreement_matrix, consensus_level, confidence,
        )

        return BoardDecision(
            session_id=session_id,
            patient_id=patient_id,
            timestamp=timestamp,
            recommended_treatment=top_treatment,
            recommended_treatment_name=top_name,
            alternative_treatments=alternatives,
            consensus_level=consensus_level,
            confidence=confidence,
            agent_opinions=opinions,
            agreement_matrix=agreement_matrix,
            unified_rationale=rationale,
            evidence_chain=evidence_chain,
            dissenting_views=dissenting,
            warnings=unique_warnings,
        )

    def run_full_analysis(
        self,
        patient_id: str,
        profile,           # PatientProfile
        twin,              # DigitalTwinConfig
        treatment_results: List,  # List[TreatmentResult]
        classification=None,      # Optional ImmuneClassification
    ) -> BoardDecision:
        """Convenience method: run all three agents and convene the board."""
        genomics_agent = GenomicsAgent()
        immune_agent = ImmuneAgent()
        clinical_agent = ClinicalAgent()

        genomics_opinion = genomics_agent.analyze(profile=profile, twin=twin)
        immune_opinion = immune_agent.analyze(
            twin=twin, profile=profile, classification=classification,
        )
        clinical_opinion = clinical_agent.analyze(
            twin=twin, treatment_results=treatment_results,
        )

        decision = self.convene(
            patient_id=patient_id,
            genomics_opinion=genomics_opinion,
            immune_opinion=immune_opinion,
            clinical_opinion=clinical_opinion,
        )

        # Enrich with clinical trial matching
        try:
            from ..genomics.clinical_trials import match_patient_to_trials
            trials = match_patient_to_trials(profile, twin, max_results=5)
            decision.matching_trials = [t.to_dict() for t in trials]
        except Exception as e:
            logger.debug("Clinical trial matching skipped: %s", e)

        return decision

    def _compute_agreement(
        self, opinions: List[AgentOpinion]
    ) -> Dict[str, Dict[str, bool]]:
        """Compute which treatments each agent has in their top N."""
        # Collect all treatment keys
        all_keys = set()
        for opinion in opinions:
            for r in opinion.treatment_rankings:
                all_keys.add(r.treatment_key)

        matrix = {}
        for key in all_keys:
            matrix[key] = {}
            for opinion in opinions:
                top_keys = [
                    r.treatment_key
                    for r in opinion.treatment_rankings[:self.top_n]
                ]
                matrix[key][opinion.agent_name] = key in top_keys

        return matrix

    def _assess_consensus(
        self, top_treatment: str, opinions: List[AgentOpinion]
    ) -> str:
        """Determine consensus level for the top treatment."""
        agents_agree = 0
        for opinion in opinions:
            if opinion.top_treatment == top_treatment:
                agents_agree += 1

        if agents_agree == 3:
            return "unanimous"
        elif agents_agree >= 2:
            return "majority"
        else:
            return "split"

    def _compute_confidence(
        self,
        opinions: List[AgentOpinion],
        consensus_level: str,
        scores: Dict[str, float],
    ) -> float:
        """Compute overall board confidence."""
        # Base: average of agent confidences
        base = sum(o.confidence for o in opinions) / len(opinions)

        # Consensus bonus/penalty
        if consensus_level == "unanimous":
            base += 0.10
        elif consensus_level == "split":
            base -= 0.15

        # Score spread: if top treatment is far ahead, more confident
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            gap = sorted_scores[0] - sorted_scores[1]
            base += gap * 0.10  # Small bonus for clear leader

        return max(0.0, min(1.0, base))

    def _generate_rationale(
        self,
        top_key: str,
        top_name: str,
        ranked: List[Tuple[str, float]],
        opinions: List[AgentOpinion],
        agreement: Dict[str, Dict[str, bool]],
        consensus_level: str,
        confidence: float,
    ) -> str:
        """Generate a human-readable unified rationale."""
        parts = []

        # Lead statement
        agree_agents = [
            name for name, agrees in agreement.get(top_key, {}).items() if agrees
        ]
        parts.append(
            f"The MAD Board recommends {top_name} as the primary treatment option "
            f"({consensus_level} consensus, confidence {confidence:.0%})."
        )

        if consensus_level == "unanimous":
            parts.append(
                "All three specialist agents (Genomics, Immune, Clinical) "
                "independently ranked this treatment highest."
            )
        elif consensus_level == "majority":
            parts.append(
                f"This recommendation is supported by the {', '.join(agree_agents)} "
                f"agents."
            )
        else:
            parts.append(
                "The agents did not reach majority consensus. "
                "The clinician should review each agent's rationale carefully."
            )

        # Top agent-specific reasons
        for opinion in opinions:
            top_ranking = next(
                (r for r in opinion.treatment_rankings if r.treatment_key == top_key),
                None,
            )
            if top_ranking and top_ranking.contributions:
                key_contribs = [
                    c for c in top_ranking.contributions
                    if abs(c.delta) > 0.05
                ]
                if key_contribs:
                    reasons = "; ".join(c.description for c in key_contribs[:2])
                    parts.append(f"  [{opinion.agent_name}]: {reasons}.")

        # Alternatives
        if len(ranked) > 1:
            alt_name = self._get_display_name(ranked[1][0])
            parts.append(
                f"Alternative: {alt_name} "
                f"(consensus score {ranked[1][1]:.2f})."
            )

        return "\n".join(parts)

    @staticmethod
    def _get_display_name(key: str) -> str:
        names = {
            "pembrolizumab": "Pembrolizumab (Keytruda)",
            "nivolumab": "Nivolumab (Opdivo)",
            "ipilimumab": "Ipilimumab (Yervoy)",
            "pembro_ipi_combo": "Pembrolizumab + Ipilimumab",
            "olaparib": "Olaparib (Lynparza)",
            "enzalutamide": "Enzalutamide (Xtandi)",
            "olaparib_pembro_combo": "Olaparib + Pembrolizumab",
            "neoantigen_vaccine": "Neoantigen mRNA Vaccine",
            "neoantigen_vaccine_pembro": "Neoantigen Vaccine + Pembrolizumab",
        }
        return names.get(key, key)


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
