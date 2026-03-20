"""
MAD Agent — Molecular AI Decision Support
==========================================

Multi-agent system for FDA-compliant immunotherapy treatment selection.
Three specialist agents (Genomics, Immune, Clinical) independently analyze
patient data, then a Board Moderator synthesizes a consensus decision with
full evidence traceability.

Non-Device CDS under 21st Century Cures Act § 3060(a):
  - Provides evidence for clinician's independent review
  - Does not replace clinical judgment
  - Every recommendation tied to traceable evidence
"""

from .evidence import EvidenceItem, FeatureContribution, TreatmentRanking
from .agents import GenomicsAgent, ImmuneAgent, ClinicalAgent, AgentOpinion
from .board import BoardModerator, BoardDecision
from .errors import MADError, MADErrorCode
from .audit import AuditRecord, AuditStore
from .provenance import DataProvenance
from .explainability import EffectivenessExplanation, explain_effectiveness
from .compliance import ContextOfUse, NonDeviceCDSChecker

__version__ = "0.1.0"

__all__ = [
    "EvidenceItem", "FeatureContribution", "TreatmentRanking",
    "GenomicsAgent", "ImmuneAgent", "ClinicalAgent", "AgentOpinion",
    "BoardModerator", "BoardDecision",
    "MADError", "MADErrorCode",
    "AuditRecord", "AuditStore",
    "DataProvenance",
    "EffectivenessExplanation", "explain_effectiveness",
    "ContextOfUse", "NonDeviceCDSChecker",
]
