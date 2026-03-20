"""
MCP Tool Wrappers for MAD Agent
================================

Each MCP tool wraps MAD Agent functionality into the existing
Tool/ToolResult pattern from cognisom/agent/tools.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..agent.tools import Tool, ToolResult
from .board import BoardModerator, BoardDecision
from .agents import GenomicsAgent, ImmuneAgent, ClinicalAgent, AgentOpinion
from .audit import AuditStore, AuditRecord
from .provenance import DataProvenance

logger = logging.getLogger(__name__)


class MADAnalyzePatientTool(Tool):
    """Full MAD Board analysis for a patient."""

    name = "mad/analyze_patient"
    description = (
        "Run the full MAD Molecular Tumor Board analysis: "
        "3 specialist agents independently analyze the patient, "
        "then a Board Moderator synthesizes a consensus decision."
    )
    parameters = {
        "patient_id": "Patient identifier",
        "profile": "PatientProfile object or dict",
        "twin": "DigitalTwinConfig object or dict",
        "treatment_results": "List of TreatmentResult objects",
        "classification": "Optional ImmuneClassification",
        "user_id": "Clinician ID for audit trail",
    }

    def run(self, **kwargs) -> ToolResult:
        try:
            patient_id = kwargs.get("patient_id", "anonymous")
            profile = kwargs["profile"]
            twin = kwargs["twin"]
            treatment_results = kwargs.get("treatment_results", [])
            classification = kwargs.get("classification")
            user_id = kwargs.get("user_id", "system")

            moderator = BoardModerator()
            decision = moderator.run_full_analysis(
                patient_id=patient_id,
                profile=profile,
                twin=twin,
                treatment_results=treatment_results,
                classification=classification,
            )

            # Record audit
            try:
                audit_store = AuditStore()
                vcf_text = kwargs.get("vcf_text", str(patient_id))
                record = AuditRecord.from_board_decision(decision, user_id, vcf_text)
                audit_store.record(record)
            except Exception as e:
                logger.warning(f"Audit recording failed: {e}")

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=decision.to_dict(),
                metadata={
                    "session_id": decision.session_id,
                    "consensus_level": decision.consensus_level,
                    "confidence": decision.confidence,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
            )


class MADGenomicsOpinionTool(Tool):
    """Run only the Genomics Agent."""

    name = "mad/genomics_opinion"
    description = "Get the Genomics Agent's independent treatment ranking"
    parameters = {
        "profile": "PatientProfile object",
        "twin": "DigitalTwinConfig object",
    }

    def run(self, **kwargs) -> ToolResult:
        try:
            agent = GenomicsAgent()
            opinion = agent.analyze(
                profile=kwargs["profile"],
                twin=kwargs["twin"],
            )
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=opinion.to_dict(),
            )
        except Exception as e:
            return ToolResult(self.name, False, error=str(e))


class MADImmuneOpinionTool(Tool):
    """Run only the Immune Agent."""

    name = "mad/immune_opinion"
    description = "Get the Immune Agent's independent treatment ranking"
    parameters = {
        "twin": "DigitalTwinConfig object",
        "profile": "PatientProfile object",
        "classification": "Optional ImmuneClassification",
    }

    def run(self, **kwargs) -> ToolResult:
        try:
            agent = ImmuneAgent()
            opinion = agent.analyze(
                twin=kwargs["twin"],
                profile=kwargs["profile"],
                classification=kwargs.get("classification"),
            )
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=opinion.to_dict(),
            )
        except Exception as e:
            return ToolResult(self.name, False, error=str(e))


class MADClinicalOpinionTool(Tool):
    """Run only the Clinical Agent."""

    name = "mad/clinical_opinion"
    description = "Get the Clinical Agent's simulation-driven treatment ranking"
    parameters = {
        "twin": "DigitalTwinConfig object",
        "treatment_results": "List of TreatmentResult objects",
    }

    def run(self, **kwargs) -> ToolResult:
        try:
            agent = ClinicalAgent()
            opinion = agent.analyze(
                twin=kwargs["twin"],
                treatment_results=kwargs["treatment_results"],
            )
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=opinion.to_dict(),
            )
        except Exception as e:
            return ToolResult(self.name, False, error=str(e))


class MADCompareTreatmentsTool(Tool):
    """Side-by-side treatment comparison with evidence."""

    name = "mad/compare_treatments"
    description = "Compare specific treatments with full evidence breakdown"
    parameters = {
        "treatment_keys": "List of treatment keys to compare",
        "twin": "DigitalTwinConfig object",
        "profile": "PatientProfile object",
    }

    def run(self, **kwargs) -> ToolResult:
        try:
            from .explainability import explain_effectiveness
            from ..genomics.treatment_simulator import TREATMENT_PROFILES

            treatment_keys = kwargs["treatment_keys"]
            twin = kwargs["twin"]

            comparisons = []
            for key in treatment_keys:
                profile = TREATMENT_PROFILES.get(key)
                if profile is None:
                    comparisons.append({
                        "treatment_key": key,
                        "error": f"Unknown treatment: {key}",
                    })
                    continue

                explanation = explain_effectiveness(profile, twin)
                comparisons.append(explanation.to_dict())

            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"comparisons": comparisons},
            )
        except Exception as e:
            return ToolResult(self.name, False, error=str(e))


class MADGetEvidenceTool(Tool):
    """Retrieve evidence chain for a specific recommendation."""

    name = "mad/get_evidence"
    description = "Get all evidence items supporting a treatment recommendation"
    parameters = {
        "session_id": "MAD Board session ID",
    }

    def run(self, **kwargs) -> ToolResult:
        try:
            session_id = kwargs["session_id"]
            audit_store = AuditStore()
            record = audit_store.get_by_session(session_id)
            if record is None:
                return ToolResult(
                    self.name, False,
                    error=f"No audit record for session {session_id}",
                )
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=record.to_dict(),
            )
        except Exception as e:
            return ToolResult(self.name, False, error=str(e))


class MADAuditTrailTool(Tool):
    """Query the audit trail."""

    name = "mad/audit_trail"
    description = "Retrieve audit records by session, patient, or recent"
    parameters = {
        "session_id": "Optional: specific session",
        "patient_id": "Optional: all sessions for a patient",
        "limit": "Optional: max records (default 50)",
    }

    def run(self, **kwargs) -> ToolResult:
        try:
            audit_store = AuditStore()
            session_id = kwargs.get("session_id")
            patient_id = kwargs.get("patient_id")
            limit = kwargs.get("limit", 50)

            if session_id:
                record = audit_store.get_by_session(session_id)
                data = record.to_dict() if record else None
            elif patient_id:
                records = audit_store.get_by_patient(patient_id, limit)
                data = [r.to_dict() for r in records]
            else:
                records = audit_store.get_recent(limit)
                data = [r.to_dict() for r in records]

            return ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
                metadata={"total_records": audit_store.count()},
            )
        except Exception as e:
            return ToolResult(self.name, False, error=str(e))


def register_mad_tools(registry) -> None:
    """Register all MAD tools with a ToolRegistry."""
    registry.register(MADAnalyzePatientTool())
    registry.register(MADGenomicsOpinionTool())
    registry.register(MADImmuneOpinionTool())
    registry.register(MADClinicalOpinionTool())
    registry.register(MADCompareTreatmentsTool())
    registry.register(MADGetEvidenceTool())
    registry.register(MADAuditTrailTool())
