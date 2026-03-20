"""
MCP Server for MAD Agent
=========================

Exposes the MAD Agent system as an MCP (Model Context Protocol) server
for interoperability with Claude, IDEs, and external tools.

Transport: HTTP+SSE on port 8600 (already in security group).

Usage:
    python -m cognisom.mad.mcp_server

Or programmatically:
    from cognisom.mad.mcp_server import create_mcp_server
    server = create_mcp_server()
    server.run(port=8600)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MCP_SERVER_PORT = 8600
MCP_SERVER_NAME = "cognisom-mad"
MCP_SERVER_VERSION = "0.1.0"


# --- MCP Tool Definitions (JSON Schema format) ---

MCP_TOOL_DEFINITIONS = [
    {
        "name": "mad_analyze_patient",
        "description": (
            "Run the full MAD Molecular Tumor Board analysis. "
            "Three specialist agents (Genomics, Immune, Clinical) "
            "independently analyze the patient, then a Board Moderator "
            "synthesizes a consensus treatment recommendation with "
            "traceable evidence."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier",
                },
                "vcf_text": {
                    "type": "string",
                    "description": "VCF file content as text",
                },
                "use_synthetic": {
                    "type": "boolean",
                    "description": "Use synthetic demo patient data",
                    "default": False,
                },
            },
            "required": ["patient_id"],
        },
    },
    {
        "name": "mad_compare_treatments",
        "description": (
            "Compare specific immunotherapy treatments with full "
            "explainability decomposition showing which biomarkers "
            "drive each treatment's predicted effectiveness."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier",
                },
                "treatment_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Treatment keys to compare. Options: pembrolizumab, "
                        "nivolumab, ipilimumab, pembro_ipi_combo, olaparib, "
                        "enzalutamide, olaparib_pembro_combo, neoantigen_vaccine, "
                        "neoantigen_vaccine_pembro"
                    ),
                },
            },
            "required": ["patient_id", "treatment_keys"],
        },
    },
    {
        "name": "mad_get_evidence",
        "description": (
            "Retrieve the full evidence chain for a previous MAD Board "
            "session, including per-agent opinions, clinical trial "
            "references, and biomarker contributions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "MAD Board session UUID",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "mad_audit_trail",
        "description": (
            "Query the MAD Agent audit trail. Retrieve records by "
            "session ID, patient ID, or get recent sessions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Specific session to look up",
                },
                "patient_id": {
                    "type": "string",
                    "description": "Get all sessions for a patient",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max records to return",
                    "default": 50,
                },
            },
        },
    },
    {
        "name": "mad_model_cards",
        "description": (
            "Retrieve model cards (FDA-aligned documentation) for all "
            "MAD Agent components: treatment simulator, neoantigen "
            "predictor, cell state classifier, and board moderator."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "component": {
                    "type": "string",
                    "description": (
                        "Specific component or 'all'. Options: "
                        "treatment_simulator, neoantigen_predictor, "
                        "cell_state_classifier, mad_board, all"
                    ),
                    "default": "all",
                },
            },
        },
    },
    {
        "name": "mad_compliance_status",
        "description": (
            "Check FDA compliance status: Non-Device CDS criteria, "
            "7-Step Credibility Framework progress, and Context of Use."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


class MCPRequestHandler:
    """Handles MCP tool call requests.

    This class bridges MCP protocol requests to the MAD Agent's
    internal tool implementations.
    """

    def __init__(self):
        self._initialized = False

    def handle_tool_call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route an MCP tool call to the appropriate handler."""
        handlers = {
            "mad_analyze_patient": self._handle_analyze,
            "mad_compare_treatments": self._handle_compare,
            "mad_get_evidence": self._handle_evidence,
            "mad_audit_trail": self._handle_audit,
            "mad_model_cards": self._handle_model_cards,
            "mad_compliance_status": self._handle_compliance,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(arguments)
        except Exception as e:
            logger.error(f"MCP tool {tool_name} failed: {e}")
            return {"error": str(e)}

    def _handle_analyze(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run full MAD Board analysis."""
        from ..genomics.patient_profile import PatientProfile
        from ..genomics.twin_config import DigitalTwinConfig
        from ..genomics.treatment_simulator import TreatmentSimulator
        from ..genomics.synthetic_vcf import generate_synthetic_vcf
        from ..genomics.vcf_parser import VCFParser
        from ..genomics.variant_annotator import VariantAnnotator
        from ..genomics.gene_protein_mapper import GeneProteinMapper
        from ..genomics.hla_typer import HLATyper
        from ..genomics.neoantigen_predictor import NeoantigenPredictor
        from ..genomics.cell_state_classifier import CellStateClassifier
        from .board import BoardModerator

        patient_id = args.get("patient_id", "MCP-PATIENT")

        # Get or generate VCF
        vcf_text = args.get("vcf_text")
        if not vcf_text or args.get("use_synthetic"):
            vcf_text = generate_synthetic_vcf()

        # Run pipeline
        parser = VCFParser()
        variants = parser.parse_vcf_text(vcf_text)
        annotator = VariantAnnotator()
        annotated = annotator.annotate(variants)
        mapper = GeneProteinMapper()
        hla_typer = HLATyper()

        profile = PatientProfile(
            patient_id=patient_id,
            variants=variants,
            coding_variants=[v for v in variants if v.is_coding],
            cancer_driver_mutations=annotated,
            affected_genes=list({v.gene for v in annotated if v.gene}),
            tumor_mutational_burden=len([v for v in variants if v.is_coding]) / 30.0,
        )

        hla_alleles = hla_typer.type_from_variants(variants, patient_id)
        profile.hla_alleles = hla_alleles

        predictor = NeoantigenPredictor()
        neoantigens = predictor.predict(profile.cancer_driver_mutations, hla_alleles)
        profile.predicted_neoantigens = neoantigens

        twin = DigitalTwinConfig.from_profile_only(profile)

        simulator = TreatmentSimulator()
        recommended = simulator.get_recommended_treatments(twin)
        results = simulator.compare_treatments(recommended, twin)

        moderator = BoardModerator()
        decision = moderator.run_full_analysis(
            patient_id=patient_id,
            profile=profile,
            twin=twin,
            treatment_results=results,
        )

        return decision.to_dict()

    def _handle_compare(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compare treatments with full explainability decomposition."""
        from ..genomics.patient_profile import PatientProfileBuilder
        from ..genomics.twin_config import DigitalTwinConfig
        from ..genomics.treatment_simulator import TreatmentSimulator, TREATMENT_PROFILES
        from ..genomics.synthetic_vcf import generate_synthetic_vcf
        from .explainability import explain_effectiveness

        patient_id = args.get("patient_id", "MCP-COMPARE")
        treatment_keys = args.get("treatment_keys", list(TREATMENT_PROFILES.keys()))

        # Build patient context (from VCF or synthetic)
        vcf_text = args.get("vcf_text")
        if not vcf_text:
            vcf_text = generate_synthetic_vcf()

        builder = PatientProfileBuilder()
        profile = builder.from_vcf_text(vcf_text, patient_id)
        twin = DigitalTwinConfig.from_profile_only(profile)

        # Run explainability decomposition for each treatment
        comparisons = []
        for key in treatment_keys:
            tx_profile = TREATMENT_PROFILES.get(key)
            if tx_profile is None:
                comparisons.append({"treatment_key": key, "error": f"Unknown: {key}"})
                continue
            explanation = explain_effectiveness(tx_profile, twin)
            comparisons.append(explanation.to_dict())

        # Also simulate to get response curves
        sim = TreatmentSimulator()
        valid_keys = [k for k in treatment_keys if k in TREATMENT_PROFILES]
        results = sim.compare_treatments(valid_keys, twin)
        simulations = [r.summary() for r in results]

        return {
            "patient_id": patient_id,
            "comparisons": comparisons,
            "simulations": simulations,
            "patient_context": {
                "tmb": profile.tumor_mutational_burden,
                "has_hrd": profile.has_dna_repair_defect,
                "has_ar_mutation": profile.has_ar_mutation,
                "vaccine_eligible": twin.neoantigen_vaccine_candidate,
                "immune_score": twin.immune_score,
            },
        }

    def _handle_evidence(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve evidence for a session."""
        from .audit import AuditStore
        session_id = args.get("session_id", "")
        store = AuditStore()
        record = store.get_by_session(session_id)
        if record:
            return record.to_dict()
        return {"error": f"No record for session {session_id}"}

    def _handle_audit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query audit trail."""
        from .audit import AuditStore
        store = AuditStore()
        session_id = args.get("session_id")
        patient_id = args.get("patient_id")
        limit = args.get("limit", 50)

        if session_id:
            record = store.get_by_session(session_id)
            return record.to_dict() if record else {"error": "Not found"}
        elif patient_id:
            records = store.get_by_patient(patient_id, limit)
            return {"records": [r.to_dict() for r in records]}
        else:
            records = store.get_recent(limit)
            return {"records": [r.to_dict() for r in records], "total": store.count()}

    def _handle_model_cards(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return model cards."""
        from .model_cards import get_all_model_cards
        cards = get_all_model_cards()
        component = args.get("component", "all")
        if component == "all":
            return {name: card.to_dict() for name, card in cards.items()}
        elif component in cards:
            return cards[component].to_dict()
        return {"error": f"Unknown component: {component}"}

    def _handle_compliance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return compliance status."""
        from .compliance import ContextOfUse, NonDeviceCDSChecker, CredibilityFramework
        return {
            "context_of_use": ContextOfUse().to_dict(),
            "non_device_cds": NonDeviceCDSChecker().to_dict(),
            "credibility_framework": CredibilityFramework().to_dict(),
        }


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return MCP tool definitions for server registration."""
    return MCP_TOOL_DEFINITIONS


def create_request_handler() -> MCPRequestHandler:
    """Create an MCP request handler instance."""
    return MCPRequestHandler()


if __name__ == "__main__":
    # Standalone server mode — requires mcp package
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server

        server = Server(MCP_SERVER_NAME)
        handler = MCPRequestHandler()

        @server.list_tools()
        async def list_tools():
            return MCP_TOOL_DEFINITIONS

        @server.call_tool()
        async def call_tool(name: str, arguments: dict):
            result = handler.handle_tool_call(name, arguments)
            return [{"type": "text", "text": json.dumps(result, indent=2)}]

        async def main():
            async with stdio_server() as (read, write):
                await server.run(read, write)

        import asyncio
        asyncio.run(main())

    except ImportError:
        logger.warning(
            "MCP package not installed. Install with: pip install mcp\n"
            "Running in standalone request handler mode."
        )
        handler = MCPRequestHandler()
        print(f"MAD MCP Server v{MCP_SERVER_VERSION}")
        print(f"Available tools: {[t['name'] for t in MCP_TOOL_DEFINITIONS]}")
