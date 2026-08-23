"""
The MAD MCP tool handlers.

``_handle_analyze`` could never have succeeded. It hand-rolled the
ingest pipeline and got four things wrong at once:

  * imported ``generate_synthetic_vcf``; the module exports
    ``get_synthetic_vcf``  -> ImportError
  * called ``parser.parse_vcf_text``; the method is ``parse_text``
    -> AttributeError
  * called ``predict(mutations, hla_alleles)``, binding the allele list
    to the ``affected_proteins`` parameter -> TypeError
  * set ``cancer_driver_mutations`` to every annotated variant rather
    than the drivers

Every one was swallowed by ``except Exception: return {"error": str(e)}``
in ``handle_tool_call``, so the tool reported a string instead of
failing, and nothing distinguished "this patient has no result" from
"this code path has never executed".

It now delegates to ``PatientProfileBuilder`` -- the same path the
dashboard uses -- so the tumor/normal comparison and provenance fields
apply here too. These tests assert the handlers return real payloads,
not error dicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics import gene_protein_mapper as gpm
from cognisom.mad.mcp_server import MCP_TOOL_DEFINITIONS, MCPRequestHandler


@pytest.fixture(autouse=True)
def offline_uniprot(monkeypatch):
    """Genes outside the built-in cache otherwise hit the network."""
    monkeypatch.setattr(
        gpm.GeneProteinMapper, "_fetch_from_uniprot", lambda self, gene: None
    )


@pytest.fixture
def handler():
    return MCPRequestHandler()


def assert_not_an_error(payload):
    assert isinstance(payload, dict)
    assert "error" not in payload, f"handler returned an error: {payload.get('error')}"
    return payload


# ── The handler that could never run ────────────────────────────────

def test_analyze_patient_returns_a_board_decision(handler):
    result = handler.handle_tool_call(
        "mad_analyze_patient",
        {"patient_id": "MCP-TEST", "use_synthetic": True},
    )
    assert_not_an_error(result)
    assert result


def test_analyze_patient_accepts_supplied_vcf_text(handler):
    from cognisom.genomics.synthetic_vcf import SYNTHETIC_PROSTATE_VCF

    result = handler.handle_tool_call(
        "mad_analyze_patient",
        {"patient_id": "MCP-VCF", "vcf_text": SYNTHETIC_PROSTATE_VCF},
    )
    assert_not_an_error(result)


def test_analyze_uses_drivers_not_every_annotated_variant(handler):
    """`cancer_driver_mutations=annotated` was every variant in the file."""
    from cognisom.genomics.patient_profile import PatientProfileBuilder
    from cognisom.genomics.synthetic_vcf import SYNTHETIC_PROSTATE_VCF

    profile = PatientProfileBuilder().from_vcf_text(
        SYNTHETIC_PROSTATE_VCF, "driver-check"
    )
    assert len(profile.cancer_driver_mutations) < len(profile.variants)
    assert all(v.is_cancer_driver for v in profile.cancer_driver_mutations)


# ── The other handlers ──────────────────────────────────────────────

def test_compare_treatments_returns_comparisons(handler):
    result = handler.handle_tool_call(
        "mad_compare_treatments",
        {"patient_id": "MCP-CMP", "treatment_keys": ["pembrolizumab", "olaparib"]},
    )
    assert_not_an_error(result)


def test_compare_treatments_reports_unknown_keys_without_failing(handler):
    result = handler.handle_tool_call(
        "mad_compare_treatments",
        {"patient_id": "MCP-CMP", "treatment_keys": ["not-a-drug"]},
    )
    assert_not_an_error(result)


def test_model_cards_handler_returns_cards(handler):
    result = handler.handle_tool_call("mad_model_cards", {"component": "all"})
    assert_not_an_error(result)


def test_compliance_handler_returns_status(handler):
    result = handler.handle_tool_call("mad_compliance_status", {})
    assert_not_an_error(result)


def test_unknown_tool_is_reported_as_unknown(handler):
    result = handler.handle_tool_call("not_a_tool", {})
    assert "Unknown tool" in result["error"]


# ── Declared surface matches the implementation ─────────────────────

def test_every_declared_tool_has_a_handler(handler):
    """A tool advertised over MCP but unroutable is a broken contract."""
    for definition in MCP_TOOL_DEFINITIONS:
        result = handler.handle_tool_call(definition["name"], {})
        assert result.get("error", "") != f"Unknown tool: {definition['name']}"
