"""
Tests for the MAD-pipeline -> patent-pipeline mutation adapter.

Covers:
  * Happy path: known curated hotspot variant maps cleanly
  * HGVS prefix is optional (``p.G12D`` and ``G12D`` both work)
  * Gene not in curated CDS set is rejected with actionable reason
  * Missing gene / protein_change annotations are rejected
  * Frameshift / nonsense / synonymous variants are rejected
  * Curated gene + parseable mutation but not in hotspot table is
    rejected
  * adapt_patient_profile collects all drivable + rejections in
    parallel
  * Duplicate (gene, mutation) tuples are deduplicated
  * Falls back to coding_variants / variants when
    cancer_driver_mutations is absent
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics.mutation_adapter import (
    AdapterRejection,
    adapt_patient_profile,
    variant_to_patent_mutation,
)


def _variant(**fields):
    """Lightweight MAD Variant stand-in; only the fields the adapter
    reads (gene, protein_change) are required."""
    fields.setdefault("gene", None)
    fields.setdefault("protein_change", None)
    return SimpleNamespace(**fields)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gene,protein_change,expected_label", [
    ("KRAS", "p.G12D", "G12D"),
    ("KRAS", "p.G12V", "G12V"),
    ("KRAS", "p.G13D", "G13D"),
    ("BRAF", "p.V600E", "V600E"),
    ("TP53", "p.R175H", "R175H"),
    ("TP53", "p.R248W", "R248W"),
])
def test_curated_hotspots_map_cleanly(gene, protein_change, expected_label):
    v = _variant(gene=gene, protein_change=protein_change)
    result, rejection = variant_to_patent_mutation(v)
    assert rejection is None
    assert result == (gene, expected_label)


def test_hgvs_prefix_is_optional():
    """Both ``p.G12D`` and bare ``G12D`` parse identically."""
    v_with = _variant(gene="KRAS", protein_change="p.G12D")
    v_without = _variant(gene="KRAS", protein_change="G12D")
    r1, _ = variant_to_patent_mutation(v_with)
    r2, _ = variant_to_patent_mutation(v_without)
    assert r1 == r2 == ("KRAS", "G12D")


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

def test_missing_gene_is_rejected_with_reason():
    v = _variant(gene=None, protein_change="p.G12D")
    result, rejection = variant_to_patent_mutation(v)
    assert result is None
    assert "no gene annotation" in rejection.reason


def test_gene_not_in_curated_set_is_rejected_with_actionable_reason():
    v = _variant(gene="EGFR", protein_change="p.L858R")
    result, rejection = variant_to_patent_mutation(v)
    assert result is None
    assert "not in the patent-pipeline curated CDS set" in rejection.reason
    assert "reference_cds.py" in rejection.reason


def test_missing_protein_change_is_rejected():
    v = _variant(gene="KRAS", protein_change=None)
    result, rejection = variant_to_patent_mutation(v)
    assert result is None
    assert "no protein_change" in rejection.reason


def test_unparseable_protein_change_is_rejected():
    """Frameshifts and complex changes don't match the HGVS missense regex."""
    cases = ["p.E1143fs", "p.K519fs", "p.G12_G13del", "p.?", "garbage"]
    for change in cases:
        v = _variant(gene="KRAS", protein_change=change)
        result, rejection = variant_to_patent_mutation(v)
        assert result is None, f"{change} should be rejected"
        assert "not a parseable missense" in rejection.reason


def test_nonsense_mutation_is_rejected():
    """Stop-gain (``p.R175*``) must NOT take the missense path."""
    v = _variant(gene="TP53", protein_change="p.R175*")
    result, rejection = variant_to_patent_mutation(v)
    assert result is None
    assert "nonsense" in rejection.reason.lower()


def test_curated_gene_but_not_curated_hotspot_is_rejected():
    """KRAS Q61H is biologically a real driver but not in the curated
    hotspot table for the patent pipeline. Adapter must reject with
    a reason that points the maintainer at the right file."""
    v = _variant(gene="KRAS", protein_change="p.Q61H")
    result, rejection = variant_to_patent_mutation(v)
    assert result is None
    assert "not in the curated hotspot table" in rejection.reason
    assert "nucleic_acids.py" in rejection.reason


# ---------------------------------------------------------------------------
# adapt_patient_profile
# ---------------------------------------------------------------------------

def _profile(cancer_driver_mutations=None, coding_variants=None, variants=None):
    """Minimal stand-in for the MAD PatientProfile object."""
    return SimpleNamespace(
        cancer_driver_mutations=cancer_driver_mutations,
        coding_variants=coding_variants,
        variants=variants,
    )


def test_adapt_profile_separates_drivable_from_rejections():
    profile = _profile(cancer_driver_mutations=[
        _variant(gene="KRAS", protein_change="p.G12D"),
        _variant(gene="EGFR", protein_change="p.L858R"),  # rejected: gene
        _variant(gene="BRAF", protein_change="p.V600E"),
        _variant(gene="BRCA2", protein_change="p.E1143fs"),  # rejected: frameshift
        _variant(gene="TP53", protein_change="p.R175H"),
    ])
    drivable, rejections = adapt_patient_profile(profile)
    assert set(drivable) == {
        ("KRAS", "G12D"),
        ("BRAF", "V600E"),
        ("TP53", "R175H"),
    }
    assert len(rejections) == 2
    rejection_genes = {r.gene for r in rejections}
    assert rejection_genes == {"EGFR", "BRCA2"}


def test_adapt_profile_deduplicates():
    """Same (gene, mutation) appearing twice (e.g., subclonal hits)
    should appear once in the drivable list."""
    profile = _profile(cancer_driver_mutations=[
        _variant(gene="KRAS", protein_change="p.G12D"),
        _variant(gene="KRAS", protein_change="p.G12D"),
        _variant(gene="KRAS", protein_change="G12D"),  # bare label
    ])
    drivable, _ = adapt_patient_profile(profile)
    assert drivable == [("KRAS", "G12D")]


def test_adapt_profile_falls_back_to_coding_variants():
    """If cancer_driver_mutations is None, use coding_variants."""
    profile = _profile(
        cancer_driver_mutations=None,
        coding_variants=[
            _variant(gene="KRAS", protein_change="p.G12V"),
        ],
    )
    drivable, _ = adapt_patient_profile(profile)
    assert drivable == [("KRAS", "G12V")]


def test_adapt_profile_falls_back_to_variants():
    """If both narrower lists are None, use the full variants list."""
    profile = _profile(
        cancer_driver_mutations=None,
        coding_variants=None,
        variants=[
            _variant(gene="BRAF", protein_change="p.V600E"),
        ],
    )
    drivable, _ = adapt_patient_profile(profile)
    assert drivable == [("BRAF", "V600E")]


def test_adapt_profile_empty_returns_empty_lists():
    profile = _profile(cancer_driver_mutations=[])
    drivable, rejections = adapt_patient_profile(profile)
    assert drivable == []
    assert rejections == []


# ---------------------------------------------------------------------------
# Real Variant integration (uses the actual VCFParser output, not a stand-in)
# ---------------------------------------------------------------------------

def test_real_variant_class_compatible():
    """The adapter must work with cognisom.genomics.vcf_parser.Variant
    instances, not just SimpleNamespace stand-ins. Guards against the
    real Variant dataclass evolving in a way that breaks the adapter."""
    from cognisom.genomics.vcf_parser import Variant
    v = Variant(
        chrom="chr12", pos=25398284, id=".",
        ref="C", alt="A", qual=99.0, filter_status="PASS",
        gene="KRAS",
        protein_change="p.G12D",
    )
    result, rejection = variant_to_patent_mutation(v)
    assert rejection is None
    assert result == ("KRAS", "G12D")


def test_real_variant_with_synthetic_prostate_vcf():
    """End-to-end: parse the synthetic-prostate VCF, run the adapter,
    confirm the expected 4 hotspots come out."""
    from cognisom.genomics.synthetic_vcf import SYNTHETIC_PROSTATE_VCF
    from cognisom.genomics.vcf_parser import VCFParser

    variants = VCFParser().parse_text(SYNTHETIC_PROSTATE_VCF)
    profile = _profile(cancer_driver_mutations=variants)
    drivable, rejections = adapt_patient_profile(profile)

    drivable_set = set(drivable)
    # Per the patent-pipeline curated table, these four are drivable
    # from the synthetic prostate VCF.
    expected = {
        ("KRAS", "G12V"),
        ("BRAF", "V600E"),
        ("TP53", "R175H"),
        ("TP53", "R248W"),
    }
    assert expected.issubset(drivable_set), (
        f"missing expected drivable mutations: {expected - drivable_set}"
    )
    # The other ~46 VCF rows (AR T877A, PTEN R130*, etc.) should appear
    # in rejections with actionable reasons.
    assert len(rejections) >= 30
