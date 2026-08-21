"""
Reference-sequence integrity on the neoantigen path.

A neoantigen is a claim about a specific residue of a specific protein.
If the reference sequence under that residue is wrong, or absent, every
downstream artifact -- the peptide, the wild-type comparison, the
agretopicity ratio, the vaccine ranking -- is wrong too, and nothing
about the output looks any different.

Two failures of that kind were live:

  1. The built-in TP53 entry was a 302-residue chimera. It carried the
     correct N-terminus and then a spliced block that repeated
     RVCACPGRDRRTEEENL and dropped the DNA-binding core, putting I at
     175 and C at 248 -- the two residues p53's best-known hotspot
     mutations act on. `apply_mutation(TP53, "R248W")` logged a warning
     that the reference had C, then applied the substitution anyway.

  2. Positions past the end of a truncated reference produced an empty
     peptide window and no message. AR is held as 173 of 919 residues,
     so AR T877A -- a headline driver for this platform's demo patient
     -- contributed nothing and reported nothing.

These tests pin the residues, and pin the refusals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.genomics.gene_protein_mapper import (
    BUILTIN_PROTEINS,
    GeneProteinMapper,
    ProteinInfo,
)
from cognisom.genomics.neoantigen_predictor import (
    NeoantigenPredictor,
    PredictionDiagnostics,
)
from cognisom.genomics.vcf_parser import Variant


# ── TP53 reference correctness ──────────────────────────────────────

# Canonical UniProt P04637 hotspot residues. These are the most
# frequently mutated positions in human cancer, so they are the ones a
# corrupted reference damages first.
TP53_HOTSPOTS = {175: "R", 245: "G", 248: "R", 249: "R", 273: "R", 282: "R"}


def test_tp53_reference_is_complete():
    tp53 = BUILTIN_PROTEINS["TP53"]
    assert len(tp53.sequence) == 393 == tp53.length
    assert tp53.is_partial is False
    assert tp53.sequence.startswith("MEEPQSDPSV")
    assert tp53.sequence.endswith("KTEGPDSD")


@pytest.mark.parametrize("pos,residue", sorted(TP53_HOTSPOTS.items()))
def test_tp53_hotspot_residues_are_correct(pos, residue):
    """Regression guard against reintroducing the chimeric sequence."""
    assert BUILTIN_PROTEINS["TP53"].sequence[pos - 1] == residue


def test_tp53_hotspot_mutations_now_apply_to_the_right_residue():
    mapper = GeneProteinMapper()
    tp53 = mapper.get_protein("TP53")

    mutant = mapper.apply_mutation(tp53, "R248W")
    assert mutant is not None
    assert mutant.sequence[247] == "W"
    # Only that residue changed.
    assert mutant.sequence[:247] == tp53.sequence[:247]
    assert mutant.sequence[248:] == tp53.sequence[248:]

    assert mapper.apply_mutation(tp53, "R175H").sequence[174] == "H"


# ── Partial-reference bookkeeping ───────────────────────────────────

def test_partial_builtins_declare_themselves_partial():
    """The excerpts must be distinguishable from complete sequences."""
    for gene in ("AR", "PTEN", "BRCA2", "SPOP"):
        protein = BUILTIN_PROTEINS[gene]
        assert protein.is_partial is True
        assert len(protein.sequence) < protein.length
        assert 0.0 < protein.coverage < 1.0


def test_covers_position_tracks_the_held_sequence_not_the_true_length():
    ar = BUILTIN_PROTEINS["AR"]
    assert ar.length == 919
    assert ar.covers_position(100) is True
    assert ar.covers_position(len(ar.sequence)) is True
    assert ar.covers_position(len(ar.sequence) + 1) is False
    assert ar.covers_position(877) is False  # AR T877A
    assert ar.covers_position(0) is False


# ── apply_mutation must refuse, not guess ───────────────────────────

def test_apply_mutation_refuses_on_wildtype_mismatch():
    """Previously this warned and proceeded, editing the wrong residue."""
    mapper = GeneProteinMapper()
    protein = ProteinInfo(
        gene="TEST", uniprot_id="X", protein_name="t",
        sequence="MAAAAAAAAA", length=10,
    )
    # Position 5 holds A, not Q.
    assert mapper.apply_mutation(protein, "Q5W") is None
    # The matching residue is accepted.
    assert mapper.apply_mutation(protein, "A5W").sequence == "MAAAWAAAAA"


def test_apply_mutation_refuses_positions_beyond_a_truncated_reference():
    mapper = GeneProteinMapper()
    ar = mapper.get_protein("AR")
    assert mapper.apply_mutation(ar, "T877A") is None
    assert mapper.apply_mutation(ar, "L702H") is None


# ── Prediction diagnostics ──────────────────────────────────────────

def _variant(gene: str, protein_change: str) -> Variant:
    return Variant(
        chrom="chr1", pos=1, id=".", ref="A", alt="T", qual=99.0,
        filter_status="PASS", gene=gene, protein_change=protein_change,
        is_coding=True, is_cancer_driver=True,
    )


HLA = ["HLA-A*02:01"]


def test_position_beyond_reference_is_counted_not_dropped():
    predictor = NeoantigenPredictor()
    ar = BUILTIN_PROTEINS["AR"]

    result = predictor.predict(
        cancer_mutations=[_variant("AR", "p.T877A")],
        affected_proteins={"AR": ar},
        hla_alleles=HLA,
    )

    assert result == []
    diag = predictor.diagnostics
    assert diag.variants_considered == 1
    assert diag.variants_yielding_peptides == 0
    assert diag.skipped_position_not_covered == 1
    assert diag.total_skipped == 1
    assert "AR" in diag.genes_with_partial_reference


def test_wildtype_mismatch_is_counted_with_detail():
    predictor = NeoantigenPredictor()
    protein = ProteinInfo(
        gene="TEST", uniprot_id="X", protein_name="t",
        sequence="M" + "A" * 40, length=41,
    )

    result = predictor.predict(
        cancer_mutations=[_variant("TEST", "p.Q20W")],
        affected_proteins={"TEST": protein},
        hla_alleles=HLA,
    )

    assert result == []
    diag = predictor.diagnostics
    assert diag.skipped_wildtype_mismatch == 1
    assert len(diag.mismatch_details) == 1
    assert "reference has A at position 20, not Q" in diag.mismatch_details[0]


def test_non_missense_variants_are_counted():
    predictor = NeoantigenPredictor()
    predictor.predict(
        cancer_mutations=[
            _variant("TP53", "p.E1143fs"),
            _variant("TP53", "p.R130*"),
        ],
        affected_proteins={"TP53": BUILTIN_PROTEINS["TP53"]},
        hla_alleles=HLA,
    )
    assert predictor.diagnostics.skipped_not_missense == 2


def test_a_valid_missense_variant_still_produces_neoantigens():
    """The guards must not block the working path."""
    predictor = NeoantigenPredictor()
    result = predictor.predict(
        cancer_mutations=[_variant("TP53", "p.R248W")],
        affected_proteins={"TP53": BUILTIN_PROTEINS["TP53"]},
        hla_alleles=HLA,
    )

    assert result
    diag = predictor.diagnostics
    assert diag.variants_yielding_peptides == 1
    assert diag.total_skipped == 0
    # Every peptide carries the mutant residue at the recorded offset.
    for neo in result:
        assert neo.peptide[neo.mutation_position_in_peptide] == "W"
        assert neo.wild_type_peptide[neo.mutation_position_in_peptide] == "R"


def test_diagnostics_are_reset_between_runs():
    predictor = NeoantigenPredictor()
    predictor.predict(
        cancer_mutations=[_variant("AR", "p.T877A")],
        affected_proteins={"AR": BUILTIN_PROTEINS["AR"]},
        hla_alleles=HLA,
    )
    assert predictor.diagnostics.skipped_position_not_covered == 1

    predictor.predict(
        cancer_mutations=[_variant("TP53", "p.R248W")],
        affected_proteins={"TP53": BUILTIN_PROTEINS["TP53"]},
        hla_alleles=HLA,
    )
    assert predictor.diagnostics.skipped_position_not_covered == 0
    assert predictor.diagnostics.variants_considered == 1


# ── Scorer provenance ───────────────────────────────────────────────

def test_binding_method_travels_with_each_neoantigen():
    """An affinity must never be separable from the scorer that made it."""
    predictor = NeoantigenPredictor()
    result = predictor.predict(
        cancer_mutations=[_variant("TP53", "p.R248W")],
        affected_proteins={"TP53": BUILTIN_PROTEINS["TP53"]},
        hla_alleles=HLA,
    )

    assert result
    for neo in result:
        assert neo.binding_method == predictor.binding_method
        assert neo.binding_method in ("mhcflurry-2.0.6", "pwm-fallback")
        assert neo.to_dict()["binding_method"] == neo.binding_method


def test_diagnostics_serialize():
    diag = PredictionDiagnostics(variants_considered=5, skipped_not_missense=2)
    payload = diag.to_dict()
    assert payload["variants_considered"] == 5
    assert payload["total_skipped"] == 2
