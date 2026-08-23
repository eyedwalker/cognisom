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
     peptide window and no message. AR was held as 173 of 919 residues,
     so AR T877A -- a headline driver for this platform's demo patient
     -- contributed nothing and reported nothing.

Both are fixed: the bundled reference proteome carries all 37 driver
genes at full length, and AR's numbering offset is declared so calls
reported in the 919-aa literature convention resolve against the 920-aa
canonical sequence.

These tests pin the residues, pin the refusals, and pin the batching that
keeps a real neural-network scorer affordable now that most variants
actually yield peptides.
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

#: Canonical Swiss-Prot lengths, as an independent check that the bundled
#: FASTA holds the sequence it claims to.
CANONICAL_LENGTHS = {
    "AR": 920, "TP53": 393, "PTEN": 403, "BRCA2": 3418, "SPOP": 374,
    "KRAS": 189, "BRAF": 766, "EGFR": 1210, "PIK3CA": 1068, "ATM": 3056,
}


def test_the_reference_proteome_is_complete():
    """These were excerpts: AR held 173 of 919, BRCA2 119 of 3418."""
    assert len(BUILTIN_PROTEINS) >= 35

    for gene, protein in BUILTIN_PROTEINS.items():
        assert protein.is_partial is False, f"{gene} is still truncated"
        assert protein.coverage == 1.0
        assert protein.uniprot_id, f"{gene} has no accession"


@pytest.mark.parametrize("gene,length", sorted(CANONICAL_LENGTHS.items()))
def test_sequences_match_their_canonical_length(gene, length):
    assert len(BUILTIN_PROTEINS[gene].sequence) == length


def test_covers_position_spans_the_whole_protein():
    ar = BUILTIN_PROTEINS["AR"]
    assert ar.covers_position(1) is True
    assert ar.covers_position(877) is True      # AR T877A, once unreachable
    assert ar.covers_position(0) is False
    assert ar.covers_position(len(ar.sequence) + 1) is False


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


def test_apply_mutation_refuses_positions_past_the_end_of_the_protein():
    mapper = GeneProteinMapper()
    ar = mapper.get_protein("AR")
    assert mapper.apply_mutation(ar, "T9999A") is None


# ── Residue numbering ───────────────────────────────────────────────

#: Documented AR ligand-binding-domain mutations, in the 919-aa numbering
#: the androgen-receptor literature uses. UniProt P10275 canonical is 920
#: aa because the exon-1 polyglutamine tract is polymorphic, so every one
#: of these sits one residue later in the canonical sequence.
AR_HOTSPOTS = [("L", 701), ("V", 715), ("W", 741), ("H", 874),
               ("F", 876), ("T", 877), ("M", 895), ("R", 629)]


@pytest.mark.parametrize("residue,pos", AR_HOTSPOTS)
def test_ar_numbering_offset_resolves_documented_hotspots(residue, pos):
    """All eight match at +1 and none at 0 -- that is what makes it a
    gene-level fact rather than a per-variant guess."""
    ar = BUILTIN_PROTEINS["AR"]
    assert ar.numbering_offset == 1
    assert ar.residue_at(pos) == residue


def test_genes_without_a_declared_offset_use_straight_numbering():
    for gene in ("TP53", "KRAS", "BRAF", "PTEN"):
        assert BUILTIN_PROTEINS[gene].numbering_offset == 0

    # And their hotspots resolve without any shift.
    assert BUILTIN_PROTEINS["TP53"].residue_at(248) == "R"
    assert BUILTIN_PROTEINS["KRAS"].residue_at(12) == "G"
    assert BUILTIN_PROTEINS["BRAF"].residue_at(600) == "V"


def test_ar_hotspot_mutations_now_apply():
    """Every one of these was refused when AR was a 173-residue excerpt."""
    mapper = GeneProteinMapper()
    ar = mapper.get_protein("AR")

    mutant = mapper.apply_mutation(ar, "T877A")
    assert mutant is not None
    # The offset means the edit lands at canonical 878, not 877.
    assert mutant.sequence[877] == "A"
    assert ar.sequence[877] == "T"

    for mutation in ("W741L", "L701H", "H874Y", "F876L"):
        assert mapper.apply_mutation(ar, mutation) is not None, mutation


def test_a_genuine_residue_mismatch_is_still_refused():
    """The offset must not become a licence to shift until something fits."""
    mapper = GeneProteinMapper()
    ar = mapper.get_protein("AR")
    # There is no Cys at 877 under any offset in this window.
    assert mapper.apply_mutation(ar, "C877A") is None


# ── Prediction diagnostics ──────────────────────────────────────────

def _variant(gene: str, protein_change: str) -> Variant:
    return Variant(
        chrom="chr1", pos=1, id=".", ref="A", alt="T", qual=99.0,
        filter_status="PASS", gene=gene, protein_change=protein_change,
        is_coding=True, is_cancer_driver=True,
    )


HLA = ["HLA-A*02:01"]


def test_ar_t877a_now_produces_neoantigens():
    """The headline prostate driver, which used to yield zero silently."""
    predictor = NeoantigenPredictor()

    result = predictor.predict(
        cancer_mutations=[_variant("AR", "p.T877A")],
        affected_proteins={"AR": BUILTIN_PROTEINS["AR"]},
        hla_alleles=HLA,
    )

    assert result, "AR T877A produced no neoantigens"
    diag = predictor.diagnostics
    assert diag.variants_yielding_peptides == 1
    assert diag.total_skipped == 0

    # Every peptide carries the substitution, and its wild-type
    # counterpart carries the original threonine.
    for neo in result:
        offset = neo.mutation_position_in_peptide
        assert neo.peptide[offset] == "A"
        assert neo.wild_type_peptide[offset] == "T"


def test_position_beyond_the_protein_is_counted_not_dropped():
    predictor = NeoantigenPredictor()

    result = predictor.predict(
        cancer_mutations=[_variant("AR", "p.T9999A")],
        affected_proteins={"AR": BUILTIN_PROTEINS["AR"]},
        hla_alleles=HLA,
    )

    assert result == []
    diag = predictor.diagnostics
    assert diag.variants_considered == 1
    assert diag.variants_yielding_peptides == 0
    assert diag.skipped_position_not_covered == 1
    assert diag.total_skipped == 1


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
        cancer_mutations=[_variant("AR", "p.T9999A")],
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


def test_binding_is_scored_in_batches_not_one_pair_at_a_time(monkeypatch):
    """MHCflurry per-invocation overhead dwarfs the marginal peptide.

    The scoring loop asks for one peptide-allele pair at a time, which was
    fine only while the PWM fallback -- pure arithmetic -- was what ran
    locally. With a real reference proteome most variants now yield
    peptides, and against MHCflurry the same loop took long enough to blow
    a 120s CI timeout on eight tests. Pairs are warmed in one batched call
    per variant; this asserts the single-pair path is not used at all.
    """
    from types import SimpleNamespace

    from cognisom.genomics import mhcflurry_binding as mb

    calls = {"batch": 0, "single": 0, "pairs": 0}

    def fake_batch(peptides, alleles):
        calls["batch"] += 1
        calls["pairs"] += len(peptides)
        return [SimpleNamespace(affinity_nm=120.0, method="mhcflurry")
                for _ in peptides]

    def fake_single(peptide, allele):
        calls["single"] += 1
        return SimpleNamespace(affinity_nm=120.0, method="mhcflurry")

    monkeypatch.setattr(mb, "is_mhcflurry_available", lambda: True)
    monkeypatch.setattr(mb, "predict_binding_batch", fake_batch)
    monkeypatch.setattr(mb, "predict_binding", fake_single)

    variants = [_variant("TP53", "p.R248W"), _variant("AR", "p.T877A")]
    proteins = {g: BUILTIN_PROTEINS[g] for g in ("TP53", "AR")}

    predictor = NeoantigenPredictor()
    result = predictor.predict(
        cancer_mutations=variants, affected_proteins=proteins,
        hla_alleles=["HLA-A*02:01", "HLA-B*07:02", "HLA-C*07:01"],
    )

    assert result
    assert calls["single"] == 0, (
        f"{calls['single']} peptide-at-a-time calls; batching was bypassed"
    )
    # Exactly one warm-up for the whole run -- not one per variant, and
    # emphatically not one per peptide-allele pair.
    assert calls["batch"] == 1
    assert calls["pairs"] > 50, "batch looks too small to be the real work"


def test_the_batch_cache_does_not_leak_between_predictors():
    """A stale affinity is worse than a slow one."""
    a, b = NeoantigenPredictor(), NeoantigenPredictor()
    assert a._binding_cache is not b._binding_cache


def test_diagnostics_serialize():
    diag = PredictionDiagnostics(variants_considered=5, skipped_not_missense=2)
    payload = diag.to_dict()
    assert payload["variants_considered"] == 5
    assert payload["total_skipped"] == 2
