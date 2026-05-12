"""
MHC loading unit tests.

Covers engine/py/immune/mhc_loading.py:
  * peptide with HLA-A*02:01 canonical anchor pattern scores < 500 nM
  * peptide with charged residue at P1 and basic anchor pattern is a
    non-binder
  * score_all drops non-binders by default
  * best_per_peptide reduces to the strongest allele per peptide
  * presentation_score is monotone with cleavage score (multiplicative)
  * allele canonicalization adds 'HLA-' prefix
  * neoantigen flag propagates from Peptide.is_mutant
"""
from __future__ import annotations

import pytest

from engine.py.immune.mhc_loading import (
    MHCLoader,
    MHCPresentation,
    _canonical_allele,
)
from engine.py.molecular.peptidome import Peptide


def _wt_peptide(seq: str, cleav: float = 0.9) -> Peptide:
    return Peptide(
        sequence=seq,
        source_gene="TEST",
        length=len(seq),
        is_mutant=False,
        wild_type_sequence=seq,
        mutation_label=None,
        anchor_position_in_peptide=-1,
        parent_position_1based=1,
        cleavage_score=cleav,
    )


def _mut_peptide(seq: str, wt_seq: str, cleav: float = 0.9) -> Peptide:
    assert len(seq) == len(wt_seq)
    anchor = next(i for i in range(len(seq)) if seq[i] != wt_seq[i])
    return Peptide(
        sequence=seq,
        source_gene="TEST",
        length=len(seq),
        is_mutant=True,
        wild_type_sequence=wt_seq,
        mutation_label="X1Y",
        anchor_position_in_peptide=anchor,
        parent_position_1based=1,
        cleavage_score=cleav,
    )


# ---------------------------------------------------------------------------
# Allele canonicalization
# ---------------------------------------------------------------------------

def test_canonical_allele_adds_prefix():
    assert _canonical_allele("A*02:01") == "HLA-A*02:01"
    assert _canonical_allele("HLA-A*02:01") == "HLA-A*02:01"


# ---------------------------------------------------------------------------
# Binding classification
# ---------------------------------------------------------------------------

def test_a0201_canonical_anchor_peptide_binds():
    # HLA-A*02:01 wants L/M/V/I at position 2 and V/L/I at the C-terminus
    # (per HLA_BINDING_PROPERTIES). A peptide matching both anchors should
    # not be a non-binder.
    loader = MHCLoader()
    pep = _wt_peptide("YLAGGVGKV")  # L@2, V@9
    pres = loader.score_peptide(pep, ["HLA-A*02:01"])
    assert len(pres) == 1
    assert pres[0].binding_level in ("strong", "weak")
    assert pres[0].ic50_nm < 500.0
    assert pres[0].is_neoantigen is False


def test_a0201_bad_anchors_is_non_binder():
    loader = MHCLoader()
    # Charged P1, no preferred anchors -- the PWM should push this past
    # 500 nM.
    pep = _wt_peptide("RRRRRRRRR")
    pres = loader.score_peptide(pep, ["HLA-A*02:01"])
    assert pres[0].binding_level == "non-binder"
    assert pres[0].ic50_nm >= 500.0


def test_score_all_drops_non_binders_by_default():
    loader = MHCLoader()
    peptides = [_wt_peptide("YLAGGVGKV"), _wt_peptide("RRRRRRRRR")]
    kept = loader.score_all(peptides, ["HLA-A*02:01"])
    seqs = {p.peptide.sequence for p in kept}
    assert "RRRRRRRRR" not in seqs
    assert "YLAGGVGKV" in seqs


def test_score_all_include_non_binders_optin():
    loader = MHCLoader()
    peptides = [_wt_peptide("YLAGGVGKV"), _wt_peptide("RRRRRRRRR")]
    all_ = loader.score_all(peptides, ["HLA-A*02:01"], include_non_binders=True)
    seqs = {p.peptide.sequence for p in all_}
    assert "RRRRRRRRR" in seqs
    assert "YLAGGVGKV" in seqs


# ---------------------------------------------------------------------------
# best_per_peptide
# ---------------------------------------------------------------------------

def test_best_per_peptide_picks_strongest_allele():
    loader = MHCLoader()
    pep = _wt_peptide("YLAGGVGKV")
    # Score against three alleles; pick the strongest.
    pres = loader.score_peptide(pep, [
        "HLA-A*02:01", "HLA-B*07:02", "HLA-C*05:01",
    ])
    best = loader.best_per_peptide(pres)
    assert len(best) == 1
    chosen = next(iter(best.values()))
    assert chosen.ic50_nm == min(p.ic50_nm for p in pres)


# ---------------------------------------------------------------------------
# Presentation score
# ---------------------------------------------------------------------------

def test_presentation_score_is_multiplicative_in_cleavage():
    loader = MHCLoader()
    pep_high_cleav = _wt_peptide("YLAGGVGKV", cleav=0.9)
    pep_low_cleav = _wt_peptide("YLAGGVGKV", cleav=0.2)
    s_high = loader.score_peptide(pep_high_cleav, ["HLA-A*02:01"])[0]
    s_low = loader.score_peptide(pep_low_cleav, ["HLA-A*02:01"])[0]
    # Same peptide, same IC50, only cleavage differs -- presentation
    # score should scale by the cleavage ratio.
    ratio = s_high.presentation_score / s_low.presentation_score
    assert ratio == pytest.approx(0.9 / 0.2, rel=1e-6)


def test_neoantigen_flag_propagates():
    loader = MHCLoader()
    mut_pep = _mut_peptide("YLAGGVGKD", "YLAGGVGKV")
    pres = loader.score_peptide(mut_pep, ["HLA-A*02:01"])
    assert pres[0].is_neoantigen is True
    assert pres[0].peptide.mutation_label == "X1Y"
