"""
Peptidome unit tests.

Patent-evidence checks for engine/py/molecular/peptidome.py:
  * full sliding window produces N - L + 1 peptides for each length L
  * mutation-anchored window covers exactly the windows that span the
    mutation site and substitutes the right residue at the anchor
  * stop codons in the window are excluded
  * misdeclared wild-type residue raises ValueError (no silent neoantigen
    fabrication against a wrong reference)
  * cleavage score is monotone with preference table
  * top_k_by_cleavage prefers high-score peptides and then shorter ones
"""
from __future__ import annotations

import pytest

from engine.py.molecular.peptidome import (
    DEFAULT_LENGTHS,
    Peptide,
    generate_neoantigen_peptides,
    generate_peptides,
    proteasomal_cleavage_score,
    top_k_by_cleavage,
)


SAMPLE_PROTEIN = "MTEYKLVVVGAGGVGKSALTIQLIQN"  # KRAS N-term (codons 1..26)


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def test_full_sliding_window_count_matches_formula():
    n = len(SAMPLE_PROTEIN)
    peps = generate_peptides(SAMPLE_PROTEIN, source_gene="KRAS")
    expected = sum(n - L + 1 for L in DEFAULT_LENGTHS if L <= n)
    assert len(peps) == expected
    assert all(p.is_self for p in peps)
    assert {p.source_gene for p in peps} == {"KRAS"}


def test_full_sliding_window_respects_length_filter():
    peps_9 = generate_peptides(SAMPLE_PROTEIN, "KRAS", lengths=(9,))
    assert all(p.length == 9 for p in peps_9)
    assert {p.sequence for p in peps_9} == {
        SAMPLE_PROTEIN[i:i + 9] for i in range(len(SAMPLE_PROTEIN) - 8)
    }


def test_empty_protein_returns_empty_pool():
    assert generate_peptides("", "KRAS") == []


def test_protein_shorter_than_min_length_returns_empty():
    assert generate_peptides("MTE", "KRAS", lengths=(8, 9)) == []


# ---------------------------------------------------------------------------
# Mutation-anchored window
# ---------------------------------------------------------------------------

def test_neoantigen_windows_cover_mutation_site():
    # KRAS G12D: codon 12 (1-indexed AA position 12) G -> D
    peps = generate_neoantigen_peptides(
        wild_type_protein=SAMPLE_PROTEIN,
        mutant_position_1based=12,
        wild_type_aa="G",
        mutant_aa="D",
        source_gene="KRAS",
        mutation_label="G12D",
    )
    assert peps, "no peptides emitted around mutation site"
    for p in peps:
        # Mutation must be inside the window
        assert 0 <= p.anchor_position_in_peptide < p.length
        # Mutant residue at the anchor, WT residue in wild_type_sequence
        assert p.sequence[p.anchor_position_in_peptide] == "D"
        assert p.wild_type_sequence[p.anchor_position_in_peptide] == "G"
        # Rest of the window is identical between mutant and WT
        for i in range(p.length):
            if i != p.anchor_position_in_peptide:
                assert p.sequence[i] == p.wild_type_sequence[i]
        # Provenance
        assert p.is_mutant is True
        assert p.mutation_label == "G12D"
        assert p.source_gene == "KRAS"


def test_neoantigen_window_count_matches_geometry():
    # For position p (0-indexed) within a length-N protein, the number of
    # length-L windows containing p equals
    #     min(p, L-1, N-L) - max(0, p-L+1) + 1
    # but more usefully: at least L windows when p is interior. We just
    # assert the count is consistent with the iter bounds.
    peps_9 = generate_neoantigen_peptides(
        SAMPLE_PROTEIN, 12, "G", "D", "KRAS", "G12D", lengths=(9,)
    )
    n = len(SAMPLE_PROTEIN)
    idx = 11
    expected = (min(idx, n - 9) - max(0, idx - 8) + 1)
    assert len(peps_9) == expected


def test_neoantigen_rejects_mismatched_wild_type_aa():
    with pytest.raises(ValueError, match="does not match"):
        generate_neoantigen_peptides(
            SAMPLE_PROTEIN, 12, "A", "D", "KRAS", "G12D"
        )


def test_neoantigen_position_out_of_range():
    with pytest.raises(IndexError):
        generate_neoantigen_peptides(
            SAMPLE_PROTEIN, 999, "G", "D", "KRAS", "G12D"
        )


def test_stop_codon_in_window_excluded():
    # Inject a stop just past the mutation site so some windows contain it
    protein_with_stop = SAMPLE_PROTEIN[:14] + "*" + SAMPLE_PROTEIN[15:]
    peps = generate_neoantigen_peptides(
        wild_type_protein=protein_with_stop,
        mutant_position_1based=12,
        wild_type_aa="G",
        mutant_aa="D",
        source_gene="KRAS",
        mutation_label="G12D",
    )
    for p in peps:
        assert "*" not in p.sequence
        assert "*" not in p.wild_type_sequence


# ---------------------------------------------------------------------------
# Cleavage scoring & ranking
# ---------------------------------------------------------------------------

def test_cleavage_score_preference_order():
    # Hydrophobic / aromatic preferred; basic disfavored; everything else
    # neutral. Exact values matter less than monotonicity.
    assert proteasomal_cleavage_score("L") > proteasomal_cleavage_score("A")
    assert proteasomal_cleavage_score("F") > proteasomal_cleavage_score("D")
    assert proteasomal_cleavage_score("R") < proteasomal_cleavage_score("A")
    assert proteasomal_cleavage_score("K") < proteasomal_cleavage_score("S")


def test_top_k_by_cleavage_orders_by_score_then_length():
    peptides = [
        Peptide("AAAAAAAAR", "X", 9, False, "AAAAAAAAR", None, -1, 1, 0.2),  # K
        Peptide("AAAAAAAAL", "X", 9, False, "AAAAAAAAL", None, -1, 2, 0.9),  # L
        Peptide("AAAAAAAAA", "X", 9, False, "AAAAAAAAA", None, -1, 3, 0.5),  # A
        Peptide("AAAAAAAL", "X", 8, False, "AAAAAAAL", None, -1, 4, 0.9),    # L, shorter
    ]
    ordered = top_k_by_cleavage(peptides, k=3)
    # Top score 0.9 -- shorter peptide first per tie-break
    assert ordered[0].length == 8
    assert ordered[1].length == 9
    assert ordered[0].cleavage_score == 0.9
    # Last of the top 3 is the neutral one (score 0.5)
    assert ordered[2].cleavage_score == 0.5


def test_top_k_zero_or_empty():
    assert top_k_by_cleavage([], k=5) == []
    assert top_k_by_cleavage([
        Peptide("AAAAAAAAL", "X", 9, False, "AAAAAAAAL", None, -1, 1, 0.9),
    ], k=0) == []


# ---------------------------------------------------------------------------
# Peptide invariants
# ---------------------------------------------------------------------------

def test_peptide_post_init_catches_length_mismatch():
    with pytest.raises(ValueError, match="disagrees"):
        Peptide("AAA", "X", 9, False, "AAA", None, -1, 1, 0.5)


def test_peptide_post_init_catches_unlabeled_mutant():
    with pytest.raises(ValueError, match="mutation_label"):
        Peptide("AAAAAAAAL", "X", 9, True, "AAAAAAAAG", None, 8, 1, 0.9)


def test_peptide_post_init_catches_inconsistent_self():
    with pytest.raises(ValueError, match="wild_type_sequence"):
        Peptide("AAAAAAAAD", "X", 9, False, "AAAAAAAAG", None, -1, 1, 0.5)
