"""
Regression tests for NucleicAcid._calculate_tm.

Reason for existence: prior implementation applied the Wallace rule
unconditionally and returned non-physical Tm values for sequences > 14
bases (e.g., 386 C for a 136-base KRAS fragment). This module asserts
that Tm stays within physical bounds across short, medium, and long
sequences, and that the short-oligo Wallace branch still matches the
textbook formula for <= 14 bases.

Run: pytest tests/test_tm_calculation.py -v
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest

from engine.py.molecular.nucleic_acids import DNA, RNA, NucleicAcidType


# -- short oligos: Wallace rule must match exactly ---------------------------


@pytest.mark.parametrize("seq,expected", [
    ("AAAA",              8.0),    # 4 AT * 2 = 8
    ("GGGG",             16.0),    # 4 GC * 4 = 16
    ("ATCG",             12.0),    # 2 AT * 2 + 2 GC * 4 = 12
    ("ATCGATCGATCGAT",   40.0),    # 14 bases, 8 AT + 6 GC = 16 + 24 = 40
])
def test_wallace_rule_short_oligos(seq, expected):
    dna = DNA(seq, gene_name="oligo")
    assert dna.melting_temp == pytest.approx(expected, rel=1e-6)


# -- long sequences: Marmur-Doty must stay within physical bounds ------------


def test_long_sequence_tm_is_physical():
    """A 136-base fragment must not melt above 100 C."""
    kras_fragment = (
        "ATGGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGC"
        "TAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTA"
    )
    dna = DNA(kras_fragment, gene_name="KRAS_fragment")
    assert 50.0 < dna.melting_temp < 100.0, (
        f"KRAS Tm out of physical range: {dna.melting_temp}")


def test_very_high_gc_long_sequence_finite():
    """100% GC, 500 bases. Marmur-Doty asymptote for 100% GC ~ 106 C
    (calibrated for typical salt buffers where high-GC DNA does melt slightly
    above 100 C). Assert finite + physically plausible upper bound; the prior
    Wallace-only bug returned 2000+ here."""
    seq = "G" * 500
    dna = DNA(seq, gene_name="poly_G")
    assert 90.0 < dna.melting_temp < 110.0


def test_very_low_gc_long_sequence_still_above_room_temp():
    """100% AT, 500 bases: still above 0 C."""
    seq = "A" * 500
    dna = DNA(seq, gene_name="poly_A")
    assert dna.melting_temp > 0.0


def test_gc_content_monotonic_with_tm():
    """Higher GC -> higher Tm at fixed length."""
    n = 100
    seq_low_gc  = "A" * 80 + "G" * 20
    seq_high_gc = "A" * 20 + "G" * 80
    tm_low  = DNA(seq_low_gc,  gene_name="low_gc").melting_temp
    tm_high = DNA(seq_high_gc, gene_name="high_gc").melting_temp
    assert tm_high > tm_low


def test_branch_boundary_at_14_bases():
    """The boundary between Wallace and Marmur-Doty is at length 14.
    The transition should not produce an absurd jump."""
    seq14 = "ATCG" * 3 + "AT"        # 14 bases, 6 GC + 8 AT
    seq15 = "ATCG" * 3 + "ATA"       # 15 bases, 6 GC + 9 AT
    tm14 = DNA(seq14, gene_name="b14").melting_temp
    tm15 = DNA(seq15, gene_name="b15").melting_temp
    # Wallace at 14: 4*6 + 2*8 = 40
    # Marmur-Doty at 15 with 6/15 GC: 64.9 + 41*0.4 - 500/15 = 48.0
    assert tm14 == pytest.approx(40.0, rel=1e-6)
    assert 40.0 < tm15 < 60.0


def test_empty_sequence_returns_zero():
    dna = DNA("", gene_name="empty")
    assert dna.melting_temp == 0.0


# -- RNA (with U) is handled the same way ------------------------------------


def test_rna_long_sequence_physical():
    rna_seq = "AUGGACUGAAUAUAAACUUGUGGUAGUUGGAGCUGGUGGCGUAGGCAAGAGUGCCUUGACGAU" \
              "ACAGCUAAUUCAGAAUCAUUUUGUGGACGAAUAUGAUCCAACAAUAGAGGAUUCCUACAGGAA" \
              "GCAAGUAGUA"
    rna = RNA(rna_seq, name="KRAS_mRNA_fragment", rna_type=NucleicAcidType.mRNA)
    assert 50.0 < rna.melting_temp < 100.0
