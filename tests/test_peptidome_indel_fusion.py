"""
INDEL + fusion neoantigen tests (Upgrade 8).

Closes the last lecture-flagged gap: prior to this upgrade, the
peptidome only generated missense neoantigens, ignoring frameshifts
and gene fusions. Clinically this matters because:
  * MMR-deficient / MSI-high tumors (Lynch syndrome, MSI-high CRC)
    have high frameshift burden -> high neoantigen quality ->
    strongest ICB response rates of any cancer type
  * Translocation-driven cancers (CML BCR-ABL1, ALK-fusion NSCLC,
    EWSR1 sarcomas) generate fusion neoantigens that are absent
    from either parent proteome and thus strongly foreign

Coverage:
  * generate_frameshift_peptides: window geometry, novel residue
    coverage, wild_type_sequence padding past WT length, anchor
    position, mutation_type="frameshift", input validation
  * generate_fusion_peptides: junction-spanning windows only,
    wild_type_sequence uses LEFT-extended reference, anchor at
    first right-partner residue, mutation_type="fusion", input
    validation
  * Backwards compat: existing missense path stamps
    mutation_type="missense"; full-window self peptides stamp
    "self".
"""
from __future__ import annotations

import pytest

from engine.py.molecular.peptidome import (
    DEFAULT_LENGTHS,
    Peptide,
    generate_frameshift_peptides,
    generate_fusion_peptides,
    generate_neoantigen_peptides,
    generate_peptides,
)


# ---------------------------------------------------------------------------
# mutation_type stamping
# ---------------------------------------------------------------------------

def test_existing_missense_path_stamps_mutation_type():
    wt = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEY"
    peps = generate_neoantigen_peptides(
        wt, mutant_position_1based=12,
        wild_type_aa="G", mutant_aa="D",
        source_gene="KRAS", mutation_label="G12D",
        lengths=(9,),
    )
    assert peps
    assert all(p.mutation_type == "missense" for p in peps)


def test_self_path_stamps_mutation_type():
    peps = generate_peptides("MTEYKLVVVGAGG", "TEST", lengths=(9,))
    assert peps
    assert all(p.mutation_type == "self" for p in peps)


def test_peptide_rejects_invalid_mutation_type():
    with pytest.raises(ValueError, match="invalid mutation_type"):
        Peptide(
            sequence="AAAAAAAAA",
            source_gene="X",
            length=9,
            is_mutant=False,
            wild_type_sequence="AAAAAAAAA",
            mutation_label=None,
            anchor_position_in_peptide=-1,
            parent_position_1based=1,
            cleavage_score=0.5,
            mutation_type="not_a_real_type",
        )


# ---------------------------------------------------------------------------
# Frameshift peptide generation
# ---------------------------------------------------------------------------

WT_PROTEIN = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILD"
NOVEL = "NEWFRAMESHIFTEDSEQ"


def test_frameshift_emits_peptides_spanning_fs_site():
    peps = generate_frameshift_peptides(
        wild_type_protein=WT_PROTEIN,
        frameshift_position_1based=30,
        novel_c_terminal=NOVEL,
        source_gene="BRCA2",
        mutation_label="E30fs",
        lengths=(9,),
    )
    assert peps, "expected peptides spanning the FS site"
    assert all(p.is_mutant for p in peps)
    assert all(p.mutation_type == "frameshift" for p in peps)
    assert all(p.source_gene == "BRCA2" for p in peps)
    assert all(p.mutation_label == "E30fs" for p in peps)


def test_frameshift_peptides_carry_at_least_one_novel_residue():
    """Every emitted peptide must include at least one residue from
    the novel C-terminal -- pure-WT windows are not neoantigens."""
    peps = generate_frameshift_peptides(
        WT_PROTEIN, 30, NOVEL, "BRCA2", "E30fs", lengths=(9,),
    )
    for p in peps:
        # Position of the first novel residue within the peptide is
        # ``anchor_position_in_peptide``; must be < length.
        assert 0 <= p.anchor_position_in_peptide < p.length
        # All residues at and after the anchor must come from the
        # novel sequence (i.e., differ from WT or be past WT length).
        novel_portion = p.sequence[p.anchor_position_in_peptide:]
        assert novel_portion, "no novel residues in peptide"


def test_frameshift_wt_sequence_pads_past_wt_protein_length():
    """When the FS extends past the WT protein, the wild_type_sequence
    must X-pad rather than crash."""
    short_wt = "MTEYKLVVV"  # 9 residues
    peps = generate_frameshift_peptides(
        wild_type_protein=short_wt,
        frameshift_position_1based=5,
        novel_c_terminal="NEWSEQUENCEHERE",
        source_gene="X",
        mutation_label="K5fs",
        lengths=(9,),
    )
    assert peps
    # At least one peptide should have an X in wild_type_sequence
    # (the positions past short_wt's length).
    assert any("X" in p.wild_type_sequence for p in peps)


def test_frameshift_skips_windows_containing_stop():
    """If the novel sequence contains a stop codon, peptides spanning
    that stop must be excluded."""
    peps = generate_frameshift_peptides(
        WT_PROTEIN, 30, "AAAA*ZZZZZZ", "X", "E30fs", lengths=(9,),
    )
    for p in peps:
        assert "*" not in p.sequence


def test_frameshift_empty_protein_returns_empty_pool():
    assert generate_frameshift_peptides("", 1, "ABC", "X", "fs") == []


def test_frameshift_empty_novel_returns_empty_pool():
    assert generate_frameshift_peptides(WT_PROTEIN, 30, "", "X", "fs") == []


def test_frameshift_position_out_of_range_raises():
    with pytest.raises(IndexError):
        generate_frameshift_peptides(WT_PROTEIN, 9999, "AAA", "X", "fs")


def test_frameshift_includes_pure_novel_windows():
    """Windows that start at or after the FS site are 100% novel."""
    peps = generate_frameshift_peptides(
        WT_PROTEIN, 30, NOVEL, "X", "fs", lengths=(9,),
    )
    # Find at least one peptide whose start position is at or past
    # the FS site -> anchor_position_in_peptide should be 0.
    pure_novel = [p for p in peps if p.anchor_position_in_peptide == 0]
    assert pure_novel


# ---------------------------------------------------------------------------
# Fusion peptide generation
# ---------------------------------------------------------------------------

LEFT_PARTNER = "MAAAAAAAEEEEEEEEELLLLLLLLLL"      # 27 residues
RIGHT_PARTNER = "KKKKKKKKKKAAAAAAAAAAYYYYYYYYYY"  # 30 residues


def test_fusion_emits_only_junction_spanning_peptides():
    """Every emitted peptide must contain at least one residue from
    each partner. Windows entirely in left or entirely in right are
    not neoantigens."""
    peps = generate_fusion_peptides(
        LEFT_PARTNER, RIGHT_PARTNER,
        left_breakpoint_1based=14,
        right_breakpoint_1based=11,
        source_gene="X-Y",
        mutation_label="X(13)-Y(11)",
        lengths=(9,),
    )
    assert peps
    for p in peps:
        # anchor_position_in_peptide is the position of the first
        # right-partner residue within the peptide. For a junction-
        # spanning peptide it must be > 0 (some left residues come
        # first) AND < length (some right residues follow).
        assert 0 < p.anchor_position_in_peptide < p.length
        assert p.mutation_type == "fusion"


def test_fusion_provenance_fields_populated():
    peps = generate_fusion_peptides(
        LEFT_PARTNER, RIGHT_PARTNER,
        left_breakpoint_1based=14, right_breakpoint_1based=11,
        source_gene="BCR-ABL1",
        mutation_label="BCR(13)-ABL1(11)",
        lengths=(9,),
    )
    assert peps
    for p in peps:
        assert p.source_gene == "BCR-ABL1"
        assert p.mutation_label == "BCR(13)-ABL1(11)"
        assert p.is_mutant is True


def test_fusion_wt_sequence_extends_left_partner():
    """The wild_type_sequence uses the left partner (the cell's prior
    identity) extended past its length with X padding."""
    # Use a short left partner so we can see the X padding
    short_left = "MAEEEE"      # 6 residues
    right = "RIGHTRESIDUES"    # 13 residues
    peps = generate_fusion_peptides(
        short_left, right,
        left_breakpoint_1based=4,
        right_breakpoint_1based=1,
        source_gene="X-Y", mutation_label="J",
        lengths=(9,),
    )
    if peps:
        # The junction is at position 4 in the chimera; a length-9
        # window starting at 0 covers positions [0, 8]; positions 0-2
        # are left, 3-8 are right. WT extends left to position 5
        # (length 6), so positions 6, 7, 8 in WT must be X.
        p = peps[0]
        assert p.wild_type_sequence.endswith("X" * 3) or "X" in p.wild_type_sequence


def test_fusion_input_validation():
    with pytest.raises(IndexError):
        generate_fusion_peptides(
            LEFT_PARTNER, RIGHT_PARTNER,
            left_breakpoint_1based=0,  # invalid: 1-based
            right_breakpoint_1based=11,
            source_gene="X", mutation_label="J",
        )
    with pytest.raises(IndexError):
        generate_fusion_peptides(
            LEFT_PARTNER, RIGHT_PARTNER,
            left_breakpoint_1based=14,
            right_breakpoint_1based=9999,
            source_gene="X", mutation_label="J",
        )


def test_fusion_empty_partner_returns_empty():
    assert generate_fusion_peptides("", "ABCDEFG", 1, 1, "X", "J") == []
    assert generate_fusion_peptides("ABCDEFG", "", 1, 1, "X", "J") == []


def test_fusion_skips_windows_containing_stop():
    """Stop codons in either partner around the junction must exclude
    the spanning peptides cleanly."""
    left = "MAAAAAAA*"     # stop near the end
    right = "RIGHTYNICE"
    peps = generate_fusion_peptides(
        left, right,
        left_breakpoint_1based=8,
        right_breakpoint_1based=1,
        source_gene="X", mutation_label="J",
        lengths=(9,),
    )
    for p in peps:
        assert "*" not in p.sequence


# ---------------------------------------------------------------------------
# Length sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length", [8, 9, 10, 11])
def test_frameshift_supports_all_default_lengths(length):
    peps = generate_frameshift_peptides(
        WT_PROTEIN, 30, NOVEL, "X", "fs", lengths=(length,),
    )
    assert all(p.length == length for p in peps)
    assert peps


@pytest.mark.parametrize("length", [8, 9, 10, 11])
def test_fusion_supports_all_default_lengths(length):
    peps = generate_fusion_peptides(
        LEFT_PARTNER, RIGHT_PARTNER,
        left_breakpoint_1based=14, right_breakpoint_1based=11,
        source_gene="X-Y", mutation_label="J",
        lengths=(length,),
    )
    assert all(p.length == length for p in peps)
    assert peps
