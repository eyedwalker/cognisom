"""A frameshift must not be converted into a fabricated missense substitution.

VEP writes `<ref>/X` in its Amino_acids field for a frameshift, where `X` marks
an unknown downstream product rather than a residue. Both sides are a single
character, so the annotator's `amino_acids` fallback used to emit `p.R248X`,
and the predictor's `([A-Z])(\\d+)([A-Z])` regex accepted it as a missense to
"X" -- producing peptides containing a literal X, some of them flagged for
vaccine inclusion, while `skipped_not_missense` recorded zero.

These tests pin both halves: the annotator must refuse to invent the
substitution, and the predictor must refuse to consume one if it arrives from
some other annotator.
"""
import pytest

from cognisom.genomics.vep_annotator import _protein_change
from cognisom.genomics.neoantigen_predictor import _CANONICAL_AA


class TestAnnotatorRefusesNonResidues:
    def test_frameshift_amino_acids_field_yields_no_substitution(self):
        # The HGVS string is a frameshift, which the three-letter parser
        # cannot read; the fallback must not rescue it as a missense.
        consequence = {
            "hgvsp": "ENSP00000269305:p.Arg248SerfsTer5",
            "amino_acids": "R/X",
            "protein_end": 248,
        }
        assert _protein_change(consequence) is None

    def test_frameshift_without_hgvsp_yields_no_substitution(self):
        assert _protein_change({"amino_acids": "R/X", "protein_end": 248}) is None

    def test_stop_gained_yields_no_substitution(self):
        assert _protein_change({"amino_acids": "Q/*", "protein_end": 248}) is None

    @pytest.mark.parametrize("alt", ["B", "Z", "J", "U", "O"])
    def test_ambiguity_and_nonstandard_codes_are_refused(self, alt):
        consequence = {"amino_acids": f"R/{alt}", "protein_end": 248}
        assert _protein_change(consequence) is None

    def test_real_missense_still_resolves_via_hgvsp(self):
        consequence = {
            "hgvsp": "ENSP00000269305:p.Arg248Trp",
            "amino_acids": "R/W",
            "protein_end": 248,
        }
        assert _protein_change(consequence) == "p.R248W"

    def test_real_missense_still_resolves_via_fallback(self):
        # No hgvsp, so the amino_acids path has to carry it.
        assert _protein_change({"amino_acids": "R/W", "protein_end": 248}) == "p.R248W"


class TestCanonicalResidueSet:
    def test_has_exactly_the_twenty_residues(self):
        assert len(_CANONICAL_AA) == 20

    @pytest.mark.parametrize("bad", ["X", "*", "B", "Z", "J", "U", "O"])
    def test_excludes_non_residues(self, bad):
        assert bad not in _CANONICAL_AA

    @pytest.mark.parametrize("good", list("ACDEFGHIKLMNPQRSTVWY"))
    def test_includes_every_residue(self, good):
        assert good in _CANONICAL_AA
