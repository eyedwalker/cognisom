"""Two ways the neoantigen path mis-read a real callset's protein changes.

Both were found by pointing the pipeline at a real prostate patient
(SU2C ``SC_9184_T``), and neither raised an error. They produced
confident wrong peptides instead, which is the harder failure to notice.

**1. A frameshift was cut into peptides as if it were a substitution.**
The predictor parsed protein changes with
``re.match(r"([A-Z])(\\d+)([A-Z])", clean_mut)`` -- not anchored at the
end -- so every ``fs``, ``del``, ``ins`` and ``dup`` suffix was silently
discarded. ``A1679Pfs*38`` became ``A1679P``. On this patient that
fabricated peptides from three frameshifts (KMT2C ``A1679Pfs*38``, APC
``T1556Nfs*3``, CASZ1 ``T1125Kfs*144``), and two of them scored among
his six strongest binders -- 34.2 nM and 36.6 nM for 9-mers the tumour
does not make, because a frameshift at residue 1679 replaces everything
downstream rather than one residue.

A frameshift IS a real and often rich neoantigen source: the novel
reading frame is entirely foreign. It needs the translated downstream
sequence, which this repository does not derive, so the honest output is
a refusal counted as ``not_missense``.

Meanwhile ``patient_engine.parse_missense`` anchored its pattern and
refused the same three correctly. So the two halves of one demo
disagreed about what the tumour presents. There is now ONE parser, in
``genomics.protein_change``, and the dashboard imports it.

**2. The AR numbering offset could only be right for one convention.**
``NUMBERING_OFFSETS["AR"] = 1`` exists because AR calls in the
literature use a 919-aa reference while the canonical sequence is 920.
But a call arrives in EITHER convention: the literature hotspot is
``F876L`` and this cohort's MAF reports the same residue as ``F877L``. A
single fixed offset necessarily refuses one of them, and it refused the
patient's own AR variant -- the one mutation in this demo the simulator
has a mechanism for.

So ``resolve_for_wildtype`` tries the curated offset and then the
canonical numbering, curated first so nothing that resolved before
moves. This is NOT the "search for an offset that fits" that
NUMBERING_OFFSETS warns against: two documented conventions for one
gene, discriminated by the wild-type residue.

The matching change was deliberately NOT made in
``patient_engine.engine_inputs``, and a pre-existing test caught the
attempt. That function emits an identity LABEL other code string-matches,
so accepting a historically numbered call would mean labelling it 877
while the residue is 878, or relabelling it and breaking every match.
The predictor emits a PEPTIDE, where resolving to the right residue is
the entire job. The two differ on purpose.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from cognisom.genomics.protein_change import parse_missense


# ── one parser, anchored ──────────────────────────────────────────


@pytest.mark.parametrize("change", [
    "p.A1679Pfs*38",     # KMT2C, this patient -- was read as A1679P
    "p.T1556Nfs*3",      # APC
    "p.T1125Kfs*144",    # CASZ1
    "p.E1143fs",
    "p.R248Pdup",
    "p.E103_P105del",
    "p.H640_H641del",
    "p.X50_splice",
    "p.Y429*",           # nonsense
    "p.R248R",           # synonymous
    "p.A0T",             # position 0 is not a residue number
    "",
    None,
])
def test_nothing_but_a_single_substitution_parses_as_missense(change):
    assert parse_missense(change) is None, (
        f"{change!r} parsed as a missense substitution. If this is a "
        "frameshift or indel the peptides built from it do not exist.")


@pytest.mark.parametrize("change,expected", [
    ("p.F877L", ("F", 877, "L")),
    ("p.N218T", ("N", 218, "T")),
    ("p.Asn218Thr", ("N", 218, "T")),
    ("F877L", ("F", 877, "L")),
    ("p.(N218T)", ("N", 218, "T")),
])
def test_a_real_substitution_still_parses(change, expected):
    assert parse_missense(change) == expected


def test_the_pattern_is_anchored_at_both_ends():
    """The ``$`` is the whole defect. Guard it directly."""
    from cognisom.genomics.protein_change import PROTEIN_CHANGE
    assert PROTEIN_CHANGE.pattern.startswith("^")
    assert PROTEIN_CHANGE.pattern.endswith("$"), (
        "without the trailing anchor every fs/del/ins/dup suffix is "
        "discarded and the variant is mistaken for a substitution")


def test_the_dashboard_and_the_predictor_share_one_parser():
    """Not two copies that agree today."""
    from cognisom.dashboard import patient_engine
    from cognisom.genomics import protein_change
    assert patient_engine.parse_missense is protein_change.parse_missense
    predictor = (
        __import__("cognisom.genomics.neoantigen_predictor",
                   fromlist=["x"]))
    assert predictor.PROTEIN_CHANGE is protein_change.PROTEIN_CHANGE


# ── AR numbering: two conventions, one residue ────────────────────


@pytest.fixture(scope="module")
def ar():
    from cognisom.genomics.gene_protein_mapper import GeneProteinMapper
    with contextlib.redirect_stdout(io.StringIO()):
        mapper = GeneProteinMapper()
    protein = mapper.get_protein("AR")
    assert protein is not None and len(protein.sequence) == 920
    return protein


def test_the_bundled_ar_sequence_is_the_canonical_920_mer(ar):
    """Pinned beside the claim: Q876, F877, T878 in canonical numbering.

    Every statement below depends on this, so if the bundled sequence
    changes these tests must be RE-MEASURED against the new one, not
    re-pinned to keep passing.
    """
    assert ar.sequence[875] == "Q"
    assert ar.sequence[876] == "F"
    assert ar.sequence[877] == "T"


def test_both_ar_numbering_conventions_reach_the_same_residue(ar):
    """F876L and F877L are the SAME substitution, named twice.

    That is the fact which makes trying both conventions safe rather
    than permissive: they do not compete for different residues, they
    agree on one.
    """
    historical = ar.resolve_for_wildtype(876, "F")
    canonical = ar.resolve_for_wildtype(877, "F")
    assert historical is not None and canonical is not None
    assert historical[0] == canonical[0] == 877
    assert historical[1] == 1 and canonical[1] == 0


def test_the_curated_offset_wins_so_nothing_that_resolved_before_moves(ar):
    """Two of the eight calibration mutations match under BOTH offsets.

    L701 and M895 carry the same residue either way, so for them the
    choice is immaterial to the wild-type check but NOT to the peptide
    window. Trying the curated offset first keeps the window they have
    always had.
    """
    for wt, pos in (("L", 701), ("M", 895)):
        at_zero = ar.sequence[pos - 1]
        at_curated = ar.sequence[pos]
        assert at_zero == at_curated == wt, (wt, pos)
        resolved = ar.resolve_for_wildtype(pos, wt)
        assert resolved == (pos + 1, 1), (
            "an ambiguous position must keep the curated offset")


@pytest.mark.parametrize("wt,pos", [
    ("V", 715), ("W", 741), ("H", 874), ("F", 876), ("T", 877), ("R", 629),
])
def test_the_historically_numbered_hotspots_still_resolve_at_plus_one(ar, wt, pos):
    """The six calibration mutations that actually discriminate."""
    assert ar.sequence[pos - 1] != wt, (
        f"{wt}{pos} is not discriminating any more; re-measure the "
        "offset rather than re-pinning this list")
    assert ar.resolve_for_wildtype(pos, wt) == (pos + 1, 1)


@pytest.mark.parametrize("wt,pos", [("F", 877), ("T", 878)])
def test_the_canonically_numbered_hotspots_now_resolve_at_zero(ar, wt, pos):
    """These are the ones a fixed +1 offset refused.

    AR F877L is how the SU2C cohort reports this patient's receptor
    mutation, and it is the mutation the whole prostate demo turns on.
    """
    assert ar.resolve_for_wildtype(pos, wt) == (pos, 0)


def test_a_residue_that_matches_neither_convention_is_still_refused(ar):
    """Trying two conventions must not become trying anything."""
    assert ar.resolve_for_wildtype(877, "W") is None
    assert ar.resolve_for_wildtype(877, "K") is None
    assert ar.resolve_for_wildtype(99999, "F") is None
    assert ar.resolve_for_wildtype(0, "M") is None
    assert ar.resolve_for_wildtype(877, "") is None


def test_a_gene_with_no_curated_offset_only_accepts_canonical_numbering():
    from cognisom.genomics.gene_protein_mapper import GeneProteinMapper
    with contextlib.redirect_stdout(io.StringIO()):
        mapper = GeneProteinMapper()
    tp53 = mapper.get_protein("TP53")
    assert tp53 is not None and tp53.numbering_offset == 0
    assert tp53.resolve_for_wildtype(248, "R") == (248, 0)
    # and no second chance it did not earn. 250 is P, and R250 is not a
    # residue of this protein under any convention -- 249 would NOT do
    # as the negative case, because R249 is itself a real TP53 hotspot
    # (my first version of this test used it and passed for the wrong
    # reason).
    assert tp53.sequence[249] == "P"
    assert tp53.resolve_for_wildtype(250, "R") is None


def test_patient_engine_still_refuses_a_historically_numbered_ar_call():
    """Deliberate, and the opposite of the predictor. See the docstring.

    ``engine_inputs`` emits "AR_T877A", a label other code string-
    matches. Renumbering it would be fabrication either way round, so it
    is reported as a skip. I changed this to share the predictor's
    resolution and ``test_the_ar_hotspot_is_reported_not_silently_
    renumbered`` caught it.
    """
    from cognisom.dashboard.patient_engine import engine_inputs

    class _V:
        gene = "AR"
        protein_change = "p.T877A"
        consequence = "missense_variant"

    class _P:
        patient_id = "T"
        coding_variants = [_V()]
        cancer_driver_mutations = []
        hla_alleles = ["HLA-A*02:01"]
        hla_typing_method = "population-default"
        has_matched_normal = True

    inputs = engine_inputs(_P())
    assert inputs.mutations == []
    assert inputs.skipped and inputs.skipped[0][0] == "AR"
    # while the canonically numbered call, which is what the demo
    # callset actually carries, crosses here unchanged
    _V.protein_change = "p.F877L"
    assert [m.label for m in engine_inputs(_P()).mutations] == ["AR_F877L"]


# ── and the real callset, end to end ──────────────────────────────


@pytest.fixture(scope="module")
def prostate_profile():
    import os
    from cognisom.genomics.demo_patients import load_demo_patient
    with contextlib.redirect_stdout(io.StringIO()):
        return load_demo_patient("su2c_prostate")


@pytest.mark.slow
def test_no_peptide_comes_from_a_frameshift_or_nonsense_call(prostate_profile):
    """The defect, stated as a property of the output.

    Pinning "10 not_missense" would break the moment the callset
    changed. What must hold for ANY callset is that no peptide is built
    from a protein change that is not a substitution.
    """
    offenders = [
        (n.source_gene, n.mutation, n.peptide)
        for n in prostate_profile.predicted_neoantigens
        if parse_missense(n.mutation) is None
    ]
    assert offenders == [], (
        f"{len(offenders)} peptides were built from protein changes that "
        f"are not substitutions: {offenders[:5]}")


@pytest.mark.slow
def test_the_three_frameshifts_are_counted_not_dropped(prostate_profile):
    """Refused, and visibly so -- an honest null, not a silent one."""
    diagnostics = prostate_profile.neoantigen_diagnostics
    assert diagnostics["skipped_not_missense"] >= 3, diagnostics
    changes = {v.protein_change for v in prostate_profile.coding_variants}
    for frameshift in ("p.A1679Pfs*38", "p.T1556Nfs*3", "p.T1125Kfs*144"):
        assert frameshift in changes, (
            f"{frameshift} left the callset; this test no longer covers "
            "what it was written for")


@pytest.mark.slow
def test_the_resistance_mutation_yields_its_peptide(prostate_profile):
    """What the numbering fix bought, on the real patient.

    Under a fixed +1 offset AR F877L was refused as a wild-type
    mismatch and this patient's only actionable mutation contributed
    nothing. The substitution must land on the F, which is what makes
    the mutant peptide differ from the wild-type one at that residue.
    """
    ar = [n for n in prostate_profile.predicted_neoantigens
          if n.source_gene == "AR"]
    assert ar, "AR F877L produced no peptides"
    assert all(n.mutation == "F877L" for n in ar)
    for neo in ar:
        assert len(neo.peptide) == len(neo.wild_type_peptide)
        differing = [(w, m) for w, m in zip(neo.wild_type_peptide, neo.peptide)
                     if w != m]
        assert differing == [("F", "L")], (
            f"{neo.wild_type_peptide} -> {neo.peptide} differs at "
            f"{differing}, not at a single F->L; the window is centred "
            "on the wrong residue")
