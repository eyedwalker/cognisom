"""Structures carry their confidence, and a missing one means missing.

Two things are easy to get wrong here, and both have already happened in this
repo this week.

**A transient failure recorded as an absence.** The first bulk run over 133
accessions reported 30 as having no AlphaFold model -- among them BCL-2,
fibronectin and cytochrome c, all of which had been fetched successfully
minutes earlier. Unthrottled requests were failing, and ``fetch_alphafold``
returned ``None`` for both "the API says there is no model" and "the network
hiccuped". That conflation writes a permanent gap into the library on the
strength of one bad response, and nothing ever retries it. The two outcomes are
now distinct: a clean miss returns ``None``, a transient failure raises.

**A prediction rendered as a measurement.** Every AlphaFold model covers 100%
of its canonical sequence, so coverage is a useless quality signal here --
confidence is the signal. Measured across this library: the fraction of
residues above pLDDT 70 runs 11% to 100%, median 79%, and nineteen proteins are
majority low-confidence. The Androgen Receptor, which the whole AR signalling
model rests on, is 38% confident with 58% of residues below pLDDT 50. Drawing
that as solidly as a folded domain is the misleading act, so confidence travels
with the coordinates and the renderer acts on it.
"""
from __future__ import annotations

import numpy as np
import pytest

from cognisom.library.structure_cache import (
    PLDDT_CONFIDENT, PLDDT_VERY_LOW, Structure, StructureFetchError,
    confidence_bands, parse_mmcif_atoms,
)
from cognisom.visualization.structure_render import (
    BAND_RADIUS, backbone_tube, band_of, confidence_summary, smooth_trace,
)

MINIMAL_CIF = """\
data_test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM 1 N   MET 1 0.000 0.000 0.000 95.00
ATOM 2 CA  MET 1 1.000 0.000 0.000 95.00
ATOM 3 C   MET 1 2.000 0.000 0.000 95.00
ATOM 4 N   ALA 2 3.000 0.000 0.000 40.00
ATOM 5 CA  ALA 2 4.000 1.000 0.000 40.00
ATOM 6 C   ALA 2 5.000 0.000 0.000 40.00
ATOM 7 CA  GLY 3 7.000 2.000 1.000 72.00
#
"""


def _structure(plddt, n_atoms=None):
    xyz = np.stack([np.arange(len(plddt), dtype=float) * 3.8,
                    np.zeros(len(plddt)), np.zeros(len(plddt))], axis=1)
    return Structure(accession="P00000", source="alphafold", identifier="AF-P00000-F1",
                     ca_xyz=xyz, ca_plddt=np.asarray(plddt, dtype=float),
                     n_atoms=n_atoms or len(plddt) * 8, n_residues=len(plddt),
                     sequence_length=len(plddt))


# ── the mmCIF reader ─────────────────────────────────────────────────────

def test_the_atom_site_loop_parses_without_a_structure_library():
    cols, rows = parse_mmcif_atoms(MINIMAL_CIF)

    assert "label_atom_id" in cols and "Cartn_x" in cols and "B_iso_or_equiv" in cols
    assert len(rows) == 7
    ix = {c: n for n, c in enumerate(cols)}
    cas = [r for r in rows if r[ix["label_atom_id"]] == "CA"]
    assert len(cas) == 3, "three alpha carbons in the fixture"


def test_a_file_with_no_atom_site_loop_returns_nothing_rather_than_guessing():
    cols, rows = parse_mmcif_atoms("data_x\n#\nloop_\n_cell.length_a\n10.0\n#\n")

    assert cols == [] and rows == []


# ── the absence/failure distinction ──────────────────────────────────────

def test_a_transient_failure_is_a_different_outcome_from_a_missing_model():
    """The regression: 30 proteins were reported absent when the network hiccuped."""
    assert issubclass(StructureFetchError, RuntimeError)


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def test_a_clean_404_is_an_absence_and_does_not_raise(monkeypatch, tmp_path):
    from cognisom.library import structure_cache as sc

    class Session:
        headers = {}

        def get(self, url, timeout=0):
            return _Resp(404)

    assert sc.fetch_alphafold("P99999", cache_dir=str(tmp_path),
                              session=Session()) is None


def test_an_empty_payload_is_an_absence_not_an_error(monkeypatch, tmp_path):
    """AlphaFold answers 200 with [] for accessions it has no model for."""
    from cognisom.library import structure_cache as sc

    class Session:
        headers = {}

        def get(self, url, timeout=0):
            return _Resp(200, payload=[])

    assert sc.fetch_alphafold("P99999", cache_dir=str(tmp_path),
                              session=Session()) is None


def test_repeated_server_errors_raise_instead_of_returning_none(monkeypatch, tmp_path):
    """A 500 is not evidence that the structure does not exist."""
    from cognisom.library import structure_cache as sc
    monkeypatch.setattr("time.sleep", lambda *_: None)

    class Session:
        headers = {}
        calls = 0

        def get(self, url, timeout=0):
            Session.calls += 1
            return _Resp(500)

    with pytest.raises(StructureFetchError):
        sc.fetch_alphafold("P10275", cache_dir=str(tmp_path), session=Session())
    assert Session.calls > 1, "it must retry before giving up"


# ── confidence ───────────────────────────────────────────────────────────

def test_confidence_bands_split_at_alphafolds_own_thresholds():
    bands = confidence_bands(_structure([95.0, 80.0, 60.0, 30.0]))

    assert bands == {"very_high": 1, "confident": 1, "low": 1, "very_low": 1}


@pytest.mark.parametrize("plddt,expected", [
    (95.0, "very_high"), (90.0, "very_high"), (89.9, "confident"),
    (70.0, "confident"), (69.9, "low"), (50.0, "low"), (49.9, "very_low"),
])
def test_band_boundaries(plddt, expected):
    assert band_of(plddt) == expected


def test_confident_fraction_uses_the_documented_threshold():
    s = _structure([95.0, 95.0, 60.0, 30.0])

    assert s.confident_fraction == pytest.approx(0.5)
    assert PLDDT_CONFIDENT == 70.0 and PLDDT_VERY_LOW == 50.0


def test_an_experimental_structure_has_no_confidence_to_report():
    s = Structure(accession="P10275", source="pdb", identifier="2AM9",
                  ca_xyz=np.zeros((10, 3)), ca_plddt=None, n_atoms=80,
                  n_residues=10, sequence_length=920)

    assert s.confident_fraction is None
    assert not s.is_prediction
    assert s.coverage == pytest.approx(10 / 920)


# ── geometry ─────────────────────────────────────────────────────────────

def test_the_backbone_is_smoothed_rather_than_drawn_as_a_zig_zag():
    xyz = np.array([[0, 0, 0], [3.8, 0, 0], [7.6, 1.0, 0], [11.4, 0, 0]], dtype=float)
    pts, idx = smooth_trace(xyz, subdivisions=4)

    assert len(pts) > len(xyz) * 3
    assert idx[0] == 0.0 and idx[-1] == pytest.approx(len(xyz) - 1)


def test_low_confidence_stretches_are_drawn_thinner_not_removed():
    """A gap reads as "nothing here"; thinning reads as "no fixed shape here"."""
    plddt = [95.0] * 20 + [20.0] * 20
    s = _structure(plddt)

    thinned = backbone_tube(s.ca_xyz, s.ca_plddt, confidence_radius=True)
    uniform = backbone_tube(s.ca_xyz, s.ca_plddt, confidence_radius=False)

    assert thinned.dropped_residues == 0, "nothing should be dropped by default"
    assert len(thinned.vertices) == len(uniform.vertices)
    assert not np.allclose(thinned.vertices, uniform.vertices), (
        "confidence must change the geometry, not just the colour")
    assert BAND_RADIUS["very_low"] < BAND_RADIUS["very_high"]


def test_dropping_low_confidence_is_possible_but_reports_the_cost():
    """For AR this would silently delete 58% of the protein, so it is opt-in."""
    plddt = [95.0] * 10 + [20.0] * 30
    s = _structure(plddt)

    kept = backbone_tube(s.ca_xyz, s.ca_plddt, drop_below=50.0)

    assert kept.dropped_residues == 30
    assert kept.n_residues == 10


def test_every_confidence_band_gets_its_own_colour():
    s = _structure([95.0, 80.0, 60.0, 30.0] * 5)
    g = backbone_tube(s.ca_xyz, s.ca_plddt)

    distinct = {tuple(np.round(c, 4)) for c in g.colours}
    assert len(distinct) == 4, f"expected four bands, got {len(distinct)}"


def test_an_experimental_structure_renders_uniformly():
    """Without pLDDT there is nothing to vary, and varying anyway would invent a signal."""
    xyz = np.stack([np.arange(20, dtype=float) * 3.8, np.zeros(20), np.zeros(20)], axis=1)
    g = backbone_tube(xyz, plddt=None)

    assert len({tuple(np.round(c, 4)) for c in g.colours}) == 1


def test_a_single_residue_cannot_make_a_tube_and_says_so_quietly():
    g = backbone_tube(np.array([[0.0, 0.0, 0.0]]), plddt=[90.0])

    assert g.n_triangles == 0
    assert len(g.vertices) == 0


def test_the_summary_reports_the_median_and_the_confident_fraction():
    out = confidence_summary([95.0, 95.0, 30.0, 30.0])

    assert out["median"] == pytest.approx(62.5)
    assert out["confident_fraction"] == pytest.approx(0.5)


# ── relative placement ───────────────────────────────────────────────────
#
# pLDDT says how sure the model is of a residue's own position. It says nothing
# about whether two confident regions are correctly placed relative to each
# other, and that second uncertainty is often the larger one. Measured on the
# live AlphaFold PAE matrices:
#
#     AR    three confident segments, pairwise mean PAE 26.5-28.9 A
#     TP53  DNA-binding halves 3.45 A, DBD to tetramerisation 20.1 A
#     TLR4  ectodomain halves 7.44 A, ectodomain to TIR 25-28 A
#
# So AR's domains are each confidently folded while their arrangement is not
# claimed at all. Rendering them in one coordinate frame asserts a quaternary
# structure the model declines to assert -- the same overclaim as drawing a
# disordered tail solid, one level up.

def test_segments_need_a_run_not_a_lucky_residue():
    from cognisom.library.structure_cache import confident_segments

    plddt = [95.0] * 30 + [20.0] * 10 + [95.0] * 5 + [20.0] * 10
    segs = confident_segments(plddt, min_length=20)

    assert segs == [(1, 30)], "a 5-residue blip must not become a segment"


def test_a_trailing_confident_run_is_not_lost():
    from cognisom.library.structure_cache import confident_segments

    assert confident_segments([20.0] * 10 + [95.0] * 25) == [(11, 35)]


def test_segments_placed_confidently_relative_to_each_other_are_one_group():
    from cognisom.library.structure_cache import rigid_groups

    pae = np.full((60, 60), 2.0)
    groups = rigid_groups([(1, 25), (31, 60)], pae)

    assert groups == [[(1, 25), (31, 60)]]


def test_segments_the_model_will_not_place_stay_apart():
    """AR's case: each domain confident, their arrangement unclaimed."""
    from cognisom.library.structure_cache import rigid_groups

    pae = np.full((60, 60), 27.0)
    groups = rigid_groups([(1, 25), (31, 60)], pae)

    assert len(groups) == 2, (
        "segments at 27 A mean PAE must not share a coordinate frame")


def test_grouping_is_transitive():
    """A rigid with B and B rigid with C puts all three in one frame."""
    from cognisom.library.structure_cache import rigid_groups

    pae = np.full((90, 90), 30.0)
    pae[0:25, 30:55] = pae[30:55, 0:25] = 2.0      # A-B close
    pae[30:55, 60:90] = pae[60:90, 30:55] = 2.0    # B-C close
    groups = rigid_groups([(1, 25), (31, 55), (61, 90)], pae)

    assert len(groups) == 1 and len(groups[0]) == 3


def test_the_threshold_is_documented_as_a_choice_not_a_fact():
    """5 A splits TLR4's own solenoid; the default is 8 A for that reason."""
    from cognisom.library.structure_cache import RIGID_PAE_MAX, rigid_groups

    assert RIGID_PAE_MAX == 8.0
    pae = np.full((60, 60), 7.44)      # TLR4's measured ectodomain value
    assert len(rigid_groups([(1, 25), (31, 60)], pae)) == 1, (
        "at the default, TLR4's continuous ectodomain stays in one group")
    assert len(rigid_groups([(1, 25), (31, 60)], pae, max_pae=5.0)) == 2, (
        "at the conventional 5 A it splits, which is why 5 A is not the default")


def test_no_segments_means_no_groups_rather_than_an_error():
    from cognisom.library.structure_cache import rigid_groups

    assert rigid_groups([], np.zeros((10, 10))) == []
