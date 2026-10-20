"""The library's molecules as geometry, and whether the geometry is right.

The entity library carries 24 formula-checked small-molecule structures.
In the USD track every one of them was a single sphere -- ``glucose`` and
``oxygen`` both have ``geometry_type="sphere"`` in ``prototype_library``,
so a C6H12O6 ring and an O=O diatomic rendered identically.

Two things here are easy to get wrong and invisible once wrong:

  * **Bond orientation.** USD's implicit cylinder lies along its axis
    attribute, so a bond is a rotation of +Z onto the vector between two
    atoms. A wrong quaternion still writes a valid stage full of
    cylinders at plausible positions pointing the wrong way, and the
    file opens without complaint. So the test recomputes each
    cylinder's own end from the transform that was authored and checks
    it lands on an atom.
  * **Provenance.** The topology is the library's and was checked
    against formula and weight; the coordinates are computed, one
    conformer, not a structure. A stage that does not say so invites
    someone to read a docetaxel conformer as its bound pose.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("rdkit")
pytest.importorskip("pxr")

from pxr import Gf, Usd, UsdGeom, UsdShade          # noqa: E402

from cognisom.visualization import molecular_usd as mu   # noqa: E402

# Small, mid and large, so the whole range is exercised in ~1 s rather
# than embedding all 24.
OXYGEN = "O=O"
GLUCOSE = "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O"
ENZALUTAMIDE = ("CC1(C(=O)N(C(=S)N1C2=CC(=C(C=C2)C(=O)NC)F)C3=CC(=C(C=C3)C#N)"
                "C(F)(F)F)C")


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    path = tmp_path_factory.mktemp("usd") / "molecules.usda"
    s = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(s, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(s, mu.METERS_PER_UNIT)
    UsdGeom.Scope.Define(s, "/Materials")
    UsdGeom.Xform.Define(s, "/Molecules")
    geoms = {
        name: mu.author_molecule(s, "/Molecules", name, smiles, formula=formula)
        for name, smiles, formula in (
            ("Oxygen", OXYGEN, "O2"),
            ("Glucose", GLUCOSE, "C6H12O6"),
            ("Enzalutamide", ENZALUTAMIDE, "C21H16F4N4O2S"),
        )
    }
    return s, geoms


def _atom_centres(stage, molecule):
    out = {}
    for prim in stage.GetPrimAtPath(f"/Molecules/{molecule}/Atoms").GetChildren():
        idx = prim.GetCustomDataByKey("atomIndex")
        t = UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Get()
        out[idx] = np.array([t[0], t[1], t[2]])
    return out


def _cylinder_ends(prim):
    """The two ends of a cylinder, from the transform actually authored."""
    cyl = UsdGeom.Cylinder(prim)
    height = cyl.GetHeightAttr().Get()
    translate, orient = UsdGeom.Xformable(prim).GetOrderedXformOps()[:2]
    centre = np.array(list(translate.Get()))
    q = orient.Get()
    rot = Gf.Rotation(Gf.Quatd(q.GetReal(), Gf.Vec3d(*q.GetImaginary())))
    tip = np.array(rot.TransformDir(Gf.Vec3d(0.0, 0.0, height / 2.0)))
    return centre - tip, centre + tip


# ── the thing this replaces ──────────────────────────────────────────

def test_glucose_and_oxygen_are_no_longer_the_same_object(stage):
    """Both were geometry_type="sphere": one prim each, indistinguishable."""
    _, geoms = stage
    assert geoms["Oxygen"].atoms == 2 and geoms["Oxygen"].bonds == 1
    assert geoms["Glucose"].atoms == 24 and geoms["Glucose"].bonds == 24
    assert geoms["Glucose"].prims > geoms["Oxygen"].prims * 10
    assert geoms["Glucose"].elements == {"C": 6, "H": 12, "O": 6}
    assert geoms["Oxygen"].elements == {"O": 2}


def test_the_prim_count_is_what_was_actually_authored(stage):
    """atoms + 2 * bonds, because each bond is two half-coloured sticks."""
    s, geoms = stage
    for name, geom in geoms.items():
        atoms = len(s.GetPrimAtPath(f"/Molecules/{name}/Atoms").GetChildren())
        bonds = len(s.GetPrimAtPath(f"/Molecules/{name}/Bonds").GetChildren())
        assert atoms == geom.atoms
        assert bonds == 2 * geom.bonds
        assert geom.prims == atoms + bonds


def test_the_floor_in_structures_agrees_with_what_is_emitted(stage):
    """``conformer_prim_count`` is the single-stick floor; the exporter
    emits one cylinder more per bond. The two must not drift apart."""
    from cognisom.library.structures import conformer_prim_count
    _, geoms = stage
    atoms, bonds = conformer_prim_count(ENZALUTAMIDE)
    geom = geoms["Enzalutamide"]
    assert (atoms, bonds) == (geom.atoms, geom.bonds)
    assert geom.prims == atoms + 2 * bonds


# ── the geometry, which is where a silent error would live ───────────

def test_every_bond_ends_on_an_atom(stage):
    """The quaternion check. A wrong rotation still writes a valid stage
    full of cylinders pointing the wrong way, and nothing complains."""
    s, geoms = stage
    for name in geoms:
        centres = _atom_centres(s, name)
        assert centres, name
        for prim in s.GetPrimAtPath(f"/Molecules/{name}/Bonds").GetChildren():
            e0, e1 = _cylinder_ends(prim)
            gap = min(min(np.linalg.norm(e0 - p), np.linalg.norm(e1 - p))
                      for p in centres.values())
            # float32 quaternion round-off, not a modelling tolerance.
            assert gap < 1e-4, f"{name} {prim.GetName()} ends {gap:.2e} A off"


def test_the_two_halves_of_a_bond_meet_at_its_midpoint(stage):
    """Each half runs atom-to-midpoint, so the halves must share an end --
    a gap or an overlap there is visible as a seam or a bulge."""
    s, _ = stage
    bonds = s.GetPrimAtPath("/Molecules/Glucose/Bonds")
    halves = {}
    for prim in bonds.GetChildren():
        halves.setdefault(prim.GetCustomDataByKey("bondIndex"), []).append(prim)
    assert halves
    for index, pair in halves.items():
        assert len(pair) == 2, f"bond {index} has {len(pair)} halves"
        ends_a, ends_b = _cylinder_ends(pair[0]), _cylinder_ends(pair[1])
        shared = min(np.linalg.norm(a - b) for a in ends_a for b in ends_b)
        assert shared < 1e-4, f"bond {index} halves meet {shared:.2e} A apart"


def test_bond_lengths_are_chemically_real(stage):
    """A C-C single bond is ~1.5 A and a C-H ~1.1 A. If the conformer were
    collapsed or the units wrong, this is what would show it."""
    s, _ = stage
    centres = _atom_centres(s, "Glucose")
    halves = {}
    for prim in s.GetPrimAtPath("/Molecules/Glucose/Bonds").GetChildren():
        halves.setdefault(prim.GetCustomDataByKey("bondIndex"), []).append(prim)
    lengths = []
    for pair in halves.values():
        total = sum(UsdGeom.Cylinder(p).GetHeightAttr().Get() for p in pair)
        lengths.append(total)
    assert lengths
    assert 0.9 < min(lengths) < 1.2, f"shortest bond {min(lengths):.2f} A"
    assert 1.3 < max(lengths) < 1.7, f"longest bond {max(lengths):.2f} A"


def test_the_atoms_are_not_all_at_the_origin(stage):
    """An unembedded molecule reports every atom at the origin rather than
    raising, so a collapsed conformer has to be caught by looking."""
    s, geoms = stage
    for name, geom in geoms.items():
        assert geom.embedded, name
        spread = np.ptp(np.array(list(_atom_centres(s, name).values())), axis=0)
        assert spread.max() > 0.5, f"{name} is flat: {spread}"


# ── materials ────────────────────────────────────────────────────────

def test_one_material_per_element_not_per_atom(stage):
    """Enzalutamide has 48 atoms over 7 elements. A material each would be
    48 shader prims saying one of 7 things."""
    s, geoms = stage
    materials = s.GetPrimAtPath("/Materials").GetChildren()
    elements = set()
    for geom in geoms.values():
        elements |= set(geom.elements)
    assert {m.GetName() for m in materials} == elements
    assert len(materials) < geoms["Enzalutamide"].atoms


def test_every_atom_resolves_to_its_element_colour(stage):
    s, _ = stage
    seen = set()
    for prim in s.GetPrimAtPath("/Molecules/Enzalutamide/Atoms").GetChildren():
        symbol = prim.GetCustomDataByKey("element")
        bound, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert bound, f"{prim.GetName()} has no material"
        assert bound.GetPrim().GetName() == symbol
        shader = UsdShade.Shader(
            s.GetPrimAtPath(f"{bound.GetPath()}/Surface"))
        assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
        colour = shader.GetInput("diffuseColor").Get()
        assert tuple(round(c, 3) for c in colour) == pytest.approx(
            mu.CPK_COLORS[symbol], abs=1e-3)
        seen.add(symbol)
    assert seen == {"C", "H", "N", "O", "F", "S"}


def test_a_bond_half_takes_the_colour_of_its_own_atom(stage):
    """What half-colouring is for: a single grey stick loses which element
    is at which end, and that is most of what a structure conveys."""
    s, _ = stage
    centres = _atom_centres(s, "Enzalutamide")
    symbols = {i: p.GetCustomDataByKey("element")
               for i, p in ((pr.GetCustomDataByKey("atomIndex"), pr)
                            for pr in s.GetPrimAtPath(
                                "/Molecules/Enzalutamide/Atoms").GetChildren())}
    mixed = 0
    for prim in s.GetPrimAtPath("/Molecules/Enzalutamide/Bonds").GetChildren():
        e0, e1 = _cylinder_ends(prim)
        # the atom this half starts from
        nearest = min(centres, key=lambda i: min(
            np.linalg.norm(e0 - centres[i]), np.linalg.norm(e1 - centres[i])))
        bound, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert bound.GetPrim().GetName() == symbols[nearest]
        mixed += 1
    assert mixed > 0


# ── provenance ───────────────────────────────────────────────────────

def test_the_stage_says_the_coordinates_are_computed(stage):
    """A conformer is not a structure. Docetaxel has nine rotatable bonds
    and its pose in a tubulin site is an experimental question this does
    not answer; a stage that does not say so invites the confusion."""
    s, _ = stage
    prim = s.GetPrimAtPath("/Molecules/Enzalutamide")
    assert prim.GetCustomDataByKey("coordinatesAreExperimental") is False
    assert "ETKDG" in prim.GetCustomDataByKey("coordinateSource")
    assert "MMFF94" in prim.GetCustomDataByKey("coordinateSource")
    assert prim.GetCustomDataByKey("coordinateSeed")
    assert "library" in prim.GetCustomDataByKey("topologySource")
    assert prim.GetCustomDataByKey("radiiSource").startswith("rdkit")
    assert prim.GetCustomDataByKey("smiles") == ENZALUTAMIDE
    assert prim.GetCustomDataByKey("formula") == "C21H16F4N4O2S"


def test_the_rendering_conventions_are_recorded_as_conventions(stage):
    """BALL_SCALE is a choice that makes bonds visible, not a measurement,
    and a radius on the stage is that choice times a real radius."""
    s, _ = stage
    prim = s.GetPrimAtPath("/Molecules/Glucose")
    assert prim.GetCustomDataByKey("ballScale") == mu.BALL_SCALE
    assert prim.GetCustomDataByKey("stickRadiusAngstrom") == mu.STICK_RADIUS

    from rdkit.Chem import GetPeriodicTable
    vdw_o = GetPeriodicTable().GetRvdw("O")
    oxygen = next(p for p in s.GetPrimAtPath(
        "/Molecules/Glucose/Atoms").GetChildren()
        if p.GetCustomDataByKey("element") == "O")
    assert UsdGeom.Sphere(oxygen).GetRadiusAttr().Get() == pytest.approx(
        vdw_o * mu.BALL_SCALE)
    # and the stick is thinner than the smallest ball, or bonds disappear
    vdw_h = GetPeriodicTable().GetRvdw("H")
    assert mu.STICK_RADIUS < vdw_h * mu.BALL_SCALE


# ── reproducibility and units ────────────────────────────────────────

def test_the_same_smiles_gives_the_same_coordinates(tmp_path):
    """Two exports of one library must not differ, or a geometry test pins
    nothing and a diff between stages is all noise."""
    def coords():
        s = Usd.Stage.CreateNew(str(tmp_path / f"m{coords.n}.usda"))
        coords.n += 1
        UsdGeom.Scope.Define(s, "/Materials")
        UsdGeom.Xform.Define(s, "/Molecules")
        mu.author_molecule(s, "/Molecules", "Glucose", GLUCOSE)
        return _atom_centres(s, "Glucose")
    coords.n = 0

    first, second = coords(), coords()
    assert set(first) == set(second)
    for i in first:
        assert first[i] == pytest.approx(second[i], abs=1e-9)


def test_the_stage_is_in_angstroms(stage):
    """The cell-scale exporters here use micrometres, four orders away. A
    molecule laid into one of those scenes without scaling is invisible."""
    s, _ = stage
    assert UsdGeom.GetStageMetersPerUnit(s) == pytest.approx(1e-10)
    assert UsdGeom.GetStageUpAxis(s) == UsdGeom.Tokens.z
    assert mu.METERS_PER_UNIT == 1e-10


def test_a_molecule_that_will_not_embed_is_marked_not_silently_flat(monkeypatch):
    """Distance geometry fails on some cages. The topology is still worth
    writing; what must not happen is a blob that looks like a conformer."""
    from cognisom.library import structures

    real = structures.embed_conformer
    monkeypatch.setattr(structures, "embed_conformer",
                        lambda smi, with_hydrogens=True: (
                            real(smi, with_hydrogens)[0], False))

    s = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(s, "/Materials")
    UsdGeom.Xform.Define(s, "/Molecules")
    geom = mu.author_molecule(s, "/Molecules", "Glucose", GLUCOSE)

    assert geom.embedded is False
    prim = s.GetPrimAtPath("/Molecules/Glucose")
    assert prim.GetCustomDataByKey("conformerEmbedded") is False
    assert "origin" in prim.GetCustomDataByKey("warning")
    # still a full topology, so the failure is legible rather than empty
    assert geom.atoms == 24 and geom.bonds == 24


def test_an_unparseable_smiles_is_refused_rather_than_drawn(stage):
    s, _ = stage
    with pytest.raises(ValueError, match="does not parse"):
        mu.author_molecule(s, "/Molecules", "Nonsense", "this is not smiles")


# ── the whole library ────────────────────────────────────────────────

def test_the_library_export_covers_every_structured_entity(tmp_path):
    entries = mu.library_molecules()
    assert len(entries) >= 20, f"only {len(entries)} entities carry a SMILES"

    out = tmp_path / "library.usda"
    geoms = mu.export_library(str(out))

    assert len(geoms) == len(entries)
    assert all(g.embedded for g in geoms), [
        g.name for g in geoms if not g.embedded]
    assert out.exists() and out.stat().st_size > 100_000

    s = Usd.Stage.Open(str(out))
    assert s.GetDefaultPrim().GetPath() == "/Molecules"
    root = s.GetPrimAtPath("/Molecules")
    assert root.GetCustomDataByKey("moleculeCount") == len(geoms)
    assert root.GetCustomDataByKey("primCount") == sum(g.prims for g in geoms)
    assert "not experimental" in root.GetCustomDataByKey("note")


def test_molecules_do_not_overlap_on_the_grid(tmp_path):
    """A grid is only readable if each cell holds its own molecule; sized
    to the largest is what keeps docetaxel out of its neighbour."""
    geoms = mu.export_library(str(tmp_path / "grid.usda"),
                              names=["Oxygen", "Glucose", "Docetaxel"])
    s = Usd.Stage.Open(str(tmp_path / "grid.usda"))
    cell = s.GetPrimAtPath("/Molecules").GetCustomDataByKey("gridCellAngstrom")
    assert cell > 2 * max(g.radius_a for g in geoms)

    origins = []
    for geom in geoms:
        prim = s.GetPrimAtPath(f"/Molecules/{mu._safe_name(geom.name)}")
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        origins.append((np.array(list(ops[-1].Get())), geom.radius_a))
    for i, (a, ra) in enumerate(origins):
        for b, rb in origins[i + 1:]:
            assert np.linalg.norm(a - b) > ra + rb, "two molecules intersect"
