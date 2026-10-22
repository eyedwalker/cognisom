"""Bio-USD is a schema worth keeping and a writer that could not carry it.

``biousd/converter.py`` never imports ``pxr``. It concatenates ``.usda``
text, so nothing it emits is checked by USD until a reader tries, and it
authors ONE PRIM PER CELL at about 456 bytes each. The schema is the good
part -- namespaced attributes, allowedTokens, nine prim types, a working
SBML path -- and the carrier is the part that cannot hold a scene.

``usd_writer`` authors the same schema through the USD API, carrying the
per-cell attributes as instance primvars on a single PointInstancer.
Measured here at 20,000 cells:

    converter.py text     8.76 MB    20,002 prims
    usd_writer            0.76 MB    17 prims

The prim count is the number that matters. It does not grow with the
population, so a stage opens in the same time at five hundred thousand
cells as at five hundred.

Two details that are easy to get wrong and silent when you do. The
interpolation for per-instance primvars is ``varying``; there is no
``instance`` token in USD 25.11 and reaching for one raises. And a camera
aimed with Euler angles instead of a look-at renders an EMPTY FRAME
SUCCESSFULLY -- a valid PNG of nothing, no error anywhere.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom                                   # noqa: E402

from cognisom.biousd import ontology as onto                   # noqa: E402
from cognisom.biousd.schema import (                           # noqa: E402
    BioCell, CellPhase, CellType)
from cognisom.biousd.usd_writer import CELLS, write_cells      # noqa: E402

#: name, mesh_type, coarse type -- four cells whose ontology terms come
#: from different places: two CL lineages, a CL generic, and an NCIT
#: oncology term CL does not cover.
POPULATION = [
    ("Macrophage", "ruffled_pseudopod", CellType.IMMUNE, "CL:0000235"),
    ("CD8+ T cell", "smooth_sphere", CellType.IMMUNE, "CL:0000625"),
    ("Stromal fibroblast", "elongated_spindle", CellType.FIBROBLAST, "CL:0000057"),
    ("Prostatic adenocarcinoma cell", "irregular_ruffled", CellType.CANCER,
     "NCIT:C2919"),
]


def _cells(n):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        name, mesh, ctype, _curie = POPULATION[i % len(POPULATION)]
        cell = BioCell(cell_id=i,
                       position=tuple(float(x) for x in rng.uniform(-150, 150, 3)),
                       cell_type=ctype, phase=CellPhase.G1, age=float(i % 24))
        cell.name, cell.mesh_type, cell.scale_um = name, mesh, 12.0
        out.append(cell)
    return out


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("biousd") / "scene.usdc")
    result = write_cells(path, _cells(600))
    return result, Usd.Stage.Open(path)


def test_openusd_opens_what_was_written(written):
    """The text path's output once did not parse at all."""
    _result, stage = written
    assert stage is not None
    assert stage.GetDefaultPrim().IsValid()
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1e-6)


def test_the_prim_count_does_not_follow_the_population(tmp_path):
    """One instancer, not one prim per cell. This is the whole point."""
    counts = {}
    for n in (100, 2000):
        path = str(tmp_path / f"s{n}.usdc")
        result = write_cells(path, _cells(n))
        counts[n] = result.prims
        assert result.cells == n
    assert counts[100] == counts[2000], (
        f"prims grew with the population: {counts}. That is the defect "
        "being replaced -- 20,000 cells was 20,002 prims.")
    assert counts[100] < 40


def test_every_bio_cell_attribute_survives_as_a_primvar(written):
    """The schema is kept; only the carrier changed."""
    result, stage = written
    api = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(CELLS))
    present = {v.GetPrimvarName(): v for v in api.GetPrimvars()
               if v.HasAuthoredValue()}
    for name in ("bio:cell:id", "bio:cell:type", "bio:cell:phase",
                 "bio:cell:age", "bio:cell:alive", "bio:cell:volume",
                 "bio:cell:ontologyTerm"):
        assert name in present, f"{name} missing; have {sorted(present)}"
        primvar = present[name]
        assert primvar.GetInterpolation() == UsdGeom.Tokens.varying, (
            f"{name} is {primvar.GetInterpolation()}; per-instance data on "
            "a PointInstancer is 'varying'")
        assert len(primvar.Get()) == result.cells


def test_the_prototypes_are_library_meshes_and_not_a_sphere(written):
    """``cells_to_usd`` instances one UsdGeom.Sphere, however many cells."""
    _result, stage = written
    instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(CELLS))
    targets = instancer.GetPrototypesRel().GetTargets()
    assert len(targets) == len({m for _n, m, _t, _c in POPULATION})

    faces = 0
    for target in targets:
        prim = stage.GetPrimAtPath(target)
        meshes = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
        assert meshes, f"{target} has no mesh beneath it"
        for mesh in meshes:
            faces += len(UsdGeom.Mesh(mesh).GetFaceVertexCountsAttr().Get())
    assert faces > 1000, faces
    assert not [p for p in stage.Traverse() if p.IsA(UsdGeom.Sphere)]


def test_each_cell_carries_a_term_someone_else_reads(written):
    """A CURIE, or nothing. Never a guess."""
    result, stage = written
    api = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(CELLS))
    terms = list(api.GetPrimvar("bio:cell:ontologyTerm").Get())
    expected = {curie for _n, _m, _t, curie in POPULATION}
    assert set(terms) == expected, sorted(set(terms))
    assert result.unmapped == []
    for term in expected:
        assert onto.iri_for(term), f"{term} does not expand to an IRI"


def test_an_unknown_cell_gets_no_term_rather_than_a_near_one(tmp_path):
    """Fuzzy ontology search returns something for everything.

    It offered 'endothelial cell of umbilical vein' for an inflamed
    endothelial cell and 'stromal cell of endometrium' for a stromal
    fibroblast -- both with the confidence of a correct answer.
    """
    cell = BioCell(cell_id=0, position=(0.0, 0.0, 0.0))
    cell.name = "Entirely Invented Cell"
    cell.mesh_type, cell.scale_um = "smooth_sphere", 10.0
    result = write_cells(str(tmp_path / "u.usdc"), [cell])
    assert result.unmapped == ["Entirely Invented Cell"]
    stage = Usd.Stage.Open(result.path)
    api = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(CELLS))
    assert list(api.GetPrimvar("bio:cell:ontologyTerm").Get()) == [""]


def test_colour_is_authored_so_a_dense_scene_is_not_a_white_mass(written):
    """And it is the mechanism for colouring by any simulation variable."""
    result, stage = written
    api = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(CELLS))
    colours = np.asarray(api.GetPrimvar("displayColor").Get())
    assert colours.shape == (result.cells, 3)
    assert len(np.unique(colours, axis=0)) >= len(POPULATION) - 1


def test_a_camera_is_framed_on_the_cells(written):
    """Aimed by look-at: Euler angles render an empty frame successfully."""
    _result, stage = written
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera"))
    assert camera, "no camera authored"
    xform = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    eye = np.array(xform.ExtractTranslation())
    forward = -np.array([xform[2][0], xform[2][1], xform[2][2]])
    forward = forward / np.linalg.norm(forward)
    toward_origin = -eye / np.linalg.norm(eye)
    assert float(np.dot(forward, toward_origin)) > 0.999


# ── the ontology table itself ────────────────────────────────────────

def test_the_duplicate_tregs_collapse_to_one_term(written):
    """Two entities, one cell. Name-based checks cannot see it."""
    assert (onto.curie_for("Regulatory T cell")
            == onto.curie_for("Regulatory T cell (Treg)") == "CL:0000815")


def test_macrophage_polarisation_is_not_flattened():
    """The census mapping sends both M1 and M2 to plain 'macrophage'."""
    m1, m2 = onto.curie_for("Macrophage M1"), onto.curie_for("Macrophage M2")
    assert m1 == "CL:0000863" and m2 == "CL:0000890"
    assert m1 != m2 != onto.curie_for("Macrophage")


def test_exhaustion_has_its_own_term():
    """It matters here specifically: this platform studies immune escape."""
    assert onto.curie_for("Exhausted CD8+ T cell") == "CL:0020031"
    assert onto.curie_for("CD8+ T cell") != onto.curie_for("Exhausted CD8+ T cell")


def test_normal_is_a_state_and_gets_no_term():
    """Conflating it with identity is why the enum cannot be mapped."""
    assert "normal" not in onto.TYPE_CURIE
    assert onto.curie_for(None, "normal") is None
    assert onto.curie_for(None, "cancer") == "CL:0001064"


def test_the_whole_cellular_library_is_covered():
    """An unmapped cell is honest; a silently growing set of them is not."""
    import contextlib
    import io

    from cognisom.visualization.morphology_usd import CELLULAR_TYPES
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        store = EntityStore()
    names = {n for t, n, _ in store._conn.execute(
        "SELECT entity_type, name, data_json FROM entities")
        if t in CELLULAR_TYPES and t not in ("tissue_type", "virus", "bacterium")}
    missing = sorted(names - set(onto.CELL_CURIE))
    assert not missing, (
        f"{len(missing)} library cells have no ontology term: {missing}. "
        "Resolve by EXACT label match against OLS, or leave unmapped "
        "deliberately -- fuzzy search returns a wrong term for everything.")

def test_the_body_is_translucent_so_the_nucleus_can_be_seen(written):
    """Three parts meshed and one visible is three parts wasted.

    Every prototype meshes a body, and most mesh a nucleus and granules
    inside it. Without an opacity on the body those inner parts are
    sealed in an opaque shell -- authored, instanced, shipped, and
    invisible. The library card has drawn the body at 0.35 since it was
    written; the export did not.
    """
    _result, stage = written
    instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(CELLS))
    bodies = translucent = 0
    for target in instancer.GetPrototypesRel().GetTargets():
        for prim in Usd.PrimRange(stage.GetPrimAtPath(target)):
            if prim.GetName() != "body":
                continue
            bodies += 1
            opacity = UsdGeom.Mesh(prim).GetDisplayOpacityAttr().Get()
            if opacity and float(opacity[0]) < 1.0:
                translucent += 1
    assert bodies and translucent == bodies, (
        f"{bodies - translucent} of {bodies} bodies are opaque")


def test_part_colour_and_per_cell_colour_cannot_both_win(tmp_path):
    """A measured USD behaviour, pinned because it is invisible.

    A prototype carrying ``displayColor`` OVERRIDES the instancer's
    per-instance primvar. Two renders of the same scene -- one with a
    green per-instance colour authored, one without -- came back pixel
    for pixel identical once the prototype had its own colours. So
    authoring part colours silently disables colouring a population by
    any simulation variable, which is what the primvar path is for.

    The default therefore authors opacity without colour, which keeps
    both as far as USD allows: the instancer tints each cell, and the
    translucent body still shows the structure inside it.
    """
    from pxr import Usd as _Usd
    from cognisom.biousd.usd_writer import author_prototypes

    def colours_on_prototypes(part_colours):
        stage = _Usd.Stage.CreateNew(
            str(tmp_path / f"p{int(part_colours)}.usda"))
        author_prototypes(stage, ["multilobed_granular"],
                          part_colours=part_colours)
        return [UsdGeom.Mesh(p).GetDisplayColorAttr().Get()
                for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]

    assert all(c is None for c in colours_on_prototypes(False)), (
        "the default authors prototype colour, which suppresses the "
        "instancer's per-cell colour over the whole population")
    explicit = colours_on_prototypes(True)
    assert explicit and all(c is not None for c in explicit)
