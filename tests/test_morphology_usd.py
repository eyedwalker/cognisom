"""Library morphologies as USD meshes, and what the stage admits about them.

``morphology`` turns the library's ``mesh_type`` vocabulary into surfaces --
32 builders, with named sub-parts. Nothing carried any of it into USD, so
the cell exporters drew a sphere per cell and a neutrophil with a
multilobed nucleus rendered identically to a lymphocyte.

Two claims need guarding, and neither is about triangles:

  * **A fallback must be visible.** An unknown ``mesh_type`` becomes a
    sphere. If the prim does not say so, a sphere goes on wearing the
    label "neutrophil" and the picture claims more than the library
    supports. Four library entities hit this today: the tissue types,
    whose ``mesh_type`` is the string ``"mesh"`` -- a placeholder naming
    a file that does not exist.
  * **A depiction must not pass for a measurement.** A
    ``multilobed_granular`` nucleus is an illustration of a word someone
    wrote; a nucleus segmented from a confocal stack is evidence. Once
    both are triangles in one scene only the metadata tells them apart.

And one structural trap: two library entities can share a name. "NK cell"
exists as both a ``cell_type`` and an ``immune_cell``, so a prim path
computed from the name alone merges them and the second one's transform
lands on the first one's prim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("pxr")
pytest.importorskip("skimage")

from pxr import Usd, UsdGeom, UsdShade          # noqa: E402

from cognisom.visualization import morphology_usd as mz   # noqa: E402

# Small grid: these tests are about metadata and structure, not fidelity.
GRID = 24


@pytest.fixture(scope="module")
def stage():
    s = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(s, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(s, mz.METERS_PER_UNIT)
    UsdGeom.Scope.Define(s, "/Materials")
    UsdGeom.Xform.Define(s, "/Entities")
    cache = {}
    geoms = {}
    for name, mesh_type, scale, etype in (
            ("Neutrophil", "multilobed_granular", 12.0, "immune_cell"),
            ("Macrophage", "ruffled_pseudopod", 20.0, "cell_type"),
            ("Influenza A", "enveloped_ha_na_spikes", 0.1, "virus"),
            ("Made Up Cell", "no_such_mesh_type", 10.0, "cell_type"),
            ("BCL-2", "mesh", 0.005, "protein")):
        geoms[name] = mz.author_entity(
            s, "/Entities", name, mesh_type, scale,
            entity_type=etype, grid=GRID, material_cache=cache)
    return s, geoms


def _mesh_points(stage, path):
    return np.array(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get())


# ── the thing this replaces ──────────────────────────────────────────

def test_cells_with_different_shapes_are_different_geometry(stage):
    """All of these were a sphere in the USD track."""
    _, geoms = stage
    tri = {n: g.triangles for n, g in geoms.items()}
    assert tri["Neutrophil"] > 0 and tri["Macrophage"] > 0
    assert len({tri["Neutrophil"], tri["Macrophage"],
                tri["Influenza A"]}) == 3, tri


def test_a_cell_is_built_as_named_parts_not_one_blob(stage):
    """The parts are the point: a nucleus can be shaded as a nucleus, and a
    viewer can hide the body to look inside."""
    s, geoms = stage
    assert set(geoms["Neutrophil"].parts) == {"body", "nucleus", "granules"}
    for part in ("body", "nucleus", "granules"):
        prim = s.GetPrimAtPath(f"/Entities/Neutrophil/{part}")
        assert prim, part
        assert prim.GetCustomDataByKey("part") == part
        assert UsdGeom.Mesh(prim).GetPointsAttr().Get()


def test_the_nucleus_is_inside_the_body(stage):
    """A nucleus whose bounds escape the cell is a build error that still
    renders, so it is worth asserting rather than eyeballing."""
    s, _ = stage
    body = _mesh_points(s, "/Entities/Neutrophil/body")
    nucleus = _mesh_points(s, "/Entities/Neutrophil/nucleus")
    assert (nucleus.min(axis=0) >= body.min(axis=0) - 1e-6).all()
    assert (nucleus.max(axis=0) <= body.max(axis=0) + 1e-6).all()
    # and it is genuinely smaller, not coincident with the membrane
    assert np.ptp(nucleus, axis=0).prod() < np.ptp(body, axis=0).prod()


def test_the_mesh_is_triangles_and_not_subdivided(stage):
    """Marching cubes gives a closed triangle surface. Leaving the default
    subdivision scheme on rounds the lobes off a nucleus built to have
    lobes -- the shape would be quietly smoothed by the renderer."""
    s, _ = stage
    mesh = UsdGeom.Mesh(s.GetPrimAtPath("/Entities/Neutrophil/nucleus"))
    counts = mesh.GetFaceVertexCountsAttr().Get()
    assert set(counts) == {3}
    assert mesh.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    assert len(indices) == 3 * len(counts)
    assert max(indices) < len(mesh.GetPointsAttr().Get())


def test_the_geometry_is_at_the_scale_the_library_states(stage):
    """scale_um is the width of the whole shape. An earlier bug in the
    builder doubled it -- a "20 um" cell measured 35 um across -- so the
    stage is the right place to catch a regression of that kind."""
    s, geoms = stage
    for name, tol in (("Neutrophil", 0.45), ("Macrophage", 0.45)):
        pts = np.vstack([_mesh_points(s, f"/Entities/{mz._safe_name(name)}/{p}")
                         for p in geoms[name].parts])
        width = float(np.ptp(pts, axis=0).max())
        want = geoms[name].scale_um
        assert abs(width - want) / want < tol, (
            f"{name}: {width:.1f} um across, library says {want} um")


# ── the fallback, which must never be silent ─────────────────────────

def test_an_unknown_mesh_type_says_it_became_a_sphere(stage):
    s, geoms = stage
    geom = geoms["Made Up Cell"]
    assert geom.fell_back is True
    assert geom.builder == "sphere"
    prim = s.GetPrimAtPath("/Entities/Made_Up_Cell")
    assert prim.GetCustomDataByKey("fellBackToSphere") is True
    assert prim.GetCustomDataByKey("meshType") == "no_such_mesh_type"
    assert "sphere wearing the label" in prim.GetCustomDataByKey("warning")


def test_a_known_mesh_type_does_not_claim_to_have_fallen_back(stage):
    s, geoms = stage
    assert geoms["Neutrophil"].fell_back is False
    prim = s.GetPrimAtPath("/Entities/Neutrophil")
    assert prim.GetCustomDataByKey("fellBackToSphere") is False
    assert prim.GetCustomDataByKey("builder") == "multilobed_granular"
    assert prim.GetCustomDataByKey("warning") is None


def test_the_library_tissue_types_are_the_ones_that_fall_back():
    """Their mesh_type is the string "mesh": a placeholder naming a file
    that does not exist. Worth pinning, because it is the live example of
    the fallback and it should not grow silently."""
    entries = mz.library_entities(None, mz.CELLULAR_TYPES)
    placeholder = [(n, t) for n, t, mt, _s in entries if mt == "mesh"]
    assert placeholder, "no entity has mesh_type 'mesh' any more"
    assert {t for _n, t in placeholder} == {"tissue_type"}, placeholder


# ── provenance ───────────────────────────────────────────────────────

def test_the_stage_calls_the_shapes_depictions(stage):
    s, _ = stage
    prim = s.GetPrimAtPath("/Entities/Neutrophil")
    assert prim.GetCustomDataByKey("geometrySource") == "procedural depiction"
    note = prim.GetCustomDataByKey("geometryNote")
    assert "not segmented" in note and "not a measurement" in note


def test_a_shared_scale_is_called_a_placeholder(stage):
    """All 34 library proteins are recorded at 0.005 um and all 5 organs at
    50,000 um. One number shared by every member of a class is a
    placeholder sitting in the right column, not a measurement."""
    s, _ = stage
    protein = s.GetPrimAtPath("/Entities/BCL_2")
    assert "placeholder" in protein.GetCustomDataByKey("scaleProvenance")

    cell = s.GetPrimAtPath("/Entities/Neutrophil")
    prov = cell.GetCustomDataByKey("scaleProvenance")
    assert "placeholder" not in prov
    assert "not traced to a measurement" in prov


def test_the_library_really_does_share_those_scales():
    """The claim above is about the library, so it should be checked
    against the library and not just asserted in a docstring."""
    import collections
    import json

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        from cognisom.library.store import EntityStore
        store = EntityStore()
    rows = [tuple(r) for r in store._conn.execute(
        "SELECT entity_type, data_json FROM entities")]
    by_type = collections.defaultdict(set)
    for etype, blob in rows:
        scale = json.loads(blob or "{}").get("scale_um")
        if scale is not None:
            by_type[etype].add(float(scale))
    for etype in ("protein", "organ", "tissue_type"):
        assert len(by_type[etype]) == 1, (
            f"{etype} no longer shares one scale: {sorted(by_type[etype])}")
        assert etype in mz._UNIFORM_SCALE_TYPES
    # cells genuinely vary, so they are not placeholders in this sense
    assert len(by_type["cell_type"]) > 5


# ── materials ────────────────────────────────────────────────────────

def test_one_material_per_part_shared_across_entities(stage):
    s, _ = stage
    names = {m.GetName() for m in s.GetPrimAtPath("/Materials").GetChildren()}
    assert "body" in names and "nucleus" in names
    for entity in ("Neutrophil", "Macrophage"):
        prim = s.GetPrimAtPath(f"/Entities/{entity}/nucleus")
        bound, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert bound.GetPath() == "/Materials/nucleus"


def test_the_body_is_translucent_so_the_parts_can_be_seen(stage):
    s, _ = stage
    shader = UsdShade.Shader(s.GetPrimAtPath("/Materials/body/Surface"))
    assert shader.GetInput("opacity").Get() == pytest.approx(
        mz.PART_OPACITY["body"])
    assert shader.GetInput("opacity").Get() < 1.0
    # The nucleus is not, or it would not read against the body. No
    # opacity input at all means UsdPreviewSurface's default of 1.0 --
    # GetInput returns a wrapper whether or not the input was authored,
    # so the value is what to test, never the handle.
    nucleus = UsdShade.Shader(s.GetPrimAtPath("/Materials/nucleus/Surface"))
    assert nucleus.GetInput("opacity").Get() in (None, 1.0)


# ── the whole library ────────────────────────────────────────────────

def test_two_entities_sharing_a_name_get_separate_prims(tmp_path):
    """"NK cell" is both a cell_type and an immune_cell. A prim path taken
    from the name alone merged them, and the second transform was refused
    because the first had already claimed the op."""
    entries = mz.library_entities(None, mz.CELLULAR_TYPES)
    duplicated = [n for n, *_ in entries
                  if sum(1 for m, *_ in entries if m == n) > 1]
    assert duplicated, "the library no longer has a duplicate name"

    geoms = mz.export_library(str(tmp_path / "dup.usda"),
                              names=duplicated, grid=GRID)
    assert len(geoms) == len(duplicated)
    paths = [g.prim_path for g in geoms]
    assert len(set(paths)) == len(paths), paths

    s = Usd.Stage.Open(str(tmp_path / "dup.usda"))
    for geom in geoms:
        prim = s.GetPrimAtPath(geom.prim_path)
        assert prim, geom.prim_path
        assert prim.GetCustomDataByKey("entityType") == geom.entity_type


def test_the_export_covers_the_cellular_entities(tmp_path):
    out = tmp_path / "cells.usda"
    geoms = mz.export_library(str(out), grid=GRID)

    assert len(geoms) >= 50, f"only {len(geoms)} entities exported"
    assert out.exists()

    s = Usd.Stage.Open(str(out))
    root = s.GetPrimAtPath("/Entities")
    assert root.GetCustomDataByKey("entityCount") == len(geoms)
    assert root.GetCustomDataByKey("triangleCount") == sum(
        g.triangles for g in geoms)
    low, high = root.GetCustomDataByKey("sizeRangeUm")
    assert low < 1.0 < high, (low, high)      # a virus and a cell in one stage
    assert "depictions" in root.GetCustomDataByKey("note")
    assert UsdGeom.GetStageMetersPerUnit(s) == pytest.approx(1e-6)


def test_the_export_leaves_out_types_the_vocabulary_does_not_describe():
    """A protein's mesh_type is "protein_structure" -- a file to load, not
    a shape to build -- and a gene's is a sphere standing in for a locus.
    Building either would be inventing a shape for something the library
    does not describe the shape of."""
    assert "protein" not in mz.CELLULAR_TYPES
    assert "gene" not in mz.CELLULAR_TYPES
    types = {t for _n, t, _m, _s in mz.library_entities(None, mz.CELLULAR_TYPES)}
    assert types <= set(mz.CELLULAR_TYPES)
    assert "cell_type" in types and "immune_cell" in types


def test_entities_do_not_overlap_on_the_grid(tmp_path):
    """Influenza at 0.1 um sits beside a fibroblast at 30 um, a factor of
    300. Scaling them to a common size would make a virus and a cell look
    comparable, which is most of what their sizes mean."""
    names = ["Influenza A", "Cancer-associated fibroblast", "Neutrophil"]
    geoms = mz.export_library(str(tmp_path / "grid.usda"), names=names,
                              grid=GRID)
    s = Usd.Stage.Open(str(tmp_path / "grid.usda"))
    cell = s.GetPrimAtPath("/Entities").GetCustomDataByKey("gridCellUm")
    assert cell > max(g.scale_um for g in geoms)

    placed = []
    for geom in geoms:
        ops = UsdGeom.Xformable(s.GetPrimAtPath(geom.prim_path)).GetOrderedXformOps()
        placed.append((np.array(list(ops[-1].Get())), geom.scale_um))
    for i, (a, sa) in enumerate(placed):
        for b, sb in placed[i + 1:]:
            assert np.linalg.norm(a - b) > (sa + sb) / 2
