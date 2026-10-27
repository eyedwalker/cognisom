"""Every lineage in a run is drawn as itself: a shape and a size, not only a colour.

What was wrong
--------------
``write_run`` authored two prototypes for a whole run -- a 10 um sphere
for every cell and an 8 um sphere for every immune cell -- so the only
thing that told a cancer cell from a normal one, or a macrophage from a
T cell, was the colour primvar, and both diameters were invented in the
exporter. The library had carried a built morphology per entity for
some time (a ruffled macrophage, a columnar luminal cell, a granular NK
cell) and a measured radius for two of them, and the run exporter read
none of it. Three other places kept their own token -> shape maps, and
one of them (``usd_animation``) had no callers at all.

What is now true
----------------
``lineage_appearance.appearance_for`` is the one resolver. Per cell per
frame, ``write_run`` resolves the frame's type token (and, for a
macrophage, its polarization) to an Appearance, authors one prototype
per distinct mesh actually used plus one named sphere for anything with
no builder, and time-samples ``protoIndices`` and ``scales`` so a cell
that transforms changes shape and size on its frame. The stage records
what it drew (``customLayerData["cognisom:appearance"]``) and the report
says how badly the drawn cells overlap on the lattice.

The sizes are the library's, and the table of what each lineage resolves
to lives in ``lineage_appearance``'s docstring -- once, and checked here
against ``appearance_table_text()`` so it cannot go stale (a copy of it
in this docstring did, saying one measured size in six the day
``seed_cell_size`` made it five). Today six of seven lineages are
cited and ``normal`` is the one estimate; what the tests assert is that
``cited`` agrees with the library row itself (a MEASURED ``radius_um``
with a citation), not with the string the resolver built from it.

At grid 16 the six prototypes separate pairwise on part set,
bounding-box aspect or surface roughness (ellipsoid-normalised radial
CV: smooth_sphere 0.016, irregular_ruffled 0.086, ruffled_pseudopod
0.155; polarized_columnar is 2.3x taller than wide, elongated_spindle
1.97 x 0.66 x 0.37). The silhouette check is at grid 16 and not 8
because at grid 8 ``elongated_spindle`` meshes as a 44-triangle blob
with a unit-cube extent: the spindle does not resolve below about 16
voxels, which also means ``write_cells``' far LOD tier draws M2
macrophages, fibroblasts and smooth muscle as blobs. Those same
anisotropic bounds are why the instancer extent is checked here frame
by frame against the visible instances' computed bounds: a pad of
diameter/2 was exact for two spheres and 4.5 um short on a real run.

The Bio-USD validator reads only the un-timed values (the last frame),
so the per-frame array lengths, index ranges and colours are checked
here on every time sample, on the hand-built run and on the real one.

The 2 h ring Baseline run (``np.random.seed(3)``, 21 frames) builds,
runs and writes in about 2 s, so nothing here is marked slow. That run
seeds no macrophage polarization mix (the mix is applied by the slab
seeding only), so its six macrophages stay 'mixed' and the M1 -> M2 flip
is exercised on a hand-built three-frame run instead.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from contextlib import redirect_stdout
from itertools import combinations

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom, UsdShade                                   # noqa: E402

from cognisom.biousd.usd_writer import (                                  # noqa: E402
    PROTOTYPE_GRID, _normalise_to_unit_diameter)
from cognisom.dashboard.engine_runner import CellSnapshot, EngineRunner  # noqa: E402
from cognisom.library.parameters import Provenance, get_parameter        # noqa: E402
from cognisom.visualization import lineage_appearance as la               # noqa: E402
from cognisom.visualization import run_usd                                # noqa: E402
from cognisom.visualization.lineage_appearance import (                   # noqa: E402
    ENGINE_LINEAGES, UNRESOLVED_PROTOTYPE, appearance_for, is_citation,
    mesh_type_for_cell_token)
from cognisom.visualization.morphology import BUILDERS, build_parts       # noqa: E402
from cognisom.visualization.run_usd import (                              # noqa: E402
    CELLS_PATH, IMMUNE_PATH, write_run)

#: The lineages a run can show that must be told apart by shape or size.
#: macrophage_M1 is absent on purpose: the library's "Macrophage" and
#: "Macrophage M1" rows carry the same mesh, the same measured radius and
#: the same colour, so an unpolarised and an M1 macrophage are one drawn
#: object until the library says otherwise.
DISTINCT = ("normal", "cancer", "T_cell", "NK_cell", "macrophage", "macrophage_M2")


def _store():
    from cognisom.library.store import EntityStore
    with contextlib.redirect_stdout(io.StringIO()):
        return EntityStore()


def _library_radius(app):
    """The ``radius_um`` of the row an appearance resolved to, or None."""
    entity = _store().find_entity_by_name(app.entity_name, app.entity_type)
    assert entity is not None, (app.key, app.entity_name, app.entity_type)
    return get_parameter(entity, "radius_um")


def _library_says_measured(app) -> bool:
    """The ground truth ``cited`` has to agree with: the library row holds
    a MEASURED radius with a citation. (Asserting ``cited`` against the
    string the resolver built from the same row was a tautology.)"""
    radius = _library_radius(app)
    return (radius is not None and radius.provenance is Provenance.MEASURED
            and bool(radius.citation))


# ── the resolver ───────────────────────────────────────────────────


def test_every_engine_lineage_resolves_to_a_library_entity_that_exists():
    store = _store()
    for key in ENGINE_LINEAGES:
        app = appearance_for(key)
        assert app.key == key
        assert app.entity_name, f"{key} resolved to no entity"
        assert store.find_entity_by_name(app.entity_name) is not None, (key, app.entity_name)


def test_every_resolved_mesh_has_a_builder():
    for key in ENGINE_LINEAGES:
        app = appearance_for(key)
        assert app.mesh_type in BUILDERS, (key, app.mesh_type)
        assert app.has_builder and app.prototype == app.mesh_type


def test_cited_means_the_library_row_holds_a_measured_radius_with_a_citation():
    """Against the library, not against the resolver's own string."""
    for key in ENGINE_LINEAGES:
        app = appearance_for(key)
        assert app.size_source.strip(), key
        measured = _library_says_measured(app)
        assert app.cited == measured, (key, app.size_source)
        if measured:
            radius = _library_radius(app)
            assert app.size_source == f"radius_um {radius.citation}"
            assert app.diameter_um == pytest.approx(2.0 * radius.value)
            assert app.size_grade == radius.provenance.value == "measured"
            assert app.size_convention, f"{key} is cited and says nothing about what was measured"
        else:
            assert app.size_source.startswith("estimate:")
            assert app.size_grade == "" and app.size_convention == ""
    assert is_citation("radius_um PMID:9400735")
    assert not is_citation("radius_um placeholder")
    assert not is_citation("scale_um (no provenance)")
    assert not is_citation("estimate: 10 um fallback")


def test_the_docstring_table_is_what_the_library_says():
    """The dated table in lineage_appearance's docstring is regenerated
    text; when the library moves, this fails until it is pasted over."""
    doc = " ".join(la.__doc__.split())
    table = la.appearance_table_text()
    assert table.count("\n") == len(ENGINE_LINEAGES) - 1
    for line in table.splitlines():
        assert " ".join(line.split()) in doc, f"docstring table is stale: {line!r}"


def test_the_lineages_a_run_can_show_differ_in_shape_or_size():
    apps = {k: appearance_for(k) for k in DISTINCT}
    for a, b in combinations(DISTINCT, 2):
        x, y = apps[a], apps[b]
        assert x.mesh_type != y.mesh_type or abs(x.diameter_um - y.diameter_um) > 0.5, (
            f"{a} and {b} would be told apart only by colour: "
            f"both {x.mesh_type} at {x.diameter_um} um")


def test_the_engine_s_normal_cell_is_drawn_as_luminal_and_says_so():
    app = appearance_for("normal")
    assert app.entity_name == "Luminal epithelial cell"
    assert app.ontology_term == "", "drawn as luminal; the identity is not asserted"
    assert "PSA" in app.note and "drawn as" in app.note
    # The still exporters keep their contract: there, a state token has
    # no shape, and 'normal' is a state (and the default for an untyped
    # cell).
    assert mesh_type_for_cell_token("normal") is None
    assert mesh_type_for_cell_token("dead") is None
    assert mesh_type_for_cell_token("cancer") == appearance_for("cancer").mesh_type


def test_macrophage_polarization_selects_the_entity():
    m1 = appearance_for("macrophage", polarization=-1.0)
    m2 = appearance_for("macrophage", polarization=+1.0)
    m0 = appearance_for("macrophage", polarization=0.0)
    assert (m1.entity_name, m2.entity_name, m0.entity_name) == (
        "Macrophage M1", "Macrophage M2", "Macrophage")
    assert (m1.key, m2.key, m0.key) == ("macrophage_M1", "macrophage_M2", "macrophage")
    # the cutoffs are strict, as in cell2sentence
    assert appearance_for("macrophage", polarization=-0.3).key == "macrophage"
    assert appearance_for("macrophage", polarization=0.3).key == "macrophage"
    assert appearance_for("macrophage").key == "macrophage"
    assert appearance_for("macrophage_M2") == m2
    assert m1.mesh_type != m2.mesh_type, "M1 and M2 must differ by shape, not only size"


def test_tokens_are_case_insensitive_and_an_unknown_one_is_an_estimate():
    assert appearance_for("nk_CELL").key == "NK_cell"
    assert appearance_for("T_CELL") == appearance_for("t_cell")
    unknown = appearance_for("plasmoid")
    assert unknown.entity_name is None and unknown.mesh_type is None
    assert not unknown.has_builder and unknown.prototype == UNRESOLVED_PROTOTYPE
    assert unknown.size_source.startswith("estimate:") and not unknown.cited
    assert unknown.diameter_um == la.ESTIMATE_DIAMETER_UM


def test_bio_usd_tokens_and_composer_names_go_through_the_same_resolver():
    from cognisom.biousd.scene_adapter import (
        LIBRARY_ENTITY_FOR_PROTOTYPE, mesh_type_for_prototype)
    from cognisom.visualization.exporters import cell_mesh_type

    assert LIBRARY_ENTITY_FOR_PROTOTYPE is la.LIBRARY_ENTITY_FOR_PROTOTYPE
    assert (mesh_type_for_prototype("macrophage_m2")
            == appearance_for("macrophage_m2").mesh_type == "elongated_spindle")
    assert mesh_type_for_prototype("generic_cell") is None
    assert appearance_for("luminal").entity_name == "Luminal epithelial cell"
    assert appearance_for("endothelial").mesh_type == "flat_squamous"
    assert appearance_for("immune").mesh_type == appearance_for("T_cell").mesh_type

    class _Cell:
        cell_type = "cancer"
    assert cell_mesh_type(_Cell()) == appearance_for("cancer").mesh_type


def test_the_appearance_table_lists_every_engine_lineage_with_a_source():
    table = la.appearance_table()
    assert set(table) == set(ENGINE_LINEAGES)
    for key, rec in table.items():
        assert rec["entity"] and rec["mesh_type"] and rec["size_source"], key
        assert rec["cited"] == _library_says_measured(appearance_for(key)), key
        assert bool(rec["size_convention"]) == bool(rec["size_grade"]) == rec["cited"], key
        assert rec["prototype"] == rec["mesh_type"]


# ── silhouettes ────────────────────────────────────────────────────


def _descriptors(mesh_type: str) -> dict:
    parts, fell_back = build_parts(mesh_type, scale_um=1.0, seed=0, grid=PROTOTYPE_GRID)
    assert not fell_back, mesh_type
    parts = _normalise_to_unit_diameter(parts)
    vertices, _ = parts["body"]
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    # Roughness: radial scatter after each axis is scaled to its own
    # half-extent, so an ellipsoid scores like a sphere and only
    # ruffles and pseudopods score high.
    unit = (vertices - vertices.mean(axis=0)) / (extent / 2.0)
    radial = np.linalg.norm(unit, axis=1)
    return {"parts": frozenset(parts),
            "aspect_y": float(extent[1] / extent[0]),
            "aspect_z": float(extent[2] / extent[0]),
            "roughness": float(radial.std() / radial.mean()),
            "triangles": int(sum(len(t) for _, t in parts.values()))}


def _distinguishable(a: dict, b: dict) -> bool:
    return (a["parts"] != b["parts"]
            or abs(a["aspect_y"] - b["aspect_y"]) > 0.15
            or abs(a["aspect_z"] - b["aspect_z"]) > 0.15
            or abs(a["roughness"] - b["roughness"]) > 0.04)


def test_the_prototypes_differ_in_silhouette_not_only_in_size():
    meshes = {k: appearance_for(k).mesh_type for k in DISTINCT}
    desc = {m: _descriptors(m) for m in set(meshes.values())}
    for a, b in combinations(DISTINCT, 2):
        if meshes[a] == meshes[b]:
            continue                       # told apart by size; none today
        assert _distinguishable(desc[meshes[a]], desc[meshes[b]]), (
            f"{a} ({meshes[a]}) and {b} ({meshes[b]}) have the same silhouette: "
            f"{desc[meshes[a]]} vs {desc[meshes[b]]}")
    # the numbers the thresholds rest on
    assert desc["smooth_sphere"]["roughness"] < 0.03 < desc["irregular_ruffled"]["roughness"]
    assert desc["ruffled_pseudopod"]["aspect_z"] < 0.8
    assert desc["elongated_spindle"]["aspect_z"] < 0.5
    assert desc["polarized_columnar"]["aspect_z"] > 1.5


def test_the_run_exporter_meshes_at_a_grid_where_a_spindle_is_a_spindle():
    """At grid 8 the spindle is a blob; the exporter's grid must resolve it."""
    parts, _ = build_parts("elongated_spindle", scale_um=1.0, seed=0, grid=run_usd.PROTOTYPE_GRID)
    vertices, _ = _normalise_to_unit_diameter(parts)["body"]
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    assert float(extent.min() / extent.max()) < 0.5
    assert run_usd.PROTOTYPE_GRID >= 16


# ── a hand-built three-frame run ───────────────────────────────────

#: Distance between the two cells that overlap in the fixture, in um.
NEAR = 5.0


def _snapshot(t, cells, immune):
    """``cells``: (id, type, position); ``immune``: (id, type, position, polarization)."""
    return CellSnapshot(
        time=float(t),
        cell_positions=np.array([p for _, _, p in cells], dtype=float),
        cell_types=[ty for _, ty, _ in cells],
        cell_phases=["G1"] * len(cells),
        cell_oxygen=[0.2] * len(cells),
        cell_mhc1=[1.0] * len(cells),
        cell_ids=[i for i, _, _ in cells],
        cell_glucose=[5.0] * len(cells),
        cell_atp=[1000.0] * len(cells),
        cell_age=[1.0] * len(cells),
        immune_positions=np.array([p for _, _, p, _ in immune], dtype=float),
        immune_types=[ty for _, ty, _, _ in immune],
        immune_activated=[False] * len(immune),
        immune_ids=[i for i, _, _, _ in immune],
        immune_polarization=[pol for _, _, _, pol in immune])


def _three_frames():
    """Cell 1 is normal throughout; cell 2 is normal on frames 0-1 and
    cancer on frame 2, 5 um from cell 1. Immune cells 10 (T), 11 (NK) and
    12 (a macrophage at -1 then +1) sit 200 um apart from everything."""
    a, b = (0.0, 0.0, 0.0), (NEAR, 0.0, 0.0)
    frames = []
    for k, (type_b, pol) in enumerate([("normal", -1.0), ("normal", -1.0), ("cancer", 1.0)]):
        frames.append(_snapshot(
            0.5 * k,
            cells=[(1, "normal", a), (2, type_b, b)],
            immune=[(10, "T_cell", (200.0, 0.0, 0.0), 0.0),
                    (11, "NK_cell", (400.0, 0.0, 0.0), 0.0),
                    (12, "macrophage", (600.0, 0.0, 0.0), pol)]))
    return frames


@pytest.fixture(scope="module")
def handmade(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("handmade") / "three.usdc")
    report = write_run(path, None, snapshots=_three_frames(), events=[])
    return path, report, Usd.Stage.Open(path)


def _instancer(stage, path):
    return UsdGeom.PointInstancer(stage.GetPrimAtPath(path))


def _proto_name(instancer, index) -> str:
    targets = instancer.GetPrototypesRel().GetTargets()
    return str(targets[int(index)]).rsplit("/", 1)[-1]


def test_a_transformed_cell_changes_prototype_and_size_on_its_frame(handmade):
    _, _, stage = handmade
    inst = _instancer(stage, CELLS_PATH)
    api = UsdGeom.PrimvarsAPI(inst.GetPrim())
    ids = list(api.GetPrimvar("bio:cell:id").Get())
    i = ids.index(2)
    normal, cancer = appearance_for("normal"), appearance_for("cancer")
    for k in (0, 1):
        assert _proto_name(inst, inst.GetProtoIndicesAttr().Get(float(k))[i]) == normal.prototype
        assert inst.GetScalesAttr().Get(float(k))[i][0] == pytest.approx(normal.diameter_um)
        assert api.GetPrimvar("cognisom:cell:appearance").Get(float(k))[i] == "normal"
    assert _proto_name(inst, inst.GetProtoIndicesAttr().Get(2.0)[i]) == cancer.prototype
    assert inst.GetScalesAttr().Get(2.0)[i][0] == pytest.approx(cancer.diameter_um)
    assert api.GetPrimvar("cognisom:cell:appearance").Get(2.0)[i] == "cancer"
    # the un-timed default is the last frame, and the type token is as before
    assert _proto_name(inst, inst.GetProtoIndicesAttr().Get()[i]) == cancer.prototype
    assert api.GetPrimvar("bio:cell:type").Get()[i] == "cancer"
    assert api.GetPrimvar("cognisom:cell:typeAtFrame").Get(0.0)[i] == "normal"
    colour = api.GetPrimvar("displayColor")
    assert tuple(colour.Get()[i]) == pytest.approx(cancer.colour)
    assert tuple(colour.Get()[ids.index(1)]) == pytest.approx(normal.colour)
    # and the colour follows the shape on its frame: it was the last
    # frame's colour on every frame, so this cell wore the cancer colour
    # while its shape and its appearance key said normal
    assert normal.colour != cancer.colour
    assert colour.GetTimeSamples() == [0.0, 1.0, 2.0]
    for k in (0, 1):
        assert tuple(colour.Get(float(k))[i]) == pytest.approx(normal.colour), k
    assert tuple(colour.Get(2.0)[i]) == pytest.approx(cancer.colour)


def test_a_macrophage_crossing_the_cutoff_changes_prototype_on_its_frame(handmade):
    _, _, stage = handmade
    inst = _instancer(stage, IMMUNE_PATH)
    api = UsdGeom.PrimvarsAPI(inst.GetPrim())
    ids = list(api.GetPrimvar("bio:cell:id").Get())
    m = ids.index(12)
    m1 = appearance_for("macrophage", polarization=-1.0)
    m2 = appearance_for("macrophage", polarization=+1.0)
    assert _proto_name(inst, inst.GetProtoIndicesAttr().Get(0.0)[m]) == m1.prototype
    assert _proto_name(inst, inst.GetProtoIndicesAttr().Get(1.0)[m]) == m1.prototype
    assert _proto_name(inst, inst.GetProtoIndicesAttr().Get(2.0)[m]) == m2.prototype
    assert m1.prototype != m2.prototype
    assert inst.GetScalesAttr().Get(2.0)[m][0] == pytest.approx(m2.diameter_um)
    assert api.GetPrimvar("cognisom:cell:appearance").Get(0.0)[m] == "macrophage_M1"
    assert api.GetPrimvar("cognisom:cell:appearance").Get(2.0)[m] == "macrophage_M2"
    assert api.GetPrimvar("cognisom:immune:polarization").Get(2.0)[m] == pytest.approx(1.0)
    assert set(api.GetPrimvar("bio:cell:type").Get()) == {"immune"}
    assert api.HasPrimvar("cognisom:immune:type")
    # the colour flips with the shape, on the frame
    assert m1.colour != m2.colour
    colour = api.GetPrimvar("displayColor")
    assert tuple(colour.Get(0.0)[m]) == pytest.approx(m1.colour)
    assert tuple(colour.Get(2.0)[m]) == pytest.approx(m2.colour)
    assert tuple(colour.Get()[m]) == pytest.approx(m2.colour)


def test_t_and_nk_cells_get_different_prototypes_and_colours(handmade):
    _, _, stage = handmade
    inst = _instancer(stage, IMMUNE_PATH)
    api = UsdGeom.PrimvarsAPI(inst.GetPrim())
    ids = list(api.GetPrimvar("bio:cell:id").Get())
    t, nk = ids.index(10), ids.index(11)
    idx = inst.GetProtoIndicesAttr().Get(0.0)
    assert idx[t] != idx[nk]
    assert _proto_name(inst, idx[t]) == appearance_for("T_cell").prototype
    assert _proto_name(inst, idx[nk]) == appearance_for("NK_cell").prototype
    scales = inst.GetScalesAttr().Get(0.0)
    assert scales[t][0] == pytest.approx(appearance_for("T_cell").diameter_um)
    assert scales[nk][0] == pytest.approx(appearance_for("NK_cell").diameter_um)
    colours = api.GetPrimvar("displayColor").Get()
    assert tuple(colours[t]) != tuple(colours[nk])


def test_the_prototypes_are_library_meshes_shared_by_both_populations(handmade):
    _, report, stage = handmade
    cells, immune = _instancer(stage, CELLS_PATH), _instancer(stage, IMMUNE_PATH)
    targets = [str(t) for t in cells.GetPrototypesRel().GetTargets()]
    assert targets == [str(t) for t in immune.GetPrototypesRel().GetTargets()] == report.prototypes
    for target in targets:
        prim = stage.GetPrimAtPath(target)
        assert prim.HasAPI(UsdShade.MaterialBindingAPI), f"{target} has no material"
        if target.endswith(UNRESOLVED_PROTOTYPE):
            assert prim.IsA(UsdGeom.Sphere)
            continue
        assert any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim)), target
    names = {t.rsplit("/", 1)[-1] for t in targets}
    expected = {appearance_for(k).prototype for k in ENGINE_LINEAGES} | {UNRESOLVED_PROTOTYPE}
    assert names == expected


def test_the_stage_records_what_it_drew(handmade):
    _, report, stage = handmade
    raw = stage.GetRootLayer().customLayerData["cognisom:appearance"]
    assert isinstance(raw, str), "a Python list will not pack into .usdc; the record is JSON"
    record = json.loads(raw)
    assert set(record) == {"normal", "cancer", "T_cell", "NK_cell", "macrophage_M1", "macrophage_M2"}
    assert record == report.appearance
    assert record["normal"]["count_last_frame"] == 1
    assert record["cancer"]["count_last_frame"] == 1
    assert record["macrophage_M1"]["count_last_frame"] == 0
    assert record["macrophage_M2"]["count_last_frame"] == 1
    for key, rec in record.items():
        assert rec["entity"] and rec["mesh_type"] and rec["size_source"], key
        assert rec["cited"] == _library_says_measured(appearance_for(key)), key
        # A cited size says what was measured: a reader of the stage can
        # tell the 10.7 um cancer cell is a circulating tumour cell in
        # blood, the T cell a 72 h blast, the macrophage alveolar.
        if rec["cited"]:
            assert rec["size_convention"] and rec["size_grade"] == "measured", key
            assert "PMID:" in rec["size_convention"], key
        else:
            assert rec["size_convention"] == "" and rec["size_grade"] == "", key
    assert "circulating" in record["cancer"]["size_convention"]
    assert "alveolar" in record["macrophage_M1"]["size_convention"]
    assert set(report.unsourced) == {k for k, r in record.items() if not r["cited"]}
    # The record's term is the one the stage asserts for the token, not
    # the entity's NCIT disease term.
    inst = _instancer(stage, CELLS_PATH)
    api = UsdGeom.PrimvarsAPI(inst.GetPrim())
    i = list(api.GetPrimvar("bio:cell:id").Get()).index(2)
    assert (record["cancer"]["ontology_term"]
            == api.GetPrimvar("bio:cell:ontologyTerm").Get()[i] == "CL:0001064")
    text = report.describe()
    assert "shapes:" in text and "overlap:" in text
    assert "mean-extent radii" in text
    assert not hasattr(run_usd, "DEFAULT_DIAMETER_UM")
    assert not hasattr(run_usd, "IMMUNE_DIAMETER_UM")


def test_the_overlap_metric_is_the_two_cells_inside_each_other(handmade):
    _, report, _ = handmade
    r1, r2 = appearance_for("normal").radius_um, appearance_for("cancer").radius_um
    assert r1 + r2 > NEAR, "the fixture's premise: these two overlap"
    assert report.overlap["cells"] == 5
    assert report.overlap["fraction"] == pytest.approx(2 / 5)
    assert report.overlap["mean_depth_um"] == pytest.approx(r1 + r2 - NEAR, abs=1e-3)


def test_the_overlap_metric_agrees_with_brute_force():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 60, (300, 3))
    radii = rng.uniform(3, 10, 300)
    tree = run_usd.overlap_metric(pts, radii)
    dist, nearest = run_usd._nearest_brute(pts, chunk=64)
    depth = radii + radii[nearest] - dist
    assert tree["fraction"] == pytest.approx(float((depth > 0).mean()))
    assert tree["mean_depth_um"] == pytest.approx(float(depth[depth > 0].mean()))
    assert run_usd.overlap_metric(pts[:1], radii[:1]) == {"fraction": 0.0, "mean_depth_um": 0.0, "cells": 1.0}


def test_the_handmade_timeline_validates_as_bio_usd(handmade):
    from cognisom.biousd.standard.validate import validate
    path, _, _ = handmade
    report = validate(path)
    assert report.ok, [str(i) for i in report.issues]


# ── every frame, not only the one the validator reads ──────────────


def _every_frame_is_consistent(inst, n_frames: int) -> None:
    """On every time sample and at the default: the instance arrays and
    the per-frame primvars are the population's length, every protoIndex
    names a prototype, every diameter is positive and invisibleIds are
    instance indices. The Bio-USD validator reads only the default."""
    api = UsdGeom.PrimvarsAPI(inst.GetPrim())
    n = len(api.GetPrimvar("bio:cell:id").Get())
    n_proto = len(inst.GetPrototypesRel().GetTargets())
    frames = [float(k) for k in range(n_frames)]
    for sampled in (inst.GetProtoIndicesAttr(), inst.GetScalesAttr(),
                    inst.GetPositionsAttr(), inst.GetInvisibleIdsAttr(),
                    inst.GetExtentAttr(), api.GetPrimvar("displayColor"),
                    api.GetPrimvar("cognisom:cell:appearance")):
        assert sampled.GetTimeSamples() == frames, sampled.GetName()
    for t in [Usd.TimeCode.Default()] + [Usd.TimeCode(f) for f in frames]:
        idx = list(inst.GetProtoIndicesAttr().Get(t))
        scales = inst.GetScalesAttr().Get(t)
        assert len(idx) == len(scales) == len(inst.GetPositionsAttr().Get(t)) == n, t
        assert len(api.GetPrimvar("displayColor").Get(t)) == n, t
        assert len(api.GetPrimvar("cognisom:cell:appearance").Get(t)) == n, t
        assert 0 <= min(idx) and max(idx) < n_proto, t
        assert min(s[0] for s in scales) > 0, t
        assert set(inst.GetInvisibleIdsAttr().Get(t) or []) <= set(range(n)), t


def _visible_bounds_lie_inside_the_extent(inst, n_frames: int) -> None:
    """On every frame the computed world bounds of the VISIBLE instances
    sit inside the authored extent for that frame, and the extent is
    not a box around the world either: within two diameters of them."""
    n = len(inst.GetPositionsAttr().Get())
    for k in range(n_frames):
        t = Usd.TimeCode(float(k))
        hidden = set(inst.GetInvisibleIdsAttr().Get(t) or [])
        shown = [i for i in range(n) if i not in hidden]
        if not shown:
            continue
        boxes = UsdGeom.BBoxCache(t, [UsdGeom.Tokens.default_]).ComputePointInstanceWorldBounds(
            inst, shown)
        lo = np.min([list(b.ComputeAlignedRange().GetMin()) for b in boxes], axis=0)
        hi = np.max([list(b.ComputeAlignedRange().GetMax()) for b in boxes], axis=0)
        ext_lo, ext_hi = (np.array(list(corner)) for corner in inst.GetExtentAttr().Get(t))
        assert np.all(lo >= ext_lo - 1e-3) and np.all(hi <= ext_hi + 1e-3), (
            f"frame {k}: instances span {lo}..{hi}, extent {ext_lo}..{ext_hi}")
        slack = 2.0 * max(s[0] for s in inst.GetScalesAttr().Get(t))
        assert np.all(ext_lo >= lo - slack) and np.all(ext_hi <= hi + slack), k


def test_every_frame_s_arrays_are_the_population_s_length(handmade):
    _, report, stage = handmade
    for path in (CELLS_PATH, IMMUNE_PATH):
        _every_frame_is_consistent(_instancer(stage, path), report.frames)


def test_the_extent_contains_every_visible_instance_on_every_frame(handmade):
    """A pad of diameter/2 round the positions was exact when every
    prototype was a sphere; a 15 um columnar cell is 24 um tall and a
    21.2 um spindle 41.7 um long, and both stuck out of it."""
    _, report, stage = handmade
    for path in (CELLS_PATH, IMMUNE_PATH):
        _visible_bounds_lie_inside_the_extent(_instancer(stage, path), report.frames)
    cells = _instancer(stage, CELLS_PATH)
    tall = appearance_for("normal")
    assert tall.prototype == "polarized_columnar"
    lo, hi = cells.GetExtentAttr().Get(0.0)
    assert hi[2] - lo[2] > 1.5 * tall.diameter_um, "the columnar cell is taller than its diameter"


# ── a real run ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ring_run(tmp_path_factory):
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    np.random.seed(3)
    runner = EngineRunner(tissue="ring", dt=0.05, duration=2.0, scenario="Baseline",
                          snapshot_interval_hours=0.1)
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    path = str(tmp_path_factory.mktemp("ring") / "run.usdc")
    report = write_run(path, runner)
    return runner, path, report, Usd.Stage.Open(path)


def test_a_real_run_instances_at_least_four_library_meshes(ring_run):
    _, _, report, stage = ring_run
    used = set()
    for path in (CELLS_PATH, IMMUNE_PATH):
        inst = _instancer(stage, path)
        targets = [str(t) for t in inst.GetPrototypesRel().GetTargets()]
        for k in range(report.frames):
            hidden = set(inst.GetInvisibleIdsAttr().Get(float(k)) or [])
            idx = inst.GetProtoIndicesAttr().Get(float(k))
            used |= {targets[idx[i]].rsplit("/", 1)[-1]
                     for i in range(len(idx)) if i not in hidden}
    assert len(used) >= 4, used
    assert UNRESOLVED_PROTOTYPE not in used, "an engine lineage fell back to the sphere"
    for name in used:
        prim = stage.GetPrimAtPath(f"{run_usd.PROTOTYPE_ROOT}/{name}")
        assert any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim)), name


def test_a_real_run_sizes_each_lineage_from_the_library(ring_run):
    runner, _, _, stage = ring_run
    inst = _instancer(stage, CELLS_PATH)
    api = UsdGeom.PrimvarsAPI(inst.GetPrim())
    ids = list(api.GetPrimvar("bio:cell:id").Get())
    snap = runner.cell_snapshots[0]
    scales = inst.GetScalesAttr().Get(0.0)
    for j, cid in enumerate(snap.cell_ids):
        want = appearance_for(snap.cell_types[j]).diameter_um
        assert scales[ids.index(cid)][0] == pytest.approx(want), (cid, snap.cell_types[j])


def test_a_real_run_s_report_lists_its_lineages_with_sources(ring_run):
    _, _, report, _ = ring_run
    assert {"normal", "cancer", "T_cell", "NK_cell"} <= set(report.appearance)
    assert any(k.startswith("macrophage") for k in report.appearance)
    for key, rec in report.appearance.items():
        assert rec["entity"] and rec["size_source"], key
        assert rec["count_last_frame"] >= 0
        assert bool(rec["size_convention"]) == rec["cited"], key
    assert set(report.unsourced) <= set(report.appearance)
    assert report.overlap and 0.0 <= report.overlap["fraction"] <= 1.0
    assert report.overlap["cells"] == sum(r["count_last_frame"] for r in report.appearance.values())
    # On this lattice most drawn cells sit inside a neighbour, and the
    # report says so rather than letting "every lineage has its shape"
    # pass for a legible picture.
    assert report.overlap["fraction"] > run_usd.OVERLAP_LEGIBILITY_FRACTION
    assert "not legible here" in report.describe()


def test_a_real_run_s_frames_are_consistent_and_inside_their_extents(ring_run):
    """Dead cells, born cells, moving immune cells: 21 frames of each."""
    _, _, report, stage = ring_run
    for path in (CELLS_PATH, IMMUNE_PATH):
        inst = _instancer(stage, path)
        _every_frame_is_consistent(inst, report.frames)
        _visible_bounds_lie_inside_the_extent(inst, report.frames)
    cells = _instancer(stage, CELLS_PATH)
    assert inst and len(set(cells.GetInvisibleIdsAttr().Get(float(report.frames - 1)))) > 0, (
        "Baseline loses cells in 2 h, so the last frame hides some")


def test_a_real_run_still_validates(ring_run):
    from cognisom.biousd.standard.validate import validate
    _, path, _, _ = ring_run
    assert validate(path).ok
