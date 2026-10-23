"""Phase 2: a large scene you can see the lesion in, and that draws at all.

Four things were found building this, each measured on a 500,000-cell
scene through usdrecord and looked at:

* Storm's render time is linear in visible instances x faces, and
  between 639M and 690M triangles submitted by ONE prototype it stops
  drawing entirely -- exit 0, a valid PNG, every pixel blank. The
  total across prototypes is not the axis: 728M drew, 735M did not.
  The writer now predicts this and says so.

* A depth peel keeping 8 cells per half-diameter screen column hides
  85,000 of 279,000 visible cells and moves the frame by a mean of
  0.6/765 per pixel -- 8 because the body is 0.35 opaque, so 95% of
  what reaches the eye is within ln(0.05)/ln(0.65) ~ 7 layers.

* Level of detail fits on one PointInstancer as extra prototypes, so
  invisibleIds and every primvar survive it. The tier is chosen by
  projected pixel size, because distance fractions sent 158,000 cells
  at 7-9 px to grid 8.

* No prototype was unit diameter. smooth_sphere spanned 0.88 at grid
  16 and 0.78 at grid 8, so every exported cell was 12-22% too small
  and a tier swap shrank it by a further 12%. That, not LOD, was what
  made LOD look wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pxr")

from cognisom.biousd import usd_writer as uw                      # noqa: E402
from cognisom.biousd.usd_writer import (                           # noqa: E402
    CAMERA_DIRECTION, COLOUR_RAMP, _normalise_to_unit_diameter, write_cells)
from cognisom.visualization.cutaway import occluded_instance_ids   # noqa: E402
from cognisom.visualization.morphology import build_parts          # noqa: E402


class _Cell:
    def __init__(self, position, cell_type="normal", mesh_type="smooth_sphere",
                 scale_um=12.0, **extra):
        self.position = tuple(float(v) for v in position)
        self.cell_type = cell_type
        self.mesh_type = mesh_type
        self.scale_um = scale_um
        self.volume = 1.0
        for key, value in extra.items():
            setattr(self, key, value)


def _population(n_stroma=600, n_lesion=120, seed=3):
    rng = np.random.default_rng(seed)
    return ([_Cell(p) for p in rng.uniform(-80, 80, (n_stroma, 3))]
            + [_Cell(p, "cancer", "irregular_ruffled", 14.0)
               for p in rng.normal(0, 25, (n_lesion, 3))])


def _instancer(path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(path)
    return stage, UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/Cells"))


# ── prototypes are the size they claim ──────────────────────────────


@pytest.mark.parametrize("mesh_type", ["smooth_sphere", "irregular_ruffled",
                                       "multilobed_granular"])
@pytest.mark.parametrize("grid", [8, 12, 16])
def test_a_prototype_body_is_one_unit_across_at_every_grid(mesh_type, grid):
    parts = _normalise_to_unit_diameter(
        build_parts(mesh_type, scale_um=1.0, seed=0, grid=grid)[0])
    vertices, _ = parts["body"]
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    assert float(extent.mean()) == pytest.approx(1.0, abs=1e-6)
    centre = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    assert np.abs(centre).max() < 1e-9


def test_the_raw_builder_was_not_unit_sized():
    """Pins the defect, so the normaliser cannot be dropped as redundant."""
    raw, _ = build_parts("smooth_sphere", scale_um=1.0, seed=0, grid=8)
    vertices, _ = raw["body"]
    assert float((vertices.max(axis=0) - vertices.min(axis=0)).mean()) < 0.85


def test_the_nucleus_scales_with_the_body():
    raw, _ = build_parts("smooth_sphere", scale_um=1.0, seed=0, grid=12)
    norm = _normalise_to_unit_diameter(raw)
    ratio_raw = (np.ptp(raw["nucleus"][0], axis=0).mean()
                 / np.ptp(raw["body"][0], axis=0).mean())
    ratio_norm = (np.ptp(norm["nucleus"][0], axis=0).mean()
                  / np.ptp(norm["body"][0], axis=0).mean())
    assert ratio_norm == pytest.approx(ratio_raw, rel=1e-6)


# ── the depth peel ──────────────────────────────────────────────────


def test_a_stack_along_the_view_keeps_only_the_nearest():
    direction = np.array([0.0, 0.0, 1.0])
    stack = np.array([[0, 0, z] for z in (0, 10, 20, 30, 40)], float)
    assert occluded_instance_ids(stack, 10.0, direction, keep=2) == [0, 1, 2]


def test_a_row_across_the_view_hides_nothing():
    direction = np.array([0.0, 0.0, 1.0])
    row = np.array([[x, 0, 0] for x in (0, 20, 40, 60, 80)], float)
    assert occluded_instance_ids(row, 10.0, direction, keep=2) == []


def test_the_peel_runs_among_what_the_cut_left():
    """Otherwise the cut's removed front layer counts as the occluders."""
    direction = np.array([0.0, 0.0, 1.0])
    stack = np.array([[0, 0, z] for z in (0, 10, 20, 30, 40)], float)
    peeled = occluded_instance_ids(stack, 10.0, direction, keep=2,
                                   already_hidden=[4, 3])
    assert peeled == [0]


def test_the_peel_is_exposed_on_the_writer(tmp_path):
    rng = np.random.default_rng(5)
    # a dense column of stroma along the camera direction
    d = np.asarray(CAMERA_DIRECTION, float); d /= np.linalg.norm(d)
    cells = [_Cell(d * t + rng.normal(0, 1.5, 3)) for t in np.linspace(-120, 120, 400)]
    path = str(tmp_path / "peel.usdc")
    result = write_cells(path, cells, occlusion=3)
    assert result.occluded > 0
    _, instancer = _instancer(path)
    hidden = list(instancer.GetInvisibleIdsAttr().Get() or [])
    assert len(hidden) == result.hidden + result.occluded


def test_the_cut_and_the_peel_compose(tmp_path):
    """The cut keeps the lesion; the peel hides what cannot be seen.

    Those are different contracts and the second is type-blind on
    purpose: a lesion cell buried behind eight others is invisible
    whatever it is, and keeping it would only spend triangles. So the
    stroma_only guarantee is checked on the cut alone, and the composed
    export is checked for its arithmetic.
    """
    cells = _population()
    lesion = set(range(600, 720))

    cut_only = str(tmp_path / "cut.usdc")
    write_cells(cut_only, cells, cutaway="section", stroma_only=True)
    _, instancer = _instancer(cut_only)
    cut = set(int(i) for i in (instancer.GetInvisibleIdsAttr().Get() or []))
    assert cut and not (cut & lesion)

    both = str(tmp_path / "both.usdc")
    result = write_cells(both, cells, cutaway="section", stroma_only=True,
                         occlusion=2)
    _, instancer = _instancer(both)
    hidden = set(int(i) for i in (instancer.GetInvisibleIdsAttr().Get() or []))
    assert len(hidden) == result.hidden + result.occluded
    assert result.hidden == len(cut) and result.occluded > 0
    assert cut <= hidden


# ── level of detail on one instancer ────────────────────────────────


def _tiers(tmp_path, lod_pixels, name):
    result = write_cells(str(tmp_path / f"{name}.usdc"), _population(),
                         grid=16, lod=True, lod_pixels=lod_pixels)
    return result


def test_lod_is_chosen_by_projected_pixel_size(tmp_path):
    """Narrow the frame and every cell drops a tier; widen it and none do."""
    tiny = _tiers(tmp_path, 20.0, "tiny")      # ~2 px cells  -> grid 8
    wide = _tiers(tmp_path, 4000.0, "wide")    # ~400 px cells -> grid 16
    assert tiny.lod_tiers.get(8, 0) == 720 and tiny.lod_tiers.get(16, 0) == 0
    assert wide.lod_tiers.get(16, 0) == 720 and wide.lod_tiers.get(8, 0) == 0


def test_only_the_tiers_in_use_are_authored(tmp_path):
    """Every cell here is one size at one depth: one tier, two prototypes.

    The first version authored all three tiers whatever the scene --
    nine prototypes and 14,644 faces for a composed scene that used
    three and 8,694.
    """
    result = write_cells(str(tmp_path / "one.usdc"), _population(), grid=16,
                         lod=True, lod_pixels=300.0)
    assert result.lod_tiers == {16: 720}
    assert result.prototypes == 2


def test_lod_keeps_one_instancer_and_valid_indices(tmp_path):
    """Two cell sizes across a 2x depth range, so more than one tier."""
    rng = np.random.default_rng(9)
    d = np.asarray(CAMERA_DIRECTION, float); d /= np.linalg.norm(d)
    cells = []
    for t in np.linspace(-300, 300, 400):
        size = 6.0 if t > 0 else 30.0
        cells.append(_Cell(d * t + rng.normal(0, 4, 3), scale_um=size))
    path = str(tmp_path / "tiers.usdc")
    result = write_cells(path, cells, grid=16, lod=True, lod_pixels=250.0)
    stage, instancer = _instancer(path)
    instancers = [p for p in stage.Traverse() if p.GetTypeName() == "PointInstancer"]
    assert len(instancers) == 1
    protos = instancer.GetPrototypesRel().GetTargets()
    indices = np.asarray(instancer.GetProtoIndicesAttr().Get())
    assert indices.min() >= 0 and indices.max() < len(protos)
    assert sum(result.lod_tiers.values()) == 400
    assert len(result.lod_tiers) >= 2, result.lod_tiers
    # one mesh type here, so one prototype per tier in use
    assert result.prototypes == len(protos) == len(result.lod_tiers)


def test_lod_submits_fewer_triangles_than_a_flat_grid(tmp_path):
    flat = write_cells(str(tmp_path / "flat.usdc"), _population(), grid=16)
    lod = write_cells(str(tmp_path / "lod.usdc"), _population(), grid=16,
                      lod=True, lod_pixels=60.0)
    assert lod.submitted_faces < flat.submitted_faces


def test_primvars_survive_lod(tmp_path):
    path = str(tmp_path / "pv.usdc")
    result = write_cells(path, _population(), grid=16, lod=True, lod_pixels=300.0)
    assert "bio:cell:type" in result.primvars and "displayColor" in result.primvars
    from pxr import UsdGeom
    _, instancer = _instancer(path)
    api = UsdGeom.PrimvarsAPI(instancer.GetPrim())
    assert len(api.GetPrimvar("bio:cell:type").Get()) == 720


# ── colour by a variable ────────────────────────────────────────────


def _aged(n=200):
    rng = np.random.default_rng(1)
    return [_Cell(p, "cancer", age=float(a))
            for p, a in zip(rng.uniform(-40, 40, (n, 3)), np.linspace(0, 72, n))]


def test_colour_by_runs_the_ramp_from_light_to_dark(tmp_path):
    from pxr import UsdGeom
    path = str(tmp_path / "age.usdc")
    result = write_cells(path, _aged(), colour_by="age")
    assert result.colour_by == "age" and result.colour_range == (0.0, 72.0)
    _, instancer = _instancer(path)
    colours = np.asarray(UsdGeom.PrimvarsAPI(instancer.GetPrim())
                         .GetPrimvar("displayColor").Get())
    assert colours[0] == pytest.approx(COLOUR_RAMP[0], abs=1e-6)
    assert colours[-1] == pytest.approx(COLOUR_RAMP[1], abs=1e-6)


def test_colour_by_is_declared_on_the_layer_for_a_legend(tmp_path):
    path = str(tmp_path / "meta.usdc")
    write_cells(path, _aged(), colour_by="age")
    stage, _ = _instancer(path)
    data = dict(stage.GetRootLayer().customLayerData)
    assert data["bio:colourBy"] == "age"
    assert (data["bio:colourMin"], data["bio:colourMax"]) == (0.0, 72.0)


def test_a_fixed_range_is_honoured(tmp_path):
    result = write_cells(str(tmp_path / "fixed.usdc"), _aged(),
                         colour_by="age", colour_range=(0.0, 100.0))
    assert result.colour_range == (0.0, 100.0)


def test_a_missing_variable_falls_back_and_says_so(tmp_path):
    """Type colour, not a ramp over garbage, and the reason is reported."""
    result = write_cells(str(tmp_path / "none.usdc"), _aged(), colour_by="oxygen")
    assert result.colour_by is None
    assert "oxygen" in (result.colour_skipped or "")


# ── the render cliff is predicted ───────────────────────────────────


def test_a_small_scene_is_under_the_limit(tmp_path):
    result = write_cells(str(tmp_path / "small.usdc"), _population())
    assert result.over_limit == []
    assert sum(result.submitted_by_prototype.values()) == result.submitted_faces


def test_a_scene_over_the_limit_is_flagged(tmp_path, monkeypatch, caplog):
    """The frame would be blank; the writer must not be silent about it."""
    monkeypatch.setattr(uw, "STORM_DRAW_TRIANGLE_LIMIT", 10_000)
    with caplog.at_level("WARNING"):
        result = write_cells(str(tmp_path / "over.usdc"), _population())
    assert result.over_limit
    assert "OVER THE RENDER LIMIT" in result.describe()
    assert any("blank frame" in r.message for r in caplog.records)


def test_hiding_cells_brings_a_scene_back_under(tmp_path, monkeypatch):
    monkeypatch.setattr(uw, "STORM_DRAW_TRIANGLE_LIMIT", 1_200_000)
    over = write_cells(str(tmp_path / "o.usdc"), _population())
    under = write_cells(str(tmp_path / "u.usdc"), _population(),
                        cutaway="section", occlusion=2)
    assert over.over_limit and not under.over_limit
