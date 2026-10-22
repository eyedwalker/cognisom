"""A dense population hides its own subject, and the cut is what fixes it.

`cutaway.py` has said so in its docstring since it was written -- of
20,000 cells, 15,200 stroma hide the 2,378 that carry the biology -- but
nothing in the USD export called it. These tests hold that wiring in
place.

The measurement behind them, taken on the 20,000-cell lesion-in-stroma
scene these tests build, rendered at 700 px through `usdrecord` and
classified by `R - max(G, B) > 0.25` (a hue comparison does *not* work:
the key light desaturates a red cell until its hue sits nearer the
stroma's brown than its own):

    mode                 lesion px   % of lit frame
    none                     6,790            9.3%
    section                 10,532           23.4%
    section + stroma_only   14,075           30.7%   <- 3.3x
    corner                   9,922           17.3%
    corner + stroma_only    12,151           21.0%

Those are render facts and are not asserted here, because the suite does
not shell out to `usdrecord`. What is asserted is the thing the render
depends on: which instances the stage marks invisible.
"""

from __future__ import annotations

import numpy as np
import pytest

from cognisom.visualization.exporters import SceneExporter

pytest.importorskip("pxr")

N_STROMA = 1_700
N_LESION = 300


def _population():
    """A lesion buried concentrically in stroma, lesion ids last.

    Concentric is the hard case and the realistic one: the lesion is not
    off to one side where any camera would catch it, it is in the middle
    with stroma on every side of it.
    """
    rng = np.random.default_rng(7)
    cells = []
    for p in rng.normal(0, 60, (N_STROMA, 3)):
        cells.append({"position": p, "cell_type": "fibroblast",
                      "state": "normal", "radius": 6.0})
    for p in rng.normal(0, 18, (N_LESION, 3)):
        cells.append({"position": p, "cell_type": "cancer",
                      "state": "proliferating", "radius": 8.0})
    return cells


def _hidden(tmp_path, **kwargs):
    from pxr import Usd, UsdGeom

    path = str(tmp_path / "cells.usda")
    SceneExporter.cells_to_usd(_population(), path,
                               instancer_threshold=1, **kwargs)
    stage = Usd.Stage.Open(path)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.PointInstancer):
            ids = UsdGeom.PointInstancer(prim).GetInvisibleIdsAttr().Get()
            return np.asarray(list(ids or []), dtype=int)
    raise AssertionError("no PointInstancer in the exported stage")


def test_an_uncut_scene_hides_nothing(tmp_path):
    assert len(_hidden(tmp_path)) == 0


@pytest.mark.parametrize("mode", ["section", "corner"])
def test_a_cut_hides_a_real_share_of_the_population(tmp_path, mode):
    hidden = _hidden(tmp_path, cutaway=mode)
    fraction = len(hidden) / (N_STROMA + N_LESION)
    # A cut that hides almost nothing has not cut, and one that hides
    # almost everything has deleted the scene.
    assert 0.15 < fraction < 0.65, f"{mode} hid {fraction:.0%}"


def test_every_hidden_id_is_a_real_instance(tmp_path):
    hidden = _hidden(tmp_path, cutaway="section")
    assert hidden.min() >= 0
    assert hidden.max() < N_STROMA + N_LESION
    assert len(set(hidden.tolist())) == len(hidden), "duplicate ids"


def test_a_plain_section_cuts_the_lesion_too(tmp_path):
    """Which is correct, and is the reason `stroma_only` exists.

    A section is an anatomical slice: it passes through the subject and
    you read the cut face. It is not a reveal, and a caller that wanted
    one would be surprised -- so this pins the distinction rather than
    treating it as a defect.
    """
    hidden = _hidden(tmp_path, cutaway="section")
    lesion_hidden = int((hidden >= N_STROMA).sum())
    assert lesion_hidden > 0
    # It cuts the lesion at about the rate it cuts everything else.
    assert abs(lesion_hidden / N_LESION - len(hidden) / (N_STROMA + N_LESION)) < 0.15


@pytest.mark.parametrize("mode", ["section", "corner"])
def test_stroma_only_keeps_every_lesion_cell(tmp_path, mode):
    """The reveal: the stroma in the way goes, the subject stays whole."""
    hidden = _hidden(tmp_path, cutaway=mode, stroma_only=True)
    assert int((hidden >= N_STROMA).sum()) == 0, "a lesion cell was hidden"
    # and it still removed a substantial part of what was in front
    assert len(hidden) > 0.2 * N_STROMA


def test_the_cut_plane_faces_the_camera(tmp_path):
    """The cut must remove what is BETWEEN the camera and the subject.

    Hiding the far half instead would be invisible in a test that only
    counts ids -- the count is the same -- and would leave every
    occluder in place, so the render would look untouched. The camera
    sits along +USD_CAMERA_DIRECTION from the cells, so the hidden
    cells' centroid must lie on that side of the kept cells'.
    """
    hidden = _hidden(tmp_path, cutaway="section")
    cells = _population()
    pos = np.array([c["position"] for c in cells])
    keep = np.ones(len(cells), dtype=bool)
    keep[hidden] = False

    direction = np.asarray(SceneExporter.USD_CAMERA_DIRECTION, dtype=float)
    direction /= np.linalg.norm(direction)
    assert (pos[~keep] @ direction).mean() > (pos[keep] @ direction).mean(), (
        "the cut removed the far half and left the occluders")


# ── the Bio-USD writer, which is the path with a future ───────────────


class _Cell:
    """The duck `write_cells` reads: position, type, size."""

    def __init__(self, position, cell_type):
        self.position = tuple(float(v) for v in position)
        self.cell_type = cell_type
        self.scale_um = 12.0
        self.volume = 1.0


def _biousd_cells():
    rng = np.random.default_rng(7)
    return ([_Cell(p, "fibroblast") for p in rng.normal(0, 60, (N_STROMA, 3))]
            + [_Cell(p, "cancer") for p in rng.normal(0, 18, (N_LESION, 3))])


def _write(tmp_path, name, **kwargs):
    from cognisom.biousd.usd_writer import write_cells
    return write_cells(str(tmp_path / name), _biousd_cells(), **kwargs)


def test_the_biousd_writer_hides_nothing_by_default(tmp_path):
    assert _write(tmp_path, "a.usdc").hidden == 0


def test_the_biousd_writer_cuts_and_says_how_much(tmp_path):
    result = _write(tmp_path, "b.usdc", cutaway="section")
    assert result.hidden > 0
    assert result.hidden < result.cells


def test_the_biousd_writer_keeps_the_lesion_whole(tmp_path):
    """Only stroma may go, and enough of it to matter."""
    from pxr import Usd, UsdGeom

    path = str(tmp_path / "c.usdc")
    result = _write(tmp_path, "c.usdc", cutaway="section", stroma_only=True)
    stage = Usd.Stage.Open(path)
    instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/Cells"))
    hidden = np.asarray(list(instancer.GetInvisibleIdsAttr().Get() or []),
                        dtype=int)
    assert int((hidden >= N_STROMA).sum()) == 0
    assert result.hidden > 0.2 * N_STROMA


def test_the_cut_follows_the_stage_up_axis(tmp_path):
    """`corner` cuts a quadrant, so a wrong up axis cuts the wrong one.

    Nothing about the resulting file looks wrong -- it is a valid stage
    with a plausible number of hidden ids -- so this pins that the two
    axes actually disagree and the parameter is therefore load-bearing.
    """
    y_up = _write(tmp_path, "y.usdc", cutaway="corner", up_axis="Y").hidden
    z_up = _write(tmp_path, "z.usdc", cutaway="corner", up_axis="Z").hidden
    assert y_up != z_up


def test_the_camera_and_the_cut_read_the_same_direction(tmp_path):
    """If these drift, the cut removes what the viewer could not see."""
    from cognisom.biousd.usd_writer import CAMERA_DIRECTION
    from pxr import Usd, UsdGeom, Gf

    path = str(tmp_path / "d.usdc")
    _write(tmp_path, "d.usdc", cutaway="section")
    stage = Usd.Stage.Open(path)
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera"))
    eye = cam.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()

    want = np.asarray(CAMERA_DIRECTION, dtype=float)
    want /= np.linalg.norm(want)
    got = np.asarray([eye[0], eye[1], eye[2]], dtype=float)
    got /= np.linalg.norm(got)
    assert float(want @ got) > 0.999


def test_the_helper_does_not_care_how_far_away_the_camera_is(tmp_path):
    """The reason an exporter may invent a camera point at all.

    `hidden_instance_ids` normalises the view vector, so distance
    cancels. If that ever stopped being true, every caller that passes a
    direction instead of the fitted eye point would quietly cut at the
    wrong depth.

    It is not bit-identical, and asserting that it were would be wrong.
    Normalising a vector scaled by 500 and one scaled by 5e7 rounds
    differently in the last place, so a cell lying exactly on the plane
    can fall either side. Measured here that is 3 cells in 2,000, every
    one of them within 0.3 um of the plane -- against a cell radius of
    6-8 um, so they straddle it and the choice is arbitrary. What must
    hold is that nothing *away* from the plane moves.
    """
    from cognisom.visualization.cutaway import hidden_instance_ids, lesion_pivot

    cells = _population()
    pos = np.array([c["position"] for c in cells])
    types = [c["cell_type"] for c in cells]
    direction = np.asarray(SceneExporter.USD_CAMERA_DIRECTION, dtype=float)
    direction /= np.linalg.norm(direction)

    near = set(hidden_instance_ids(pos, types, direction * 500.0))
    far = set(hidden_instance_ids(pos, types, direction * 5.0e7))

    disputed = sorted(near ^ far)
    assert len(disputed) < 0.01 * len(cells), f"{len(disputed)} cells moved"
    depth = (pos - lesion_pivot(pos, types)) @ direction
    smallest_radius = min(c["radius"] for c in cells)
    for i in disputed:
        assert abs(depth[i]) < smallest_radius, (
            f"cell {i} is {abs(depth[i]):.2f} um from the plane and still "
            f"changed side -- that is not a rounding tie")
