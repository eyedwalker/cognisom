"""Surface and interior detail, at scales the occupancy grid cannot reach.

``morphology`` fixes a cell's silhouette by filling a grid and running
marching cubes over it, and that is also its ceiling: at 40 voxels across
a 7 um lymphocyte one voxel is 0.18 um, so a microvillus -- 0.1 um across
-- is narrower than a voxel and comes out as noise or as nothing. That is
why the library's eighteen ``smooth_sphere`` entities were bare balls. It
is not a missing builder, it is the wrong representation for the feature.

So detail is instanced over the MESH and sampled from the MASK, and three
things have to hold or it is visibly wrong rather than subtly wrong:

  * **Features sit on the surface and point out of it.** A microvillus
    pointing inward is invisible, which is worse than one pointing the
    wrong way, because nothing looks broken.
  * **The mask frame and the mesh frame are not the same frame.**
    ``build_parts`` puts a vertex at ``index * resolution`` and then
    subtracts the mean of every part's vertices -- a quantity the mask
    knows nothing about, and which has already been subtracted, so it
    cannot be read back off the result. Before this was aligned,
    mitochondria sat 0.16 um outside a lymphocyte's membrane.
  * **A count follows from a volume fraction**, not from what looks
    right. Morphometry reports a fraction and a size; choosing a count
    directly is choosing a fraction without saying so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("skimage")

from cognisom.visualization import cell_detail as cd            # noqa: E402
from cognisom.visualization.morphology import (                  # noqa: E402
    LYMPHOCYTE_NUCLEAR_RADIUS, build_masks, build_parts)

GRID = 32
SCALE = 7.0


@pytest.fixture(scope="module")
def lymphocyte():
    parts, _ = build_parts("smooth_sphere", scale_um=SCALE, seed=0, grid=GRID)
    masks, _ = build_masks("smooth_sphere", seed=0, grid=GRID)
    return parts, masks


# ── the ratio that decides whether it reads as a lymphocyte ──────────

def test_a_lymphocyte_is_mostly_nucleus(lymphocyte):
    """It was 0.095 -- a dot in an empty ball, and the main reason the
    eighteen smooth_sphere entities read as featureless spheres. A resting
    lymphocyte is described everywhere as a nucleus with a rim of
    cytoplasm."""
    _parts, masks = lymphocyte
    body = cd.mask_volume_um3(masks["body"], SCALE)
    nucleus = cd.mask_volume_um3(masks["nucleus"], SCALE)
    assert 0.55 < nucleus / body < 0.78, nucleus / body
    assert LYMPHOCYTE_NUCLEAR_RADIUS > 0.7


def test_the_cytoplasm_is_a_thin_shell(lymphocyte):
    """Which is why the frame alignment matters: there is very little room
    between the nucleus and the membrane for anything to sit in."""
    _parts, masks = lymphocyte
    cyto = masks["body"] & ~masks["nucleus"]
    body = cd.mask_volume_um3(masks["body"], SCALE)
    assert 0.2 < cd.mask_volume_um3(cyto, SCALE) / body < 0.45


# ── surface sampling ─────────────────────────────────────────────────

def test_points_land_on_the_surface(lymphocyte):
    parts, _ = lymphocyte
    v, f = parts["body"]
    rng = np.random.default_rng(0)
    points, normals = cd.sample_surface(v, f, 400, rng)
    assert len(points) == 400 and len(normals) == 400
    # every point is within a voxel of some surface vertex
    d = np.sqrt(((points[:, None, :] - v[None, ::5, :]) ** 2).sum(-1)).min(1)
    assert d.max() < SCALE / GRID * 2, d.max()
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)


def test_sampling_is_by_area_not_by_vertex(lymphocyte):
    """Marching cubes puts vertices where the surface curves, so sampling
    vertices crowds features into the creases and leaves flat ground bare
    -- a ruffled macrophage would grow all its microvilli in the folds."""
    parts, _ = build_parts("ruffled_pseudopod", scale_um=20.0, seed=0,
                           grid=GRID)
    v, f = parts["body"]
    areas = cd.triangle_areas(v, f)
    assert areas.min() >= 0
    # The area distribution is uneven, which is what makes the two
    # weightings differ; if it were uniform this test would prove nothing.
    # Marching cubes over a voxel grid produces a limited set of triangle
    # shapes, so the spread is about threefold rather than orders.
    assert areas.max() / max(areas[areas > 0].min(), 1e-12) > 2.5

    rng = np.random.default_rng(1)
    points, _ = cd.sample_surface(v, f, 3000, rng)
    # Area weighting puts about as many points on each half of the shape
    # as that half has area.
    mid = v[:, 0].mean()
    big = areas[v[f].mean(axis=1)[:, 0] > mid].sum() / areas.sum()
    got = float((points[:, 0] > mid).mean())
    assert abs(got - big) < 0.08, (got, big)


def test_every_projection_points_away_from_the_cell(lymphocyte):
    """A microvillus pointing inward is invisible rather than wrong-looking,
    which is the worse failure: nothing appears broken."""
    parts, _ = lymphocyte
    v, f = parts["body"]
    inst = cd.projections(v, f, count=300, length_um=0.45,
                          radius_um=0.045, seed=2)
    assert len(inst) == 300

    axes = _instance_axes(inst.orientations)
    radial = inst.positions - v.mean(axis=0)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    dots = np.einsum("ij,ij->i", axes, radial)
    assert (dots > 0).all(), f"{(dots <= 0).sum()} point inward"
    assert dots.mean() > 0.8


def test_projections_vary_in_length(lymphocyte):
    """An exactly uniform brush reads as a machined object, not a cell."""
    parts, _ = lymphocyte
    v, f = parts["body"]
    inst = cd.projections(v, f, count=300, length_um=0.45, radius_um=0.045,
                          seed=3)
    lengths = inst.scales[:, 2]
    assert lengths.std() > 0.05 * lengths.mean()
    assert 0.3 < lengths.mean() / 0.45 < 1.7
    assert (lengths > 0).all()


def test_projections_stand_off_the_surface_not_inside_it(lymphocyte):
    parts, _ = lymphocyte
    v, f = parts["body"]
    inst = cd.projections(v, f, count=200, length_um=0.5, radius_um=0.05,
                          seed=4, embed=0.25)
    centre = v.mean(axis=0)
    r_surface = np.linalg.norm(v - centre, axis=1)
    r_inst = np.linalg.norm(inst.positions - centre, axis=1)
    # Centres sit around the membrane, pushed out by half the length less
    # the embedded part -- not buried, not floating.
    assert r_inst.mean() > r_surface.mean()
    assert r_inst.max() < r_surface.max() + 0.5


def test_no_surface_no_crash():
    empty_v = np.zeros((0, 3), np.float32)
    empty_f = np.zeros((0, 3), np.int32)
    rng = np.random.default_rng(0)
    pts, nrm = cd.sample_surface(empty_v, empty_f, 10, rng)
    assert len(pts) == 0 and len(nrm) == 0
    assert len(cd.projections(empty_v, empty_f, count=10, length_um=1.0,
                              radius_um=0.1)) == 0


# ── the frame, which is where the visible bug was ────────────────────

def test_the_mask_frame_matches_the_mesh(lymphocyte):
    """``build_parts`` subtracts the mean of every part's vertices, so the
    mesh is centred on a quantity the mask cannot see. Aligning the two
    bounding boxes recovers it."""
    parts, masks = lymphocyte
    v, _f = parts["body"]
    resolution, offset = cd.mask_frame(masks["body"], v, SCALE)
    assert resolution == pytest.approx(SCALE / GRID)

    idx = np.argwhere(masks["body"])
    placed = idx * resolution + offset
    # The mask's box and the mesh's box agree to well under a voxel.
    for axis in range(3):
        assert abs(placed[:, axis].min() - v[:, axis].min()) < resolution
        assert abs(placed[:, axis].max() - v[:, axis].max()) < resolution


def test_organelles_stay_inside_the_membrane(lymphocyte):
    """The regression. With a lymphocyte's cytoplasm a thin shell under the
    membrane, an unaligned frame put mitochondria 0.16 um outside the
    cell."""
    parts, masks = lymphocyte
    v, _f = parts["body"]
    frame = cd.mask_frame(masks["body"], v, SCALE)
    cyto = masks["body"] & ~masks["nucleus"]

    inst = cd.organelles(cyto, scale_um=SCALE, count=120, radius_um=0.18,
                         seed=5, frame=frame)
    assert len(inst) == 120

    r_body = np.linalg.norm(v - v.mean(axis=0), axis=1).max()
    r_org = np.linalg.norm(inst.positions - v.mean(axis=0), axis=1)
    # Within one voxel of the membrane at worst. Sub-voxel is the accuracy
    # available: a point is drawn from a voxel and jittered inside it, so
    # the floor on any placement error is the voxel itself. Before the
    # frames were aligned the error was 0.16 um -- most of a voxel at this
    # resolution, and plainly visible as mitochondria outside the cell.
    voxel = SCALE / GRID
    assert r_org.max() <= r_body + voxel, (r_org.max(), r_body, voxel)
    # And the bulk sits properly inside, not pressed against the membrane.
    assert np.median(r_org) < r_body * 0.97


def test_the_frame_matters_most_on_an_asymmetric_cell():
    """A sphere hides this bug: it is symmetric about the grid centre, so
    the unaligned fallback happens to coincide and nothing looks wrong.
    A plasma cell does not -- its nucleus is eccentric by definition, so
    the body is off-centre in the grid and the two frames genuinely
    disagree. Measured here, the unaligned fallback displaces the
    interior by 0.66 um on a 14 um cell -- about three voxels -- and
    leaves 13% of it outside the cell altogether.
    """
    scale, grid = 14.0, GRID
    parts, _ = build_parts("eccentric_nucleus_er", scale_um=scale, seed=0,
                           grid=grid)
    masks, _ = build_masks("eccentric_nucleus_er", seed=0, grid=grid)
    v, _f = parts["body"]
    frame = cd.mask_frame(masks["body"], v, scale)
    rng = np.random.default_rng(7)

    aligned = cd.sample_interior(masks["body"], 500, rng, scale, frame=frame)
    rng = np.random.default_rng(7)
    fallback = cd.sample_interior(masks["body"], 500, rng, scale)

    # The two placements differ by much more than a voxel, so the frame is
    # doing real work rather than agreeing by coincidence.
    voxel = scale / grid
    assert np.abs(aligned.mean(axis=0) - fallback.mean(axis=0)).max() > voxel

    # Then the question that matters: is the point actually in the cell?
    # Read each world point back into the mesh's own frame -- which
    # test_the_mask_frame_matches_the_mesh checks independently against
    # the mesh's bounds -- and ask the mask. Points placed with the
    # unaligned frame land outside the cell; that is the bug, and it is
    # what was visible as mitochondria floating off a membrane.
    resolution, offset = frame
    body = masks["body"]

    def fraction_inside(points):
        idx = np.rint((np.asarray(points, np.float64) - offset) / resolution)
        idx = idx.astype(int)
        ok = ((idx >= 0) & (idx < np.array(body.shape))).all(axis=1)
        hit = np.zeros(len(idx), bool)
        inside = idx[ok]
        hit[ok] = body[inside[:, 0], inside[:, 1], inside[:, 2]]
        return float(hit.mean())

    # Measured: aligned puts essentially everything inside; unaligned
    # leaves 13% of the interior outside the cell.
    assert fraction_inside(aligned) > 0.98, fraction_inside(aligned)
    assert fraction_inside(fallback) < 0.90, fraction_inside(fallback)

    # And every aligned point is inside the mesh's own bounds.
    lo, hi = v.min(axis=0), v.max(axis=0)
    assert ((aligned >= lo - voxel) & (aligned <= hi + voxel)).all()


def test_organelles_avoid_the_nucleus(lymphocyte):
    """They are placed in the cytoplasm, which is the body minus the
    nucleus -- if the mask subtraction were dropped they would sit inside
    the nucleus, where they would be hidden rather than obviously wrong."""
    parts, masks = lymphocyte
    v, _f = parts["body"]
    nv, _nf = parts["nucleus"]
    frame = cd.mask_frame(masks["body"], v, SCALE)
    cyto = masks["body"] & ~masks["nucleus"]
    inst = cd.organelles(cyto, scale_um=SCALE, count=150, radius_um=0.15,
                         seed=6, frame=frame)
    # The nucleus fills most of the cell, so a uniform placement over the
    # whole body would put the great majority inside it.
    nucleus_r = np.linalg.norm(nv - nv.mean(axis=0), axis=1).max()
    inside_nucleus = (np.linalg.norm(inst.positions - nv.mean(axis=0), axis=1)
                      < nucleus_r * 0.9).mean()
    assert inside_nucleus < 0.35, inside_nucleus


def test_an_empty_mask_places_nothing():
    mask = np.zeros((8, 8, 8), bool)
    assert len(cd.organelles(mask, scale_um=5.0, count=10,
                             radius_um=0.2)) == 0


# ── orientation maths ────────────────────────────────────────────────

def test_the_quaternion_turns_z_onto_the_direction():
    dirs = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0],
                     [1, 1, 1], [-0.3, 0.5, -0.8]], float)
    axes = _instance_axes(cd.quat_z_to(dirs))
    want = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    assert np.allclose(axes, want, atol=1e-6), axes - want


def test_the_degenerate_directions_are_handled():
    """Parallel needs no rotation; antiparallel has no unique axis, so the
    cross product supplies none and the general case divides by zero."""
    q = cd.quat_z_to(np.array([[0, 0, 1.0]]))
    assert np.allclose(q[0], [1, 0, 0, 0])
    q = cd.quat_z_to(np.array([[0, 0, -1.0]]))
    assert np.allclose(_instance_axes(q)[0], [0, 0, -1], atol=1e-6)
    # a zero vector must not produce NaN
    q = cd.quat_z_to(np.array([[0.0, 0, 0]]))
    assert np.isfinite(q).all()


# ── counts from fractions ────────────────────────────────────────────

def test_a_count_comes_from_a_volume_fraction():
    """Morphometry reports a fraction and a size; the count follows.
    Choosing a count directly is choosing a fraction without saying so."""
    import math
    volume, radius, fraction = 500.0, 0.25, 0.06
    n = cd.volume_fraction_count(fraction, volume, radius)
    one = 4.0 / 3.0 * math.pi * radius ** 3
    assert n * one == pytest.approx(fraction * volume, rel=0.02)

    # Bigger organelles, same fraction, fewer of them.
    assert cd.volume_fraction_count(fraction, volume, 0.5) < n
    assert cd.volume_fraction_count(0.0, volume, radius) == 0
    assert cd.volume_fraction_count(fraction, volume, 0.0) == 0


def test_mask_volume_is_in_cubic_micrometres():
    mask = np.zeros((20, 20, 20), bool)
    mask[:10, :10, :10] = True          # an eighth of the grid
    # 10 um across the grid, so the filled block is 5 um on a side.
    assert cd.mask_volume_um3(mask, 10.0) == pytest.approx(125.0, rel=0.01)


def _instance_axes(quaternions: np.ndarray) -> np.ndarray:
    """The +Z axis of each quaternion, which is where a cone points."""
    w, x, y, z = (np.asarray(quaternions)[:, i] for i in range(4))
    return np.stack([2 * (x * z + w * y),
                     2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)], axis=1)
