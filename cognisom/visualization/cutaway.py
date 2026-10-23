"""Cutaway views for a dense cell population — prepared for integration.

PROBLEM, measured: at 20,000 cells the exported slab renders as a solid
brick. Only the surface layer is visible and the tumour is entirely buried;
of 20,000 cells, 15,200 stroma hide the 2,378 that carry the biology. The
render is fast and clean and shows nothing.

Three approaches were built and rendered through /usr/bin/usdrecord and
looked at:

  section  cut everything in front of a plane through the lesion     <- default
  corner   remove one quadrant; keeps the block readable as a solid
  thin     shrink the stroma so the lesion shows through the gaps

`section` is the default because it mirrors this domain's own visual
convention -- it reads as a histology section -- and because the gradient it
exposes is the thing worth seeing: hypoxic core, cancer mass, immune margin,
stroma.

TWO THINGS LEARNED BY GETTING THEM WRONG FIRST, both worth keeping:

1. CUT THROUGH THE LESION, NOT THE BOUNDING BOX. The first version cut at
   the geometric centroid of all cells. That happens to work when the tumour
   is central and silently misses it entirely when it is not -- the section
   would show clean stroma and look like a healthy sample.

2. A SECTION CUTS EVERYTHING. The second version preserved all non-stroma
   cells on the grounds that they were the interesting ones. The immune
   cells in the near half then floated in front of the tumour and obscured
   exactly what the section was cut to reveal. A microtome does not spare
   the interesting cells, and neither should this.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


#: Cell types that constitute the bulk the cut removes.
STROMA_TYPES = frozenset({"normal", "stroma", "epithelial", "fibroblast"})

#: Cell types that ARE the lesion. The cut plane is positioned from these and
#: these only.
#:
#: Defining the lesion as "everything that is not stroma" looks equivalent and
#: is not: the immune compartment is scattered through the entire slab, so its
#: near face is the slab's own surface. Measured on a real 20,000-cell export,
#: that put the plane at the front face and removed 769 cells of 20,000 --
#: leaving the tumour exactly as buried as before.
LESION_TYPES = frozenset({
    "cancer", "tumor", "tumour", "malignant", "hypoxic", "dying", "necrotic",
})


def lesion_pivot(positions: np.ndarray, cell_types: Sequence[str]) -> np.ndarray:
    """Centroid of the non-stroma cells, which is where a section should cut.

    Falls back to the whole population when a run has no lesion, so a healthy
    control still sections sensibly rather than dividing by zero.
    """
    positions = np.asarray(positions, dtype=float)
    mask = _is_lesion(cell_types)
    chosen = positions[mask] if mask.any() else positions
    return chosen.mean(axis=0)


def _is_lesion(cell_types: Sequence[str]) -> np.ndarray:
    return np.array(
        [str(t).lower() in LESION_TYPES for t in cell_types], dtype=bool)


def hidden_instance_ids(
    positions: np.ndarray,
    cell_types: Sequence[str],
    camera_position: Sequence[float],
    mode: str = "section",
    *,
    offset: float | str = 0.0,
    stroma_only: bool = False,
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> list[int]:
    """Instance ids to hide for a cutaway view.

    Args:
        positions: (N, 3) cell positions.
        cell_types: N cell-type names, used to locate the lesion.
        camera_position: where the camera sits; the cut is perpendicular to
            the view direction, so the section always faces the viewer.
        mode: "section", "corner", or "none".
        offset: where the plane sits along the view axis, relative to the
            lesion centroid. Defaults to 0.0 -- a bisecting section through
            the lesion, which is the view that actually reads.

            "near_face" peels back only the stroma in front of the lesion,
            keeping every lesion cell. It sounds strictly better and renders
            worse: measured on a 20,000-cell slab it removed 3,355 cells and
            left the tumour muddy behind a residual layer of stroma, while
            the bisecting plane removed 9,994 and showed the hypoxic core
            cleanly. A roughly spherical tumour occludes its own centre, so
            keeping its near hemisphere hides the thing worth seeing. Kept as
            an option because it is the honest choice when the question is
            "how big is the lesion" rather than "what is inside it".
        stroma_only: keep every non-stroma cell regardless of the plane. This
            is NOT the default, and the docstring above says why. Note that
            it is a different knob from ``offset="near_face"`` and does not
            share its problem: the plane still bisects, so the lesion's near
            hemisphere is not left in front of its own centre. Measured on
            the 20,000-cell scene, section alone took the lesion from 9.3%
            to 23.4% of the lit frame and section with ``stroma_only`` took
            it to 30.7%, without hiding a single lesion cell.
        up: the stage's up axis. ``corner`` cuts a quadrant rather than a
            half, so it needs to know which way is up; USD stages here are
            written both Y-up and Z-up and a wrong guess silently cuts the
            wrong quadrant.

    Returns:
        Instance ids, suitable for UsdGeomPointInstancer.GetInvisibleIdsAttr.
    """
    if mode == "none":
        return []

    positions = np.asarray(positions, dtype=float)
    pivot = lesion_pivot(positions, cell_types)

    view = pivot - np.asarray(camera_position, dtype=float)
    norm = np.linalg.norm(view)
    if norm == 0.0:
        return []
    view /= norm

    # Depth along the view axis, measured from the plane. Positive is between
    # the camera and the plane, i.e. the material a section removes.
    depth = (positions - pivot) @ (-view)

    if offset == "near_face":
        # Near face of the lesion along the view axis: cut away what hides it,
        # not the lesion itself. A centroid plane discards half the tumour,
        # which is right for a section and wrong for simply seeing it.
        lesion_m = _is_lesion(cell_types)
        lesion_depth = depth[lesion_m] if lesion_m.any() else depth
        offset = float(lesion_depth.max()) if len(lesion_depth) else 0.0

    if mode == "section":
        hide = depth > offset
    elif mode == "corner":
        up_v = np.asarray(up, dtype=float)
        hide = (depth > offset) & ((positions - pivot) @ up_v > 0.0)
    else:
        raise ValueError(f"unknown cutaway mode {mode!r}")

    if stroma_only:
        is_stroma = np.array(
            [str(t).lower() in STROMA_TYPES for t in cell_types], dtype=bool)
        hide &= is_stroma

    return [int(i) for i in np.flatnonzero(hide)]


def hidden_for_camera_direction(
    positions: np.ndarray,
    cell_types: Sequence[str],
    direction: Sequence[float],
    mode: str = "section",
    *,
    offset: float | str = 0.0,
    stroma_only: bool = False,
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> list[int]:
    """:func:`hidden_instance_ids` for a camera given as a direction.

    Every exporter that wants a cut knows which way its camera faces long
    before it knows where the camera ends up, because the eye point falls
    out of a fit over the cells. :func:`hidden_instance_ids` only ever
    uses the camera to form a unit view vector, so the distance cancels
    and a point far along ``direction`` from the lesion is exactly as
    good as the real one.

    ``direction`` points *from the cells toward the camera*, matching
    ``SceneExporter.USD_CAMERA_DIRECTION``. Passing it reversed is the
    one mistake here that no test counting hidden ids can see -- the
    count is identical, every occluder stays put, and the render looks
    untouched -- so it is worth being sure of the sign.
    """
    positions = np.asarray(positions, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return []
    direction = direction / norm

    pivot = lesion_pivot(positions, cell_types)
    span = float(np.abs(positions - pivot).max()) if len(positions) else 1.0
    camera = pivot + direction * (span + 1.0) * 1000.0
    return hidden_instance_ids(positions, cell_types, camera, mode,
                               offset=offset, stroma_only=stroma_only, up=up)


def occluded_instance_ids(
    positions: np.ndarray,
    diameters: np.ndarray | float,
    direction: Sequence[float],
    *,
    keep: int = 2,
    up: Sequence[float] = (0.0, 0.0, 1.0),
    column: Optional[float] = None,
    already_hidden: Optional[Iterable[int]] = None,
) -> list[int]:
    """Instances buried behind others along the view, to hide.

    "Draw only what is potentially visible" is Mesoscale Explorer's
    third optimisation, and it matters here for a reason the plan did
    not state. Storm's render time is linear in visible instances x
    faces (measured: grid 12, 100k cells 0.8 s, 400k cells 2.8 s), and
    somewhere between ~463M and ~579M submitted triangles it stops
    drawing altogether -- exit code 0, a valid PNG, every pixel blank.
    500,000 cells at grid 12 is over that line; the same scene with a
    section cut is under it. invisibleIds is honoured before
    submission, so hiding what cannot be seen is the lever that keeps
    a scene on the right side of the cliff.

    This is a depth peel. The screen is tiled into columns `column`
    wide (default: the median cell diameter), each cell lands in one,
    and within a column only the `keep` nearest to the camera stay.
    Everything deeper is behind at least `keep` cells and, in a packed
    tissue, cannot be seen.

    It is approximate in two honest ways. A cell straddles up to four
    columns and is binned by its centre, so a cell can be kept in one
    column while a neighbour it half-covers is dropped from the next;
    `keep` = 2 absorbs that. And the body material is translucent, so
    deeper cells do contribute a little -- measured on the 500k scene
    the peel changes the frame by a few percent of pixels, which is the
    price of staying under the cliff and is reported, not hidden.

    `already_hidden` lets this run after a section cut, peeling only
    among what the cut left; otherwise the cut's own removed front layer
    would count as the occluders and nothing would be peeled.
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n == 0 or keep < 1:
        return []
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return []
    forward = -direction / norm
    up_v = np.asarray(up, dtype=float)
    right = np.cross(forward, up_v)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up_cam = np.cross(right, forward)

    d = np.asarray(diameters, dtype=float)
    if d.ndim == 0:
        d = np.full(n, float(d))
    if column is None:
        column = float(np.median(d)) if n else 1.0
    column = max(float(column), 1e-6)

    visible = np.ones(n, dtype=bool)
    if already_hidden is not None:
        hidden_idx = np.fromiter((int(i) for i in already_hidden), dtype=int)
        visible[hidden_idx] = False
    idx = np.flatnonzero(visible)
    if len(idx) == 0:
        return []
    rel = positions[idx]
    a = np.floor(rel @ right / column).astype(np.int64)
    b = np.floor(rel @ up_cam / column).astype(np.int64)
    depth = rel @ (-forward)            # larger = nearer the camera
    # One key per column; sort by column then by depth descending.
    key = (a - a.min()) * (b.max() - b.min() + 1) + (b - b.min())
    order = np.lexsort((-depth, key))
    key_sorted = key[order]
    # rank within column = position since the column's first entry
    starts = np.r_[0, np.flatnonzero(np.diff(key_sorted)) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    hide = idx[order[rank >= keep]]
    return [int(i) for i in np.sort(hide)]


def stroma_scales(
    cell_types: Sequence[str],
    *,
    stroma_scale: float = 0.34,
) -> np.ndarray:
    """Per-instance scales that shrink the stroma so the lesion shows through.

    The alternative to cutting. It keeps the whole volume visible and conveys
    the lesion's three-dimensional extent, at the cost of the stroma reading
    as dust rather than tissue. Useful as a second view, poor as the only one.
    """
    scale = np.array(
        [stroma_scale if str(t).lower() in STROMA_TYPES else 1.0 for t in cell_types],
        dtype=float,
    )
    return np.repeat(scale[:, None], 3, axis=1)
