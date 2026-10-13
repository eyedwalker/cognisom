#!/usr/bin/env python3
"""
Tissue Slab Geometry
====================

Seeding geometry for the ``TissueSlab`` scenario: a close-packed slab of
stroma with an embedded tumour nodule, perfused by a capillary lattice.

WHY THIS EXISTS
---------------
The seeding every other preset uses is a RING. ``CellularModule``'s
``_seed_ring`` places each normal cell on a circle of radius 50 um about
(100, 100, 50), so the whole "tissue" is one bead necklace 1 um thick in
radius with a 100 um hole in the middle, and the cancer cells are a
separate 30 x 30 x 10 um cloud about (120, 120, 50) that does not touch
the ring at all. Measured on the Baseline preset: the normal cells span
radius 50.00-50.97 um -- a distribution of two values -- and their
median nearest-neighbour distance is 4.5 um, taken ALONG a ring that is
100 um across. Rendered, it is a circle of beads, which is why no amount
of work in the USD exporter could make it look like tissue.

This module builds the alternative: a packed slab in which every cell is
in contact with its neighbours, with a solid tumour nodule carved out of
it off-centre, and a vessel lattice dense enough that the slab does not
asphyxiate.

THE NUMBERS ARE READ OFF THE REST OF THE ENGINE
-----------------------------------------------
* SPACING. ``TissueScaleConfig.cell_radius_um`` is 5.0 um -- the same
  radius ``SpatialModule`` uses to turn a per-cell demand into a
  volumetric uptake density, and the same one ``CellMechanicsModule``
  passes to ``set_radii``. So two cells touch at 10.0 um centre to centre
  and are still inside the E-cadherin adhesion band out to
  ``contact + CellMechanicsConfig.adhesion_range`` = 12.0 um
  (cognisom/physics/cell_mechanics.py). ``DEFAULT_SPACING_UM`` is the
  lattice constant; what matters biologically is the resulting
  nearest-neighbour DISTRIBUTION after jitter, measured by
  ``nearest_neighbour_distances``. At the shipped defaults that
  distribution is, over six seeds, median 9.2 um with the tightest pair
  at 7.0-7.5 um and the loosest at 11.4-12.4 um -- the typical cell 12%
  inside contact, the tightest pair 28% inside it, and at most one cell
  in 2,040 with its nearest neighbour past the 12.0 um adhesion cutoff.
  Nothing floats.

  Centre distances below 2R are the expected state, not an error. A
  confluent tissue has no interstitial space, so its cells cannot be
  hard spheres at 0.74 packing fraction; they deform, and the distance
  between their centres is correspondingly less than twice the
  free-cell radius. What would be wrong is a gap: a nearest neighbour
  beyond 12.0 um is a cell adhering to nothing.

* THE DOMAIN. ``SpatialModule`` defaults to a (20, 20, 10) voxel grid at
  10 um/voxel and ``MorphogenModule`` declares
  ``tissue_size = (200, 200, 100)``, so positions must stay inside
  200 x 200 x 100 um with the origin at a corner. That is a hard ceiling
  on how big a packed slab can be: 2,040 cells at an 11 um lattice
  constant occupy 165 x 156 x 63 um, which leaves a ~18 um margin and no
  room for a larger one. It is also why the slab is centred on
  (100, 100, 50) like every other module's geometry.

* VESSEL PITCH. ``Capillary.exchange_radius`` is 50 um and it is the
  LETHAL CORNER: by construction a normal cell 50 um from its nearest
  vessel sits exactly at its death threshold, and
  ``VascularModule._exchange_with_tissue`` documents the resulting
  zonation as normoxic out to 25 um and hypoxic from 25 to 50 um. Only
  the NEAREST vessel contributes -- conductances are deliberately not
  summed -- so a slab is perfused only if EVERY cell is inside ~25 um of
  a vessel, and healthy only if the typical cell is far closer. At the
  default 22 um pitch, measured over six seeds of the shipped preset:
  median 8.8-9.0 um, p95 13.4-14.4 um, worst 16.2-19.7 um. So even the
  worst-supplied cell in the slab has a predicted steady state of
  0.068-0.083 -- above the 0.05 hypoxia flag with room to spare -- and
  the slab mean is 0.144. ``perfusion_report`` measures this rather than
  trusting the paragraph, and the TissueSlab tests assert it.

The resulting density -- 26 vessels through a 165 x 63 um cross section,
~2,500 per mm^2 -- is myocardial (intercapillary distance ~20 um in
heart, against ~40-60 um in brain grey matter). It is dense tissue,
deliberately: this engine's own 25 um normoxic limit leaves no room for
a sparse network, so the slab is modelled on the best-perfused tissue in
the body rather than on an average one.

JITTER
------
Every position is jittered, because a perfect lattice renders as a
crystal and reads as fake, and because it collapses the
nearest-neighbour distance to a single number instead of a distribution.
The jitter is a fraction of the lattice constant applied independently
per axis. Note that it lowers the median nearest-neighbour distance
rather than leaving it alone: each cell has twelve neighbours at exactly
the lattice constant, and jitter takes the MINIMUM over twelve perturbed
distances, so the median falls by ~12% at the default amplitude. The
lattice constant is set with that bias already accounted for.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cognisom.core.tissue_config import TissueScaleConfig


# ── Constants read off the rest of the engine ────────────────────────

#: Cell radius in um. This project's own default, from
#: ``TissueScaleConfig.cell_radius_um``; ``SpatialModule.cell_radius_um``
#: and ``CellMechanicsModule.add_cells`` both use the same 5.0 um.
CELL_RADIUS_UM = float(
    TissueScaleConfig.__dataclass_fields__["cell_radius_um"].default
)

#: Centre-to-centre distance at which two cells are in contact.
CONTACT_DISTANCE_UM = 2.0 * CELL_RADIUS_UM

#: How far beyond contact cell-cell adhesion still acts, from
#: ``CellMechanicsConfig.adhesion_range`` in
#: cognisom/physics/cell_mechanics.py. Stated rather than imported: that
#: module pulls in the Warp backend at import time and this one must stay
#: importable with no GPU stack present.
ADHESION_RANGE_UM = 2.0

#: Furthest two cells can be apart and still adhere.
ADHESION_CUTOFF_UM = CONTACT_DISTANCE_UM + ADHESION_RANGE_UM

#: Lattice constant for the packed slab. See the module docstring: chosen
#: so that the jittered nearest-neighbour distribution lands inside
#: [0.8 * contact, adhesion cutoff], and so that 2,040 cells still fit
#: inside the 200 x 200 x 100 um domain with a margin.
DEFAULT_SPACING_UM = 11.0

#: The simulation domain, from ``MorphogenModule.tissue_size`` and the
#: default ``SpatialModule`` grid (20, 20, 10) at 10 um/voxel.
DOMAIN_UM = (200.0, 200.0, 100.0)

#: Centre of the domain, which is where every other module puts its
#: geometry (``VascularModule.initialize``, ``LymphaticModule.initialize``).
DOMAIN_CENTER_UM = (100.0, 100.0, 50.0)

#: Close-packed cell layers through the slab thickness.
DEFAULT_LAYERS = 8

#: Per-axis positional jitter, as a fraction of the lattice constant.
#: Set by rendering both: at 0.10 the lattice rows are still legible on a
#: flat face of the slab and it reads as a crystal; at 0.14 they are not.
#: The cost is that the tightest pair in 2,040 cells closes to 7.2 um
#: rather than 8.2 um, which is not an error -- see the note on contact
#: distance in the module docstring.
DEFAULT_JITTER = 0.14

#: Capillary lattice pitch in um. Bounded above by the vascular module's
#: 25 um normoxic limit, not chosen for looks: measured over the shipped
#: slab over six seeds, a 25 um pitch leaves the worst-supplied cell
#: 18.0-20.1 um from a vessel and 22 um leaves it 16.5-18.4 um, against
#: that limit of 25. A 20 um pitch would reach 14.7-16.2 um but needs 34
#: vessels rather than 26, which is past myocardial density.
DEFAULT_VESSEL_PITCH_UM = 22.0


# ── The shipped TissueSlab geometry ──────────────────────────────────
#
# One dict, because three modules have to agree about where the tissue
# is and they are initialised independently of each other: the cellular
# module fills the slab, the vascular module has to wrap a vessel
# lattice around the same box, and the immune module has to scatter
# inside it. If each carried its own copy of the geometry they would
# drift, and the failure mode is silent -- a slab whose outer cells are
# simply outside every Krogh cylinder.

SLAB_SPEC: Dict[str, Any] = {
    # Population. 15 x 17 x 8 = 2,040 close-packed sites, of which the
    # 80 nearest the nodule centre become the tumour.
    "n_normal": 1960,
    "n_cancer": 80,
    # Packing.
    "spacing": DEFAULT_SPACING_UM,
    "layers": DEFAULT_LAYERS,
    "center": DOMAIN_CENTER_UM,
    "jitter": DEFAULT_JITTER,
    # Where the nodule sits, relative to the slab centre. Off-centre in
    # x and y so there is stroma on every side of it and a visible
    # tumour/stroma boundary; centred in z so the 26 um-radius mass is
    # fully enclosed by the 63 um-thick slab rather than cropped by its
    # faces.
    "nodule_offset": (40.0, -28.0, 0.0),
    # Vasculature.
    "vessel_pitch": DEFAULT_VESSEL_PITCH_UM,
    "vessel_axis": 1,
    "vessel_jitter": 0.12,
    "vessel_tilt": 0.06,
    # Immune scatter is inset from the slab faces by this much, so a
    # patrolling cell starts inside the tissue rather than on its skin.
    "immune_inset": 4.0,
    # None means "draw from the global numpy stream", so that
    # ``np.random.seed`` still makes a whole run reproducible. An int
    # pins the layout regardless of the global stream.
    "seed": None,
}


def slab_spec(**overrides: Any) -> Dict[str, Any]:
    """``SLAB_SPEC`` with ``overrides`` applied, rejecting unknown keys.

    Unknown keys are an error rather than a no-op: a misspelled
    ``spacing`` would otherwise silently seed the default geometry, and
    the only symptom would be a slab that does not match its vessels.
    """
    unknown = set(overrides) - set(SLAB_SPEC)
    if unknown:
        raise ValueError(
            f"unknown slab parameter(s) {sorted(unknown)}; "
            f"known: {sorted(SLAB_SPEC)}"
        )
    spec = dict(SLAB_SPEC)
    spec.update(overrides)
    return spec


def rng_from(seed: Optional[int]) -> np.random.Generator:
    """A Generator for ``seed``, or one drawn from the global stream.

    ``seed=None`` takes one draw from ``np.random`` so that a caller who
    did ``np.random.seed(n)`` -- which is how every test and every
    dashboard run in this project fixes its randomness -- still gets a
    reproducible slab.
    """
    if seed is None:
        return np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
    return np.random.default_rng(int(seed))


# ── Close-packed lattice ─────────────────────────────────────────────

def lattice_shape(n_sites: int, layers: int) -> Tuple[int, int, int]:
    """Pick (nx, ny, nz) for a close-packed slab of >= ``n_sites`` sites.

    ``layers`` fixes nz. The in-plane counts make the footprint as close
    to square as a triangular lattice allows: the row pitch is
    ``spacing * sqrt(3)/2``, so ny-1 has to be 2/sqrt(3) times nx-1 for
    the x and y extents to match.
    """
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    if n_sites <= 0:
        return (0, 0, 0)

    per_layer = math.ceil(n_sites / layers)
    aspect = 2.0 / math.sqrt(3.0)
    nx = 1
    while True:
        ny = int(round(aspect * (nx - 1))) + 1
        if nx * ny >= per_layer:
            return (nx, ny, layers)
        nx += 1


def packed_sites(
    n_sites: int,
    spacing: float = DEFAULT_SPACING_UM,
    layers: int = DEFAULT_LAYERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hexagonal close-packed sites for a slab, centred on the origin.

    Returns ``(sites, extents)``: ``sites`` is (M, 3) with
    M >= ``n_sites``, ``extents`` is the (Lx, Ly, Lz) bounding box.

    The stacking is ABAB. Each layer is a triangular lattice of pitch
    ``spacing``; odd layers drop into the hollows of the layer below, at
    an in-plane offset of (spacing/2, spacing/(2*sqrt(3))) and a z pitch
    of ``spacing * sqrt(2/3)``. Every nearest neighbour -- in plane and
    between layers -- is then exactly ``spacing`` away, which is what
    makes the unjittered lattice's nearest-neighbour distance a single
    number and the jittered one a narrow distribution about it.
    """
    if spacing <= 0.0:
        raise ValueError(f"spacing must be > 0, got {spacing}")

    nx, ny, nz = lattice_shape(n_sites, layers)
    if nx == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(3)

    row_pitch = spacing * math.sqrt(3.0) / 2.0
    layer_pitch = spacing * math.sqrt(2.0 / 3.0)
    hollow = (spacing / 2.0, spacing / (2.0 * math.sqrt(3.0)))

    ii, jj, kk = np.meshgrid(
        np.arange(nx, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nz, dtype=np.float64),
        indexing="ij",
    )
    row_odd = np.mod(jj, 2.0)
    layer_odd = np.mod(kk, 2.0)

    x = ii * spacing + row_odd * (spacing / 2.0) + layer_odd * hollow[0]
    y = jj * row_pitch + layer_odd * hollow[1]
    z = kk * layer_pitch

    sites = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    # Centre on the origin using the lattice's own bounding box, so the
    # row and layer offsets cannot push the slab off its nominal centre.
    lo = sites.min(axis=0)
    hi = sites.max(axis=0)
    return sites - 0.5 * (lo + hi), hi - lo


# ── Slab seeding ─────────────────────────────────────────────────────

def seed_slab(
    n_normal: int,
    n_cancer: int,
    *,
    spacing: float = DEFAULT_SPACING_UM,
    layers: int = DEFAULT_LAYERS,
    center: Sequence[float] = DOMAIN_CENTER_UM,
    nodule_offset: Sequence[float] = (40.0, -28.0, 0.0),
    jitter: float = DEFAULT_JITTER,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Lay out a tissue slab with an embedded tumour nodule.

    The nodule is not a separate cloud dropped on top of the stroma: the
    ``n_cancer`` lattice sites nearest ``center + nodule_offset`` BECOME
    the nodule and leave the stromal pool, so the mass is solid, is as
    spherical as a close-packed lattice permits, and meets the stroma at
    a sharp boundary with no interpenetration. Its radius is therefore an
    OUTPUT of the cell count and the packing density, not a free
    parameter that could disagree with either.

    If the lattice yields more sites than ``n_normal + n_cancer``, the
    surplus is trimmed from the stroma by largest normalised radius
    ``sqrt((dx/Lx)^2 + (dy/Ly)^2 + (dz/Lz)^2)``, which rounds the corners
    of the box. Trimming on a box metric instead would remove a
    contiguous run of lattice rows and leave a notch in one face.

    Returns a dict with ``normal`` and ``cancer`` ((N, 3) float arrays),
    plus ``center``, ``extents``, ``bounds`` ((lo, hi) of the jittered
    cells), ``nodule_center``, ``nodule_radius``, ``spacing``, ``layers``
    and ``n_sites``.
    """
    if n_normal < 0 or n_cancer < 0:
        raise ValueError("cell counts must be >= 0")
    rng = np.random.default_rng() if rng is None else rng
    center = np.asarray(center, dtype=np.float64)
    nodule_center = center + np.asarray(nodule_offset, dtype=np.float64)

    n_total = n_normal + n_cancer
    if n_total == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return {
            "normal": empty, "cancer": empty.copy(),
            "center": center, "extents": np.zeros(3),
            "bounds": (center.copy(), center.copy()),
            "nodule_center": nodule_center, "nodule_radius": 0.0,
            "spacing": float(spacing), "layers": int(layers), "n_sites": 0,
        }

    sites, extents = packed_sites(n_total, spacing=spacing, layers=layers)
    sites = sites + center

    if n_cancer > 0:
        d_nodule = np.linalg.norm(sites - nodule_center, axis=1)
        cancer_idx = np.argsort(d_nodule, kind="stable")[:n_cancer]
    else:
        cancer_idx = np.zeros(0, dtype=np.int64)

    stroma_mask = np.ones(len(sites), dtype=bool)
    stroma_mask[cancer_idx] = False
    stroma_idx = np.flatnonzero(stroma_mask)

    if len(stroma_idx) > n_normal:
        half = np.where(extents > 0.0, 0.5 * extents, 1.0)
        rel = (sites[stroma_idx] - center) / half
        keep = np.argsort(np.linalg.norm(rel, axis=1), kind="stable")[:n_normal]
        stroma_idx = stroma_idx[np.sort(keep)]

    normal = sites[stroma_idx]
    cancer = sites[np.sort(cancer_idx)]

    amplitude = float(jitter) * float(spacing)
    if amplitude > 0.0:
        normal = normal + rng.uniform(-amplitude, amplitude, size=normal.shape)
        cancer = cancer + rng.uniform(-amplitude, amplitude, size=cancer.shape)

    # Measured AFTER the jitter, so the reported radius describes the
    # cells that exist rather than the lattice sites they came from.
    nodule_radius = (
        float(np.linalg.norm(cancer - nodule_center, axis=1).max())
        if len(cancer) else 0.0
    )

    occupied = np.vstack([normal, cancer]) if len(cancer) else normal
    return {
        "normal": normal,
        "cancer": cancer,
        "center": center,
        "extents": extents,
        "bounds": (occupied.min(axis=0), occupied.max(axis=0)),
        "nodule_center": nodule_center,
        "nodule_radius": nodule_radius,
        "spacing": float(spacing),
        "layers": int(layers),
        "n_sites": int(len(sites)),
    }


def slab_bounds(
    n_normal: int,
    n_cancer: int,
    *,
    spacing: float = DEFAULT_SPACING_UM,
    layers: int = DEFAULT_LAYERS,
    center: Sequence[float] = DOMAIN_CENTER_UM,
) -> Tuple[np.ndarray, np.ndarray]:
    """The slab's (lo, hi) bounding box, without seeding anything.

    The vascular and immune modules need the box the cellular module is
    going to fill, and all three are initialised independently, so the
    box has to be derivable from the shared ``SLAB_SPEC`` alone. This is
    the unjittered box; the jitter is small compared with the vessel
    overhang that ``capillary_lattice`` applies.
    """
    _, extents = packed_sites(n_normal + n_cancer, spacing=spacing, layers=layers)
    center = np.asarray(center, dtype=np.float64)
    return center - 0.5 * extents, center + 0.5 * extents


# ── Capillary lattice ────────────────────────────────────────────────

def capillary_lattice(
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    pitch: float = DEFAULT_VESSEL_PITCH_UM,
    axis: int = 1,
    overhang: float = 6.0,
    jitter: float = 0.12,
    tilt: float = 0.06,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Capillaries running the length of the slab on a staggered lattice.

    Every vessel runs parallel to ``axis`` (y by default) and they are
    arranged on a staggered lattice in the other two axes, so the
    interstitium is tiled by triangles rather than squares: the
    worst-supplied point of a triangle is pitch/sqrt(3) from a vessel
    against pitch/sqrt(2) for a square, an 18% smaller lethal corner for
    the same vessel count.

    Along the SLAB THICKNESS the rows are inset by half a row pitch, so
    no vessel lies on the top or bottom face: a capillary bed runs
    through a slab, not across its skin, and a vessel on a face supplies
    tissue on one side only. Across the slab's width the even rows run
    from edge to edge and the odd rows take the midpoints between them.
    That asymmetry is deliberate and was measured: insetting BOTH axes
    leaves the mid-depth corners of the width faces 20.5 um from the
    nearest vessel -- a predicted steady state of 0.0556, a whisker above
    the 0.05 hypoxia flag -- because the staggered middle row is the one
    that stops short there. Running the even rows to the edges instead
    costs three vessels and brings the worst cell in the slab back under
    18.4 um.

    Each vessel overhangs the slab by ``overhang`` um at both ends.
    ``Capillary.distance_to`` clamps the projection to the segment, so
    without the overhang the cells near the two end faces would be
    measured against an ENDPOINT -- a longer distance than the real
    perpendicular one -- and the slab would be falsely hypoxic exactly
    where the vessels enter it.

    ``jitter`` (a fraction of the pitch) displaces each vessel and
    ``tilt`` (likewise) displaces its two ends independently, so the
    bundle is neither a grid nor perfectly parallel. Both default low,
    because this lattice is a perfusion budget first: measured over six
    seeds, the shipped jitter moves the worst-supplied cell from
    15.5-15.8 um to 16.5-18.4 um, and it is that number rather than the
    look of the bundle that decides whether the slab lives.

    Returns a list of ``(start, end)`` pairs.
    """
    if pitch <= 0.0:
        raise ValueError(f"pitch must be > 0, got {pitch}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    rng = np.random.default_rng() if rng is None else rng

    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    u, v = [a for a in (0, 1, 2) if a != axis]  # u staggered, v carries rows

    span_u = float(hi[u] - lo[u])
    span_v = float(hi[v] - lo[v])

    # Rows at the triangular row pitch, columns at the pitch, both
    # rounded to whole counts and then spread evenly over the span, so
    # the realised spacing follows the slab rather than the other way
    # round.
    n_rows = max(1, int(round(span_v / (pitch * math.sqrt(3.0) / 2.0))))
    n_cols = max(1, int(round(span_u / pitch)))
    du = span_u / n_cols
    dv = span_v / n_rows

    amp = float(jitter) * float(pitch)
    tilt_amp = float(tilt) * float(pitch)

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for r in range(n_rows):
        v_pos = lo[v] + (r + 0.5) * dv
        odd = bool(r % 2)
        # Even rows span the width edge to edge; odd rows take the
        # midpoints between them, which is the stagger.
        row_u = [lo[u] + (c + 0.5) * du for c in range(n_cols)] if odd \
            else [lo[u] + c * du for c in range(n_cols + 1)]
        for u_pos in row_u:
            start = np.zeros(3)
            end = np.zeros(3)
            start[axis] = lo[axis] - overhang
            end[axis] = hi[axis] + overhang
            for a, p in ((u, u_pos), (v, v_pos)):
                base = p + (rng.uniform(-amp, amp) if amp > 0.0 else 0.0)
                start[a] = base + (rng.uniform(-tilt_amp, tilt_amp)
                                   if tilt_amp > 0.0 else 0.0)
                end[a] = base + (rng.uniform(-tilt_amp, tilt_amp)
                                 if tilt_amp > 0.0 else 0.0)
            segments.append((start, end))
    return segments


# ── Immune seeding ───────────────────────────────────────────────────

def scatter_in_box(
    n: int,
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    inset: float = 4.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """``n`` uniformly random positions inside the box, inset from its faces.

    Immune seeding. The other presets scatter immune cells over a fixed
    100 x 100 x 20 um window about the domain centre, which has nothing
    to do with where the cells are; on a 165 x 156 x 63 um slab that
    window would leave the whole periphery unpatrolled.
    """
    rng = np.random.default_rng() if rng is None else rng
    lo = np.asarray(lo, dtype=np.float64) + inset
    hi = np.maximum(np.asarray(hi, dtype=np.float64) - inset, lo)
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    return rng.uniform(lo, hi, size=(int(n), 3))


# ── Measurement ──────────────────────────────────────────────────────

def nearest_neighbour_distances(positions: np.ndarray) -> np.ndarray:
    """Distance from each position to its nearest OTHER position.

    Brute force: 2,000 cells is a 2,000 x 2,000 matrix, and no spatial
    index means nothing to get wrong in a measurement that the tests use
    to judge the packing.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) < 2:
        return np.zeros(0, dtype=np.float64)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def perfusion_report(
    positions: np.ndarray,
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """Distance from each position to its nearest vessel SEGMENT.

    Mirrors ``Capillary.distance_to``: the projection onto the segment is
    clamped to the endpoints, so this is the radial coordinate of the
    Krogh cylinder the vascular module actually uses and not the distance
    to an infinite line.

    Returns min / median / p95 / max, plus ``n_beyond_25um`` (the
    vascular module's normoxic limit) and ``n_beyond_50um`` (its lethal
    corner, beyond which a cell receives nothing at all).
    """
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) == 0 or not len(segments):
        return {"min": float("inf"), "median": float("inf"),
                "p95": float("inf"), "max": float("inf"),
                "n_beyond_25um": len(pos), "n_beyond_50um": len(pos)}

    best = np.full(len(pos), np.inf)
    for start, end in segments:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        along = end - start
        along_sq = float(np.dot(along, along))
        if along_sq <= 0.0:
            d = np.linalg.norm(pos - start, axis=1)
        else:
            s = np.clip(((pos - start) @ along) / along_sq, 0.0, 1.0)
            d = np.linalg.norm(pos - (start + s[:, None] * along), axis=1)
        np.minimum(best, d, out=best)

    return {
        "min": float(best.min()),
        "median": float(np.median(best)),
        "p95": float(np.percentile(best, 95)),
        "max": float(best.max()),
        "n_beyond_25um": int((best > 25.0).sum()),
        "n_beyond_50um": int((best >= 50.0).sum()),
    }
