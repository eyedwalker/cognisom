"""Turn the library's ``mesh_type`` vocabulary into geometry.

The entity library carries 35 distinct ``mesh_type`` values and they are not
placeholders -- ``multilobed_granular``, ``ruffled_pseudopod``,
``enveloped_corona_spikes``, ``rod_flagellated``, ``stellate_branching``. Someone
who knew what these cells and organisms look like wrote them down.

Nothing read any of it. The only code touching ``mesh_type`` was the enrichment
table that writes it, and the USD exporter drew a sphere for every entity, so a
neutrophil with a multilobed nucleus and a flagellated Vibrio rendered
identically. This module is the reader.

HOW THE SHAPES ARE BUILT
------------------------
Each builder fills an occupancy grid and the surface comes out of marching
cubes, through the same ``MeshGenerator`` the imaging path uses. That is
deliberate: a procedural cell and a segmented one then arrive in the same form,
with the same units, and can sit in one scene without a second code path.

WHAT THESE ARE AND ARE NOT
--------------------------
These are *depictions*, drawn from the words in the library, not measurements.
A segmented nucleus from a confocal stack is evidence; a ``multilobed_granular``
nucleus built here is an illustration of a label. :func:`build` reports which
builder ran and whether it fell back, so a caller can tell the difference and a
viewer can say so. A shape that silently falls back to a sphere while claiming
to be a neutrophil is the failure this module exists to prevent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: Grid resolution per axis. 64 is enough for a lobed nucleus or a flagellum to
#: read at a glance and cheap enough to build a whole catalogue in seconds.
GRID = 64


@dataclass
class Morphology:
    """One built shape, and an honest account of where it came from."""

    vertices: np.ndarray          # (N, 3) in micrometres
    triangles: np.ndarray         # (M, 3) indices
    mesh_type: str                # what was asked for
    builder: str                  # what actually ran
    fell_back: bool               # True when the request had no builder
    parts: Dict[str, int]         # label -> voxel count, for sub-structures

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)


# ── grid helpers ─────────────────────────────────────────────────────────

def _grid(n: int = GRID):
    """Coordinates on [-1, 1]^3."""
    a = np.linspace(-1.0, 1.0, n)
    return np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).astype(np.float32)


def _variation(seed, *, radius=0.05, axes=0.04, offset=0.03):
    """Per-cell variation: how this one differs from its neighbours.

    Seventeen of the thirty-two builders ignored their ``seed``
    argument, and ``smooth_sphere`` -- shared by eighteen entities,
    every CD4 and CD8 subset, both B cells, all three ILCs -- was one of
    them. So :func:`morphology_usd.shape_seed`, added precisely to stop
    entities sharing a mesh_type from being byte-identical, had no
    effect on the builder that collides most. A Th1 and a Th2 were the
    same object.

    THIS ADDS NO INFORMATION, and it is important to be clear about
    that. It does not claim a Th1 is rounder than a Th2. It says a
    population of lymphocytes is a population rather than one cell
    stamped out repeatedly, which is a weaker claim and a true one.
    The cells still differ meaningfully only in the ways the library
    records.

    Amplitudes are small on purpose: a few per cent, against the ~10%
    diameter spread the morphometry histograms show for a real WBC
    population (PMID 6775712, Fig. 7). Large enough that two cells are
    not the same mesh, small enough that nobody reads structure into it.

    Returns ``(radius_scale, squash, offset)``.
    """
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    scale = 1.0 + rng.normal(0.0, radius)
    squash = 1.0 + rng.normal(0.0, axes, size=3)
    shift = rng.normal(0.0, offset, size=3)
    return float(np.clip(scale, 0.85, 1.15)), np.clip(squash, 0.88, 1.12), shift


def _sphere(g, centre=(0, 0, 0), r=1.0, squash=(1, 1, 1)):
    d = (g - np.asarray(centre, np.float32)) / np.asarray(squash, np.float32)
    return np.linalg.norm(d, axis=-1) <= r


def _rounded_box(g, half, roundness=5.0, centre=(0, 0, 0)):
    """A superellipsoid: a box with rounded edges.

    Epithelial cells were built as literal boxes -- ``abs(x) < 0.62`` and
    the same in y and z -- which renders as a glass cube with sharp 90
    degree corners. No cell has those. A superellipsoid keeps the flat
    faces that make cuboidal epithelium cuboidal and rounds the edges,
    which is both what the tissue looks like and what distinguishes a
    cuboidal cell from a columnar one in the first place.

    ``roundness`` 2 is an ellipsoid, large is a box; 4-6 reads as a cell.
    """
    d = np.abs((g - np.asarray(centre, np.float32))
               / np.asarray(half, np.float32))
    return (d ** roundness).sum(axis=-1) <= 1.0


def _helix(g, start, axis, length, turns, amplitude, radius):
    """A helical filament: a bacterial flagellum.

    Flagella were a random walk, which wandered out past the grid edge and
    came back as a clipped fragment floating beside the cell. A real
    flagellum is a helix of fairly regular pitch, which is also what makes
    it legible as a flagellum rather than as debris.
    """
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(axis @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, tmp); u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    out = np.zeros(g.shape[:3], bool)
    steps = 26
    prev = None
    for i in range(steps + 1):
        t = i / steps
        angle = 2.0 * np.pi * turns * t
        # The amplitude eases in, so the filament leaves the cell along
        # its axis rather than starting mid-swing.
        a = amplitude * min(1.0, t * 3.0)
        point = (np.asarray(start, np.float64) + axis * length * t
                 + u * np.cos(angle) * a + v * np.sin(angle) * a)
        if prev is not None:
            out |= _capsule(g, prev, point, radius)
        prev = point
    return out


def _capsule(g, a, b, r):
    """Filled capsule between two points: the body of a rod bacterium."""
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    ab = b - a
    t = np.clip(((g - a) @ ab) / float(ab @ ab), 0.0, 1.0)[..., None]
    return np.linalg.norm(g - (a + t * ab), axis=-1) <= r


def _lobes(g, n_lobes, radius, spread, seed):
    """A chain of overlapping spheres: a segmented nucleus."""
    rng = np.random.default_rng(seed)
    out = np.zeros(g.shape[:3], bool)
    angles = np.linspace(0, 2 * np.pi, n_lobes, endpoint=False)
    for i, ang in enumerate(angles):
        c = (spread * np.cos(ang), spread * np.sin(ang),
             float(rng.uniform(-0.12, 0.12)))
        out |= _sphere(g, c, radius)
    return out


def _veil(g, centre, direction, reach, width, thickness):
    """A flat sheet extending along ``direction``: a lamellipodium.

    A macrophage's processes are broad flattened veils, not rods. They
    were nine radial capsules of radius 0.14-0.24, which renders as a ball
    with sausages stuck in it -- closer to a jack than to a cell, and it
    loses the one thing the shape is for, which is the large thin leading
    edge a phagocyte crawls and engulfs on.

    An axis-aligned squash cannot express this, because each veil points
    somewhere different, so the ellipsoid is built in its own frame.
    """
    u = np.asarray(direction, np.float64)
    u = u / np.linalg.norm(u)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(u @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    w = np.cross(u, tmp); w /= np.linalg.norm(w)
    n = np.cross(u, w)
    basis = np.stack([u, w, n]).astype(np.float32)        # rows
    d = (g - np.asarray(centre, np.float32)) @ basis.T
    radii = np.asarray([reach, width, thickness], np.float32)
    return ((d / radii) ** 2).sum(axis=-1) <= 1.0


#: Granule volume fraction of the cell, by cell type. One table, one
#: method, one paper -- Schmid-Schonbein PMID 6775712 Table 2, read back
#: through BNIDs 115197 / 115195 / 115196 / 115194:
#:
#:      eosinophil  23.0%
#:      neutrophil  15.4%
#:      monocyte     2.6%
#:      lymphocyte   0.2%
#:
#: A 77-fold measured spread, and the strongest single discriminator
#: between these cells. The builders gave an eosinophil 2.4% and a
#: neutrophil 1.5% -- tenfold low, and close enough to each other that
#: the discriminator was flattened to nothing.


def _granules_to_fraction(g, body, fraction, radius, seed, spread=0.35):
    """Granules filling ``fraction`` of ``body`` by volume.

    Placed until the fraction is reached rather than by a precomputed
    count, because granules overlap: at an eosinophil's 23% the overlap
    is substantial and a count derived from the single-granule volume
    falls short of the target by a third.
    """
    rng = np.random.default_rng(seed)
    target = fraction * float(body.sum())
    if target <= 0:
        return np.zeros(g.shape[:3], bool)
    out = np.zeros(g.shape[:3], bool)
    # Candidate centres must sit a granule's radius inside the body, or
    # granules placed against the membrane push through it -- which they
    # did, on 1.4-3.6% of granule vertices across the three granulocytes.
    try:
        from scipy.ndimage import distance_transform_edt
        voxel = 2.0 / body.shape[0]            # the grid spans [-1, 1]
        deep = distance_transform_edt(body) * voxel >= radius
        idx = np.argwhere(deep if deep.any() else body)
    except ImportError:                                 # pragma: no cover
        idx = np.argwhere(body)
    for _ in range(4000):
        if float((out & body).sum()) >= target:
            break
        centre = (g[tuple(idx[rng.integers(len(idx))])]
                  if len(idx) else np.zeros(3))
        r = radius * (1.0 + rng.normal(scale=spread))
        out |= _sphere(g, centre, max(r, radius * 0.4))
    return out & body


def _spikes(g, n, inner, outer, thickness, seed):
    """Radial projections: viral envelope spikes, or cell processes."""
    rng = np.random.default_rng(seed)
    out = np.zeros(g.shape[:3], bool)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for d in v:
        out |= _capsule(g, d * inner, d * outer, thickness)
    return out


@dataclass(frozen=True)
class ShapeParams:
    """The proportions an ENTITY supplies, as opposed to its type.

    These numbers lived as module constants, and the library stored the
    same numbers again as provenanced parameters. Two copies of one
    measurement is one copy too many, and the module's copy won:
    ``cell_profiles.library_detail`` fetched seven parameters off the
    entity and ``FROM_LIBRARY`` mapped two of them, so
    ``nuclear_volume_fraction`` -- 0.444, read out of PMID 6775712 Table
    2 by hand, agreeing with Petrzilka's independent 0.458, and cleared
    by that paper's own Appendix as the one fraction whose
    section-thickness error is negligible -- was fetched, graded
    MEASURED on 20 entities, and then overridden by a constant.

    The copies agreed, which is why it went unnoticed, and they agreed
    only because one person wrote both in one pass. One had already
    drifted: ``granular_sphere`` carved granules at 0.020 while the NK
    cell entity stored 0.002, a factor of ten, and the hardcoded one was
    what rendered.

    A field left None means the caller supplied nothing and the builder
    falls back to the documented default below. That is not the same as
    a measurement and the scene records which it got.
    """

    nuclear_volume_fraction: Optional[float] = None
    granule_volume_fraction: Optional[float] = None

    def nuclear_radius(self, body_radius: float = 0.92) -> float:
        """Nuclear radius as a fraction of the grid, from the fraction.

        A RADIUS is the cube root of a volume fraction. Using the
        fraction directly would put a lymphocyte nucleus at 8.8% of the
        cell instead of 76%.
        """
        fraction = (self.nuclear_volume_fraction
                    if self.nuclear_volume_fraction is not None
                    else DEFAULT_NUCLEAR_VOLUME_FRACTION)
        return body_radius * float(fraction) ** (1.0 / 3.0)

    def granules(self, default: float) -> float:
        """Granule volume fraction, or the builder's own default."""
        return (float(self.granule_volume_fraction)
                if self.granule_volume_fraction is not None else default)


#: Used only when no entity supplies one. The measured value for a
#: resting human lymphocyte (PMID 6775712 Table 2, 44.4%); it is here as
#: a fallback for a bare build_parts() call, not as the source of truth.
#: An entity that carries the parameter overrides it.
DEFAULT_NUCLEAR_VOLUME_FRACTION = 0.444

#: Per-builder granule fallbacks, same status: the measured values, used
#: when the caller supplies nothing. The eosinophil and neutrophil
#: numbers are PMID 6775712 Table 2 and agree with the entities; the
#: 0.020 that granular_sphere used to carry agreed with nothing and is
#: now the lymphocyte's measured 0.002.
DEFAULT_GRANULE_FRACTION: Dict[str, float] = {
    "eosinophil": 0.230, "neutrophil": 0.154,
    "monocyte": 0.026, "lymphocyte": 0.002,
}

#: No entity supplied anything.
NO_PARAMS = ShapeParams()


# ── builders ─────────────────────────────────────────────────────────────
# Each returns a dict of label -> boolean volume. Label 1 is always the body.

def _b_sphere(g, seed, p):
    return {1: _sphere(g, r=0.92)}


def _b_smooth_sphere(g, seed, p):
    # Offset slightly, because a perfectly concentric nucleus reads as a
    # diagram; a real one sits a little to one side. The variation is
    # what makes this cell this cell rather than the shared prototype --
    # eighteen entities use this builder.
    scale, squash, shift = _variation(seed)
    return {1: _sphere(g, r=0.92 * scale, squash=squash),
            2: _sphere(g, (0.06 + shift[0], -0.04 + shift[1], shift[2]),
                       p.nuclear_radius() * scale, squash=squash)}


def _b_granular_sphere(g, seed, p):
    """NK / NKT: a lymphocyte carrying lytic granules.

    Still a lymphocyte, so it keeps the lymphocyte nucleus; the granules
    are what set it apart, and they are few -- a primary human NK cell
    holds 63 +/- 23 of them (PMID 29592963), not the ~200 that reviews
    quote from the NK-92 and YTS cell lines.
    """
    scale, squash, shift = _variation(seed)
    body = _sphere(g, r=0.92 * scale, squash=squash)
    nuc = _sphere(g, (0.06 + shift[0], -0.04 + shift[1], shift[2]),
                  p.nuclear_radius() * scale, squash=squash)
    # 0.002 and not the 0.020 this carried: the NK cell entity stores
    # the measured lymphocyte fraction and the hardcoded one was ten
    # times it.
    gran = _granules_to_fraction(
        g, body & ~nuc, p.granules(DEFAULT_GRANULE_FRACTION["lymphocyte"]),
        0.052, seed)
    return {1: body, 2: nuc, 3: gran}


def _b_dense_granular(g, seed, p):
    """Mast cell / basophil: so densely granulated that the granules,
    not the nucleus, are what you see. Taken at the eosinophil's measured
    fraction, which is the highest in the table and the closest
    available: Schmid-Schonbein's series does not separate basophils."""
    body = _sphere(g, r=0.92)
    nuc = _sphere(g, (0.10, 0.06, 0), 0.40)
    gran = _granules_to_fraction(g, body & ~nuc,
                                 p.granules(DEFAULT_GRANULE_FRACTION["eosinophil"]),
                                 0.062, seed)
    return {1: body, 2: nuc, 3: gran}


def _b_multilobed_granular(g, seed, p):
    """Neutrophil: three to five nuclear lobes on a thread, dense granules.

    15.4% of the cell by volume (PMID 6775712), against the 1.5% this
    built before. Finer granules than an eosinophil's, and fewer by
    volume, which is the pair of facts that separates the two.
    """
    body = _sphere(g, r=0.92)
    nuc = _lobes(g, 4, 0.30, 0.44, seed)
    gran = _granules_to_fraction(g, body & ~nuc,
                                 p.granules(DEFAULT_GRANULE_FRACTION["neutrophil"]),
                                 0.058, seed)
    return {1: body, 2: nuc & body, 3: gran}


def _b_bilobed_granular(g, seed, p):
    """Eosinophil: two nuclear lobes, and packed with coarse granules.

    23.0% of the cell by volume (PMID 6775712), the highest in the
    series and nearly ten times what this built before. An eosinophil is
    defined by how full of granules it is -- that is what the stain
    picks out -- so an eosinophil at 2.4% is not a dimmer eosinophil, it
    is a different cell.
    """
    body = _sphere(g, r=0.92)
    nuc = _sphere(g, (-0.34, 0.10, 0), 0.36) | _sphere(g, (0.34, -0.10, 0), 0.36)
    gran = _granules_to_fraction(g, body & ~nuc,
                                 p.granules(DEFAULT_GRANULE_FRACTION["eosinophil"]),
                                 0.085, seed)
    return {1: body, 2: nuc & body, 3: gran}


def _b_ruffled_pseudopod(g, seed, p):
    """Macrophage: a spread body with broad lamellipodia and ruffles."""
    rng = np.random.default_rng(seed)
    # Adherent macrophages spread, so the body is flattened rather than
    # round, and the veils come off its edge.
    body = _sphere(g, r=0.60, squash=(1.0, 1.0, 0.62))
    for k in range(6):
        angle = 2.0 * np.pi * k / 6.0 + rng.uniform(-0.3, 0.3)
        tilt = rng.uniform(-0.28, 0.28)
        d = np.array([np.cos(angle), np.sin(angle), tilt])
        d /= np.linalg.norm(d)
        body |= _veil(g, d * 0.42, d,
                      reach=rng.uniform(0.46, 0.60),
                      width=rng.uniform(0.30, 0.44),
                      thickness=rng.uniform(0.07, 0.11))
    # Membrane ruffles: folds standing off the dorsal surface, which is
    # what "ruffled" names and what a phagocyte's surface actually does.
    x, y, z = g[..., 0], g[..., 1], g[..., 2]
    folds = np.sin(x * 11.0 + 0.7) * np.sin(y * 9.0) > 0.35
    body |= folds & _sphere(g, (0, 0, 0.10), 0.80, squash=(1.0, 1.0, 0.72))
    return {1: body, 2: _sphere(g, (0.10, 0.05, 0), 0.30)}


def _b_irregular_ruffled(g, seed, p):
    rng = np.random.default_rng(seed)
    body = _sphere(g, r=0.70)
    for _ in range(14):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        body |= _sphere(g, d * rng.uniform(0.6, 0.85), rng.uniform(0.14, 0.22))
    return {1: body, 2: _sphere(g, (0, 0, 0), 0.30)}


def _b_stellate_branching(g, seed, p):
    """Dendritic cell: a small body and long thin radiating processes."""
    body = _sphere(g, r=0.38)
    body |= _spikes(g, 11, 0.25, 0.98, 0.055, seed)
    return {1: body, 2: _sphere(g, (0, 0, 0), 0.21)}


def _b_eccentric_nucleus_er(g, seed, p):
    """Plasma cell: nucleus pushed to one side, ER filling the rest."""
    body = _sphere(g, r=0.92)
    nuc = _sphere(g, (-0.34, 0.0, 0.0), 0.42)
    # Rough ER is stacked lamellae -- flattened cisternae wrapped round
    # the nucleus -- not a cloud. It was a spherical shell speckled at 55%
    # random, which renders as one solid blob filling half the cell and
    # says nothing about what an ER is. Concentric sheets read as the
    # stack they are, and are what makes a plasma cell recognisable.
    rng = np.random.default_rng(seed)
    # Cisternae wrapped round the nucleus, which is how rough ER is
    # arranged in a plasma cell. They must be THIN: at grid 64 a voxel is
    # 0.031 of the grid, so a period of 0.14 with a quarter duty cycle is
    # a sheet about one voxel thick with three voxels of lumen between.
    # Concentric shells 8 voxels thick merged into one solid ball that
    # filled the cell and hid the nucleus.
    # PARALLEL cisternae, not concentric shells. Concentric shells wrap
    # the whole cell, so whatever their thickness the outermost one is all
    # you ever see and the ER reads as a solid ball that hides the
    # nucleus. Parallel sheets present their EDGES, which is what makes a
    # stack legible as a stack -- and it is also what rough ER looks like
    # in a plasma cell: flattened cisternae in register, not onion skin.
    x, y = g[..., 0], g[..., 1]
    # tilted slightly off-axis so the stack is never seen exactly
    # edge-on or exactly face-on from a default view
    across = 0.94 * y + 0.34 * x
    period, duty = 0.17, 0.34
    sheets = np.sin(across * (2.0 * np.pi / period)) > np.cos(np.pi * duty)
    sheets &= rng.random(g.shape[:3]) < 0.88      # fenestrated
    # Confined to a stack in the cytoplasm rather than filling the whole
    # hemisphere. Sheets that reach the membrane on every side still
    # present a closed outer surface, so the cell reads as a striped ball;
    # leaving cytoplasm around the stack is what lets the sheets be seen
    # as sheets, with ends.
    region = (_sphere(g, (0.30, 0.0, 0.0), 0.60, squash=(1.0, 1.15, 1.15))
              | _sphere(g, (-0.05, 0.0, 0.0), 0.46))
    er = sheets & region
    return {1: body, 2: nuc & body, 4: er & body & ~nuc}


def _b_elongated_spindle(g, seed, p):
    """Fibroblast: a long thin cell with tapered ends.

    The nucleus has to fit. It was r=0.30 squashed (1.5, 1, 1), giving
    half-extents (0.45, 0.30, 0.30) inside a body of (0.95, 0.247, 0.19)
    -- wider than the cell in two axes out of three, so 37% of its voxels
    lay outside the membrane and it visibly burst through the side. It is
    also elongated along the cell, as a fibroblast nucleus is.
    """
    body = _sphere(g, r=0.95, squash=(1.0, 0.26, 0.20))
    return {1: body,
            2: _sphere(g, (0, 0, 0), 0.155, squash=(2.4, 1.0, 0.82))}


def _b_flat_squamous(g, seed, p):
    """Endothelium: a thin plate that bulges where the nucleus is.

    The nucleus was half-extent 0.17 in z inside a cell of 0.152 -- the
    nucleus was thicker than the cell it is in, and 16% of its vertices
    lay outside the membrane. Thinning the nucleus would fix the
    containment and lose the fact: a squamous cell really is thinner than
    its nucleus everywhere except over the nucleus, and the bulge is how
    you pick endothelial nuclei out of a vessel wall. So the body bulges
    instead.
    """
    nuc = _sphere(g, (0, 0, 0), 0.34, squash=(1, 1, 0.52))
    body = _sphere(g, r=0.95, squash=(1.0, 1.0, 0.16))
    body |= _sphere(g, (0, 0, 0), 0.40, squash=(1, 1, 0.58))   # the bulge
    return {1: body, 2: nuc & body}


def _b_polarized_columnar(g, seed, p):
    """Columnar epithelium: tall, with the nucleus held basally."""
    body = _rounded_box(g, (0.44, 0.44, 0.96), roundness=4.0)
    return {1: body, 2: _sphere(g, (0, 0, -0.56), 0.28, squash=(1, 1, 1.15))}


def _b_cuboidal(g, seed, p):
    """Basal epithelium: about as tall as it is wide, edges rounded."""
    body = _rounded_box(g, (0.70, 0.70, 0.70), roundness=4.5)
    return {1: body, 2: _sphere(g, (0, 0, 0), 0.30)}


def _b_cuboidal_tall(g, seed, p):
    """HEV cell: plump for endothelium, which is how it is recognised."""
    body = _rounded_box(g, (0.52, 0.52, 0.90), roundness=4.0)
    return {1: body, 2: _sphere(g, (0, 0, -0.38), 0.27)}


def _b_flat_activated_adhesion(g, seed, p):
    """Inflamed endothelium: flat, thickened, with a nuclear bulge."""
    nuc = _sphere(g, (0, 0, 0), 0.28, squash=(1, 1, 0.55))
    body = _sphere(g, r=0.95, squash=(1.0, 0.85, 0.18))
    body |= _sphere(g, (0, 0, 0), 0.34, squash=(1, 1, 0.62))
    body |= _spikes(g, 7, 0.5, 0.98, 0.06, seed) & (np.abs(g[..., 2]) < 0.18)
    return {1: body, 2: nuc & body}


# viruses
def _b_enveloped_corona_spikes(g, seed, p):
    return {1: _sphere(g, r=0.62) | _spikes(g, 54, 0.58, 0.95, 0.055, seed),
            5: _sphere(g, r=0.34)}


def _b_enveloped_ha_na_spikes(g, seed, p):
    return {1: _sphere(g, r=0.66) | _spikes(g, 72, 0.62, 0.92, 0.042, seed),
            5: _sphere(g, r=0.36)}


def _b_enveloped_sparse_spikes(g, seed, p):
    return {1: _sphere(g, r=0.74) | _spikes(g, 16, 0.70, 0.95, 0.05, seed),
            5: _sphere(g, r=0.40)}


def _b_enveloped_smooth(g, seed, p):
    return {1: _sphere(g, r=0.85), 5: _sphere(g, r=0.48)}


def _b_enveloped_small(g, seed, p):
    return {1: _sphere(g, r=0.55), 5: _sphere(g, r=0.30)}


def _b_enveloped_smooth_small(g, seed, p):
    return {1: _sphere(g, r=0.52), 5: _sphere(g, r=0.28)}


def _b_enveloped_pleomorphic(g, seed, p):
    rng = np.random.default_rng(seed)
    body = _sphere(g, r=0.60, squash=(1.0, 0.78, 0.62))
    for _ in range(5):
        body |= _sphere(g, rng.normal(scale=0.26, size=3), rng.uniform(0.26, 0.40))
    return {1: body, 5: _sphere(g, r=0.26)}


def _b_bullet_shape(g, seed, p):
    """Rhabdovirus: one rounded end, one flat."""
    z = g[..., 2]
    # Rhabdovirus: a cylinder with one hemispherical end and one flat, the
    # shape the family is named for. Cutting an ellipsoid at z > -0.62
    # removed almost nothing -- its base was already at -0.70 -- so it
    # rendered as a rounded capsule with no flat end at all.
    body = (_capsule(g, (0, 0, -0.30), (0, 0, 0.34), 0.42)
            & (z > -0.30))
    core = (_capsule(g, (0, 0, -0.22), (0, 0, 0.26), 0.24) & (z > -0.22))
    return {1: body, 5: core}


# bacteria
def _b_rod_flagellated(g, seed, p):
    rng = np.random.default_rng(seed)
    body = _capsule(g, (-0.46, 0, 0), (0.46, 0, 0), 0.26)
    fla = np.zeros(g.shape[:3], bool)
    # Peritrichous: flagella over the whole surface, not one tuft. Each is
    # a helix, and each stays inside the grid -- the random walk this
    # replaces reached x = 1.29 on a grid that ends at 1.0, so it was
    # clipped and rendered as a fragment floating beside the cell.
    for k in range(5):
        angle = 2.0 * np.pi * k / 5.0 + 0.3
        anchor = np.array([0.30 - 0.30 * k / 4.0,
                           0.22 * np.cos(angle), 0.22 * np.sin(angle)])
        direction = np.array([0.55, np.cos(angle), np.sin(angle)])
        fla |= _helix(g, anchor, direction, length=0.62, turns=2.1,
                      amplitude=0.065, radius=0.022)
    return {1: body, 6: fla}


def _b_rod_acid_fast(g, seed, p):
    return {1: _capsule(g, (-0.58, 0, 0), (0.58, 0, 0), 0.21)}


def _b_rod_spore_forming(g, seed, p):
    body = _capsule(g, (-0.58, 0, 0), (0.46, 0, 0), 0.24)
    return {1: body, 7: _sphere(g, (0.40, 0, 0), 0.22)}


def _b_coccus_chain(g, seed, p):
    out = np.zeros(g.shape[:3], bool)
    for i in range(5):
        out |= _sphere(g, (-0.72 + i * 0.36, 0, 0), 0.20)
    return {1: out}


def _b_coccus_cluster(g, seed, p):
    """Staphylococcus: cocci dividing in several planes, so they stay in
    an irregular cluster rather than a chain or a pair.

    The centres were drawn from an unbounded normal, so a tail draw put a
    coccus past the edge of the grid and it came out sliced flat -- 21
    voxels of it against the -x face. Clamping the draw keeps the whole
    cluster inside, and keeping the cocci touching is what makes it a
    cluster rather than scattered cells.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(g.shape[:3], bool)
    radius = 0.21
    reach = 0.92 - radius                 # the furthest a centre may sit
    for _ in range(11):
        centre = rng.normal(scale=0.34, size=3)
        length = float(np.linalg.norm(centre))
        if length > reach:
            centre *= reach / length
        out |= _sphere(g, centre, radius)
    return {1: out}


def _b_diplococcus(g, seed, p):
    return {1: _sphere(g, (-0.26, 0, 0), 0.34, squash=(0.9, 1, 1))
               | _sphere(g, (0.26, 0, 0), 0.34, squash=(0.9, 1, 1))}


# molecular
def _b_y_shape(g, seed, p):
    """Immunoglobulin: two Fab arms and an Fc stem."""
    body = _capsule(g, (0, 0, -0.85), (0, 0, -0.05), 0.15)
    body |= _capsule(g, (0, 0, -0.05), (-0.62, 0, 0.72), 0.13)
    body |= _capsule(g, (0, 0, -0.05), (0.62, 0, 0.72), 0.13)
    return {1: body}


def _b_groove(g, seed, p):
    """MHC: a platform carrying a peptide-binding groove."""
    x, y, z = g[..., 0], g[..., 1], g[..., 2]
    plat = (np.abs(x) < 0.82) & (np.abs(y) < 0.52) & (np.abs(z) < 0.26)
    groove = (np.abs(y) < 0.14) & (z > 0.02)
    helix = _capsule(g, (-0.78, 0.30, 0.30), (0.78, 0.30, 0.30), 0.14)
    helix |= _capsule(g, (-0.78, -0.30, 0.30), (0.78, -0.30, 0.30), 0.14)
    return {1: (plat & ~groove) | helix}


BUILDERS: Dict[str, Callable] = {
    "sphere": _b_sphere, "smooth_sphere": _b_smooth_sphere,
    "granular_sphere": _b_granular_sphere, "dense_granular": _b_dense_granular,
    "multilobed_granular": _b_multilobed_granular,
    "bilobed_granular": _b_bilobed_granular,
    "ruffled_pseudopod": _b_ruffled_pseudopod,
    "irregular_ruffled": _b_irregular_ruffled,
    "stellate_branching": _b_stellate_branching,
    "eccentric_nucleus_er": _b_eccentric_nucleus_er,
    "elongated_spindle": _b_elongated_spindle,
    "flat_squamous": _b_flat_squamous,
    "polarized_columnar": _b_polarized_columnar,
    "cuboidal": _b_cuboidal, "cuboidal_tall": _b_cuboidal_tall,
    "flat_activated_adhesion": _b_flat_activated_adhesion,
    "enveloped_corona_spikes": _b_enveloped_corona_spikes,
    "enveloped_ha_na_spikes": _b_enveloped_ha_na_spikes,
    "enveloped_sparse_spikes": _b_enveloped_sparse_spikes,
    "enveloped_smooth": _b_enveloped_smooth,
    "enveloped_small": _b_enveloped_small,
    "enveloped_smooth_small": _b_enveloped_smooth_small,
    "enveloped_pleomorphic": _b_enveloped_pleomorphic,
    "bullet_shape": _b_bullet_shape,
    "rod_flagellated": _b_rod_flagellated,
    "rod_acid_fast": _b_rod_acid_fast,
    "rod_spore_forming": _b_rod_spore_forming,
    "coccus_chain": _b_coccus_chain, "coccus_cluster": _b_coccus_cluster,
    "diplococcus": _b_diplococcus,
    "y_shape": _b_y_shape, "groove": _b_groove,
}

#: Sub-structure labels, so a caller can colour a nucleus differently from a
#: granule without knowing which builder ran.
PART_NAMES = {1: "body", 2: "nucleus", 3: "granules", 4: "er",
              5: "capsid", 6: "flagella", 7: "spore"}


#: Taubin smoothing passes applied to every extracted surface.
#:
#: Marching cubes over an occupancy grid produces terraces: the surface
#: steps from one voxel layer to the next, and those steps catch the light
#: as concentric rings. It is the single thing that made these cells read
#: as contoured blobs rather than cells, and no grid resolution removes it
#: -- a finer grid makes the steps smaller and more numerous, not absent.
#:
#: ``MeshGenerator`` has had a ``smoothing`` parameter since it was
#: written, documented and defaulting to 0.5. It is assigned to
#: ``self.smoothing`` and read nowhere, and the ``smooth_mesh`` method
#: beside it is never called -- and needs trimesh, which is not installed.
#: So nothing has ever been smoothed.
#:
#: 25 is where the benefit stops. Measured on a lymphocyte nucleus at
#: grid 64, as mean local deviation from the neighbour average:
#:
#:      passes    ripple    volume drift
#:           0    0.0290           --
#:           8    0.0109        +0.02%
#:          14    0.0076        +0.04%
#:          25    0.0045        -0.04%
#:          40    0.0042        +0.11%
#:          60    0.0037        +0.17%
#:
#: A 6.5-fold reduction by 25 and almost nothing after, while the volume
#: holds to a fifth of a percent throughout -- which is the property a
#: plain Laplacian smoother would not have.
SMOOTH_PASSES = 25

#: Taubin's two step sizes. A plain Laplacian smoother shrinks a closed
#: surface a little on every pass, so a cell smoothed enough to lose its
#: terraces would also have lost volume -- and volume here is a quantity
#: other code reads. Taubin alternates a positive step with a slightly
#: larger negative one, which cancels the shrinkage while still removing
#: the high-frequency steps.
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53


def smooth_surface(vertices: np.ndarray, triangles: np.ndarray,
                   passes: int = SMOOTH_PASSES,
                   lam: float = TAUBIN_LAMBDA,
                   mu: float = TAUBIN_MU) -> np.ndarray:
    """Taubin-smoothed vertices. Topology is untouched.

    Each pass moves every vertex a fraction of the way towards the mean of
    its neighbours, alternating the sign so the volume comes back.
    """
    if len(vertices) == 0 or len(triangles) == 0 or passes <= 0:
        return vertices

    edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]],
                       triangles[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    edges = np.unique(edges, axis=0)            # each neighbour once
    counts = np.bincount(edges[:, 0], minlength=len(vertices))
    counts = np.maximum(counts, 1)[:, None]

    out = np.asarray(vertices, np.float64).copy()
    for step in range(passes):
        neighbour_sum = np.zeros_like(out)
        np.add.at(neighbour_sum, edges[:, 0], out[edges[:, 1]])
        delta = neighbour_sum / counts - out
        out += (lam if step % 2 == 0 else mu) * delta
    return out


def build_masks(mesh_type: str, seed: int = 0, grid: int = GRID,
                params: Optional[ShapeParams] = None
                ) -> Tuple[Dict[str, np.ndarray], bool]:
    """The builder's own occupancy masks, before any surface is extracted.

    ``(part name -> boolean grid, fell_back)``. The masks are what define
    the shape; the meshes are derived from them. A caller placing
    organelles inside a cell wants the mask, because a point-in-mask test
    is exact where a point-in-mesh test against a marching-cubes surface
    is not, and because the volume of a mask is the volume the shape
    actually has.
    """
    builder = BUILDERS.get(mesh_type)
    fell_back = builder is None
    if fell_back:
        builder = _b_sphere
    # The entity's own proportions, or the documented fallbacks.
    p = params if params is not None else NO_PARAMS
    g = _grid(grid)
    return ({PART_NAMES.get(label, str(label)): mask
             for label, mask in builder(g, seed, p).items()}, fell_back)


def build_parts(mesh_type: str, scale_um: float = 10.0, seed: int = 0,
                grid: int = GRID, cut: bool = False,
                smooth: int = SMOOTH_PASSES,
                params: Optional[ShapeParams] = None
                ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], bool]:
    """Geometry per named part, so a nucleus can be coloured as a nucleus.

    ``cut`` removes one half along y before meshing. A neutrophil viewed from
    outside is a sphere -- that is true of the real cell and useless in a
    catalogue of shapes, because the four nuclear lobes that make it a
    neutrophil are inside an opaque body. Sectioning is what makes the label
    visible.

    Returns ``(parts, fell_back)``.
    """
    from cognisom.imaging.mesh_generation import MeshGenerator

    builder = BUILDERS.get(mesh_type)
    fell_back = builder is None
    if fell_back:
        builder = _b_sphere
    # The entity's own proportions, or the documented fallbacks.
    p = params if params is not None else NO_PARAMS

    g = _grid(grid)
    raw = builder(g, seed, p)
    resolution = float(scale_um) / grid

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, volume in sorted(raw.items()):
        vol = volume.copy()
        if cut:
            # Inner parts keep a couple of extra voxel layers so they stand
            # proud of the body's cut face. Cutting every part on the same
            # plane makes their surfaces coplanar, and the sectioned cell
            # renders as a flat disc of body colour with the nucleus z-fighting
            # invisibly behind it.
            plane = vol.shape[1] // 2 - (0 if label == 1 else 3)
            vol[:, :plane, :] = False
        if not vol.any():
            continue
        labels = np.zeros(vol.shape, np.int32)
        labels[vol] = 1
        sim = MeshGenerator(resolution=resolution,
                            simplify=False).labels_to_mesh(labels)
        verts, tris, off = [], [], 0
        for comp in sim.compartments:
            if comp.mesh is None:
                continue
            v = np.asarray(comp.mesh.vertices, dtype=float)
            verts.append(v)
            tris.append(np.asarray(comp.mesh.triangles, dtype=int) + off)
            off += len(v)
        if not verts:
            continue
        V = np.concatenate(verts)
        T = np.concatenate(tris)
        out[PART_NAMES.get(label, str(label))] = (
            smooth_surface(V, T, passes=smooth), T)

    if out:
        centre = np.concatenate([v for v, _ in out.values()]).mean(axis=0)
        out = {k: (v - centre, t) for k, (v, t) in out.items()}
    return out, fell_back


def build(mesh_type: str, scale_um: float = 10.0, seed: int = 0,
          grid: int = GRID,
          params: Optional[ShapeParams] = None) -> Morphology:
    """Build geometry for one ``mesh_type``, in micrometres.

    An unknown type falls back to a sphere and says so in ``fell_back``. The
    saying-so is the point: a shape that silently becomes a sphere while still
    being labelled a neutrophil is how a picture starts claiming more than the
    data supports.
    """
    from cognisom.imaging.mesh_generation import MeshGenerator

    builder = BUILDERS.get(mesh_type)
    fell_back = builder is None
    if fell_back:
        log.info("no builder for mesh_type %r; falling back to a sphere", mesh_type)
        builder = _b_sphere

    # The entity's own proportions, or the documented fallbacks.
    p = params if params is not None else NO_PARAMS
    g = _grid(grid)
    parts = builder(g, seed, p)

    labels = np.zeros(g.shape[:3], np.int32)
    for label in sorted(parts):
        labels[parts[label]] = label

    # The grid spans [-1, 1], so its full width is 2 units across `grid`
    # voxels. For scale_um to mean the width of the whole shape, one voxel is
    # scale_um / grid. Multiplying by 2 here made every shape 2x its stated
    # size: a "20 um" cell measured 35 um across.
    resolution = float(scale_um) / grid
    sim = MeshGenerator(resolution=resolution, simplify=False).labels_to_mesh(labels)

    verts: List[np.ndarray] = []
    tris: List[np.ndarray] = []
    offset = 0
    for comp in sim.compartments:
        if comp.mesh is None:
            continue
        v = np.asarray(comp.mesh.vertices, dtype=float)
        t = np.asarray(comp.mesh.triangles, dtype=int) + offset
        verts.append(v); tris.append(t); offset += len(v)

    V = np.concatenate(verts) if verts else np.zeros((0, 3))
    T = np.concatenate(tris) if tris else np.zeros((0, 3), int)
    if len(V):
        V = V - V.mean(axis=0)

    return Morphology(
        vertices=V, triangles=T, mesh_type=mesh_type,
        builder=("sphere" if fell_back else mesh_type), fell_back=fell_back,
        parts={PART_NAMES.get(k, str(k)): int(v.sum()) for k, v in parts.items()},
    )
