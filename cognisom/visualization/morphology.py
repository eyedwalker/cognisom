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


def _sphere(g, centre=(0, 0, 0), r=1.0, squash=(1, 1, 1)):
    d = (g - np.asarray(centre, np.float32)) / np.asarray(squash, np.float32)
    return np.linalg.norm(d, axis=-1) <= r


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


def _spikes(g, n, inner, outer, thickness, seed):
    """Radial projections: viral envelope spikes, or cell processes."""
    rng = np.random.default_rng(seed)
    out = np.zeros(g.shape[:3], bool)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for d in v:
        out |= _capsule(g, d * inner, d * outer, thickness)
    return out


# ── builders ─────────────────────────────────────────────────────────────
# Each returns a dict of label -> boolean volume. Label 1 is always the body.

def _b_sphere(g, seed):
    return {1: _sphere(g, r=0.92)}


def _b_smooth_sphere(g, seed):
    return {1: _sphere(g, r=0.92), 2: _sphere(g, (0.12, -0.08, 0), 0.42)}


def _b_granular_sphere(g, seed):
    rng = np.random.default_rng(seed)
    body = _sphere(g, r=0.92)
    gran = np.zeros(g.shape[:3], bool)
    for _ in range(26):
        c = rng.normal(scale=0.42, size=3)
        gran |= _sphere(g, c, 0.09)
    return {1: body, 3: gran & body}


def _b_dense_granular(g, seed):
    rng = np.random.default_rng(seed)
    body = _sphere(g, r=0.92)
    gran = np.zeros(g.shape[:3], bool)
    for _ in range(70):
        gran |= _sphere(g, rng.normal(scale=0.46, size=3), 0.075)
    return {1: body, 3: gran & body}


def _b_multilobed_granular(g, seed):
    """Neutrophil: three to five nuclear lobes on a thread, dense granules."""
    rng = np.random.default_rng(seed)
    body = _sphere(g, r=0.92)
    nuc = _lobes(g, 4, 0.30, 0.44, seed)
    gran = np.zeros(g.shape[:3], bool)
    for _ in range(60):
        gran |= _sphere(g, rng.normal(scale=0.5, size=3), 0.07)
    return {1: body, 2: nuc & body, 3: gran & body & ~nuc}


def _b_bilobed_granular(g, seed):
    """Eosinophil: two lobes, coarser granules."""
    rng = np.random.default_rng(seed)
    body = _sphere(g, r=0.92)
    nuc = _sphere(g, (-0.34, 0.10, 0), 0.36) | _sphere(g, (0.34, -0.10, 0), 0.36)
    gran = np.zeros(g.shape[:3], bool)
    for _ in range(34):
        gran |= _sphere(g, rng.normal(scale=0.48, size=3), 0.10)
    return {1: body, 2: nuc & body, 3: gran & body & ~nuc}


def _b_ruffled_pseudopod(g, seed):
    """Macrophage: a body throwing broad irregular processes."""
    rng = np.random.default_rng(seed)
    body = _sphere(g, r=0.66)
    for _ in range(9):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        body |= _capsule(g, d * 0.3, d * rng.uniform(0.85, 1.0), rng.uniform(0.14, 0.24))
    return {1: body, 2: _sphere(g, (0.1, 0.05, 0), 0.30)}


def _b_irregular_ruffled(g, seed):
    rng = np.random.default_rng(seed)
    body = _sphere(g, r=0.70)
    for _ in range(14):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        body |= _sphere(g, d * rng.uniform(0.6, 0.85), rng.uniform(0.14, 0.22))
    return {1: body, 2: _sphere(g, (0, 0, 0), 0.30)}


def _b_stellate_branching(g, seed):
    """Dendritic cell: a small body and long thin radiating processes."""
    body = _sphere(g, r=0.38)
    body |= _spikes(g, 11, 0.25, 0.98, 0.055, seed)
    return {1: body, 2: _sphere(g, (0, 0, 0), 0.21)}


def _b_eccentric_nucleus_er(g, seed):
    """Plasma cell: nucleus pushed to one side, ER filling the rest."""
    body = _sphere(g, r=0.92)
    nuc = _sphere(g, (-0.40, 0.0, 0.0), 0.38)
    rng = np.random.default_rng(seed)
    er = _sphere(g, (0.18, 0, 0), 0.62) & ~_sphere(g, (0.18, 0, 0), 0.30)
    er &= rng.random(g.shape[:3]) < 0.55
    return {1: body, 2: nuc & body, 4: er & body & ~nuc}


def _b_elongated_spindle(g, seed):
    """Fibroblast: a long thin cell with tapered ends."""
    body = _sphere(g, r=0.95, squash=(1.0, 0.26, 0.20))
    return {1: body, 2: _sphere(g, (0, 0, 0), 0.30, squash=(1.5, 1, 1))}


def _b_flat_squamous(g, seed):
    body = _sphere(g, r=0.95, squash=(1.0, 1.0, 0.16))
    return {1: body, 2: _sphere(g, (0, 0, 0), 0.34, squash=(1, 1, 0.5))}


def _b_polarized_columnar(g, seed):
    """Columnar epithelium: tall, with the nucleus held basally."""
    x, y, z = g[..., 0], g[..., 1], g[..., 2]
    body = (np.abs(x) < 0.40) & (np.abs(y) < 0.40) & (np.abs(z) < 0.95)
    return {1: body, 2: _sphere(g, (0, 0, -0.52), 0.26)}


def _b_cuboidal(g, seed):
    x, y, z = g[..., 0], g[..., 1], g[..., 2]
    body = (np.abs(x) < 0.62) & (np.abs(y) < 0.62) & (np.abs(z) < 0.62)
    return {1: body, 2: _sphere(g, (0, 0, 0), 0.26)}


def _b_cuboidal_tall(g, seed):
    x, y, z = g[..., 0], g[..., 1], g[..., 2]
    body = (np.abs(x) < 0.46) & (np.abs(y) < 0.46) & (np.abs(z) < 0.86)
    return {1: body, 2: _sphere(g, (0, 0, -0.40), 0.24)}


def _b_flat_activated_adhesion(g, seed):
    body = _sphere(g, r=0.95, squash=(1.0, 0.85, 0.18))
    body |= _spikes(g, 7, 0.5, 0.98, 0.06, seed) & (np.abs(g[..., 2]) < 0.18)
    return {1: body, 2: _sphere(g, (0, 0, 0), 0.28, squash=(1, 1, 0.55))}


# viruses
def _b_enveloped_corona_spikes(g, seed):
    return {1: _sphere(g, r=0.62) | _spikes(g, 54, 0.58, 0.95, 0.055, seed),
            5: _sphere(g, r=0.34)}


def _b_enveloped_ha_na_spikes(g, seed):
    return {1: _sphere(g, r=0.66) | _spikes(g, 72, 0.62, 0.92, 0.042, seed),
            5: _sphere(g, r=0.36)}


def _b_enveloped_sparse_spikes(g, seed):
    return {1: _sphere(g, r=0.74) | _spikes(g, 16, 0.70, 0.95, 0.05, seed),
            5: _sphere(g, r=0.40)}


def _b_enveloped_smooth(g, seed):
    return {1: _sphere(g, r=0.85), 5: _sphere(g, r=0.48)}


def _b_enveloped_small(g, seed):
    return {1: _sphere(g, r=0.55), 5: _sphere(g, r=0.30)}


def _b_enveloped_smooth_small(g, seed):
    return {1: _sphere(g, r=0.52), 5: _sphere(g, r=0.28)}


def _b_enveloped_pleomorphic(g, seed):
    rng = np.random.default_rng(seed)
    body = _sphere(g, r=0.60, squash=(1.0, 0.78, 0.62))
    for _ in range(5):
        body |= _sphere(g, rng.normal(scale=0.26, size=3), rng.uniform(0.26, 0.40))
    return {1: body, 5: _sphere(g, r=0.26)}


def _b_bullet_shape(g, seed):
    """Rhabdovirus: one rounded end, one flat."""
    z = g[..., 2]
    body = _sphere(g, (0, 0, 0.1), 0.80, squash=(0.62, 0.62, 1.0)) & (z > -0.62)
    return {1: body, 5: _sphere(g, (0, 0, 0.1), 0.34, squash=(0.5, 0.5, 0.9)) & (z > -0.5)}


# bacteria
def _b_rod_flagellated(g, seed):
    rng = np.random.default_rng(seed)
    body = _capsule(g, (-0.50, 0, 0), (0.50, 0, 0), 0.26)
    fla = np.zeros(g.shape[:3], bool)
    for k in range(3):
        prev = np.array([0.52, 0.0, 0.0])
        d = np.array([1.0, 0.0, 0.0])
        for i in range(9):
            d = d + rng.normal(scale=0.42, size=3); d /= np.linalg.norm(d)
            nxt = prev + d * 0.085
            fla |= _capsule(g, prev, nxt, 0.028)
            prev = nxt
    return {1: body, 6: fla}


def _b_rod_acid_fast(g, seed):
    return {1: _capsule(g, (-0.58, 0, 0), (0.58, 0, 0), 0.21)}


def _b_rod_spore_forming(g, seed):
    body = _capsule(g, (-0.58, 0, 0), (0.46, 0, 0), 0.24)
    return {1: body, 7: _sphere(g, (0.40, 0, 0), 0.22)}


def _b_coccus_chain(g, seed):
    out = np.zeros(g.shape[:3], bool)
    for i in range(5):
        out |= _sphere(g, (-0.72 + i * 0.36, 0, 0), 0.20)
    return {1: out}


def _b_coccus_cluster(g, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros(g.shape[:3], bool)
    for _ in range(11):
        out |= _sphere(g, rng.normal(scale=0.36, size=3), 0.21)
    return {1: out}


def _b_diplococcus(g, seed):
    return {1: _sphere(g, (-0.26, 0, 0), 0.34, squash=(0.9, 1, 1))
               | _sphere(g, (0.26, 0, 0), 0.34, squash=(0.9, 1, 1))}


# molecular
def _b_y_shape(g, seed):
    """Immunoglobulin: two Fab arms and an Fc stem."""
    body = _capsule(g, (0, 0, -0.85), (0, 0, -0.05), 0.15)
    body |= _capsule(g, (0, 0, -0.05), (-0.62, 0, 0.72), 0.13)
    body |= _capsule(g, (0, 0, -0.05), (0.62, 0, 0.72), 0.13)
    return {1: body}


def _b_groove(g, seed):
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


def build_parts(mesh_type: str, scale_um: float = 10.0, seed: int = 0,
                grid: int = GRID, cut: bool = False
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

    g = _grid(grid)
    raw = builder(g, seed)
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
        out[PART_NAMES.get(label, str(label))] = (
            np.concatenate(verts), np.concatenate(tris))

    if out:
        centre = np.concatenate([v for v, _ in out.values()]).mean(axis=0)
        out = {k: (v - centre, t) for k, (v, t) in out.items()}
    return out, fell_back


def build(mesh_type: str, scale_um: float = 10.0, seed: int = 0,
          grid: int = GRID) -> Morphology:
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

    g = _grid(grid)
    parts = builder(g, seed)

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
