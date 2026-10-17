"""Turn a protein structure into geometry, carrying its confidence with it.

A predicted model covers the whole sequence, which is why AlphaFold is the
practical source for a library of 137 proteins. It is also why rendering one
naively is the most misleading thing this project could do: the model always
produces coordinates, including through regions where the protein has no single
shape. Drawing those as solidly as a folded domain shows a viewer a structure
the model does not claim.

Measured on this library's own proteins, the fraction of residues above pLDDT 70
runs from 11% to 100%, median 79%, and nineteen proteins are majority
low-confidence:

    PSGL-1              11% confident
    FOXA1               20%
    MYC                 23%
    Androgen Receptor   38%   (58% below pLDDT 50)
    TLR4                90%

AR matters most here, because the whole AR signalling model is built on it, and
its 920-residue prediction is majority disorder. So confidence is not an
optional overlay in this module: :func:`backbone_tube` varies radius and colour
by pLDDT, and a caller can thin or drop the low-confidence stretches outright.
The default is to thin rather than drop, because a gap in a chain reads as
"nothing here" when the truth is "no fixed shape here".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: AlphaFold's own confidence bands, and the colours its viewer uses.
BANDS = (
    (90.0, "very_high", (0.051, 0.341, 0.827)),   # dark blue
    (70.0, "confident", (0.396, 0.792, 0.937)),   # light blue
    (50.0, "low",       (0.996, 0.859, 0.333)),   # yellow
    (0.0,  "very_low",  (0.996, 0.490, 0.255)),   # orange
)

#: Tube radius multiplier per band. A disordered stretch is drawn thin rather
#: than removed: a break in the chain reads as "nothing is here", which is a
#: different and equally wrong claim from "no fixed shape is here".
BAND_RADIUS = {"very_high": 1.0, "confident": 0.85, "low": 0.5, "very_low": 0.3}


def band_of(plddt: float) -> str:
    for threshold, name, _ in BANDS:
        if plddt >= threshold:
            return name
    return "very_low"


def band_colour(plddt: float) -> Tuple[float, float, float]:
    for threshold, _, colour in BANDS:
        if plddt >= threshold:
            return colour
    return BANDS[-1][2]


def smooth_trace(xyz: np.ndarray, subdivisions: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Catmull-Rom through the alpha carbons.

    A raw CA trace is a 3.8 angstrom zig-zag; drawn as a tube it looks like a
    chain of beads rather than a backbone. Returns the smoothed points and, for
    each one, the fractional index into the original residues so per-residue
    values can be carried onto the finer curve.
    """
    xyz = np.asarray(xyz, dtype=float)
    n = len(xyz)
    if n < 2:
        return xyz, np.arange(n, dtype=float)
    padded = np.vstack([xyz[0], xyz, xyz[-1]])

    pts: List[np.ndarray] = []
    idx: List[float] = []
    for i in range(n - 1):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        for s in range(subdivisions):
            t = s / subdivisions
            t2, t3 = t * t, t * t * t
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
            idx.append(i + t)
    pts.append(xyz[-1])
    idx.append(float(n - 1))
    return np.array(pts), np.array(idx)


def _frames(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Parallel-transported normal and binormal along a curve.

    Frenet frames flip at inflection points, which puts a visible twist in the
    tube. Transporting one frame along the curve avoids that.
    """
    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-9)

    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(tangents[0] @ seed)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    normal = seed - tangents[0] * float(tangents[0] @ seed)
    normal /= max(np.linalg.norm(normal), 1e-9)

    normals = np.zeros_like(points)
    binormals = np.zeros_like(points)
    for i, t in enumerate(tangents):
        normal = normal - t * float(t @ normal)
        n = np.linalg.norm(normal)
        if n < 1e-7:
            alt = np.array([0.0, 0.0, 1.0])
            if abs(float(t @ alt)) > 0.9:
                alt = np.array([1.0, 0.0, 0.0])
            normal = alt - t * float(t @ alt)
            n = np.linalg.norm(normal)
        normal = normal / max(n, 1e-9)
        normals[i] = normal
        binormals[i] = np.cross(t, normal)
    return normals, binormals


@dataclass
class BackboneGeometry:
    vertices: np.ndarray        # (N, 3) angstrom
    triangles: np.ndarray       # (M, 3)
    colours: np.ndarray         # (N, 3) per-vertex RGB from the confidence band
    plddt: np.ndarray           # (N,) per-vertex confidence carried from residues
    n_residues: int
    dropped_residues: int = 0

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)


def backbone_tube(
    ca_xyz: np.ndarray,
    plddt: Optional[Sequence[float]] = None,
    radius: float = 1.6,
    sides: int = 8,
    subdivisions: int = 3,
    confidence_radius: bool = True,
    drop_below: Optional[float] = None,
) -> BackboneGeometry:
    """A tube along the backbone, thinned and coloured by confidence.

    Args:
        ca_xyz: (n_res, 3) alpha-carbon coordinates.
        plddt: per-residue confidence. Without it the tube is uniform and
            uncoloured, which is the right behaviour for an experimental
            structure, where every modelled residue is equally observed.
        confidence_radius: thin the tube through low-confidence stretches.
        drop_below: remove residues under this pLDDT entirely. Off by default:
            a break in the chain reads as "nothing here" rather than "no fixed
            shape here", and for AR that would silently delete more than half
            the protein.
    """
    ca_xyz = np.asarray(ca_xyz, dtype=float)
    n_res = len(ca_xyz)
    if n_res < 2:
        empty = np.zeros((0, 3))
        return BackboneGeometry(empty, np.zeros((0, 3), int), empty,
                                np.zeros(0), n_res)

    conf = None if plddt is None else np.asarray(plddt, dtype=float)
    dropped = 0
    if drop_below is not None and conf is not None:
        keep = conf >= drop_below
        dropped = int((~keep).sum())
        ca_xyz, conf = ca_xyz[keep], conf[keep]
        n_res = len(ca_xyz)
        if n_res < 2:
            empty = np.zeros((0, 3))
            return BackboneGeometry(empty, np.zeros((0, 3), int), empty,
                                    np.zeros(0), n_res, dropped)

    points, idx = smooth_trace(ca_xyz, subdivisions)
    normals, binormals = _frames(points)

    if conf is not None:
        at_point = np.interp(idx, np.arange(len(conf)), conf)
    else:
        at_point = np.full(len(points), 100.0)

    radii = np.full(len(points), radius)
    if conf is not None and confidence_radius:
        radii = radius * np.array([BAND_RADIUS[band_of(p)] for p in at_point])

    ring = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos_r, sin_r = np.cos(ring), np.sin(ring)

    verts = np.empty((len(points) * sides, 3))
    cols = np.empty((len(points) * sides, 3))
    vconf = np.empty(len(points) * sides)
    for i in range(len(points)):
        offset = (normals[i][None, :] * cos_r[:, None]
                  + binormals[i][None, :] * sin_r[:, None]) * radii[i]
        lo, hi = i * sides, (i + 1) * sides
        verts[lo:hi] = points[i] + offset
        cols[lo:hi] = band_colour(at_point[i])
        vconf[lo:hi] = at_point[i]

    tris: List[Tuple[int, int, int]] = []
    for i in range(len(points) - 1):
        a, b = i * sides, (i + 1) * sides
        for s in range(sides):
            s2 = (s + 1) % sides
            tris.append((a + s, b + s, a + s2))
            tris.append((a + s2, b + s, b + s2))

    return BackboneGeometry(verts, np.array(tris, dtype=int), cols, vconf,
                            n_res, dropped)


def confidence_summary(plddt: Sequence[float]) -> dict:
    """Residue counts per band, for the record and for a caption."""
    p = np.asarray(plddt, dtype=float)
    if not len(p):
        return {}
    out = {name: int((p >= lo).sum() - (p >= hi).sum())
           for (lo, name, _), hi in zip(BANDS, [1e9] + [b[0] for b in BANDS[:-1]])}
    out["median"] = float(np.median(p))
    out["confident_fraction"] = float((p > 70).mean())
    return out
