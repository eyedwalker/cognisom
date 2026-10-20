"""Surface and interior detail for a cell, at scales a voxel grid cannot reach.

``morphology`` builds a cell's gross shape by filling an occupancy grid and
running marching cubes over it. That settles the silhouette and it cannot
do anything finer, because the grid sets the smallest feature: at 64 voxels
across a 7 um lymphocyte one voxel is 0.11 um, so a microvillus -- 0.1 um
across -- is one voxel wide and comes out as noise or as nothing.

That is why the library's lymphocytes are bare spheres. It is not a missing
builder; it is the wrong representation for the feature. A resting T cell is
densely covered in microvilli, and they are not decoration: TCR concentrates
on microvillar tips and the villi are what sweep a surface for antigen. A
smooth sphere is not a plain T cell, it is a wrong one.

So this module works on the MESH, not the grid, and on the MASKS for
interiors:

* :func:`sample_surface` draws points uniformly per unit AREA, with outward
  normals, so a feature density means the same thing on a 7 um lymphocyte
  and a 30 um fibroblast.
* :func:`sample_interior` draws points inside a mask, so organelles can be
  placed at a real volume fraction instead of scattered by eye.
* :func:`quat_z_to` orients a prototype's own axis onto a normal, which is
  what lets one microvillus prototype stand in for hundreds of instances.

Nothing here is a measurement. A count and a length come from the caller,
and whether those have provenance is the caller's problem to declare -- see
``morphology_usd``, which records every shape as a depiction.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Instances:
    """Positions, orientations and scales for one instanced prototype.

    The shape a USD ``PointInstancer`` wants, and the shape three.js wants
    for an ``InstancedMesh``: hundreds of microvilli are one prototype and
    a transform each, never hundreds of meshes.
    """

    positions: np.ndarray        # (N, 3) micrometres
    orientations: np.ndarray     # (N, 4) quaternion, real part first
    scales: np.ndarray           # (N, 3)

    def __len__(self) -> int:
        return len(self.positions)


# ── sampling ────────────────────────────────────────────────────────────

def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area of each triangle."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def triangle_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Unit normal of each triangle, in the winding's own direction."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, length, out=np.zeros_like(n), where=length > 0)


def sample_surface(vertices: np.ndarray, triangles: np.ndarray, count: int,
                   rng: np.random.Generator,
                   outward_from: Optional[Sequence[float]] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """``(points, normals)``: ``count`` points spread evenly over the area.

    Area-weighted, not vertex-weighted. Marching cubes puts vertices where
    the surface curves, so sampling vertices crowds features into the
    creases and leaves flat ground bare -- a ruffled macrophage would grow
    all its microvilli in the folds.

    Normals are forced to point away from ``outward_from`` (the shape's
    centroid by default). Marching-cubes winding is consistent but its
    direction depends on the sign convention of the field it ran over, and
    a microvillus pointing inward is invisible rather than wrong-looking,
    which is worse.
    """
    if len(triangles) == 0 or count <= 0:
        empty = np.zeros((0, 3), np.float32)
        return empty, empty

    areas = triangle_areas(vertices, triangles)
    total = float(areas.sum())
    if total <= 0:
        empty = np.zeros((0, 3), np.float32)
        return empty, empty

    picked = rng.choice(len(triangles), size=count, p=areas / total)
    a, b, c = (vertices[triangles[picked, i]] for i in range(3))

    # Uniform barycentric coordinates over a triangle: the square-root
    # folds the unit square onto it without bunching along one edge.
    u = np.sqrt(rng.random((count, 1)))
    v = rng.random((count, 1))
    points = a + u * (b - a) + u * v * (c - b)

    normals = triangle_normals(vertices, triangles)[picked]
    centre = (np.asarray(outward_from, np.float64) if outward_from is not None
              else vertices.mean(axis=0))
    flip = np.einsum("ij,ij->i", normals, points - centre) < 0
    normals[flip] *= -1.0
    return points.astype(np.float32), normals.astype(np.float32)


def mask_frame(mask: np.ndarray, vertices: np.ndarray, scale_um: float
               ) -> Tuple[float, np.ndarray]:
    """``(resolution, offset)`` taking mask voxel indices into mesh space.

    The two frames do not coincide and cannot be assumed to.
    ``morphology.build_parts`` puts a vertex at ``index * resolution`` and
    then subtracts the mean of EVERY part's vertices, so the mesh it
    returns is centred on a quantity the mask knows nothing about -- and
    which has already been subtracted, so it cannot be read back off the
    result.

    Aligning the two bounding boxes recovers it without an API change.
    Marching cubes puts its surface at the half-way isolevel, so the
    surface sits about half a voxel outside the last filled voxel on
    every side; the box centres therefore agree to well under a voxel,
    which is the accuracy that matters when the question is whether an
    organelle is inside the membrane.

    Getting this wrong is not subtle in the output: with a lymphocyte's
    cytoplasm a thin shell under the membrane, a frame off by one voxel
    puts mitochondria outside the cell.
    """
    idx = np.argwhere(mask)
    resolution = float(scale_um) / mask.shape[0]
    if len(idx) == 0 or len(vertices) == 0:
        return resolution, np.zeros(3)
    mask_centre = (idx.min(axis=0) + idx.max(axis=0)) / 2.0 * resolution
    mesh_centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    return resolution, np.asarray(mesh_centre, np.float64) - mask_centre


def sample_interior(mask: np.ndarray, count: int, rng: np.random.Generator,
                    scale_um: float, *, margin: float = 0.0,
                    frame: Optional[Tuple[float, np.ndarray]] = None
                    ) -> np.ndarray:
    """``count`` points inside ``mask``, in the mesh's own coordinates.

    The mask is the builder's own occupancy grid, so membership is exact
    rather than a point-in-mesh test against a marching-cubes surface --
    and it is the grid that defines the shape in the first place.

    ``margin`` in micrometres keeps points off the boundary, so an
    organelle of that radius sits wholly inside the cell instead of
    poking through the membrane. ``frame`` comes from :func:`mask_frame`
    and is what puts the result in the same space as the mesh; without it
    the points are placed in the grid's own centred frame, which is only
    correct when the shape is symmetric about the grid centre.
    """
    if count <= 0:
        return np.zeros((0, 3), np.float32)

    inside = mask
    if margin > 0:
        try:
            from scipy.ndimage import distance_transform_edt
            grid = mask.shape[0]
            voxel_um = scale_um / grid
            depth = distance_transform_edt(mask) * voxel_um
            inside = depth >= margin
            if not inside.any():
                # Nothing is that deep -- the organelle does not fit. Fall
                # back to the unshrunk mask and say so, rather than
                # returning nothing and silently losing the compartment.
                log.debug("margin %.3f um leaves no interior; using the "
                          "whole mask", margin)
                inside = mask
        except ImportError:                     # pragma: no cover
            inside = mask

    idx = np.argwhere(inside)
    if len(idx) == 0:
        return np.zeros((0, 3), np.float32)
    chosen = idx[rng.choice(len(idx), size=count,
                            replace=len(idx) < count)].astype(np.float64)
    # Jitter within the voxel, or every point lands on a lattice site and
    # a hundred organelles line up in rows.
    chosen += rng.random(chosen.shape) - 0.5
    if frame is not None:
        resolution, offset = frame
        return (chosen * resolution + offset).astype(np.float32)
    grid = mask.shape[0]
    return (((chosen / (grid - 1.0)) * 2.0 - 1.0)
            * (scale_um / 2.0)).astype(np.float32)


# ── orientation ─────────────────────────────────────────────────────────

def quat_z_to(directions: np.ndarray) -> np.ndarray:
    """Quaternions rotating +Z onto each direction, real part first.

    USD's implicit cylinder and cone lie along an axis attribute, and
    three.js's lie along +Y, so an instanced projection is placed by
    rotating a fixed axis onto the surface normal. Two cases have to be
    handled apart: parallel needs no rotation, and antiparallel has no
    unique axis, so the cross product supplies none and any perpendicular
    will do.
    """
    v = np.atleast_2d(np.asarray(directions, np.float64))
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    # A zero-length direction has no rotation to describe. It is not a
    # hypothetical: a degenerate triangle has a zero normal, and left in
    # the general branch below it divides by a zero cross product and
    # yields a NaN quaternion -- which renders as nothing at all, so the
    # instance silently disappears instead of looking wrong.
    usable = (norm > 0).ravel()
    v = np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)

    out = np.zeros((len(v), 4), np.float64)
    out[:, 0] = 1.0                                   # identity by default

    dot = np.clip(v[:, 2], -1.0, 1.0)                 # dot with +Z
    opposed = usable & (dot < -1.0 + 1e-9)
    out[opposed] = (0.0, 1.0, 0.0, 0.0)               # 180 deg about X

    general = usable & (~opposed) & (dot < 1.0 - 1e-9)
    if general.any():
        vg = v[general]
        axis = np.cross(np.array([0.0, 0.0, 1.0]), vg)
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        angle = np.arccos(dot[general])
        s = np.sin(angle / 2.0)[:, None]
        out[general, 0] = np.cos(angle / 2.0)
        out[general, 1:] = axis * s
    return out


# ── the features themselves ─────────────────────────────────────────────

def projections(vertices: np.ndarray, triangles: np.ndarray, *,
                count: int, length_um: float, radius_um: float,
                seed: int = 0, length_spread: float = 0.35,
                embed: float = 0.25) -> Instances:
    """Microvilli, filopodia or spikes standing off a surface.

    ``count`` is a number per cell, so a caller can hand over a measured
    density and have it mean that. ``length_spread`` is the fractional
    standard deviation of length: a real brush border is not uniform, and
    an exactly uniform one reads as a machined object rather than a cell.

    ``embed`` sinks each projection into the surface by that fraction of
    its length, so the base meets the membrane instead of floating on it.
    """
    rng = np.random.default_rng(seed)
    points, normals = sample_surface(vertices, triangles, count, rng)
    if len(points) == 0:
        return Instances(points, np.zeros((0, 4), np.float32), points)

    lengths = length_um * np.clip(
        1.0 + rng.normal(scale=length_spread, size=len(points)), 0.25, 2.5)
    # A prototype of unit height centred on its own origin: its base sits
    # at the membrane once it is pushed out by half its height, less the
    # embedded part.
    offset = lengths * (0.5 - embed)
    positions = points + normals * offset[:, None]
    scales = np.stack([np.full(len(points), radius_um * 2.0),
                       np.full(len(points), radius_um * 2.0),
                       lengths], axis=1)
    return Instances(positions.astype(np.float32),
                     quat_z_to(normals).astype(np.float32),
                     scales.astype(np.float32))


def organelles(mask: np.ndarray, *, scale_um: float, count: int,
               radius_um: float, seed: int = 0,
               radius_spread: float = 0.3,
               frame: Optional[Tuple[float, np.ndarray]] = None) -> Instances:
    """A population of rounded organelles inside a compartment.

    ``count`` and ``radius_um`` together set a volume fraction, which is
    the quantity morphometry actually reports -- so a caller with a
    measured fraction and a measured size can derive the count rather than
    picking one that looks right.
    """
    rng = np.random.default_rng(seed)
    positions = sample_interior(mask, count, rng, scale_um,
                                margin=radius_um, frame=frame)
    if len(positions) == 0:
        return Instances(positions, np.zeros((0, 4), np.float32), positions)
    radii = radius_um * np.clip(
        1.0 + rng.normal(scale=radius_spread, size=len(positions)), 0.3, 2.2)
    # Unit sphere prototype, so the scale IS the radius; mitochondria are
    # given a long axis because they are not round.
    scales = np.stack([radii, radii, radii], axis=1)
    orient = quat_z_to(rng.normal(size=(len(positions), 3)))
    return Instances(positions.astype(np.float32),
                     orient.astype(np.float32),
                     scales.astype(np.float32))


def volume_fraction_count(fraction: float, compartment_um3: float,
                          radius_um: float) -> int:
    """How many spheres of ``radius_um`` fill ``fraction`` of a volume.

    The honest direction of the calculation: morphometry reports a volume
    fraction and a size, and the count follows. Choosing a count directly
    is choosing a fraction without saying so.
    """
    if radius_um <= 0 or compartment_um3 <= 0 or fraction <= 0:
        return 0
    one = 4.0 / 3.0 * math.pi * radius_um ** 3
    return max(0, int(round(fraction * compartment_um3 / one)))


def mask_volume_um3(mask: np.ndarray, scale_um: float) -> float:
    """Volume of a mask in cubic micrometres."""
    grid = mask.shape[0]
    voxel = (scale_um / grid) ** 3
    return float(mask.sum()) * voxel
