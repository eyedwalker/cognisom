"""Which way the surface faces.

Every mesh this repository built was inside-out, and had been since the
first one. All 62 parts of all 32 morphology builders came out with
NEGATIVE signed volume, which for a closed surface is the definition of
inverted winding. 2,604 tests passed over it, because not one of them
asked.

It is not a mistake anybody made. ``skimage.measure.marching_cubes``
contradicts itself: the vertex normals it returns point OUTWARD -- on a
control ball, mean dot with the outward radial +0.93 -- while the face
order it returns puts the right-hand-rule normal INWARD on 100% of
faces. Its own documentation mentions the left-hand rule for the default
``gradient_direction``. Two halves of one return value, disagreeing
about which way the surface faces, and nothing downstream reconciled
them.

It is invisible until it is rendered. No exporter here authors
``orientation``, so USD applies its default of ``rightHanded`` with
``doubleSided`` false, and an inverted mesh is backface-culled or lit
from inside. The single place it did not show is
``interaction_usd.py:644``, which sets ``doubleSided`` -- hiding the
symptom rather than fixing it.

Signed volume is the test that matters: ``sum(dot(a, cross(b, c))) / 6``
is positive only when faces are counter-clockwise seen from outside, and
it does not care whether the shape is convex. The outward-normal
fraction is reported too, but it is only a sanity check -- a concave
shape such as ``stellate_branching`` legitimately has faces pointing
back toward its own centroid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.visualization.morphology import BUILDERS, build_parts  # noqa: E402

GRID = 40


def signed_volume(vertices, faces) -> float:
    """Six times the enclosed volume; positive when wound outward."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def outward_fraction(vertices, faces) -> float:
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    normal = np.cross(b - a, c - a)
    midpoint = (a + b + c) / 3.0
    return float((np.einsum("ij,ij->i", normal,
                            midpoint - v.mean(axis=0)) > 0).mean())


@pytest.mark.parametrize("mesh_type", sorted(BUILDERS))
def test_every_part_of_every_builder_faces_outward(mesh_type):
    parts, _fell_back = build_parts(mesh_type, seed=1, grid=GRID)
    assert parts, mesh_type
    for name, (vertices, faces) in sorted(parts.items()):
        if len(faces) == 0:
            continue
        volume = signed_volume(vertices, faces)
        assert volume > 0, (
            f"{mesh_type}/{name} is inside-out: signed volume {volume:+.3f}. "
            "USD defaults to rightHanded and single-sided, so this renders "
            "as a hole or lights from within.")


def test_the_control_shows_what_the_raw_library_does():
    """The check is only worth having if it can tell the two apart, and
    the inverted case is what skimage hands us before the fix.
    """
    from skimage import measure

    axis = np.linspace(-1, 1, 48)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    ball = ((x ** 2 + y ** 2 + z ** 2) < 0.6 ** 2).astype(np.float32)
    vertices, faces, normals, _ = measure.marching_cubes(ball, level=0.5)

    # as returned: inverted
    assert signed_volume(vertices, faces) < 0
    assert outward_fraction(vertices, faces) == 0.0
    # reversed: correct
    assert signed_volume(vertices, faces[:, ::-1]) > 0
    assert outward_fraction(vertices, faces[:, ::-1]) == 1.0

    # and the normals skimage returns were pointing outward all along,
    # which is why the two disagreed
    radial = vertices - vertices.mean(axis=0)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    assert np.einsum("ij,ij->i", normals, radial).mean() > 0.8


def test_the_generator_itself_returns_outward_triangles():
    """Fixed at the chokepoint, so anything that meshes a volume gets
    it -- not only the morphology builders."""
    from cognisom.imaging.mesh_generation import MeshGenerator

    axis = np.linspace(-1, 1, 40)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    blob = ((x ** 2 + y ** 2 + z ** 2) < 0.55 ** 2).astype(np.float32)
    mesh = MeshGenerator()._marching_cubes(blob, level=0.5)
    assert mesh is not None
    assert signed_volume(mesh.vertices, mesh.triangles) > 0


def test_a_rendered_cell_encloses_a_positive_volume():
    """The consequence, stated in the units a reader cares about: a cell
    has an inside."""
    parts, _ = build_parts("smooth_sphere", scale_um=8.0, seed=1, grid=GRID)
    vertices, faces = parts["body"]
    # /6 is already applied; this is the enclosed volume in the builder's
    # own units, and it must be a real volume rather than its negative.
    assert signed_volume(vertices, faces) > 0
    assert outward_fraction(vertices, faces) > 0.95
