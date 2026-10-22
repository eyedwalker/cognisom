"""A 500,000-cell export was 500,000 copies of one sphere.

``cells_to_usd`` authored its PointInstancer with a single prototype --
``UsdGeom.Sphere``, radius 1 -- while the library held 32 real
morphologies that the entity library page had been drawing all along. The
same cell was a ruffled macrophage on its card and a ball in the scene,
and nothing said so.

The prototypes now come from the library, resolved through
``ARCHETYPE_ENTITY``, which is the route ``cell_scene`` takes for a card.
So the macrophage in a population export and the macrophage on its card
are one object, which is a claim this repository has made in a docstring
for some time without it being true of this path.

**A state is not a shape.** Several of the exporter's cell types --
``dead``, ``dividing``, ``differentiated``, ``progenitor``, ``normal`` --
describe what a cell is doing, not what kind of cell it is, so they
resolve to nothing and instance an explicitly named sphere. Drawing them
as whichever morphology sorted first would have been worse than the
original defect, because it would have looked deliberate.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom                                    # noqa: E402

from cognisom.visualization.exporters import (                  # noqa: E402
    UNRESOLVED_PROTOTYPE, SceneExporter, cell_mesh_type)

#: Types that name a kind of cell, and the library shape each resolves to.
SHAPED = {
    "cancer": "irregular_ruffled",
    "t_cell": "smooth_sphere",
    "macrophage": "ruffled_pseudopod",
    "fibroblast": "elongated_spindle",
    "endothelial": "flat_squamous",
    "nk_cell": "granular_sphere",
}

#: Types that name a STATE. These have no morphology and must not borrow
#: one.
STATES = ("normal", "dead", "dividing", "differentiated", "progenitor")


class _Cell:
    __slots__ = ("position", "cell_type", "state", "radius", "cell_id")

    def __init__(self, position, cell_type, index):
        self.position = position
        self.cell_type = cell_type
        self.state = "alive"
        self.radius = 6.0
        self.cell_id = index


def _population(n, types):
    rng = np.random.default_rng(0)
    return [_Cell(tuple(float(x) for x in rng.uniform(0, 400, 3)),
                  types[i % len(types)], i) for i in range(n)]


def _instancer(stage):
    found = [p for p in stage.Traverse() if p.IsA(UsdGeom.PointInstancer)]
    assert len(found) == 1, f"expected one instancer, got {len(found)}"
    return UsdGeom.PointInstancer(found[0])


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("usd") / "pop.usda")
    cells = _population(1200, list(SHAPED) + list(STATES))
    SceneExporter.cells_to_usd(cells, path)
    return Usd.Stage.Open(path)


def test_each_shaped_type_resolves_to_its_library_morphology():
    for cell_type, mesh_type in SHAPED.items():
        assert cell_mesh_type(_Cell((0, 0, 0), cell_type, 0)) == mesh_type


def test_a_state_resolves_to_no_shape_at_all():
    for state in STATES:
        assert cell_mesh_type(_Cell((0, 0, 0), state, 0)) is None, (
            f"{state!r} is something a cell is doing, not a kind of cell; "
            "giving it a morphology invents one")


def test_the_prototypes_are_meshes_and_not_one_sphere(exported):
    instancer = _instancer(exported)
    targets = instancer.GetPrototypesRel().GetTargets()
    names = {str(t).rsplit("/", 1)[-1] for t in targets}
    assert set(SHAPED.values()) <= names, sorted(names)

    faces = 0
    for target in targets:
        for prim in Usd.PrimRange(exported.GetPrimAtPath(target)):
            if prim.IsA(UsdGeom.Mesh):
                faces += len(
                    UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())
    assert faces > 2000, f"only {faces} faces of prototype geometry"


def test_the_unresolved_cells_get_a_sphere_that_says_so(exported):
    """Named, so a reader can see which cells had no shape."""
    instancer = _instancer(exported)
    targets = [str(t) for t in instancer.GetPrototypesRel().GetTargets()]
    assert any(t.endswith(UNRESOLVED_PROTOTYPE) for t in targets), targets
    fallback = exported.GetPrimAtPath(f"/Prototypes/{UNRESOLVED_PROTOTYPE}")
    assert fallback.IsA(UsdGeom.Sphere)

    index = targets.index(f"/Prototypes/{UNRESOLVED_PROTOTYPE}")
    used = np.asarray(instancer.GetProtoIndicesAttr().Get())
    share = float((used == index).mean())
    expected = len(STATES) / (len(SHAPED) + len(STATES))
    assert share == pytest.approx(expected, abs=0.05), (
        f"{share:.0%} of instances are unresolved; expected {expected:.0%}")


def test_one_instancer_carries_them_all(exported):
    """Several prototypes share one instancer via protoIndices.

    NestedInstancingManager authors one instancer per prototype, which is
    why it is no longer used on this path: the export's own tests read
    ``_instancers(stage)[0]``.
    """
    instancer = _instancer(exported)
    n = len(instancer.GetPositionsAttr().Get())
    assert n == 1200
    assert len(instancer.GetProtoIndicesAttr().Get()) == n
    assert len(instancer.GetPrototypesRel().GetTargets()) == len(SHAPED) + 1


def test_per_instance_state_and_colour_survive(exported):
    """Both were primvars before and must stay primvars."""
    api = UsdGeom.PrimvarsAPI(_instancer(exported).GetPrim())
    for name in ("displayColor", "cognisom:cellState"):
        primvar = api.GetPrimvar(name)
        assert primvar and primvar.HasAuthoredValue(), name
        assert len(primvar.Get()) == 1200
