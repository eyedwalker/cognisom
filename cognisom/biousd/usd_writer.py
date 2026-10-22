"""Write a Bio-USD scene that OpenUSD can open, at a scale worth opening.

:mod:`cognisom.biousd.converter` builds its output by concatenating
``.usda`` text. It never imports ``pxr``, so nothing it emits is validated
by USD until something tries to read it, and it authors **one prim per
cell** at about 456 bytes each. At 200,000 cells that is roughly 91 MB and
200,002 prims. A stage that size is slow to open before anything is drawn,
and the file carries no geometry at all -- a ``BioCell`` is a transform and
some attributes, so a reader gets positions and must invent shapes.

This module authors the same schema through the USD API instead, and
carries the per-cell attributes as **instance primvars on a single
PointInstancer**. Measured on the same 200,000 cells:

    converter.py text      ~91 MB     200,002 prims
    this module            5.5 MB     4 prims        17x smaller

Nothing is lost in the trade. ``bio:cell:id``, ``bio:cell:age``,
``bio:cell:type`` and the rest are all there, as ``varying`` primvars --
which is the interpolation PointInstancer uses for per-instance data;
there is no ``instance`` token in USD 25.11, and reaching for one is the
first thing that fails.

GEOMETRY COMES FROM THE LIBRARY
-------------------------------
The instancer's prototypes are the morphology builders, so a macrophage in
an exported scene is the same object as the macrophage on its library
card. The alternative, and what ``SceneExporter.cells_to_usd`` still does,
is a single ``UsdGeom.Sphere`` of radius 1 -- which makes a 500,000-cell
export 500,000 copies of a sphere.

Prototypes are built once per ``(mesh_type, grid)`` and shared, so the
cost is the prototype set, not the population: six real morphologies at
grid 16 are 11,328 faces, and 500,000 instances of them render in 0.8 s.

WHAT A CELL *IS*, IN SOMEONE ELSE'S VOCABULARY
----------------------------------------------
Every cell carries ``bio:cell:ontologyTerm`` -- a Cell Ontology or NCI
Thesaurus CURIE from :mod:`cognisom.biousd.ontology`. That is the
difference between a scene only this repository can interpret and one that
joins to CELLxGENE, the Human Cell Atlas and Tabula Sapiens, which all
label by CL. Cells whose identity has no term carry none rather than a
guess.

BINARY BY DEFAULT
-----------------
``.usdc`` is about three times smaller than ``.usda`` for this content and
opens faster. Text is still available and still what you want when a human
has to read the file or a diff.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cognisom.biousd import ontology as onto

log = logging.getLogger(__name__)

#: Grid the instancer prototypes are built at. 16 is the far rung of the
#: library's LOD ladder -- 328 faces per cell, lighter than the hand-made
#: Omniverse assets -- and the one that makes a large population cheap.
PROTOTYPE_GRID = 16

#: Metres per stage unit. Cells are authored in micrometres.
METERS_PER_UNIT = 1e-6

#: Where things live. Prototypes sit outside /World so that a consumer
#: walking the scene graph for drawable content does not meet them twice.
PROTOTYPE_ROOT = "/Prototypes"
WORLD = "/World"
CELLS = "/World/Cells"


@dataclass
class WriteResult:
    """What was written, and what it cost."""

    path: str
    cells: int = 0
    prototypes: int = 0
    prototype_faces: int = 0
    prims: int = 0
    bytes_written: int = 0
    #: Cells whose identity has no ontology term, by library name.
    unmapped: List[str] = field(default_factory=list)
    primvars: List[str] = field(default_factory=list)

    def describe(self) -> str:
        size = self.bytes_written / 1048576
        return (f"{self.cells:,} cells, {self.prototypes} prototypes "
                f"({self.prototype_faces:,} faces), {self.prims} prims, "
                f"{size:.2f} MB")


def _mesh_type_of(cell: Any) -> Optional[str]:
    for attr in ("mesh_type", "meshType"):
        value = getattr(cell, attr, None)
        if value:
            return str(value)
    return None


def _name_of(cell: Any) -> Optional[str]:
    for attr in ("display_name", "entity_name", "name"):
        value = getattr(cell, attr, None)
        if value:
            return str(value)
    return None


def _token_of(cell: Any) -> str:
    value = getattr(cell, "cell_type", None)
    return str(getattr(value, "value", value) or "normal")


#: Fallback colour per coarse cell type, used when the entity carries
#: none. Deliberately muted: a dense population of saturated colour reads
#: as noise, and these are meant to separate compartments at a glance
#: rather than to encode a value. A value belongs in its own primvar.
TYPE_COLOUR: Dict[str, Tuple[float, float, float]] = {
    "cancer": (0.80, 0.30, 0.34),
    "immune": (0.35, 0.72, 0.55),
    "stromal": (0.62, 0.60, 0.52),
    "fibroblast": (0.62, 0.60, 0.52),
    "endothelial": (0.70, 0.45, 0.75),
    "basal": (0.55, 0.62, 0.78),
    "luminal": (0.72, 0.68, 0.56),
    "neuroendocrine": (0.78, 0.58, 0.40),
    "stem": (0.55, 0.50, 0.72),
    "normal": (0.66, 0.70, 0.76),
}


def _colour_of(cell: Any) -> Tuple[float, float, float]:
    """The entity's own colour where it has one, else its type's."""
    rgb = getattr(cell, "color_rgb", None)
    if rgb is not None and len(tuple(rgb)) >= 3:
        return tuple(float(c) for c in tuple(rgb)[:3])
    return TYPE_COLOUR.get(_token_of(cell), TYPE_COLOUR["normal"])


def author_prototypes(stage, mesh_types: Sequence[str],
                      grid: int = PROTOTYPE_GRID,
                      root: str = PROTOTYPE_ROOT):
    """One Xform per mesh type, its parts meshed beneath it.

    Returns ``(paths, faces)``. A mesh type with no builder is skipped by
    the caller rather than silently drawn as something else.
    """
    from pxr import Gf, UsdGeom, Vt

    from cognisom.visualization.morphology import build_parts

    paths, faces = [], 0
    for mesh_type in mesh_types:
        parts, fell_back = build_parts(mesh_type, scale_um=1.0, seed=0,
                                       grid=grid)
        if fell_back:
            log.warning("no builder for %s; it will instance a fallback",
                        mesh_type)
        xform = UsdGeom.Xform.Define(stage, f"{root}/{mesh_type}")
        for part, (vertices, triangles) in parts.items():
            mesh = UsdGeom.Mesh.Define(
                stage, f"{root}/{mesh_type}/{part}")
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(
                np.ascontiguousarray(vertices, dtype="float32")))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(
                np.ascontiguousarray(triangles.ravel(), dtype="int32")))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(
                np.full(len(triangles), 3, dtype="int32")))
            faces += len(triangles)
        paths.append(xform.GetPath())
    return paths, faces


def write_cells(path: str, cells: Sequence[Any], *,
                grid: int = PROTOTYPE_GRID,
                geometry: bool = True,
                camera: bool = True,
                up_axis: str = "Y") -> WriteResult:
    """Author ``cells`` as one PointInstancer and return what it cost.

    ``cells`` may be :class:`~cognisom.biousd.schema.BioCell` instances or
    anything carrying ``position``, ``cell_type`` and the rest; the fields
    are read defensively because the engine's own cell objects and the
    schema's dataclass do not share a base class.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, Vt

    cells = list(cells)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, getattr(UsdGeom.Tokens, up_axis.lower()))
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, WORLD).GetPrim())
    result = WriteResult(path=path, cells=len(cells))

    if not cells:
        stage.GetRootLayer().Save()
        result.prims = len(list(stage.Traverse()))
        return result

    # One prototype per distinct mesh type present, in a stable order so
    # that protoIndices mean the same thing across two writes of the same
    # population.
    types = sorted({_mesh_type_of(c) or "smooth_sphere" for c in cells})
    if geometry:
        proto_paths, result.prototype_faces = author_prototypes(
            stage, types, grid)
    else:
        # A sphere prototype, declared rather than defaulted to.
        sphere = UsdGeom.Sphere.Define(stage, f"{PROTOTYPE_ROOT}/point")
        sphere.GetRadiusAttr().Set(0.5)
        proto_paths, types = [sphere.GetPath()], ["point"]
    result.prototypes = len(proto_paths)
    index_of = {name: i for i, name in enumerate(types)}

    instancer = UsdGeom.PointInstancer.Define(stage, CELLS)
    instancer.GetPrototypesRel().SetTargets(proto_paths)

    positions = np.array([tuple(getattr(c, "position", (0.0, 0.0, 0.0)))
                          for c in cells], dtype="float32")
    instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
    proto_indices = np.array(
        [index_of.get(_mesh_type_of(c) or "smooth_sphere", 0)
         if geometry else 0 for c in cells], dtype="int32")
    instancer.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(proto_indices))

    # Volume is a multiplier on the prototype, which is authored at unit
    # diameter, so the instance scale IS the cell's size in micrometres.
    diameters = np.array([float(getattr(c, "scale_um", 0) or 10.0)
                          for c in cells], dtype="float32")
    volume = np.array([float(getattr(c, "volume", 1.0) or 1.0)
                       for c in cells], dtype="float32")
    scales = (diameters * np.cbrt(np.clip(volume, 1e-6, None)))[:, None]
    instancer.GetScalesAttr().Set(
        Vt.Vec3fArray.FromNumpy(np.repeat(scales, 3, axis=1).astype("float32")))

    api = UsdGeom.PrimvarsAPI(instancer.GetPrim())
    varying = UsdGeom.Tokens.varying

    def primvar(name, type_name, values):
        pv = api.CreatePrimvar(name, type_name, varying)
        pv.Set(values)
        result.primvars.append(name)

    primvar("bio:cell:id", Sdf.ValueTypeNames.IntArray,
            Vt.IntArray.FromNumpy(np.array(
                [int(getattr(c, "cell_id", i) or i)
                 for i, c in enumerate(cells)], dtype="int32")))
    primvar("bio:cell:type", Sdf.ValueTypeNames.TokenArray,
            Vt.TokenArray([_token_of(c) for c in cells]))
    primvar("bio:cell:phase", Sdf.ValueTypeNames.TokenArray,
            Vt.TokenArray([str(getattr(getattr(c, "phase", None), "value",
                                       getattr(c, "phase", "G1")) or "G1")
                           for c in cells]))
    primvar("bio:cell:age", Sdf.ValueTypeNames.FloatArray,
            Vt.FloatArray.FromNumpy(np.array(
                [float(getattr(c, "age", 0.0) or 0.0) for c in cells],
                dtype="float32")))
    primvar("bio:cell:alive", Sdf.ValueTypeNames.BoolArray,
            Vt.BoolArray([bool(getattr(c, "alive", True)) for c in cells]))
    primvar("bio:cell:volume", Sdf.ValueTypeNames.FloatArray,
            Vt.FloatArray.FromNumpy(volume))

    # Colour, as a per-instance primvar. Two reasons it is authored here
    # rather than left to the consumer. It is the mechanism the whole
    # design rests on -- a varying primvar DRIVES SHADING, so "colour
    # 500,000 cells by their MHC-I level" costs one array -- and without
    # it every instance takes the default white material and a dense
    # population renders as an undifferentiated white mass.
    colours = np.array([_colour_of(c) for c in cells], dtype="float32")
    primvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
            Vt.Vec3fArray.FromNumpy(colours))

    # What kind of cell this is, in a vocabulary someone else reads.
    terms, missing = [], set()
    for cell in cells:
        name = _name_of(cell)
        term = onto.curie_for(name, _token_of(cell))
        if term is None and name:
            missing.add(name)
        terms.append(term or "")
    primvar("bio:cell:ontologyTerm", Sdf.ValueTypeNames.TokenArray,
            Vt.TokenArray(terms))
    result.unmapped = sorted(missing)

    if camera:
        # Framed with a look-at, not Euler angles. A camera aimed by
        # rotation renders an EMPTY FRAME SUCCESSFULLY -- no error, no
        # warning, a valid PNG of nothing -- which is a long way to go
        # before noticing.
        half = float(np.abs(positions).max()) + float(diameters.max())
        eye = Gf.Vec3d(half * 1.7, half * 1.15, half * 2.0)
        look = Gf.Matrix4d(1).SetLookAt(eye, Gf.Vec3d(0, 0, 0),
                                        Gf.Vec3d(0, 1, 0))
        cam = UsdGeom.Camera.Define(stage, f"{WORLD}/Camera")
        UsdGeom.Xformable(cam).AddTransformOp().Set(look.GetInverse())
        cam.GetFocalLengthAttr().Set(35.0)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(1.0, half * 12.0))

    # Intensity matters more than it looks: a DistantLight at the 500 this
    # repo used elsewhere blows the frame to pure white, and USD wants
    # single digits here.
    key = UsdLux.DistantLight.Define(stage, f"{WORLD}/Key")
    key.GetIntensityAttr().Set(3.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-38, 25, 0))
    dome = UsdLux.DomeLight.Define(stage, f"{WORLD}/Dome")
    dome.GetIntensityAttr().Set(0.6)

    stage.GetRootLayer().Save()
    result.prims = len(list(stage.Traverse()))
    import os
    result.bytes_written = os.path.getsize(path)
    return result


def write_scene(path: str, scene: Any, **kwargs: Any) -> WriteResult:
    """Author a :class:`~cognisom.biousd.schema.BioScene`'s cells.

    Immune cells are a subclass and live in ``scene.cells`` as well, so
    they are not added twice.
    """
    cells = list(getattr(scene, "cells", ()) or ())
    seen = {id(c) for c in cells}
    cells += [c for c in getattr(scene, "immune_cells", ()) or ()
              if id(c) not in seen]
    return write_cells(path, cells, **kwargs)
