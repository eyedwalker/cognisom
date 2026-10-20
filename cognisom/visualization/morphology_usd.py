"""Library cell, virus and bacterium morphologies as OpenUSD meshes.

``morphology`` reads the library's ``mesh_type`` vocabulary and builds a
surface for each: a neutrophil gets a multilobed nucleus, influenza gets
HA/NA spikes, a flagellated rod gets flagella. Thirty-two builders, with
named sub-parts -- body, nucleus, granules, ER, capsid, flagella, spore.

Nothing carried any of it into USD. The exporters in this package drew a
sphere per cell, so the whole vocabulary stopped at the function that
built it, and ``molecular_usd`` had the same gap on the chemistry side
until it was closed.

This is the reader. Each part becomes its own ``UsdGeomMesh`` under the
entity, so a nucleus can be shaded as a nucleus and a viewer can hide the
body to look inside, which is what the parts are for.

DEPICTIONS, NOT MEASUREMENTS
----------------------------
A ``multilobed_granular`` nucleus is an illustration of a label somebody
wrote in the library. A nucleus segmented from a confocal stack is
evidence. The two must never be confused once they are both triangles in
the same scene, so every prim here carries
``geometrySource = "procedural depiction"`` and the name of the builder
that ran -- and ``fellBackToSphere`` when the requested ``mesh_type`` had
no builder, because a shape that silently becomes a sphere while still
labelled a neutrophil is the failure worth guarding against.

The *size* is a separate question from the shape, and a weaker one. Each
entity's ``scale_um`` is a top-level library field, outside the parameter
system that carries provenance grades, and much of it is placeholder: all
34 proteins are recorded at 0.005 um and all 5 organs at 50,000 um, which
cannot be true of both a lymph node and a liver. Cell diameters are
plausible -- a CD8 T cell at 8 um, a fibroblast at 30 um -- but nothing
traces them to a measurement either. ``scaleProvenance`` says so on every
prim rather than letting a confident-looking micrometre figure pass for a
sourced one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: Micrometres, matching the cell-scale exporters in this package
#: (``exporters``, ``usd_animation``). ``molecular_usd`` uses 1e-10 for
#: angstroms, so the two cannot share a stage without a scaled reference.
METERS_PER_UNIT = 1e-6

#: Colour per named part. Chosen so a nucleus reads as a nucleus at a
#: glance and the body stays translucent enough to see through -- the
#: convention cell-biology figures have used since textbook plates.
PART_COLORS: Dict[str, Tuple[float, float, float]] = {
    "body": (0.74, 0.80, 0.88),
    "nucleus": (0.33, 0.38, 0.78),
    "granules": (0.92, 0.62, 0.26),
    "er": (0.42, 0.74, 0.58),
    "capsid": (0.82, 0.32, 0.38),
    "flagella": (0.58, 0.54, 0.70),
    "spore": (0.90, 0.84, 0.44),
}

#: The body is a shell you look through; the contents are not.
PART_OPACITY: Dict[str, float] = {"body": 0.35}

#: Colours for instanced detail. Microvilli take the membrane's colour
#: because they ARE membrane; organelles take their own.
DETAIL_COLORS: Dict[str, Tuple[float, float, float]] = {
    "microvilli": (0.80, 0.86, 0.94),
    "mitochondria": (0.86, 0.46, 0.24),
    "lysosomes": (0.62, 0.40, 0.72),
    "lytic_granules": (0.96, 0.78, 0.30),
    "ribosomes": (0.55, 0.60, 0.70),
}

#: ``scale_um`` values the library repeats across every entity of a type.
#: A single number shared by every member of a class is a placeholder that
#: happens to be in the right column, not a measurement, and a prim built
#: at that size should say which it is.
_UNIFORM_SCALE_TYPES = {"protein", "organ", "tissue_type"}


@dataclass
class EntityGeometry:
    """What one entity's geometry is, and where it came from."""

    name: str
    entity_type: str
    mesh_type: str
    builder: str
    fell_back: bool
    scale_um: float
    parts: Dict[str, Tuple[int, int]]
    """part -> (vertices, triangles)."""
    prim_path: str = ""
    """Where it actually landed. Carried rather than recomputed from the
    name, because two library entities can share one: "NK cell" exists as
    both a ``cell_type`` and an ``immune_cell``, and recomputing the path
    sent the second one's transform to the first one's prim."""

    @property
    def triangles(self) -> int:
        return sum(t for _v, t in self.parts.values())

    @property
    def vertices(self) -> int:
        return sum(v for v, _t in self.parts.values())


def _safe_name(name: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in name)
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    return out if out and not out[0].isdigit() else f"_{out}"


def _material_for(stage, scope_path: str, part: str, cache: Dict[str, object]):
    """One material per part name, shared across every entity on the stage."""
    from pxr import Sdf, UsdShade

    if part in cache:
        return cache[part]

    path = f"{scope_path}/{_safe_name(part)}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    r, g, b = PART_COLORS.get(part, (0.6, 0.6, 0.6))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((r, g, b))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    opacity = PART_OPACITY.get(part, 1.0)
    if opacity < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface")
    cache[part] = material
    return material


def author_entity(
    stage,
    parent_path: str,
    name: str,
    mesh_type: str,
    scale_um: float,
    *,
    entity_type: str = "",
    prim_name: Optional[str] = None,
    seed: int = 0,
    grid: int = 64,
    material_scope: str = "/Materials",
    material_cache: Optional[Dict[str, object]] = None,
) -> EntityGeometry:
    """Write one entity's morphology, one mesh prim per named part."""
    from pxr import Sdf, UsdGeom, UsdShade, Vt

    from cognisom.visualization.morphology import BUILDERS, build_parts

    parts, fell_back = build_parts(mesh_type, scale_um=scale_um, seed=seed,
                                   grid=grid)
    builder = "sphere" if fell_back else mesh_type

    prim_path = f"{parent_path}/{_safe_name(prim_name or name)}"
    root = UsdGeom.Xform.Define(stage, prim_path)
    cache = material_cache if material_cache is not None else {}

    counts: Dict[str, Tuple[int, int]] = {}
    for part, (verts, tris) in parts.items():
        if verts.size == 0 or tris.size == 0:
            continue
        mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/{_safe_name(part)}")
        mesh.CreatePointsAttr(Vt.Vec3fArray(
            [tuple(float(c) for c in v) for v in verts]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(
            [int(i) for i in tris.reshape(-1)]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
        # Marching cubes gives a closed surface of triangles; declaring it
        # as such stops a subdivision scheme being applied and rounding the
        # lobes off a nucleus that was built to have lobes.
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
        mesh.CreateExtentAttr([tuple(float(c) for c in lo),
                               tuple(float(c) for c in hi)])
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(
            _material_for(stage, material_scope, part, cache))
        mesh.GetPrim().SetCustomDataByKey("part", part)
        counts[part] = (int(verts.shape[0]), int(tris.shape[0]))

    uniform = entity_type in _UNIFORM_SCALE_TYPES
    prim = root.GetPrim()
    prim.SetCustomDataByKey("meshType", mesh_type)
    prim.SetCustomDataByKey("builder", builder)
    prim.SetCustomDataByKey("fellBackToSphere", bool(fell_back))
    prim.SetCustomDataByKey("entityType", entity_type)
    prim.SetCustomDataByKey("scaleUm", float(scale_um))
    prim.SetCustomDataByKey("parts", Vt.StringArray(sorted(counts)))
    prim.SetCustomDataByKey("triangles", sum(t for _v, t in counts.values()))
    prim.SetCustomDataByKey("geometrySource", "procedural depiction")
    prim.SetCustomDataByKey(
        "geometryNote",
        "Built from the library's mesh_type label, not segmented from an "
        "image. A depiction of a word, not a measurement of a cell.")
    prim.SetCustomDataByKey(
        "scaleProvenance",
        "library scale_um, shared by every entity of this type and so a "
        "placeholder" if uniform else
        "library scale_um, a top-level field outside the parameter "
        "provenance system; plausible but not traced to a measurement")
    if fell_back:
        prim.SetCustomDataByKey(
            "warning",
            f"mesh_type {mesh_type!r} has no builder among the "
            f"{len(BUILDERS)} available; this is a sphere wearing the label")
    return EntityGeometry(
        name=name, entity_type=entity_type, mesh_type=mesh_type,
        builder=builder, fell_back=fell_back, scale_um=float(scale_um),
        parts=counts, prim_path=prim_path)


def author_detail(
    stage,
    prim_path: str,
    feature: str,
    instances,
    *,
    prototype: str = "cone",
    material_scope: str = "/Materials",
    material_cache: Optional[Dict[str, object]] = None,
) -> int:
    """Write one instanced feature under an entity, as a PointInstancer.

    Microvilli are the reason this exists. A resting lymphocyte carries
    hundreds of them and each is a few hundred nanometres long on a 7 um
    cell, so they cannot come off the occupancy grid the body is built
    from -- one voxel at grid 64 is 0.11 um, which is the whole width of a
    microvillus. They are instead instanced over the body's mesh surface
    (see ``cell_detail``), which decouples the feature's size from the
    grid's resolution entirely.

    A PointInstancer and not a prim each: at a few hundred features per
    cell over fifty-six cells, individual prims would be a hundred
    thousand of them saying the same thing. Instancing is also what the
    renderer wants, and what the repo's own ``NestedInstancingManager``
    assumes for anything at this count.
    """
    from pxr import Gf, Sdf, UsdGeom, Vt

    if len(instances) == 0:
        return 0

    cache = material_cache if material_cache is not None else {}
    path = f"{prim_path}/{_safe_name(feature)}"
    instancer = UsdGeom.PointInstancer.Define(stage, path)

    proto_path = f"{path}/prototypes/shape"
    if prototype == "cone":
        # A cone, because a microvillus tapers and a cylinder of the same
        # radius reads as a bristle. Unit height and unit diameter, so the
        # instance scale is the real size.
        shape = UsdGeom.Cone.Define(stage, proto_path)
        shape.CreateHeightAttr(1.0)
        shape.CreateRadiusAttr(0.5)
        shape.CreateAxisAttr(UsdGeom.Tokens.z)
    elif prototype == "capsule":
        shape = UsdGeom.Capsule.Define(stage, proto_path)
        shape.CreateHeightAttr(1.0)
        shape.CreateRadiusAttr(0.5)
        shape.CreateAxisAttr(UsdGeom.Tokens.z)
    else:
        shape = UsdGeom.Sphere.Define(stage, proto_path)
        shape.CreateRadiusAttr(1.0)

    from pxr import UsdShade
    colour = DETAIL_COLORS.get(feature, (0.7, 0.7, 0.7))
    UsdShade.MaterialBindingAPI(shape.GetPrim()).Bind(
        _material_for_colour(stage, material_scope, feature, colour, cache))

    instancer.CreatePrototypesRel().SetTargets([shape.GetPath()])
    instancer.CreateProtoIndicesAttr(
        Vt.IntArray([0] * len(instances)))
    instancer.CreatePositionsAttr(Vt.Vec3fArray(
        [Gf.Vec3f(*p) for p in instances.positions.tolist()]))
    instancer.CreateOrientationsAttr(Vt.QuathArray(
        [Gf.Quath(float(q[0]), Gf.Vec3h(float(q[1]), float(q[2]), float(q[3])))
         for q in instances.orientations.tolist()]))
    instancer.CreateScalesAttr(Vt.Vec3fArray(
        [Gf.Vec3f(*sc) for sc in instances.scales.tolist()]))

    prim = instancer.GetPrim()
    prim.SetCustomDataByKey("feature", feature)
    prim.SetCustomDataByKey("count", len(instances))
    prim.SetCustomDataByKey(
        "note",
        "Instanced over the body's mesh surface, not built from the "
        "occupancy grid: at grid 64 one voxel spans 0.11 um on a 7 um "
        "cell, which is the entire width of a microvillus.")
    return len(instances)


def _material_for_colour(stage, scope_path: str, name: str,
                         colour: Tuple[float, float, float],
                         cache: Dict[str, object]):
    from pxr import Sdf, UsdShade

    key = f"detail:{name}"
    if key in cache:
        return cache[key]
    path = f"{scope_path}/{_safe_name(name)}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(colour)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface")
    cache[key] = material
    return material


def library_entities(store=None, types: Optional[Iterable[str]] = None
                     ) -> List[Tuple[str, str, str, float]]:
    """Entities with both a ``mesh_type`` and a ``scale_um``.

    Returns ``(name, entity_type, mesh_type, scale_um)`` sorted by name, so
    an export is stable between runs.
    """
    import json

    if store is None:
        import contextlib
        import io

        from cognisom.library.store import EntityStore
        with contextlib.redirect_stdout(io.StringIO()):
            store = EntityStore()

    wanted = set(types) if types else None
    rows = [tuple(r) for r in store._conn.execute(
        "SELECT entity_type, name, data_json FROM entities")]
    out = []
    for entity_type, name, blob in rows:
        if wanted is not None and entity_type not in wanted:
            continue
        data = json.loads(blob or "{}")
        mesh_type = data.get("mesh_type")
        scale = data.get("scale_um")
        if not mesh_type or scale is None:
            continue
        out.append((name, entity_type, mesh_type, float(scale)))
    return sorted(out)


#: Types whose shape the ``mesh_type`` vocabulary actually describes. A
#: protein's ``mesh_type`` is ``protein_structure``, which names a file to
#: load rather than a shape to build, and a gene's is a sphere standing in
#: for a locus; neither belongs in a morphology export.
CELLULAR_TYPES = ("cell_type", "immune_cell", "virus", "bacterium",
                  "tissue_type")


def export_library(
    path: str,
    *,
    store=None,
    types: Optional[Iterable[str]] = CELLULAR_TYPES,
    names: Optional[Sequence[str]] = None,
    grid: int = 64,
    gap_um: float = 4.0,
) -> List[EntityGeometry]:
    """Write every entity whose ``mesh_type`` describes a real shape.

    Laid out on a grid in XY. Unlike the molecular export, the cells here
    differ in size by a factor of 700 -- influenza at 0.1 um beside a
    fibroblast at 30 um -- so each cell of the grid is sized to the
    largest entity and the small ones sit in the middle of theirs. That is
    the honest layout: scaling them to a common size would make a virus
    and a cell look comparable, which is most of what their sizes mean.
    """
    import math

    from pxr import Gf, Usd, UsdGeom, Vt

    entries = library_entities(store, types)
    if names is not None:
        keep = {n.lower() for n in names}
        entries = [e for e in entries if e[0].lower() in keep]
    if not entries:
        raise ValueError("no library entities carry both mesh_type and scale_um")

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)
    UsdGeom.Scope.Define(stage, "/Materials")
    root = UsdGeom.Xform.Define(stage, "/Entities")
    stage.SetDefaultPrim(root.GetPrim())

    # Disambiguate before authoring. Two entities can share a name across
    # types, so a prim name taken from the name alone silently merges them.
    seen: Dict[str, int] = {}
    for name, entity_type, _mt, _s in entries:
        seen[_safe_name(name)] = seen.get(_safe_name(name), 0) + 1

    cache: Dict[str, object] = {}
    geoms: List[EntityGeometry] = []
    for name, entity_type, mesh_type, scale in entries:
        prim_name = (f"{name}_{entity_type}" if seen[_safe_name(name)] > 1
                     else name)
        geoms.append(author_entity(
            stage, "/Entities", name, mesh_type, scale,
            entity_type=entity_type, prim_name=prim_name, grid=grid,
            material_cache=cache))

    cell = max(g.scale_um for g in geoms) + gap_um
    columns = max(1, int(math.ceil(math.sqrt(len(geoms)))))
    for index, geom in enumerate(geoms):
        row, col = divmod(index, columns)
        prim = stage.GetPrimAtPath(geom.prim_path)
        if prim:
            UsdGeom.Xformable(prim).AddTranslateOp().Set(
                Gf.Vec3d(col * cell, -row * cell, 0.0))

    prim = root.GetPrim()
    prim.SetCustomDataByKey("entityCount", len(geoms))
    prim.SetCustomDataByKey("triangleCount", sum(g.triangles for g in geoms))
    prim.SetCustomDataByKey(
        "fellBack", Vt.StringArray([g.name for g in geoms if g.fell_back]))
    prim.SetCustomDataByKey("gridCellUm", float(cell))
    prim.SetCustomDataByKey("gridColumns", columns)
    prim.SetCustomDataByKey("sizeRangeUm", Vt.DoubleArray(
        [min(g.scale_um for g in geoms), max(g.scale_um for g in geoms)]))
    prim.SetCustomDataByKey(
        "note",
        "Shapes are procedural depictions of the library's mesh_type "
        "labels, not segmented from images. Sizes are the library's "
        "scale_um, which sits outside its parameter provenance system. "
        "Authored in micrometres; the molecular stage uses angstroms.")
    stage.GetRootLayer().Save()
    log.info("wrote %d entities, %d triangles to %s",
             len(geoms), sum(g.triangles for g in geoms), path)
    return geoms
