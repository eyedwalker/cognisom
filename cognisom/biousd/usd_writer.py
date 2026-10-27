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
    #: Instances marked invisible by a cutaway, if one was asked for.
    hidden: int = 0
    #: Instances the depth peel hid on top of the cut.
    occluded: int = 0
    #: Triangles actually submitted: each visible instance times the
    #: faces of the prototype it points at. This is the number the
    #: render cliff is measured in; `prototype_faces` is only what was
    #: authored once.
    submitted_faces: int = 0
    #: grid -> instances pointed at that tier, when LOD is on.
    lod_tiers: Dict[int, int] = field(default_factory=dict)
    #: prototype path -> triangles it will submit. The render cliff is
    #: per prototype, so this is the number to watch.
    submitted_by_prototype: Dict[str, int] = field(default_factory=dict)
    #: Prototypes whose submission is over STORM_DRAW_TRIANGLE_LIMIT;
    #: the frame will almost certainly come back blank.
    over_limit: List[str] = field(default_factory=list)
    #: The attribute driving displayColor, if colour_by was honoured.
    colour_by: Optional[str] = None
    colour_range: Optional[Tuple[float, float]] = None
    #: Why colour_by fell back to type colour, if it did.
    colour_skipped: Optional[str] = None
    #: Cells whose identity has no ontology term, by library name.
    unmapped: List[str] = field(default_factory=list)
    primvars: List[str] = field(default_factory=list)
    #: cell_type token -> what the cells WITHOUT their own ``scale_um``
    #: were drawn at: ``diameter_um``, ``size_source`` (the library's
    #: ``radius_um <citation>`` or an ``estimate: ...``), ``cited`` and
    #: ``cells``. The record the run exporter keeps as
    #: ``RunWriteReport.appearance``; a BioCell has no size of its own,
    #: so this is where every cell through sync and write_scene lands.
    sizes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: Cells drawn at the ``scale_um`` they carried.
    sized_by_cell: int = 0

    def describe(self) -> str:
        size = self.bytes_written / 1048576
        text = (f"{self.cells:,} cells, {self.prototypes} prototypes "
                f"({self.prototype_faces:,} faces), {self.prims} prims, "
                f"{size:.2f} MB")
        visible = self.cells - self.hidden - self.occluded
        if self.hidden or self.occluded:
            text += (f"; {visible:,} visible ({self.hidden:,} cut, "
                     f"{self.occluded:,} peeled)")
        if self.submitted_faces:
            text += f"; {self.submitted_faces / 1e6:.0f}M triangles submitted"
        if self.over_limit:
            worst = max(self.submitted_by_prototype[k] for k in self.over_limit)
            text += (f"; OVER THE RENDER LIMIT on {len(self.over_limit)} "
                     f"prototype(s), largest {worst / 1e6:.0f}M -- "
                     f"expect a blank frame")
        if self.lod_tiers:
            text += "; LOD " + ", ".join(
                f"grid {g}: {n:,}" for g, n in sorted(self.lod_tiers.items(),
                                                       reverse=True))
        if self.colour_by:
            lo, hi = self.colour_range or (float("nan"), float("nan"))
            text += f"; coloured by {self.colour_by} [{lo:g}, {hi:g}]"
        elif self.colour_skipped:
            text += f"; colour-by skipped: {self.colour_skipped}"
        if self.sizes:
            text += "; sized from the library: " + ", ".join(
                f"{token} {rec['diameter_um']:.1f} um "
                f"({rec['size_source'].split(' ', 1)[1] if rec['cited'] else 'est'}) "
                f"x{rec['cells']:,}"
                for token, rec in sorted(self.sizes.items()))
        if self.sized_by_cell and self.sizes:
            text += f"; {self.sized_by_cell:,} at their own scale_um"
        return text


def _mesh_type_of(cell: Any) -> Optional[str]:
    for attr in ("mesh_type", "meshType"):
        value = getattr(cell, attr, None)
        if value:
            return str(value)
    return None


def _resolved_mesh_of(cell: Any) -> str:
    """The prototype a cell instances: its own ``mesh_type`` if it carries
    one, else the library shape its ``cell_type`` token resolves to, else
    a smooth sphere.

    The middle step is what :mod:`cognisom.biousd.sync` and
    :func:`write_scene` were missing. A ``BioCell`` carries a type and
    no mesh, so every cancer cell those paths wrote was a smooth sphere
    while the same cell in a run export was the library's
    irregular_ruffled epithelium. The resolver is
    :func:`~cognisom.visualization.lineage_appearance.mesh_type_for_cell_token`,
    which answers ``None`` for a state token -- and ``normal`` is one
    there, because it is also this writer's default for an untyped cell
    (:func:`_token_of`), and the default bucket must not wear a lineage's
    shape.
    """
    explicit = _mesh_type_of(cell)
    if explicit:
        return explicit
    token = getattr(cell, "cell_type", None)
    if token is not None:
        from cognisom.visualization.lineage_appearance import mesh_type_for_cell_token
        resolved = mesh_type_for_cell_token(token)
        if resolved:
            return resolved
    return "smooth_sphere"


#: Diameter for a cell that carries no ``scale_um`` and whose type the
#: library cannot size: a state token, or one with no row. This was the
#: writer's silent constant for EVERY cell without a scale_um -- which
#: is every BioCell, the dataclass having no such field -- so a cancer
#: cell through sync or write_scene was the library's shape at a made-up
#: size. It is now the last resort, and the result says so.
DEFAULT_DIAMETER_UM = 10.0

#: ``size_source`` for a cell drawn at its own ``scale_um``.
EXPLICIT_SIZE = "scale_um on the cell"


def _resolved_diameter_of(cell: Any) -> Tuple[float, str]:
    """The diameter a cell is drawn at and where it came from: its own
    ``scale_um`` if it carries one; else the library's size for its
    ``cell_type`` token, with the library's source, which is what the
    run exporter draws the same lineage at; else
    :data:`DEFAULT_DIAMETER_UM`, labelled an estimate.

    The size half of :func:`_resolved_mesh_of`, and with the same rule
    for a state token: ``normal`` is this writer's default for an
    untyped cell, so it takes the default size as it takes the sphere,
    rather than the luminal cell's 15 um.
    """
    explicit = getattr(cell, "scale_um", None)
    if explicit:
        return float(explicit), EXPLICIT_SIZE
    token = getattr(cell, "cell_type", None)
    if token is not None:
        from cognisom.visualization.lineage_appearance import diameter_for_cell_token
        resolved = diameter_for_cell_token(token)
        if resolved is not None:
            return float(resolved[0]), str(resolved[1])
    return DEFAULT_DIAMETER_UM, (
        f"estimate: {DEFAULT_DIAMETER_UM:g} um default (no scale_um on the "
        "cell and no library size for its type)")


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


#: Sequential ramp for colour-by: one hue, light to dark. Chosen so no
#: step resembles a TYPE_COLOUR, which would read as a lineage.
COLOUR_RAMP: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
    (0.93, 0.95, 0.98), (0.03, 0.19, 0.42))


def _colour_of(cell: Any) -> Tuple[float, float, float]:
    """The entity's own colour where it has one, else its type's."""
    rgb = getattr(cell, "color_rgb", None)
    if rgb is not None and len(tuple(rgb)) >= 3:
        return tuple(float(c) for c in tuple(rgb)[:3])
    return TYPE_COLOUR.get(_token_of(cell), TYPE_COLOUR["normal"])


#: Surface properties for a cell. Cytoplasm is mostly water with
#: protein in it, so it is neither matte nor a mirror: a moderate
#: roughness with a little specular reads wet without reading metallic.
#: UsdPreviewSurface has no subsurface scattering, which is the thing
#: that would most improve this, so these are the knobs available rather
#: than the ones wanted.
CELL_ROUGHNESS = 0.38
CELL_SPECULAR = 0.35


def author_cell_material(stage, path: str = "/World/Looks/BioCell"):
    """One UsdPreviewSurface for every cell, driven by primvars.

    Without a bound material the stage renders under whatever default
    surface the viewer supplies -- a flat lambert in usdrecord -- which
    is neither reproducible across renderers nor anything like a cell.

    Two primvar readers, and the second one is the trap. Binding a
    material means ``displayOpacity`` STOPS GOVERNING: the material's
    own ``opacity`` input does. So a translucent body silently turns
    opaque the moment a material is bound, and the nucleus inside it
    disappears again. The opacity reader is what keeps that from
    happening, and its fallback is 1.0 so a part that authored none
    stays solid.
    """
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)

    colour = UsdShade.Shader.Define(stage, f"{path}/ColorReader")
    colour.CreateIdAttr("UsdPrimvarReader_float3")
    colour.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    colour.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(0.74, 0.80, 0.88))
    colour.CreateOutput("result", Sdf.ValueTypeNames.Float3)

    alpha = UsdShade.Shader.Define(stage, f"{path}/OpacityReader")
    alpha.CreateIdAttr("UsdPrimvarReader_float")
    alpha.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayOpacity")
    alpha.CreateInput("fallback", Sdf.ValueTypeNames.Float).Set(1.0)
    alpha.CreateOutput("result", Sdf.ValueTypeNames.Float)

    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)\
        .ConnectToSource(colour.ConnectableAPI(), "result")
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)\
        .ConnectToSource(alpha.ConnectableAPI(), "result")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(CELL_ROUGHNESS)
    shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(CELL_SPECULAR)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface")
    return material


def _normalise_to_unit_diameter(parts):
    """Scale a built prototype so its body really is one unit across.

    The instancer multiplies each prototype by the cell's `scale_um`
    on the stated grounds that the prototype is authored at unit
    diameter. It was not. Measured from the builders at scale_um=1.0:

        smooth_sphere      grid 8   body 0.776    grid 16   body 0.881
        irregular_ruffled  grid 8   body 0.596    grid 16   body 0.744-0.824

    A marching-cubes surface sits inside the voxel boundary, and how
    far inside depends on the grid. So a 12 um cell was drawn at 9.3 um
    at grid 8 and 10.6 um at grid 16 -- every export 12-22% too small,
    and a level-of-detail swap between grids shrank each cell by a
    further 12%, which is what made LOD look wrong when it was the
    prototypes that were.

    The body is the reference because it is the cell; its other parts
    scale with it so the nucleus keeps its proportion. The diameter is
    the mean of the three axis extents, which is what a measured
    diameter of an irregular cell is closest to -- its longest axis
    would make a ruffled cell smaller than its scale_um says, and its
    shortest would make it larger. The body is also centred, so the
    instance position is the cell's centre rather than a corner of the
    voxel grid it was built on.
    """
    body = parts.get("body")
    if body is None:
        return parts
    vertices, _ = body
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    extent = float(np.mean(hi - lo))
    if extent <= 0.0:
        return parts
    centre = (lo + hi) / 2.0
    factor = 1.0 / extent
    return {part: ((np.asarray(v, dtype="float64") - centre) * factor, t)
            for part, (v, t) in parts.items()}


def author_prototypes(stage, mesh_types: Sequence[str],
                      grid: int = PROTOTYPE_GRID,
                      root: str = PROTOTYPE_ROOT,
                      part_colours: bool = False,
                      suffix: str = ""):
    """One Xform per mesh type, its parts meshed beneath it.

    Returns ``(paths, faces)``. A mesh type with no builder is skipped by
    the caller rather than silently drawn as something else.

    ``part_colours`` trades per-cell colour for per-organelle colour; see
    the note at the author site. The default keeps both as far as USD
    allows: the body is translucent so the nucleus reads through it, and
    the instancer still tints each cell.
    """
    from pxr import Gf, UsdGeom, Vt

    from cognisom.visualization import morphology_usd as mu
    from cognisom.visualization.morphology import build_parts

    paths, faces = [], 0
    for mesh_type in mesh_types:
        parts, fell_back = build_parts(mesh_type, scale_um=1.0, seed=0,
                                       grid=grid)
        if fell_back:
            log.warning("no builder for %s; it will instance a fallback",
                        mesh_type)
        parts = _normalise_to_unit_diameter(parts)
        xform = UsdGeom.Xform.Define(stage, f"{root}/{mesh_type}{suffix}")
        for part, (vertices, triangles) in parts.items():
            mesh = UsdGeom.Mesh.Define(
                stage, f"{root}/{mesh_type}{suffix}/{part}")
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(
                np.ascontiguousarray(vertices, dtype="float32")))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(
                np.ascontiguousarray(triangles.ravel(), dtype="int32")))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(
                np.full(len(triangles), 3, dtype="int32")))
            # No normals are authored, deliberately. A UsdGeom.Mesh
            # defaults to catmullClark subdivision, under which Hydra
            # discards authored normals and shades the limit surface
            # smooth -- measured: a prototype rendered with and without
            # a full set of vertex normals came back identical to the
            # pixel. It is also why a grid-8 prototype has never looked
            # faceted. The far tier's remaining frame difference is
            # tessellation noise under seven translucent layers: a grid
            # 16 prototype re-meshed with a different seed moves the
            # frame as much as dropping to grid 8 does (32.5 vs 36.2
            # mean |dRGB| on the same 500k scene).
            # The body is a shell you look through; the contents are not.
            # Without this every prototype meshes a nucleus and granules
            # nobody can see, because the body around them is opaque
            # default white -- three parts authored, one ever visible.
            opacity = mu.PART_OPACITY.get(part, 1.0)
            if opacity < 1.0:
                mesh.GetDisplayOpacityAttr().Set(Vt.FloatArray([opacity]))
            # Colour is the one that has to be chosen rather than copied.
            # A prototype carrying displayColor WINS OVER the instancer's
            # per-instance primvar -- measured: two renders, one with a
            # green per-instance colour and one without, came back pixel
            # for pixel the same. So authoring part colours here silently
            # disables colour-by-variable over the whole population, which
            # is the capability the primvar path exists for.
            #
            # Default is therefore opacity without colour: the instancer
            # tints each cell and the translucent body still shows the
            # structure inside it, which is both. part_colours is for a
            # detail view of a few cells, where anatomy matters more than
            # telling one cell from the next.
            if part_colours:
                mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray(
                    [Gf.Vec3f(*mu.PART_COLORS.get(part, (0.74, 0.80, 0.88)))]))
            faces += len(triangles)
        paths.append(xform.GetPath())
    return paths, faces


#: Where the camera sits, as a direction from the cells. The cutaway
#: plane is perpendicular to this, so the two must not drift apart: a
#: cut aimed down a different axis than the camera removes material the
#: viewer could not see and leaves what it could.
CAMERA_DIRECTION = (1.7, 1.15, 2.0)

#: The authored camera. USD's default horizontal aperture is 20.955 mm;
#: with a 35 mm focal length that is a 33.4 degree field of view, and
#: the LOD tiers project through exactly this lens.
CAMERA_FOCAL_MM = 35.0
CAMERA_APERTURE_MM = 20.955

#: Projected cell width, in pixels at `lod_pixels` frame width, below
#: which a cell drops a tier. Under 5 px a grid-8 sphere is a dot at
#: any grid; under 12 px the ruffles of a grid-16 prototype are below
#: the pixel and only its silhouette survives.
LOD_PIXEL_TIERS = (5.0, 12.0)

#: Triangles one prototype can submit before Storm draws nothing at all.
#:
#: Measured, not read anywhere: on this machine (Storm on Metal, USD
#: 25.11) every scene whose largest single prototype submitted up to
#: 639M triangles drew, and every scene at 690M or more came back as a
#: valid PNG with alpha 0 in every pixel, exit code 0, no warning. The
#: total across prototypes is NOT the axis -- 728M in total drew and
#: 735M did not -- it is the largest single draw. A first estimate of
#: 330M was wrong by two, from guessing a grid-16 prototype at ~1,500
#: faces when it is 2,916 with its nucleus; the bracket here is from
#: the writer's own exact count. It will differ on another machine; it
#: exists so the export can say "this will be blank" instead of
#: letting a user find out.
STORM_DRAW_TRIANGLE_LIMIT = 640_000_000


def write_cells(path: str, cells: Sequence[Any], *,
                grid: int = PROTOTYPE_GRID,
                geometry: bool = True,
                camera: bool = True,
                up_axis: str = "Y",
                cutaway: str = "none",
                stroma_only: bool = False,
                occlusion: Optional[int] = None,
                occlusion_column: float = 0.5,
                lod: bool = False,
                lod_pixels: float = 1000.0,
                colour_by: Optional[str] = None,
                colour_range: Optional[Tuple[float, float]] = None
                ) -> WriteResult:
    """Author ``cells`` as one PointInstancer and return what it cost.

    ``cells`` may be :class:`~cognisom.biousd.schema.BioCell` instances or
    anything carrying ``position``, ``cell_type`` and the rest; the fields
    are read defensively because the engine's own cell objects and the
    schema's dataclass do not share a base class.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

    cells = list(cells)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, getattr(UsdGeom.Tokens, up_axis.lower()))
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, WORLD).GetPrim())
    # The vocabulary version this stage is written to, which the
    # standard's validator requires; the vocabulary itself lives in
    # cognisom/biousd/standard and a test holds the primvars authored
    # below equal to it.
    from cognisom.biousd.standard.vocabulary import VERSION as BIOUSD_VERSION
    stage.GetRootLayer().customLayerData = {"bio:version": BIOUSD_VERSION}
    result = WriteResult(path=path, cells=len(cells))

    if not cells:
        stage.GetRootLayer().Save()
        result.prims = len(list(stage.Traverse()))
        return result

    # One prototype per distinct mesh type present, in a stable order so
    # that protoIndices mean the same thing across two writes of the same
    # population.
    material = author_cell_material(stage)
    mesh_of = [_resolved_mesh_of(c) for c in cells]
    types = sorted(set(mesh_of))

    positions = np.array([tuple(getattr(c, "position", (0.0, 0.0, 0.0)))
                          for c in cells], dtype="float32")
    # Volume is a multiplier on the prototype, which is authored at unit
    # diameter, so the instance scale IS the cell's size in micrometres.
    sized = [_resolved_diameter_of(c) for c in cells]
    diameters = np.array([d for d, _ in sized], dtype="float32")
    volume = np.array([float(getattr(c, "volume", 1.0) or 1.0)
                       for c in cells], dtype="float32")
    # The camera's eye point, needed before the prototypes because LOD
    # tiers are assigned by distance from it.
    half = float(np.abs(positions).max()) + float(diameters.max())
    eye_np = half * np.asarray(CAMERA_DIRECTION, dtype=float)

    if geometry and lod:
        # Level of detail without a second instancer. A PointInstancer
        # can target any number of prototypes, so the far tier is just
        # more prototypes -- the same mesh types built at a coarser grid
        # -- and a cell picks its tier through protoIndices. One
        # instancer, one invisibleIds, every primvar untouched.
        #
        # A tier is chosen by how many pixels a cell will cover, not
        # by its distance. The first version used fractions of the eye
        # distance and sent 158,000 cells to grid 8 in a frame where
        # they were 7-9 px wide; 20% of pixels changed. In a 1 mm block
        # seen from 1.5 mm nothing is far, and a rule has to know that.
        #
        # The frame width is not known at export time, so it is a
        # parameter: `lod_pixels` is the width the tiers are tuned for,
        # and a cell narrower than LOD_PIXEL_TIERS[0] px at that width
        # takes grid 8, narrower than LOD_PIXEL_TIERS[1] takes a grid
        # four coarser than asked, and the rest take `grid`. The grid
        # ladder is 328 / 904 / 2,316 faces per two prototypes at
        # 8 / 12 / 16, so the far tier costs a seventh of the near one.
        tan_half = (CAMERA_APERTURE_MM / 2.0) / CAMERA_FOCAL_MM
        dist = np.linalg.norm(positions - eye_np, axis=1)
        px = lod_pixels * diameters / (np.maximum(dist, 1e-6) * 2.0 * tan_half)
        tiers = [(LOD_PIXEL_TIERS[0], 8),
                 (LOD_PIXEL_TIERS[1], max(8, grid - 4)),
                 (float("inf"), grid)]
        tier_of = np.searchsorted([t for t, _ in tiers[:-1]], px, side="right")
        # Only the grids some cell actually lands in are authored. A
        # composed scene whose every cell is over 12 px wide wants one
        # tier, not three with two of them empty -- measured, that was
        # nine prototypes and 14,644 faces authored for a scene that
        # used three and 8,694.
        used = {tiers[int(t)][1] for t in np.unique(tier_of)}
        grids = []
        for _, g in tiers:
            if g in used and g not in grids:
                grids.append(g)
        proto_paths, index_of, result.prototype_faces = [], {}, 0
        for g in grids:
            paths_g, faces_g = author_prototypes(stage, types, g,
                                                 suffix=f"_g{g}")
            result.prototype_faces += faces_g
            for name, path_g in zip(types, paths_g):
                index_of[(name, g)] = len(proto_paths)
                proto_paths.append(path_g)
        tier_grid = np.array([grids.index(tiers[int(t)][1]) for t in tier_of])
        proto_indices = np.array(
            [index_of[(mesh_of[i], grids[tier_grid[i]])]
             for i in range(len(cells))], dtype="int32")
        for g in grids:
            n = int(sum(1 for i in range(len(cells)) if grids[tier_grid[i]] == g))
            result.lod_tiers[g] = n
    elif geometry:
        proto_paths, result.prototype_faces = author_prototypes(
            stage, types, grid)
        index_of = {name: i for i, name in enumerate(types)}
        proto_indices = np.array([index_of[m] for m in mesh_of], dtype="int32")
    else:
        # A sphere prototype, declared rather than defaulted to.
        sphere = UsdGeom.Sphere.Define(stage, f"{PROTOTYPE_ROOT}/point")
        sphere.GetRadiusAttr().Set(0.5)
        proto_paths, types = [sphere.GetPath()], ["point"]
        proto_indices = np.zeros(len(cells), dtype="int32")
    result.prototypes = len(proto_paths)
    faces_of_proto = []
    for proto_path in proto_paths:
        prim = stage.GetPrimAtPath(proto_path)
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(material)
        n_faces = 0
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                counts = UsdGeom.Mesh(child).GetFaceVertexCountsAttr().Get()
                n_faces += len(counts) if counts else 0
        faces_of_proto.append(n_faces)

    instancer = UsdGeom.PointInstancer.Define(stage, CELLS)
    instancer.GetPrototypesRel().SetTargets(proto_paths)
    instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
    instancer.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(proto_indices))

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
    if colour_by:
        # Colour by a variable: one hue, light to dark, so that "more"
        # reads as "darker" and nothing in the ramp looks like a type
        # colour. The ramp is a real control rather than a flag because
        # the primvar path is the point of the whole design -- the
        # colour of 500,000 cells is one array, and which variable fills
        # it is the question a user asks.
        raw = [getattr(c, colour_by, None) for c in cells]
        try:
            values = np.array([float(getattr(v, "value", v)) for v in raw],
                              dtype="float64")
        except (TypeError, ValueError):
            values = None
        if values is None or not np.isfinite(values).all():
            result.colour_skipped = (
                f"{colour_by!r} is missing or non-numeric on some cells")
        else:
            lo, hi = colour_range or (float(values.min()), float(values.max()))
            span = (hi - lo) if hi > lo else 1.0
            t = np.clip((values - lo) / span, 0.0, 1.0)[:, None]
            light = np.array(COLOUR_RAMP[0], dtype="float32")
            dark = np.array(COLOUR_RAMP[1], dtype="float32")
            colours = (light + (dark - light) * t).astype("float32")
            result.colour_by, result.colour_range = colour_by, (lo, hi)
            layer = stage.GetRootLayer()
            data = dict(layer.customLayerData)
            data.update({"bio:colourBy": colour_by,
                         "bio:colourMin": float(lo),
                         "bio:colourMax": float(hi)})
            layer.customLayerData = data
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

    # What size each cell was drawn at and why, for the cells that did
    # not bring their own: the library's record, by type token.
    from cognisom.visualization.lineage_appearance import is_citation
    for cell, (diameter, source) in zip(cells, sized):
        if source == EXPLICIT_SIZE:
            result.sized_by_cell += 1
            continue
        record = result.sizes.setdefault(_token_of(cell), {
            "diameter_um": float(diameter), "size_source": source,
            "cited": is_citation(source), "cells": 0})
        record["cells"] += 1

    # A dense population occludes its own subject -- of 20,000 cells,
    # 15,200 stroma hide the 2,378 that carry the biology. invisibleIds
    # is the instancer's own mechanism, so the cut costs an int array
    # rather than a second export, and it needs only the direction to
    # the camera, not the eye point fitted below.
    up_vec = {"y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}.get(
        up_axis.lower(), (0.0, 0.0, 1.0))
    hidden: List[int] = []
    if cutaway and cutaway != "none":
        from cognisom.visualization.cutaway import (
            hidden_for_camera_direction)

        hidden = [int(i) for i in hidden_for_camera_direction(
            positions, [_token_of(c) for c in cells], CAMERA_DIRECTION,
            mode=cutaway, stroma_only=stroma_only, up=up_vec)]
        result.hidden = len(hidden)

    if occlusion:
        # Depth peel: of the cells left after the cut, keep only the
        # `occlusion` nearest the camera in each screen column. The
        # body is 0.35 opaque, so 95% of what reaches the eye comes
        # from the first ln(0.05)/ln(0.65) ~ 7 layers; 8 is the
        # default. Measured on the 500k section: visible 279k -> 194k,
        # the frame moved by a mean of 2.7/765 per pixel.
        from cognisom.visualization.cutaway import occluded_instance_ids

        peeled = occluded_instance_ids(
            positions, diameters, CAMERA_DIRECTION, keep=int(occlusion),
            up=up_vec, column=float(np.median(diameters)) * occlusion_column,
            already_hidden=hidden)
        result.occluded = len(peeled)
        hidden = sorted(set(hidden) | set(peeled))

    if hidden:
        instancer.CreateInvisibleIdsAttr().Set(Vt.Int64Array(hidden))
    visible = np.ones(len(cells), dtype=bool)
    if hidden:
        visible[np.asarray(hidden, dtype=int)] = False
    faces_arr = np.asarray(faces_of_proto, dtype=np.int64)
    result.submitted_faces = int(faces_arr[proto_indices[visible]].sum())
    per_proto = np.bincount(proto_indices[visible], minlength=len(proto_paths))
    for proto_path, count, n_faces in zip(proto_paths, per_proto, faces_of_proto):
        submitted = int(count) * int(n_faces)
        result.submitted_by_prototype[str(proto_path)] = submitted
        if submitted > STORM_DRAW_TRIANGLE_LIMIT:
            result.over_limit.append(str(proto_path))
    if result.over_limit:
        log.warning(
            "%s: %d prototype(s) submit more than %dM triangles each; Storm "
            "renders such a scene as a blank frame. Cut, peel, or lower the "
            "grid.", path, len(result.over_limit),
            STORM_DRAW_TRIANGLE_LIMIT // 1_000_000)

    if camera:
        # Framed with a look-at, not Euler angles. A camera aimed by
        # rotation renders an EMPTY FRAME SUCCESSFULLY -- no error, no
        # warning, a valid PNG of nothing -- which is a long way to go
        # before noticing.
        eye = Gf.Vec3d(*(float(v) for v in eye_np))
        look = Gf.Matrix4d(1).SetLookAt(eye, Gf.Vec3d(0, 0, 0),
                                        Gf.Vec3d(0, 1, 0))
        cam = UsdGeom.Camera.Define(stage, f"{WORLD}/Camera")
        UsdGeom.Xformable(cam).AddTransformOp().Set(look.GetInverse())
        cam.GetFocalLengthAttr().Set(CAMERA_FOCAL_MM)
        cam.GetHorizontalApertureAttr().Set(CAMERA_APERTURE_MM)
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
