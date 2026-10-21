"""Everything to draw for one library entity, in one object.

This module exists because the detail layer had no way in. An audit of
the import graph -- every dashboard page, every simulation module, the
app, the REST server and the orchestrator -- reached 249 of 409 modules
and **none** of ``morphology``, ``cell_detail``, ``cell_profiles``,
``morphology_usd`` or ``molecular_usd``. The 32 shape builders, the
measured microvillus density, the sourced granule counts and the
compartment placement were all computed only by tests.

Meanwhile the page a user actually opens drew cells from a parallel
system that picked a shape by SUBSTRING-MATCHING THE ENTITY NAME --
``fibroblast`` and ``endothelial`` both to ``macrophage_m2``,
``epithelial`` to ``plasma_cell``, every B cell and ILC to ``t_cell`` --
and then printed the entity's real ``mesh_type`` underneath as a
caption. The page asserted a shape it had not drawn.

So the bridge is here rather than in the dashboard, and it returns data
rather than a figure: a scene can then be checked without a browser, and
the USD exporter can be given the same one. What is NOT here is any
drawing. Plotly, USD and anything else convert from these dataclasses.

WHAT IS LIBRARY-DRIVEN AND WHAT IS NOT
--------------------------------------
:attr:`CellScene.sources` records, per element, whether the number
behind it came from the entity's own provenanced parameters or from a
constant in this repository. The distinction is the point: a scene that
cannot say which is which lets a default pass for evidence, which is
the failure this project keeps finding.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cognisom.visualization import cell_detail as cd
from cognisom.visualization import cell_profiles as cp
from cognisom.visualization.morphology import (
    BUILDERS, ShapeParams, build_parts)
from cognisom.visualization.morphology_usd import (
    DETAIL_COLORS, PART_COLORS, PART_OPACITY, shape_seed)

log = logging.getLogger(__name__)

#: Grid for an interactive card. Measured on this machine: a median
#: builder takes 34 ms at grid 32 and the whole 32-builder catalogue
#: 1.48 s, against 163 ms and 12 s at grid 64. 32 is the resolution that
#: makes real geometry affordable on a page load; the exporter still
#: uses 64.
CARD_GRID = 32


@dataclass
class Surface:
    """One meshed part of a cell: body, nucleus, granules, capsid."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    color: Tuple[float, float, float]
    opacity: float = 1.0

    @property
    def triangles(self) -> int:
        return int(len(self.faces))


@dataclass
class Detail:
    """A population of small features instanced over or inside a part.

    Too small to come off the occupancy grid -- a microvillus is 85 nm
    across where one voxel at grid 32 is 250 nm -- so they are carried
    as positions and a radius and drawn as instances.
    """

    name: str
    positions: np.ndarray
    radius_um: float
    color: Tuple[float, float, float]
    #: How many the cell actually has, before the draw cap. Reporting
    #: only the drawn count would make a capped population look sparse
    #: rather than clipped.
    actual: int = 0
    drawn: int = 0
    directions: Optional[np.ndarray] = None
    length_um: float = 0.0


@dataclass
class CellScene:
    """What to draw for one entity, and where each number came from."""

    entity_name: str
    mesh_type: str
    scale_um: float
    surfaces: List[Surface] = field(default_factory=list)
    details: List[Detail] = field(default_factory=list)
    #: True when ``mesh_type`` named a real builder. False means the
    #: entity fell back to a sphere, and the caption must say so rather
    #: than printing a shape nobody drew.
    from_library: bool = False
    fell_back: bool = False
    profile: Optional[cp.Profile] = None
    sources: Dict[str, str] = field(default_factory=dict)

    @property
    def triangles(self) -> int:
        return sum(s.triangles for s in self.surfaces)

    def caption(self) -> str:
        """What the picture may honestly claim to be.

        It names the shape that was BUILT. The old caption printed the
        entity's ``mesh_type`` beside a shape chosen by substring, so a
        fibroblast drawn as a macrophage was captioned "Elongated
        Spindle".
        """
        shape = (self.mesh_type if self.from_library and not self.fell_back
                 else "sphere (no builder for this type)")
        bits = [shape.replace("_", " ")]
        if self.scale_um:
            bits.append(f"~{self.scale_um:g} um")
        measured = sum(1 for v in self.sources.values() if v == "library")
        if measured:
            bits.append(f"{measured} measured")
        return "  |  ".join(bits)


def has_builder(mesh_type: Optional[str]) -> bool:
    """Whether the library's ``mesh_type`` names a real shape."""
    return bool(mesh_type) and mesh_type in BUILDERS


def _entity_field(entity, name, default=None):
    value = getattr(entity, name, None)
    if value in (None, ""):
        props = getattr(entity, "properties", None) or {}
        value = props.get(name)
    return default if value in (None, "") else value


def scene_for(entity, *, grid: int = CARD_GRID,
              compartments: Optional[Dict[str, str]] = None,
              max_detail: int = cd.DRAW_CAP) -> CellScene:
    """Build the drawable scene for one library entity.

    ``compartments`` is the library-derived index from
    :func:`cell_profiles.library_compartment_index`. Without it a marker
    is placed only if the hand-written table happens to name it, which
    covered 60 of the 173 strings this library uses.
    """
    name = getattr(entity, "name", "") or ""
    mesh_type = _entity_field(entity, "mesh_type", "") or ""
    scale_um = float(_entity_field(entity, "scale_um", 10.0) or 10.0)

    scene = CellScene(entity_name=name, mesh_type=mesh_type,
                      scale_um=scale_um, from_library=has_builder(mesh_type))

    # The proportions the entity itself records, read BEFORE the mesh is
    # built because they change the mesh. library_detail fetched these
    # and nothing consumed them, so an entity's measured 0.444 nuclear
    # fraction was overridden by a module constant that happened to
    # agree -- and the granule fraction by one that did not, 0.020
    # against a stored 0.002.
    library = cp.library_detail(entity)
    shape = ShapeParams(
        nuclear_volume_fraction=library.get("nuclear_volume_fraction"),
        granule_volume_fraction=library.get("granule_volume_fraction"))

    # The seed is the entity's own name, so two cells sharing a
    # mesh_type and a size are not the same object.
    seed = shape_seed(name)
    parts, fell_back = build_parts(mesh_type or "sphere", scale_um=scale_um,
                                   seed=seed, grid=grid, params=shape)
    scene.fell_back = bool(fell_back)

    for part, (vertices, faces) in sorted(parts.items()):
        if len(faces) == 0:
            continue
        scene.surfaces.append(Surface(
            name=part, vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(faces, dtype=int),
            color=PART_COLORS.get(part, (0.6, 0.6, 0.6)),
            opacity=PART_OPACITY.get(part, 1.0)))

    # ── the detail the library can pay for ───────────────────────────
    markers = cp.markers_of(getattr(entity, "properties", None) or {})
    profile = cp.profile_for(name, mesh_type, markers, scale_um=scale_um,
                             library=library, compartments=compartments)
    scene.profile = profile
    scene.sources = dict(profile.sources)

    body = next((s for s in scene.surfaces if s.name == "body"), None)
    if body is None:
        return scene

    if profile.microvilli_count:
        actual = int(profile.microvilli_count)
        drawn, _ = cd.capped(actual, max_detail)
        instances = cd.projections(
            body.vertices, body.faces, count=drawn,
            length_um=profile.microvillus_length_um,
            radius_um=profile.microvillus_radius_um, seed=seed)
        scene.details.append(Detail(
            "microvilli", instances.positions,
            profile.microvillus_radius_um,
            DETAIL_COLORS.get("microvilli", (0.8, 0.86, 0.94)),
            actual=actual, drawn=drawn,
            length_um=profile.microvillus_length_um))

    # Granules come from one place or the other, never both.
    #
    # A granular_sphere builder already carves granules into the
    # occupancy grid, and the profile also reports a measured lytic
    # granule count, so drawing both put an NK cell's granules on the
    # screen twice -- visible the moment a cell was actually looked at,
    # and invisible to every numeric check.
    #
    # The instanced population wins where there is one: it carries the
    # sourced count (63 +/- 23, PMID 29592963) and the sourced diameter
    # (251 nm, PMID 21931537), and being instanced it is independent of
    # the grid. The builder's are voxel-carved, so at a card's grid 32 a
    # 0.25 um granule is under one voxel and they merge into lumps that
    # preserve volume while losing the count entirely.
    if profile.lytic_granule_count:
        carved = [s for s in scene.surfaces if s.name == "granules"]
        if carved:
            # Recorded rather than done quietly: this drops a part the
            # entity's own mesh_type asked for, and a renderer that
            # overrules the library without saying so is how the
            # hardcoded 0.020 granule fraction outlived the measured
            # 0.002 for as long as it did.
            scene.sources["granules"] = "library:superseded_builder"
            scene.surfaces = [s for s in scene.surfaces if s.name != "granules"]
        actual = int(profile.lytic_granule_count)
        drawn, _ = cd.capped(actual, max_detail)
        mask = _body_mask(mesh_type, seed, grid, shape)
        if mask is not None:
            frame = cd.mask_frame(mask, body.vertices, scale_um)
            instances = cd.organelles(
                mask, scale_um=scale_um, count=drawn,
                radius_um=profile.lytic_granule_radius_um,
                seed=seed + 1, frame=frame)
            scene.details.append(Detail(
                "lytic granules", instances.positions,
                profile.lytic_granule_radius_um,
                DETAIL_COLORS.get("granules", (0.92, 0.62, 0.26)),
                actual=actual, drawn=drawn))

    return scene


def _body_mask(mesh_type: str, seed: int, grid: int,
               params: Optional["ShapeParams"] = None):
    """The builder's own occupancy grid for the body, or None.

    Organelles are placed inside the MASK rather than inside the mesh:
    membership is exact there, where a point-in-mesh test on an
    extracted surface is not.
    """
    try:
        from cognisom.visualization.morphology import build_masks
        masks, _ = build_masks(mesh_type, seed=seed, grid=grid, params=params)
    except Exception as exc:                            # pragma: no cover
        log.debug("no mask for %s: %s", mesh_type, exc)
        return None
    return masks.get("body")
