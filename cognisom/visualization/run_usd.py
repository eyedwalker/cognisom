"""A whole run as one scrubbable USD stage.

Everything a viewer needs to *watch* a simulation rather than inspect
its last frame: the cells, moving and dying on the frames they moved
and died; the immune cells, patrolling; and, on the frame each one
happened, the kills, divisions, deaths, recruitments and metastases the
engine emitted -- drawn by :mod:`.interaction_usd` from the positions
the events themselves carried.

Why a time-sampled cell cloud is new
------------------------------------
:func:`cognisom.biousd.usd_writer.write_cells` authors one moment. Every
Bio-USD export so far has been a still: the final state, or one
snapshot. The recorder and the marker writer have been able to draw an
event *on its frame* for some time, and were attached to nothing,
because a kill marker flashing over a cell cloud that never moves is a
caption on the wrong photograph. Here the cloud itself is authored with
one time sample per recorded snapshot, so the marker and the cell it
marks are on the same frame.

Every lineage is drawn as itself
--------------------------------
The first version of this file drew every cell as one of two spheres --
10 um for a cell, 8 um for an immune cell, both invented here -- so the
lineages differed only by colour. Now each cell on each frame is
resolved through :mod:`.lineage_appearance` from the frame's type token
(``normal`` / ``cancer``; ``T_cell`` / ``NK_cell`` / ``macrophage`` with
the macrophage's polarization), and what comes back is a library entity
with a built morphology, a diameter with a stated source, and a colour.
One prototype is authored per distinct mesh actually used, under
``/World/Prototypes/<mesh_type>``, plus ``unresolved_sphere`` for any
appearance with no builder; both instancers share that prototype set, so
a protoIndex means the same thing in either. ``protoIndices``,
``scales`` and ``displayColor`` are time-sampled together: a cell that
transforms changes shape, size and colour on its frame, and a macrophage
crossing the +0.3 polarization cutoff changes from pseudopod to spindle
on its frame. (The colour was authored once, from the last frame, for a
while -- so a cell wore the cancer colour on the frames its shape said
it was normal.)

What was drawn is recorded, not implied. The root layer carries
``cognisom:appearance``, a JSON string listing every appearance key
used with its entity, mesh, diameter, size source, the library's
measurement convention for a cited size (a circulating tumour cell, a
72 h blast, an alveolar macrophage: none of them a cell in a tissue
section) and last-frame count, and the report says which sizes are
still estimates and how badly the drawn cells overlap on the lattice
(the slab packs at 11 um; a measured macrophage is 21.2 um across).

Where the shapes are legible
----------------------------
On a sparse population -- the hand-built fixture in the tests, cells
hundreds of micrometres apart -- every lineage reads by silhouette. In
the shipped slab and ring presets it does not: on the last frame of the
2 h ring Baseline run 99% of the 116 drawn cells sit inside a
neighbour (mean depth 9.9 um), because the lattice spacing is below the
sourced diameters. The stage data is right and the picture is a pile;
:meth:`RunWriteReport.describe` says so whenever more than half the
drawn cells interpenetrate, and the fix is the engine's lattice, not a
smaller cell.

What is authored
----------------
``/World/Cells`` -- a ``PointInstancer`` whose instances are the union
of every cell id the run recorded. ``positions``, ``invisibleIds``,
``protoIndices``, ``scales`` and the per-cell primvars are time-sampled,
one sample per snapshot frame, so scrubbing shows each cell where it was
and hides it before it was born and after it died. The *default*
(un-timed) values are the last frame, so a consumer that reads the stage
without a time -- the Bio-USD validator, a still renderer -- sees the
end state and a conforming population. A cell not present on a frame
keeps its last known position and its full scale and is listed in
``invisibleIds`` for that frame; a hidden instance is not drawn
(verified against usdrecord by the marker writer). ``invisibleIds`` is
the one hiding mechanism, deliberately: the parallel exporter this
replaced also zeroed a corpse's scale for viewers that ignore
``invisibleIds``, but the Bio-USD validator refuses a non-positive
diameter in the un-timed view, and the un-timed view is the last
frame, on which the dead are many. ``extent`` is time-sampled too, and
is the bound of every authored instance on that frame -- each
prototype's local bound scaled by the instance's diameter, not a
sphere of that diameter: prototypes are normalised to a MEAN axis
extent of one, so a columnar cell is 1.6 diameters tall and a spindle
1.97 long, and a diameter/2 pad (exact for the two spheres this file
started with) cut both off.

``/World/ImmuneCells`` -- the same for immune cells, by ``immune_ids``.
Their type token is ``immune``; the finer type and activation state
are ``cognisom:immune:type`` and ``cognisom:immune:activated``, outside
the ``bio:`` namespace because the vocabulary has no such primvars.
``cognisom:cell:appearance`` on both instancers is the per-frame
appearance key (``macrophage_M2``, ``cancer``, ...).

Both instancers apply the ``BioCellAPI`` schema (Bio-USD 0.2.0) and
``/World`` carries the UsdProfiles claims: ``biousd.cells`` is ``hard``,
``biousd.ontology`` an ``enhancement`` when any cell of either population
asserts a term (an immune population always does, ``CL:0000738``; a run
of ``normal`` cells alone asserts none). Both are metadata authored
after the primvars and need no plugin to write; the plugin is registered
anyway, because importing this module imports ``cognisom.biousd`` before
any stage is created here, so in a Cognisom process the schema is known
and a reader of the written stage sees its ``allowedTokens`` and docs.

``/World/Interactions/*``, ``/World/Fields/*``, ``/World/Cameras/EventCam``
-- from :mod:`.interaction_usd`, unchanged.

Frames are the runner's snapshots: frame ``k`` is ``cell_snapshots[k]``
at its recorded time, and an event lands on the first frame at or after
its time (:class:`~.interaction_usd.FrameMap`). The snapshot cadence is
therefore the time resolution of everything here.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Grid the prototypes are meshed at: the writer's own default (16),
#: the far rung of the library's LOD ladder. Not 8 -- at grid 8 an
#: elongated_spindle meshes as a 44-triangle blob with a unit-cube
#: extent, so an M2 macrophage would be a ball again.
from cognisom.biousd.usd_writer import PROTOTYPE_GRID
from cognisom.biousd.standard.api_schema import apply_api
from cognisom.biousd.standard.profiles import write_claims

from .interaction_recorder import VisualEvent
from .interaction_usd import (
    FrameMap, author_cell_oxygen, author_engagement_bonds,
    author_event_camera, author_interaction_markers, author_legend,
    ensure_dome_light, ensure_stage_timeline, events_bounds,
    legend_entries_for, oxygen_frames_from_snapshots)
from .lineage_appearance import UNRESOLVED_PROTOTYPE, Appearance, appearance_for

log = logging.getLogger(__name__)

CELLS_PATH = "/World/Cells"
IMMUNE_PATH = "/World/ImmuneCells"
PROTOTYPE_ROOT = "/World/Prototypes"
CAMERA_PATH = "/World/Cameras/EventCam"

#: Above this fraction of drawn cells inside a neighbour, the report
#: says the shapes are not legible on this lattice. The ring Baseline
#: run sits at 0.99; the hand-built fixture at 0.4 (two cells placed to
#: overlap on purpose).
OVERLAP_LEGIBILITY_FRACTION = 0.5


@dataclass
class RunWriteReport:
    path: str
    frames: int = 0
    first_time_h: float = 0.0
    last_time_h: float = 0.0
    cells: int = 0                 # distinct cell ids over the run
    immune_cells: int = 0
    events_recorded: int = 0
    events_by_kind: Dict[str, int] = field(default_factory=dict)
    markers: Dict[str, Any] = field(default_factory=dict)
    bonds: Dict[str, Any] = field(default_factory=dict)
    oxygen: Dict[str, Any] = field(default_factory=dict)
    legend: Optional[str] = None
    bytes: int = 0
    #: Appearance key -> what was drawn for it (entity, mesh_type,
    #: prototype, diameter_um, size_source, cited, colour, note,
    #: count_last_frame). The same record is in the stage's root layer
    #: as ``cognisom:appearance``.
    appearance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: Appearance keys whose size is not a radius_um citation.
    unsourced: List[str] = field(default_factory=list)
    #: Prototype prim paths, in protoIndex order.
    prototypes: List[str] = field(default_factory=list)
    #: On the last frame, over every drawn cell of both populations:
    #: the fraction whose nearest neighbour is closer than the sum of
    #: the two drawn radii (diameter/2 each: the mean-extent sphere, not
    #: the anisotropic prototype), and the mean overlap depth over those.
    overlap: Dict[str, float] = field(default_factory=dict)
    #: Instancer paths that apply ``BioCellAPI`` (Bio-USD 0.2.0).
    api_applied: List[str] = field(default_factory=list)
    #: The prim carrying the UsdProfiles claims (``/World``).
    claims_prim: Optional[str] = None

    def describe(self) -> str:
        drawn = ", ".join(f"{k} {v}" for k, v in sorted(
            self.markers.get("drawn", {}).items())) or "none"
        skipped = (self.markers.get("no_position", 0)
                   + self.markers.get("after_last_frame", 0)
                   + self.markers.get("suppressed_duplicate_death", 0))
        text = (f"{self.frames} frames, {self.first_time_h:.2f}-{self.last_time_h:.2f} h; "
                f"{self.cells} cells, {self.immune_cells} immune; "
                f"{self.events_recorded} events recorded, drawn: {drawn}"
                + (f"; {skipped} not drawn (no position / past the last frame / "
                   f"duplicate of a kill)" if skipped else ""))
        if self.appearance:
            shapes = ", ".join(
                f"{key} {rec.get('prototype') or 'sphere'} {rec['diameter_um']:.1f} um "
                f"({rec['size_source'].split(' ', 1)[1] if rec.get('cited') else 'est'})"
                for key, rec in sorted(self.appearance.items()))
            text += f"; shapes: {shapes}"
        if self.overlap:
            fraction = self.overlap.get("fraction", 0.0)
            text += (f"; overlap: {fraction:.0%} of "
                     f"{int(self.overlap.get('cells', 0))} drawn cells sit closer than their "
                     f"mean-extent radii, mean depth {self.overlap.get('mean_depth_um', 0.0):.1f} um")
            if fraction > OVERLAP_LEGIBILITY_FRACTION:
                text += (" -- most drawn cells are inside a neighbour on this lattice, so "
                         "the shapes are not legible here; the stage data is right and the "
                         "spacing is the engine's to fix")
        if self.api_applied:
            text += f"; BioCellAPI on {len(self.api_applied)} population(s)"
            if self.claims_prim:
                text += f", profile claims on {self.claims_prim}"
        return text


def _events_of(source: Any) -> List[VisualEvent]:
    recorder = getattr(source, "recorder", None)
    if recorder is not None:
        return list(recorder.events)
    return []


def _type_token(value: Any) -> str:
    text = str(getattr(value, "value", value) or "normal")
    return text


def _ontology_term(token: str) -> str:
    """The coarse type's term, or empty. ``normal`` is a state, not a
    lineage, and has none -- an unasserted identity, not a wrong one."""
    try:
        from cognisom.biousd.standard.terms import curie_for
        return curie_for(None, cell_type=token) or ""
    except Exception:                                # noqa: BLE001
        return ""


class _Timeline:
    """Per-id, per-frame values for one population, built once."""

    def __init__(self, frames: Sequence[Tuple[List[int], Optional[np.ndarray], Dict[str, list]]]):
        self.ids: List[int] = []
        seen: Dict[int, int] = {}
        for ids, _, _ in frames:
            for cid in ids:
                if cid not in seen:
                    seen[cid] = len(self.ids)
                    self.ids.append(int(cid))
        self.index = seen
        n = len(self.ids)
        self.n_frames = len(frames)
        self.positions = np.zeros((self.n_frames, n, 3), dtype="float32")
        self.present = np.zeros((self.n_frames, n), dtype=bool)
        self.fields: Dict[str, List[list]] = {}
        last = np.full((n, 3), np.nan, dtype="float32")
        for k, (ids, pos, fields) in enumerate(frames):
            if ids and pos is not None:
                rows = [self.index[c] for c in ids]
                last[rows] = np.asarray(pos, dtype="float32").reshape(len(ids), 3)
                self.present[k, rows] = True
            self.positions[k] = last
            for name, values in fields.items():
                col = self.fields.setdefault(name, [[None] * n for _ in range(self.n_frames)])
                for cid, v in zip(ids, values):
                    col[k][self.index[cid]] = v
        # A cell not yet born keeps the position it will be born at, so
        # nothing sits at the origin; it is invisible there anyway.
        for i in range(n):
            born = np.argmax(self.present[:, i]) if self.present[:, i].any() else 0
            self.positions[:born, i] = self.positions[born, i]
        # Fill a field's gaps (absent frames) with the nearest known value.
        for name, col in self.fields.items():
            for i in range(n):
                known = None
                for k in range(self.n_frames):
                    if col[k][i] is not None:
                        known = col[k][i]
                    elif known is not None:
                        col[k][i] = known
                known = None
                for k in range(self.n_frames - 1, -1, -1):
                    if col[k][i] is not None:
                        known = col[k][i]
                    elif known is not None:
                        col[k][i] = known

    def hidden_at(self, k: int) -> List[int]:
        return [self.ids[i] for i in np.flatnonzero(~self.present[k])]

    def field_at(self, name: str, k: int, i: int, default: Any = None) -> Any:
        col = self.fields.get(name)
        if col is None:
            return default
        value = col[k][i]
        return default if value is None else value


def _resolve_appearances(tl: _Timeline, *, token_field: str, default_token: str,
                         polarization_field: Optional[str] = None,
                         apps: Dict[str, Appearance]) -> List[List[str]]:
    """Per frame, per id: the appearance key, resolving each cell from the
    token it carried on that frame. ``apps`` collects every Appearance used."""
    keys: List[List[str]] = []
    for k in range(tl.n_frames):
        row: List[str] = []
        for i in range(len(tl.ids)):
            token = tl.field_at(token_field, k, i, default_token) or default_token
            pol = tl.field_at(polarization_field, k, i) if polarization_field else None
            app = appearance_for(token, polarization=pol)
            apps.setdefault(app.key, app)
            row.append(app.key)
        keys.append(row)
    return keys


def _author_prototype_set(stage, apps: Dict[str, Appearance], grid: int
                          ) -> Tuple[List[Any], Dict[str, int], np.ndarray]:
    """One prototype per distinct mesh actually used, plus the named
    sphere, every one bound to the primvar-driven cell material (a
    prototype with no material renders under the viewer's default and
    ignores the instancer's colour).

    Returns the prototype paths, ``name -> protoIndex``, and each
    prototype's local bound as a ``(n_prototypes, 2, 3)`` array of
    ``(lo, hi)`` corners at unit scale -- what the instancers' extents
    are computed from. At unit scale that bound is NOT the unit cube:
    the mean axis extent is one, so a polarized_columnar cell spans
    0.70 x 0.70 x 1.60 and an elongated_spindle 1.97 x 0.66 x 0.37.
    """
    from pxr import Usd, UsdGeom, UsdShade

    from cognisom.biousd.usd_writer import author_cell_material, author_prototypes

    mesh_types = sorted({a.prototype for a in apps.values()} - {UNRESOLVED_PROTOTYPE})
    paths, _faces = author_prototypes(stage, mesh_types, grid=grid, root=PROTOTYPE_ROOT)
    paths = list(paths)
    sphere = UsdGeom.Sphere.Define(stage, f"{PROTOTYPE_ROOT}/{UNRESOLVED_PROTOTYPE}")
    sphere.GetRadiusAttr().Set(0.5)
    sphere.GetExtentAttr().Set([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    paths.append(sphere.GetPath())
    names = mesh_types + [UNRESOLVED_PROTOTYPE]
    material = author_cell_material(stage)
    bounds = np.zeros((len(paths), 2, 3), dtype="float64")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for j, path in enumerate(paths):
        prim = stage.GetPrimAtPath(path)
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(material)
        box = cache.ComputeLocalBound(prim).ComputeAlignedRange()
        bounds[j, 0] = tuple(box.GetMin())
        bounds[j, 1] = tuple(box.GetMax())
    return paths, {name: i for i, name in enumerate(names)}, bounds


def _nearest_brute(points: np.ndarray, chunk: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    n = len(points)
    dist = np.empty(n, dtype=float)
    nearest = np.empty(n, dtype=int)
    for start in range(0, n, chunk):
        block = points[start:start + chunk]
        d = np.linalg.norm(block[:, None, :] - points[None, :, :], axis=2)
        rows = np.arange(len(block))
        d[rows, rows + start] = np.inf
        nearest[start:start + chunk] = d.argmin(axis=1)
        dist[start:start + chunk] = d[rows, nearest[start:start + chunk]]
    return dist, nearest


def overlap_metric(positions: Any, radii: Any) -> Dict[str, float]:
    """How badly drawn cells overlap: the fraction whose nearest
    neighbour is closer than the sum of the two drawn radii, and the mean
    overlap depth (um) over those cells.

    The radius is ``diameter / 2`` -- the mean-extent sphere of each
    cell, not the drawn body. Prototypes are anisotropic (a 21.2 um
    spindle is 41.7 um long and 7.9 um thick; a 15 um columnar cell is
    24 um tall), so the sphere under-counts overlap along a long axis
    and over-counts it along a short one. It is the one radius the
    record carries, and the question it answers -- are these cells
    packed tighter than their measured sizes -- does not need the
    orientation the engine does not model.

    Reported, not corrected. The slab packs cells on an 11 um lattice
    (median nearest neighbour 9.2 um) and the ring run tighter, while
    sourced suspension diameters run 8-21 um; drawing cells at their
    size on that lattice overlaps them, and shrinking them to fit would
    hide the one number that says the lattice is too tight.
    """
    pts = np.asarray(positions, dtype=float).reshape(-1, 3)
    r = np.asarray(radii, dtype=float).reshape(-1)
    n = len(pts)
    if n < 2:
        return {"fraction": 0.0, "mean_depth_um": 0.0, "cells": float(n)}
    try:
        from scipy.spatial import cKDTree
        d, j = cKDTree(pts).query(pts, k=2)
        # With coincident points the self-match may land in column 1.
        own = np.arange(n)
        nearest = np.where(j[:, 1] == own, j[:, 0], j[:, 1])
        dist = np.where(j[:, 1] == own, d[:, 0], d[:, 1])
    except Exception:                                # noqa: BLE001 - no scipy
        dist, nearest = _nearest_brute(pts)
    depth = r + r[nearest] - dist
    over = depth > 0.0
    return {"fraction": float(over.mean()),
            "mean_depth_um": float(depth[over].mean()) if over.any() else 0.0,
            "cells": float(n)}


def _extent_of(lo: np.ndarray, hi: np.ndarray) -> list:
    """``[min corner, max corner]`` of per-instance ``(n, 3)`` corners,
    widened by one float32 ulp so the authored (float) extent never
    rounds inside the bound it was computed from."""
    from pxr import Gf

    a = np.nextafter(lo.min(axis=0).astype("float32"), np.float32(-np.inf))
    b = np.nextafter(hi.max(axis=0).astype("float32"), np.float32(np.inf))
    return [Gf.Vec3f(*map(float, a)), Gf.Vec3f(*map(float, b))]


def _author_population(stage, path: str, tl: _Timeline, *,
                       prototype_paths: Sequence[Any], proto_bounds: np.ndarray,
                       proto_indices: np.ndarray, diameters: np.ndarray,
                       type_tokens: Sequence[str], colours: np.ndarray,
                       extra_primvars: Dict[str, Tuple[Any, Any]]) -> Tuple[bool, bool]:
    """One time-sampled PointInstancer. ``proto_indices`` and
    ``diameters`` are ``(n_frames, n)`` and ``colours`` is
    ``(n_frames, n, 3)``; ``type_tokens`` is per id and constant;
    ``proto_bounds`` is the ``(n_prototypes, 2, 3)`` unit-scale bound of
    each prototype; ``extra_primvars`` maps name -> (Sdf type, per-frame
    list of per-id values).

    Returns ``(api_applied, ontology_asserted)``: whether ``BioCellAPI``
    was applied to the instancer, and whether any cell carries an
    ontology term -- what the stage's claims must say about it."""
    from pxr import Sdf, UsdGeom, Vt

    n = len(tl.ids)
    last = tl.n_frames - 1
    instancer = UsdGeom.PointInstancer.Define(stage, path)
    instancer.GetPrototypesRel().SetTargets(list(prototype_paths))

    def _indices(k):
        return Vt.IntArray.FromNumpy(np.ascontiguousarray(proto_indices[k], dtype="int32"))

    def _scales(k):
        return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(
            np.repeat(diameters[k].astype("float32")[:, None], 3, axis=1)))

    def _colours(k):
        return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(colours[k], dtype="float32"))

    proto_attr = instancer.GetProtoIndicesAttr()
    scale_attr = instancer.GetScalesAttr()
    pos_attr = instancer.GetPositionsAttr()
    inv_attr = instancer.CreateInvisibleIdsAttr()
    # Defaults: the last frame, for a reader with no notion of time.
    proto_attr.Set(_indices(last))
    scale_attr.Set(_scales(last))
    pos_attr.Set(Vt.Vec3fArray.FromNumpy(tl.positions[last]))
    inv_attr.Set(Vt.Int64Array([int(i) for i in np.flatnonzero(~tl.present[last])]))
    for k in range(tl.n_frames):
        proto_attr.Set(_indices(k), float(k))
        scale_attr.Set(_scales(k), float(k))
        pos_attr.Set(Vt.Vec3fArray.FromNumpy(tl.positions[k]), float(k))
        # invisibleIds are instance INDICES, not ids.
        inv_attr.Set(Vt.Int64Array([int(i) for i in np.flatnonzero(~tl.present[k])]), float(k))
    if n:
        # The extent is the bound of what is authored on each frame:
        # every instance's prototype bound (unit scale) scaled by its
        # diameter about its position. Instances are unrotated, so the
        # scaled corners are the bound. Hidden instances are included
        # -- they sit at a last known position inside the cloud and a
        # conservative extent is still an extent -- so the attribute is
        # exactly the bound of the arrays beside it.
        lo = tl.positions.astype("float64") + diameters[..., None] * proto_bounds[proto_indices, 0]
        hi = tl.positions.astype("float64") + diameters[..., None] * proto_bounds[proto_indices, 1]
        extent_attr = instancer.CreateExtentAttr()
        extent_attr.Set(_extent_of(lo[last], hi[last]))
        for k in range(tl.n_frames):
            extent_attr.Set(_extent_of(lo[k], hi[k]), float(k))

    api = UsdGeom.PrimvarsAPI(instancer.GetPrim())
    varying = UsdGeom.Tokens.varying
    api.CreatePrimvar("bio:cell:id", Sdf.ValueTypeNames.IntArray, varying).Set(
        Vt.IntArray.FromNumpy(np.asarray(tl.ids, dtype="int32")))
    api.CreatePrimvar("bio:cell:type", Sdf.ValueTypeNames.TokenArray, varying).Set(
        Vt.TokenArray(list(type_tokens)))
    terms = [_ontology_term(t) for t in type_tokens]
    api.CreatePrimvar("bio:cell:ontologyTerm", Sdf.ValueTypeNames.TokenArray, varying).Set(
        Vt.TokenArray(terms))
    alive = api.CreatePrimvar("bio:cell:alive", Sdf.ValueTypeNames.BoolArray, varying)
    alive.Set(Vt.BoolArray([bool(v) for v in tl.present[last]]))
    for k in range(tl.n_frames):
        alive.Set(Vt.BoolArray([bool(v) for v in tl.present[k]]), float(k))
    # Colour follows the shape frame by frame: the appearance a cell has
    # on a frame is the colour it wears on that frame.
    colour = api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, varying)
    colour.Set(_colours(last))
    for k in range(tl.n_frames):
        colour.Set(_colours(k), float(k))
    for name, (type_name, per_frame) in extra_primvars.items():
        pv = api.CreatePrimvar(name, type_name, varying)
        pv.Set(per_frame[last])
        for k in range(tl.n_frames):
            pv.Set(per_frame[k], float(k))
    # The schema goes on last: every primvar above, timed or not, carries
    # its own `varying`, so the file never depends on the schema's
    # fallback. AddAppliedSchema needs no plugin.
    return apply_api(instancer.GetPrim()), any(terms)


def write_run(path: str, source: Any, *,
              snapshots: Optional[Sequence[Any]] = None,
              events: Optional[Sequence[VisualEvent]] = None,
              up_axis: str = "Y",
              fps: float = 24.0,
              oxygen: bool = True,
              bonds: bool = True,
              camera: bool = True,
              legend_png: Optional[str] = None,
              grid: int = PROTOTYPE_GRID) -> RunWriteReport:
    """Author ``source`` (an :class:`EngineRunner` after ``run()``) at ``path``.

    ``snapshots`` and ``events`` override what is read from ``source``
    (``cell_snapshots`` and ``recorder.events``), so a saved run can be
    written without the runner. ``legend_png`` is a path to write the
    legend texture to; it must sit beside the stage, so it is off by
    default and a single-file download does without it. ``grid`` is the
    voxel grid the prototypes are meshed at.
    """
    from pxr import Sdf, Usd, UsdGeom, Vt

    from cognisom.biousd.standard.vocabulary import METERS_PER_UNIT, VERSION

    snaps = list(snapshots if snapshots is not None else getattr(source, "cell_snapshots", []))
    evs = list(events if events is not None else _events_of(source))
    if not snaps:
        raise ValueError("nothing to write: the run recorded no cell snapshots")

    report = RunWriteReport(path=path, frames=len(snaps),
                            first_time_h=float(snaps[0].time), last_time_h=float(snaps[-1].time),
                            events_recorded=len(evs))
    for ev in evs:
        report.events_by_kind[ev.kind] = report.events_by_kind.get(ev.kind, 0) + 1
    frame_map = FrameMap([float(s.time) for s in snaps])

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, getattr(UsdGeom.Tokens, up_axis.lower()))
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    ensure_stage_timeline(stage, frame_map, fps=fps)
    ensure_dome_light(stage)

    # ── timelines ─────────────────────────────────────────────────
    cell_frames = []
    for s in snaps:
        ids = [int(i) for i in (s.cell_ids or [])]
        fields = {"type": [_type_token(t) for t in (s.cell_types or [])],
                  "phase": [str(p) for p in (s.cell_phases or [])],
                  "oxygen": [float(o) for o in (s.cell_oxygen or [])],
                  "age": [float(a) for a in (s.cell_age or [])]}
        fields = {k: v for k, v in fields.items() if len(v) == len(ids)}
        cell_frames.append((ids, s.cell_positions, fields))
    cells = _Timeline(cell_frames)
    report.cells = len(cells.ids)

    immune_frames = []
    for s in snaps:
        ids = [int(i) for i in (s.immune_ids or [])]
        fields = {"itype": [str(t) for t in (s.immune_types or [])],
                  "activated": [bool(a) for a in (s.immune_activated or [])],
                  # Older saved runs have no polarization; the field is
                  # then absent and every macrophage resolves unpolarised.
                  "polarization": [float(p) for p in
                                   (getattr(s, "immune_polarization", None) or [])]}
        fields = {k: v for k, v in fields.items() if len(v) == len(ids)}
        immune_frames.append((ids, s.immune_positions, fields))
    immune = _Timeline(immune_frames)
    report.immune_cells = len(immune.ids)

    # ── appearance: who is drawn as what, per frame ───────────────
    apps: Dict[str, Appearance] = {}
    cell_keys = _resolve_appearances(cells, token_field="type", default_token="normal", apps=apps)
    immune_keys = _resolve_appearances(immune, token_field="itype", default_token="immune",
                                       polarization_field="polarization", apps=apps)
    proto_paths, index_of, proto_bounds = _author_prototype_set(stage, apps, grid)
    report.prototypes = [str(p) for p in proto_paths]

    def _arrays(keys: List[List[str]], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per frame, per id: protoIndex, diameter and colour."""
        idx = np.zeros((len(keys), n), dtype="int32")
        diam = np.zeros((len(keys), n), dtype="float32")
        rgb = np.zeros((len(keys), n, 3), dtype="float32")
        for k, row in enumerate(keys):
            for i, key in enumerate(row):
                app = apps[key]
                idx[k, i] = index_of[app.prototype]
                diam[k, i] = app.diameter_um
                rgb[k, i] = app.colour
        return idx, diam, rgb

    counts: Dict[str, int] = {}
    drawn_positions: List[np.ndarray] = []
    drawn_radii: List[np.ndarray] = []
    ontology_asserted = False

    def _count_last(tl: _Timeline, keys: List[List[str]], diam: np.ndarray) -> None:
        last = tl.n_frames - 1
        present = np.flatnonzero(tl.present[last])
        for i in present:
            counts[keys[last][i]] = counts.get(keys[last][i], 0) + 1
        if len(present):
            drawn_positions.append(tl.positions[last][present])
            drawn_radii.append(diam[last][present] / 2.0)

    # ── cells ──────────────────────────────────────────────────────
    if cells.ids:
        last = cells.n_frames - 1
        n = len(cells.ids)
        proto_idx, diam, colours = _arrays(cell_keys, n)
        # bio:cell:type is the LAST known and un-timed, as the Bio-USD
        # validator reads it: a transformed cell reads as cancer in the
        # un-timed view, and cognisom:cell:typeAtFrame carries the
        # per-frame token. Shape, size and colour are per frame.
        types_last = [cells.field_at("type", last, i, "normal") for i in range(n)]
        extra: Dict[str, Tuple[Any, Any]] = {}
        if "type" in cells.fields:
            extra["cognisom:cell:typeAtFrame"] = (
                Sdf.ValueTypeNames.TokenArray,
                [Vt.TokenArray([str(v or "normal") for v in row]) for row in cells.fields["type"]])
        extra["cognisom:cell:appearance"] = (
            Sdf.ValueTypeNames.TokenArray, [Vt.TokenArray(row) for row in cell_keys])
        if "phase" in cells.fields:
            extra["bio:cell:phase"] = (
                Sdf.ValueTypeNames.TokenArray,
                [Vt.TokenArray([str(v or "G1") for v in row]) for row in cells.fields["phase"]])
        if "age" in cells.fields:
            extra["bio:cell:age"] = (
                Sdf.ValueTypeNames.FloatArray,
                [Vt.FloatArray([float(v or 0.0) for v in row]) for row in cells.fields["age"]])
        if "oxygen" in cells.fields:
            extra["cognisom:cell:oxygen"] = (
                Sdf.ValueTypeNames.FloatArray,
                [Vt.FloatArray([float(v if v is not None else 0.0) for v in row])
                 for row in cells.fields["oxygen"]])
        applied, asserted = _author_population(
            stage, CELLS_PATH, cells, prototype_paths=proto_paths, proto_bounds=proto_bounds,
            proto_indices=proto_idx, diameters=diam, type_tokens=types_last, colours=colours,
            extra_primvars=extra)
        if applied:
            report.api_applied.append(CELLS_PATH)
        ontology_asserted = ontology_asserted or asserted
        _count_last(cells, cell_keys, diam)

    # ── immune cells ──────────────────────────────────────────────
    if immune.ids:
        last = immune.n_frames - 1
        n = len(immune.ids)
        proto_idx, diam, colours = _arrays(immune_keys, n)
        extra = {}
        if "itype" in immune.fields:
            extra["cognisom:immune:type"] = (
                Sdf.ValueTypeNames.TokenArray,
                [Vt.TokenArray([str(v or "") for v in row]) for row in immune.fields["itype"]])
        if "activated" in immune.fields:
            extra["cognisom:immune:activated"] = (
                Sdf.ValueTypeNames.BoolArray,
                [Vt.BoolArray([bool(v) for v in row]) for row in immune.fields["activated"]])
        if "polarization" in immune.fields:
            extra["cognisom:immune:polarization"] = (
                Sdf.ValueTypeNames.FloatArray,
                [Vt.FloatArray([float(v if v is not None else 0.0) for v in row])
                 for row in immune.fields["polarization"]])
        extra["cognisom:cell:appearance"] = (
            Sdf.ValueTypeNames.TokenArray, [Vt.TokenArray(row) for row in immune_keys])
        applied, asserted = _author_population(
            stage, IMMUNE_PATH, immune, prototype_paths=proto_paths, proto_bounds=proto_bounds,
            proto_indices=proto_idx, diameters=diam, type_tokens=["immune"] * n, colours=colours,
            extra_primvars=extra)
        if applied:
            report.api_applied.append(IMMUNE_PATH)
        ontology_asserted = ontology_asserted or asserted
        _count_last(immune, immune_keys, diam)

    # ── the claims: what this stage uses, on the default prim ─────
    # Written whenever a population was: the cells are a hard usage, the
    # ontology terms an enhancement claimed only when some cell asserts
    # one, so the validator's consistency check (profile.inconsistent)
    # agrees with the primvars beside it.
    if report.api_applied:
        report.claims_prim = write_claims(stage, ontology=ontology_asserted)

    # ── the record of what was drawn ──────────────────────────────
    for key, app in apps.items():
        rec = app.as_dict()
        rec["count_last_frame"] = int(counts.get(key, 0))
        report.appearance[key] = rec
    report.unsourced = sorted(k for k, a in apps.items() if not a.cited)
    if drawn_positions:
        report.overlap = overlap_metric(np.vstack(drawn_positions), np.concatenate(drawn_radii))
    # A plain Python list will not pack into a crate file; an array
    # does, and so does a string -- which is why the record is JSON.
    stage.GetRootLayer().customLayerData = {
        "bio:version": VERSION,
        "cognisom:frameTimesHours": Vt.DoubleArray([float(s.time) for s in snaps]),
        "cognisom:appearance": json.dumps(report.appearance, sort_keys=True)}

    # ── events, on their frames ───────────────────────────────────
    report.markers = author_interaction_markers(stage, evs, frame_map)
    if bonds:
        report.bonds = author_engagement_bonds(stage, evs, frame_map)
    if oxygen:
        try:
            report.oxygen = author_cell_oxygen(
                stage, oxygen_frames_from_snapshots(snaps), frame_map)
        except Exception as exc:                     # noqa: BLE001
            report.oxygen = {"skipped": f"{type(exc).__name__}: {exc}"}

    # ── camera and legend ─────────────────────────────────────────
    if camera:
        extra_pts = [tuple(map(float, p)) for p in cells.positions.reshape(-1, 3)[::max(1, cells.positions.size // 3000 or 1)]] if cells.ids else None
        center, radius = events_bounds(evs, extra_points=extra_pts)
        author_event_camera(stage, center, radius, path=CAMERA_PATH)
        if legend_png:
            from .legend_texture import render_legend_png
            entries = legend_entries_for(sorted(report.markers.get("drawn", {})))
            png, aspect = render_legend_png(entries, legend_png,
                                            subtitle=f"{report.frames} frames, "
                                                     f"{report.last_time_h:.1f} h")
            author_legend(stage, CAMERA_PATH, os.path.relpath(png, os.path.dirname(path) or "."),
                          texture_aspect=aspect)
            report.legend = png

    stage.GetRootLayer().Save()
    report.bytes = os.path.getsize(path)
    log.info("run written: %s", report.describe())
    return report
