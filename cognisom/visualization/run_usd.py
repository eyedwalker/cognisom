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

What is authored
----------------
``/World/Cells`` -- a ``PointInstancer`` whose instances are the union
of every cell id the run recorded. ``positions``, ``invisibleIds`` and
the per-cell primvars are time-sampled, one sample per snapshot frame,
so scrubbing shows each cell where it was and hides it before it was
born and after it died. The *default* (un-timed) values are the last
frame, so a consumer that reads the stage without a time -- the Bio-USD
validator, a still renderer -- sees the end state and a conforming
population. A cell not present on a frame keeps its last known position
and is listed in ``invisibleIds`` for that frame; a hidden instance is
not drawn (verified against usdrecord by the marker writer).

``/World/ImmuneCells`` -- the same for immune cells, by ``immune_ids``.
Their type token is ``immune``; the finer type and activation state
are ``cognisom:immune:type`` and ``cognisom:immune:activated``, outside
the ``bio:`` namespace because the vocabulary has no such primvars.

``/World/Interactions/*``, ``/World/Fields/*``, ``/World/Cameras/EventCam``
-- from :mod:`.interaction_usd`, unchanged.

Frames are the runner's snapshots: frame ``k`` is ``cell_snapshots[k]``
at its recorded time, and an event lands on the first frame at or after
its time (:class:`~.interaction_usd.FrameMap`). The snapshot cadence is
therefore the time resolution of everything here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .interaction_recorder import VisualEvent
from .interaction_usd import (
    FrameMap, author_cell_oxygen, author_engagement_bonds,
    author_event_camera, author_interaction_markers, author_legend,
    ensure_dome_light, ensure_stage_timeline, events_bounds,
    legend_entries_for, oxygen_frames_from_snapshots)

log = logging.getLogger(__name__)

CELLS_PATH = "/World/Cells"
IMMUNE_PATH = "/World/ImmuneCells"
PROTOTYPE_ROOT = "/World/Prototypes"
CAMERA_PATH = "/World/Cameras/EventCam"

#: Diameter authored for every cell. Snapshots carry no size, and
#: inventing one per type would be a second copy of a number the
#: library already struggles to source (see the cell-sizes note).
DEFAULT_DIAMETER_UM = 10.0
IMMUNE_DIAMETER_UM = 8.0


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

    def describe(self) -> str:
        drawn = ", ".join(f"{k} {v}" for k, v in sorted(
            self.markers.get("drawn", {}).items())) or "none"
        skipped = (self.markers.get("no_position", 0)
                   + self.markers.get("after_last_frame", 0)
                   + self.markers.get("suppressed_duplicate_death", 0))
        return (f"{self.frames} frames, {self.first_time_h:.2f}-{self.last_time_h:.2f} h; "
                f"{self.cells} cells, {self.immune_cells} immune; "
                f"{self.events_recorded} events recorded, drawn: {drawn}"
                + (f"; {skipped} not drawn (no position / past the last frame / "
                   f"duplicate of a kill)" if skipped else ""))


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


def _author_population(stage, path: str, tl: _Timeline, *, diameter: float,
                       type_tokens: Sequence[str], colours: Sequence[Sequence[float]],
                       extra_primvars: Dict[str, Tuple[Any, Any]],
                       frame_map: FrameMap, prototype_path: str) -> None:
    """One time-sampled PointInstancer. ``type_tokens`` / ``colours`` are
    per id and constant; ``extra_primvars`` maps name -> (Sdf type, per-frame
    list of per-id values)."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    n = len(tl.ids)
    last = tl.n_frames - 1
    instancer = UsdGeom.PointInstancer.Define(stage, path)
    instancer.GetPrototypesRel().SetTargets([prototype_path])
    instancer.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(np.zeros(n, dtype="int32")))
    instancer.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(
        np.full((n, 3), float(diameter), dtype="float32")))

    pos_attr = instancer.GetPositionsAttr()
    inv_attr = instancer.CreateInvisibleIdsAttr()
    # Defaults: the last frame, for a reader with no notion of time.
    pos_attr.Set(Vt.Vec3fArray.FromNumpy(tl.positions[last]))
    inv_attr.Set(Vt.Int64Array([int(i) for i in np.flatnonzero(~tl.present[last])]))
    for k in range(tl.n_frames):
        pos_attr.Set(Vt.Vec3fArray.FromNumpy(tl.positions[k]), float(k))
        # invisibleIds are instance INDICES, not ids.
        inv_attr.Set(Vt.Int64Array([int(i) for i in np.flatnonzero(~tl.present[k])]), float(k))
    if n:
        lo = tl.positions.reshape(-1, 3).min(axis=0) - diameter / 2
        hi = tl.positions.reshape(-1, 3).max(axis=0) + diameter / 2
        instancer.CreateExtentAttr().Set([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])

    api = UsdGeom.PrimvarsAPI(instancer.GetPrim())
    varying = UsdGeom.Tokens.varying
    api.CreatePrimvar("bio:cell:id", Sdf.ValueTypeNames.IntArray, varying).Set(
        Vt.IntArray.FromNumpy(np.asarray(tl.ids, dtype="int32")))
    api.CreatePrimvar("bio:cell:type", Sdf.ValueTypeNames.TokenArray, varying).Set(
        Vt.TokenArray(list(type_tokens)))
    api.CreatePrimvar("bio:cell:ontologyTerm", Sdf.ValueTypeNames.TokenArray, varying).Set(
        Vt.TokenArray([_ontology_term(t) for t in type_tokens]))
    alive = api.CreatePrimvar("bio:cell:alive", Sdf.ValueTypeNames.BoolArray, varying)
    alive.Set(Vt.BoolArray([bool(v) for v in tl.present[last]]))
    for k in range(tl.n_frames):
        alive.Set(Vt.BoolArray([bool(v) for v in tl.present[k]]), float(k))
    api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, varying).Set(
        Vt.Vec3fArray.FromNumpy(np.asarray(colours, dtype="float32").reshape(n, 3)))
    for name, (type_name, per_frame) in extra_primvars.items():
        pv = api.CreatePrimvar(name, type_name, varying)
        pv.Set(per_frame[last])
        for k in range(tl.n_frames):
            pv.Set(per_frame[k], float(k))


def write_run(path: str, source: Any, *,
              snapshots: Optional[Sequence[Any]] = None,
              events: Optional[Sequence[VisualEvent]] = None,
              up_axis: str = "Y",
              fps: float = 24.0,
              oxygen: bool = True,
              bonds: bool = True,
              camera: bool = True,
              legend_png: Optional[str] = None) -> RunWriteReport:
    """Author ``source`` (an :class:`EngineRunner` after ``run()``) at ``path``.

    ``snapshots`` and ``events`` override what is read from ``source``
    (``cell_snapshots`` and ``recorder.events``), so a saved run can be
    written without the runner. ``legend_png`` is a path to write the
    legend texture to; it must sit beside the stage, so it is off by
    default and a single-file download does without it.
    """
    from pxr import Sdf, Usd, UsdGeom, Vt

    from cognisom.biousd.standard.vocabulary import METERS_PER_UNIT, VERSION
    from cognisom.biousd.usd_writer import TYPE_COLOUR

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
    # A plain Python list will not pack into a crate file; an array does.
    stage.GetRootLayer().customLayerData = {
        "bio:version": VERSION,
        "cognisom:frameTimesHours": Vt.DoubleArray([float(s.time) for s in snaps])}
    ensure_stage_timeline(stage, frame_map, fps=fps)
    ensure_dome_light(stage)

    # ── cells ──────────────────────────────────────────────────────
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
    sphere = UsdGeom.Sphere.Define(stage, f"{PROTOTYPE_ROOT}/cell")
    sphere.GetRadiusAttr().Set(0.5)
    if cells.ids:
        last = cells.n_frames - 1
        # type is the LAST known type: a transformed cell reads as cancer
        # for the whole run, which the type primvar cannot time-sample
        # without a token per frame -- and it can, so it does, below.
        types_last = [cells.fields["type"][last][i] if "type" in cells.fields else "normal"
                      for i in range(len(cells.ids))]
        colours = [TYPE_COLOUR.get(t, TYPE_COLOUR["normal"]) for t in types_last]
        extra: Dict[str, Tuple[Any, Any]] = {}
        if "type" in cells.fields:
            extra["cognisom:cell:typeAtFrame"] = (
                Sdf.ValueTypeNames.TokenArray,
                [Vt.TokenArray([str(v or "normal") for v in row]) for row in cells.fields["type"]])
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
        _author_population(stage, CELLS_PATH, cells, diameter=DEFAULT_DIAMETER_UM,
                           type_tokens=types_last, colours=colours, extra_primvars=extra,
                           frame_map=frame_map, prototype_path=sphere.GetPath())

    # ── immune cells ──────────────────────────────────────────────
    immune_frames = []
    for s in snaps:
        ids = [int(i) for i in (s.immune_ids or [])]
        fields = {"itype": [str(t) for t in (s.immune_types or [])],
                  "activated": [bool(a) for a in (s.immune_activated or [])]}
        fields = {k: v for k, v in fields.items() if len(v) == len(ids)}
        immune_frames.append((ids, s.immune_positions, fields))
    immune = _Timeline(immune_frames)
    report.immune_cells = len(immune.ids)
    if immune.ids:
        isphere = UsdGeom.Sphere.Define(stage, f"{PROTOTYPE_ROOT}/immune")
        isphere.GetRadiusAttr().Set(0.5)
        extra = {}
        if "itype" in immune.fields:
            extra["cognisom:immune:type"] = (
                Sdf.ValueTypeNames.TokenArray,
                [Vt.TokenArray([str(v or "") for v in row]) for row in immune.fields["itype"]])
        if "activated" in immune.fields:
            extra["cognisom:immune:activated"] = (
                Sdf.ValueTypeNames.BoolArray,
                [Vt.BoolArray([bool(v) for v in row]) for row in immune.fields["activated"]])
        _author_population(stage, IMMUNE_PATH, immune, diameter=IMMUNE_DIAMETER_UM,
                           type_tokens=["immune"] * len(immune.ids),
                           colours=[TYPE_COLOUR["immune"]] * len(immune.ids),
                           extra_primvars=extra, frame_map=frame_map,
                           prototype_path=isphere.GetPath())

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
