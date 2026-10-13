"""Time-sampled USD export: a run, not a frame.

``SceneExporter.cells_to_usd`` writes one ``Sphere`` prim per cell at one
instant and sets ``startTimeCode``/``endTimeCode`` to 0 and 1 without
authoring a single time sample underneath them -- a still life with a
time axis bolted on.  This module writes the sequence instead: one
``UsdGeomPointInstancer`` for the tissue and one for the infiltrate, with
``positions``, ``protoIndices``, ``scales`` and ``invisibleIds`` sampled
at every recorded frame, an explicit camera framed on the union of all
frames, and stage time metadata that makes a viewer play it at a sane
speed.

What actually moves
-------------------
Measured on a 24 h Baseline run at dt=0.05 (``np.random.seed(12345)``,
241 recorded frames):

* **Somatic cells never move.**  ``CellularModule`` assigns
  ``cell.position`` at birth and never writes it again, so the
  displacement of every cell alive at both 0 h and 24 h is exactly
  0.0 um.  The tissue changes by *turnover*: 90 cells at t=0, 188 at
  t=24 h, 4 deaths.
* **Immune cells do migrate.**  ``ImmuneModule`` integrates
  ``position += velocity * dt``; measured displacement up to ~1.3 um
  over the run.  It is the only continuous motion in the scene.
* **Cell state changes.**  Mean cell O2 falls from the arterial 0.2100
  at t=0 to 0.1234 at 24 h, which is what the hypoxia colouring shows.
* **``cell.phase`` does not change.**  It is assigned ``'G1'`` at birth
  (``cellular_module.py``) and is never reassigned anywhere in the
  repository, so a "dividing" colour keyed on phase would be dead code
  that never fires.  Newborn cells are marked from the recorded id
  sequence instead: a cell whose id first appears at frame k divided
  between k-1 and k.  That is data the run produced.

Births and deaths with a fixed-length array
-------------------------------------------
A PointInstancer's arrays are one length for the whole prim, but the
population is not.  The stage therefore allocates one slot per cell id
*ever* seen, in first-appearance order, authors the stable ``ids`` array
once, and hides the slots that are not alive at a given frame two ways
at once:

* ``invisibleIds`` -- the idiom USD defines for exactly this, honoured by
  Hydra (``UsdGeomPointInstancer.ComputeMaskAtTime``) and by Omniverse;
  and
* ``scales`` set to (0, 0, 0) for the same slots.

Either alone suffices for a conforming viewer.  Both together mean a
viewer that ignores ``invisibleIds`` still shows nothing rather than a
corpse frozen at its last position -- a dead cell left on screen is a
fabricated cell, which is the failure mode that matters here.

Usage
-----
::

    python -m cognisom.visualization.usd_animation \\
        --scenario Baseline --duration 24 --out run.usda
    usdrecord --imageWidth 960 --frames 0:240 --camera /World/Camera \\
        run.usda frame.###.###.png

Use the two-group ``###.###`` placeholder, not ``###``.  Apple's
usdrecord (USD Tools 0.25.2, /usr/bin/usdrecord) formats a bare ``###``
with what looks like one significant digit -- frames 118, 119, 120 and
121 all land on ``f.1e+02.png`` and overwrite each other, so a 241-frame
range silently leaves three files.  ``###.###`` writes ``f.0000118.png``
and is unique per frame; measured, both forms.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by the absence of the import too
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
    USD_AVAILABLE = True
except ImportError:  # pragma: no cover
    USD_AVAILABLE = False


# ── Palette ──────────────────────────────────────────────────────────
#
# The base colours are the ones SceneExporter.cells_to_usd already uses,
# so a still and an animation of the same run agree.  The state variants
# are derived: hypoxic darkens and shifts violet, newborn brightens and
# emits for the one frame it lasts.

BASE_COLORS: Dict[str, Tuple[float, float, float]] = {
    "normal": (0.30, 0.60, 0.90),
    "cancer": (0.90, 0.20, 0.20),
    "stem": (0.40, 0.90, 0.50),
    "T_cell": (0.20, 0.90, 0.30),
    "NK_cell": (0.90, 0.90, 0.20),
    "macrophage": (0.90, 0.50, 0.20),
    "unknown": (0.50, 0.50, 0.50),
}

#: (diffuse, emissive) per state suffix, applied on top of a base colour.
STATE_STYLE: Dict[str, Dict[str, Any]] = {
    "": {"mix": None, "emissive": 0.0},
    "hypoxic": {"mix": (0.25, 0.10, 0.45), "weight": 0.65, "emissive": 0.0},
    "newborn": {"mix": (1.00, 1.00, 1.00), "weight": 0.55, "emissive": 0.35},
    "activated": {"mix": (1.00, 1.00, 0.55), "weight": 0.50, "emissive": 0.45},
}


#: The state suffixes a prototype key may carry, longest first.
STATE_SUFFIXES = ("hypoxic", "newborn", "activated")


def _split_key(key: str) -> Tuple[str, str]:
    """"normal_hypoxic" -> ("normal", "hypoxic"); "T_cell_activated" ->
    ("T_cell", "activated").  A plain type name splits to no state, which
    is why this cannot be a str.partition on the underscore."""
    for state in STATE_SUFFIXES:
        suffix = "_" + state
        if key.endswith(suffix):
            return key[: -len(suffix)], state
    return key, ""


def _styled_color(base: str, state: str) -> Tuple[Tuple[float, float, float], float]:
    """Return (diffuseColor, emissiveStrength) for a (type, state) pair."""
    rgb = BASE_COLORS.get(base, BASE_COLORS["unknown"])
    style = STATE_STYLE.get(state, STATE_STYLE[""])
    mix = style.get("mix")
    if mix:
        w = float(style.get("weight", 0.5))
        rgb = tuple((1.0 - w) * c + w * m for c, m in zip(rgb, mix))
    return rgb, float(style.get("emissive", 0.0))


# ── Results ──────────────────────────────────────────────────────────

@dataclass
class AnimationStats:
    """What was written, so a caller can assert on it instead of trusting it."""
    path: str = ""
    n_frames: int = 0
    fps: float = 24.0
    start_time_code: float = 0.0
    end_time_code: float = 0.0
    frame_times_hours: List[float] = field(default_factory=list)
    n_cell_slots: int = 0
    n_immune_slots: int = 0
    cell_prototypes: List[str] = field(default_factory=list)
    immune_prototypes: List[str] = field(default_factory=list)
    n_position_samples: int = 0
    n_hidden_total: int = 0
    max_cell_displacement: float = 0.0
    max_immune_displacement: float = 0.0
    bbox_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_path: str = ""

    def summary(self) -> str:
        return (
            f"{self.path}: {self.n_frames} frames "
            f"({self.frame_times_hours[0] if self.frame_times_hours else 0:.2f}h.."
            f"{self.frame_times_hours[-1] if self.frame_times_hours else 0:.2f}h) "
            f"@ {self.fps:g} fps, {self.n_cell_slots} cell slots, "
            f"{self.n_immune_slots} immune slots, "
            f"{self.n_position_samples} position samples, "
            f"{self.n_hidden_total} slot-frames hidden"
        )


# ── Snapshot access (CellSnapshot objects or plain dicts) ────────────

def _get(snap: Any, name: str, default: Any = None) -> Any:
    if isinstance(snap, dict):
        value = snap.get(name, default)
    else:
        value = getattr(snap, name, default)
    return default if value is None else value


def _positions(snap: Any, key: str) -> np.ndarray:
    raw = _get(snap, key, None)
    if raw is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(-1, 3))


def _cell_ids(snap: Any, n: int) -> List[Any]:
    ids = list(_get(snap, "cell_ids", []) or [])
    if len(ids) == n:
        return ids
    # No recorded ids (or a mismatched length): fall back to row order,
    # which is what a snapshot without ids can honestly support.
    return list(range(n))


def _immune_ids(snap: Any, n: int) -> List[Any]:
    ids = list(_get(snap, "immune_ids", []) or [])
    if len(ids) == n:
        return ids
    return list(range(n))


# ── Track assembly ───────────────────────────────────────────────────

class _Track:
    """One instancer's worth of slots: a stable id per row, for all frames."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.slot_of_id: Dict[Any, int] = {}
        self.ids: List[Any] = []
        self.first_frame: List[int] = []
        self.proto_keys: List[str] = []          # in authoring order
        self._proto_index: Dict[str, int] = {}

    def slot(self, cell_id: Any, frame: int) -> int:
        idx = self.slot_of_id.get(cell_id)
        if idx is None:
            idx = len(self.ids)
            self.slot_of_id[cell_id] = idx
            self.ids.append(cell_id)
            self.first_frame.append(frame)
        return idx

    def proto(self, key: str) -> int:
        idx = self._proto_index.get(key)
        if idx is None:
            idx = len(self.proto_keys)
            self._proto_index[key] = idx
            self.proto_keys.append(key)
        return idx

    @property
    def n_slots(self) -> int:
        return len(self.ids)


def _cell_state(oxygen: Optional[float], is_new: bool,
                hypoxia_threshold: float, mark_newborns: bool) -> str:
    """Hypoxia outranks newborn: it is a sustained state that explains the
    deaths, while newborn lasts a single frame."""
    if oxygen is not None and oxygen < hypoxia_threshold:
        return "hypoxic"
    if is_new and mark_newborns:
        return "newborn"
    return ""


def _proto_key(base: str, state: str) -> str:
    return f"{base}_{state}" if state else base


def _plan(
    snapshots: Sequence[Any],
    hypoxia_threshold: float,
    mark_newborns: bool,
    include_immune: bool,
) -> Tuple[_Track, _Track, List[List[Tuple[int, int, int]]], List[List[Tuple[int, int, int]]], List[np.ndarray], List[np.ndarray]]:
    """First pass: allocate slots and prototypes, frame by frame.

    Returns (cell_track, immune_track, cell_rows, immune_rows,
    cell_positions, immune_positions) where ``*_rows[f]`` is a list of
    ``(slot, row_in_frame, proto_index)``.
    """
    cells = _Track("Cells")
    immune = _Track("Immune")
    cell_rows: List[List[Tuple[int, int, int]]] = []
    immune_rows: List[List[Tuple[int, int, int]]] = []
    cell_pos: List[np.ndarray] = []
    immune_pos: List[np.ndarray] = []

    for f, snap in enumerate(snapshots):
        pos = _positions(snap, "cell_positions")
        n = len(pos)
        ids = _cell_ids(snap, n)
        types = list(_get(snap, "cell_types", []) or [])
        oxy = list(_get(snap, "cell_oxygen", []) or [])

        rows: List[Tuple[int, int, int]] = []
        for i in range(n):
            slot = cells.slot(ids[i], f)
            is_new = f > 0 and cells.first_frame[slot] == f
            base = types[i] if i < len(types) else "unknown"
            o2 = float(oxy[i]) if i < len(oxy) else None
            key = _proto_key(
                base, _cell_state(o2, is_new, hypoxia_threshold, mark_newborns)
            )
            rows.append((slot, i, cells.proto(key)))
        cell_rows.append(rows)
        cell_pos.append(pos)

        if include_immune:
            ipos = _positions(snap, "immune_positions")
            m = len(ipos)
            iids = _immune_ids(snap, m)
            itypes = list(_get(snap, "immune_types", []) or [])
            iact = list(_get(snap, "immune_activated", []) or [])
            irows: List[Tuple[int, int, int]] = []
            for i in range(m):
                slot = immune.slot(iids[i], f)
                base = itypes[i] if i < len(itypes) else "unknown"
                state = "activated" if (i < len(iact) and bool(iact[i])) else ""
                irows.append((slot, i, immune.proto(_proto_key(base, state))))
            immune_rows.append(irows)
            immune_pos.append(ipos)
        else:
            immune_rows.append([])
            immune_pos.append(np.zeros((0, 3), dtype=np.float32))

    return cells, immune, cell_rows, immune_rows, cell_pos, immune_pos


# ── USD authoring ────────────────────────────────────────────────────

def _define_material(stage: "Usd.Stage", looks_path: str, key: str,
                     rgb: Tuple[float, float, float],
                     emissive: float) -> "UsdShade.Material":
    safe = key.replace("-", "_")
    mat_path = f"{looks_path}/{safe}"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if emissive > 0.0:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*(c * emissive for c in rgb))
        )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _author_prototypes(stage: "Usd.Stage", instancer: "UsdGeom.PointInstancer",
                       instancer_path: str, looks_path: str,
                       keys: Sequence[str], radius: float) -> None:
    """One Sphere prototype per (type, state) pair, as children of the
    instancer -- Hydra draws the children of a PointInstancer only
    through the instancing, never directly."""
    targets = []
    for key in keys:
        base, state = _split_key(key)
        rgb, emissive = _styled_color(base, state)
        proto_path = f"{instancer_path}/Protos/{key}"
        sphere = UsdGeom.Sphere.Define(stage, proto_path)
        sphere.GetRadiusAttr().Set(float(radius))
        sphere.GetExtentAttr().Set(
            [Gf.Vec3f(-radius, -radius, -radius), Gf.Vec3f(radius, radius, radius)]
        )
        # displayColor so a viewer with no material support still
        # distinguishes the prototypes.
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        material = _define_material(stage, looks_path, key, rgb, emissive)
        UsdShade.MaterialBindingAPI(sphere).Bind(material)
        targets.append(Sdf.Path(proto_path))
    instancer.CreatePrototypesRel().SetTargets(targets)


def _author_track(
    stage: "Usd.Stage",
    path: str,
    looks_path: str,
    track: _Track,
    rows_per_frame: Sequence[Sequence[Tuple[int, int, int]]],
    positions_per_frame: Sequence[np.ndarray],
    radius: float,
) -> Tuple[int, int, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Author one PointInstancer with a time sample per frame.

    Returns (n_position_samples, n_hidden_slot_frames, max_displacement,
    bbox_min, bbox_max).
    """
    n_slots = track.n_slots
    if n_slots == 0:
        return 0, 0, 0.0, None, None

    instancer = UsdGeom.PointInstancer.Define(stage, path)
    _author_prototypes(stage, instancer, path, looks_path, track.proto_keys, radius)

    # The engine's own cell ids, so an id in the stage is the id in the
    # run.  invisibleIds is matched against this array, so the two must
    # agree; anything not uniquely int-able falls back to slot order.
    try:
        usd_ids = np.array([int(i) for i in track.ids], dtype=np.int64)
        if len(np.unique(usd_ids)) != n_slots:
            raise ValueError("duplicate ids")
    except (TypeError, ValueError):
        usd_ids = np.arange(n_slots, dtype=np.int64)
    instancer.CreateIdsAttr().Set(Vt.Int64Array.FromNumpy(usd_ids))

    pos_attr = instancer.CreatePositionsAttr()
    proto_attr = instancer.CreateProtoIndicesAttr()
    scale_attr = instancer.CreateScalesAttr()
    hide_attr = instancer.CreateInvisibleIdsAttr()

    # Slots that have not yet been born are parked at the position they
    # will be born at, and dead slots keep their last position.  Both are
    # hidden, two ways, so neither is ever drawn; parking them somewhere
    # finite just keeps the bounds honest.
    parked = np.zeros((n_slots, 3), dtype=np.float32)
    for f, rows in enumerate(rows_per_frame):
        pos = positions_per_frame[f]
        for slot, row, _proto in rows:
            if track.first_frame[slot] == f:
                parked[slot] = pos[row]

    proto_state = np.zeros(n_slots, dtype=np.int32)
    current = parked.copy()
    n_samples = 0
    n_hidden = 0
    max_disp = 0.0
    bmin = np.full(3, np.inf, dtype=np.float64)
    bmax = np.full(3, -np.inf, dtype=np.float64)
    previous: Optional[np.ndarray] = None
    previous_live: Optional[np.ndarray] = None

    for f, rows in enumerate(rows_per_frame):
        pos = positions_per_frame[f]
        live = np.zeros(n_slots, dtype=bool)
        for slot, row, proto in rows:
            current[slot] = pos[row]
            proto_state[slot] = proto
            live[slot] = True

        scales = np.zeros((n_slots, 3), dtype=np.float32)
        scales[live] = 1.0
        hidden = np.ascontiguousarray(usd_ids[~live])

        time = Usd.TimeCode(float(f))
        pos_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(current)), time)
        proto_attr.Set(Vt.IntArray.FromNumpy(proto_state.copy()), time)
        scale_attr.Set(Vt.Vec3fArray.FromNumpy(scales), time)
        hide_attr.Set(Vt.Int64Array.FromNumpy(hidden), time)

        n_samples += 1
        n_hidden += int(hidden.size)
        if pos.size:
            bmin = np.minimum(bmin, pos.min(axis=0))
            bmax = np.maximum(bmax, pos.max(axis=0))
        if previous is not None and previous_live is not None:
            both = live & previous_live
            if both.any():
                step = np.linalg.norm(current[both] - previous[both], axis=1)
                max_disp = max(max_disp, float(step.max()))
        previous = current.copy()
        previous_live = live

    # Bounds for the instancer itself, so a viewer can frame it without
    # computing the point extent.
    if np.isfinite(bmin).all():
        pad = float(radius) * 1.5
        instancer.CreateExtentAttr(
            [Gf.Vec3f(*(bmin - pad)), Gf.Vec3f(*(bmax + pad))]
        )
    return n_samples, n_hidden, max_disp, bmin, bmax


def _author_camera(stage: "Usd.Stage", path: str,
                   bmin: np.ndarray, bmax: np.ndarray,
                   radius_pad: float) -> str:
    """One camera framed on the union of every frame's bounds."""
    center = (bmin + bmax) / 2.0
    extent = np.maximum(bmax - bmin, 1.0)
    radius = float(np.linalg.norm(extent)) / 2.0 + radius_pad

    camera = UsdGeom.Camera.Define(stage, path)
    focal, aperture = 24.0, 20.955          # ~47 deg horizontal FOV
    half_fov = math.atan(aperture / (2.0 * focal))
    distance = radius / math.tan(half_fov) * 1.05

    direction = np.array([0.72, -0.60, 0.35], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    eye = center + direction * distance
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0.0, 0.0, 1.0)
    )
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())

    camera.GetFocalLengthAttr().Set(focal)
    camera.GetHorizontalApertureAttr().Set(aperture)
    camera.GetVerticalApertureAttr().Set(aperture * 9.0 / 16.0)
    camera.GetClippingRangeAttr().Set(
        Gf.Vec2f(max(0.1, distance - radius * 4.0), distance + radius * 4.0)
    )
    camera.GetFocusDistanceAttr().Set(float(distance))
    return path


def write_cell_animation(
    snapshots: Sequence[Any],
    output_path: str,
    fps: float = 24.0,
    cell_radius: float = 1.6,
    immune_radius: float = 1.1,
    include_immune: bool = True,
    hypoxia_threshold: float = 0.05,
    mark_newborns: bool = True,
    camera: bool = True,
    scene_name: str = "cognisom_run",
    metadata: Optional[Dict[str, Any]] = None,
) -> AnimationStats:
    """Write a recorded run to ONE USD stage as time samples.

    Args:
        snapshots: ``EngineRunner.cell_snapshots`` (or any sequence of
            objects/dicts carrying ``time``, ``cell_positions``,
            ``cell_types``, ``cell_oxygen``, ``cell_ids`` and optionally
            the ``immune_*`` fields).
        output_path: ``.usda`` for a readable stage, ``.usdc``/``.usd``
            for the binary crate format, which is what you want above a
            few hundred frames.
        fps: time codes per second, i.e. the playback rate.  One frame is
            one recorded snapshot, so 24 fps plays a 24 h run recorded
            every 0.1 h in ten seconds.
        hypoxia_threshold: cell O2 below this gets the hypoxic colour.
            0.05 is the threshold VascularModule's own hypoxia count uses.
        mark_newborns: colour a cell for the one frame in which its id
            first appears.  Derived from the id sequence, because
            ``cell.phase`` never leaves 'G1'.

    Returns:
        :class:`AnimationStats` describing what was authored.
    """
    if not USD_AVAILABLE:
        raise RuntimeError(
            "OpenUSD (pxr) is required for write_cell_animation; "
            "install with: pip install usd-core"
        )
    snapshots = list(snapshots)
    if not snapshots:
        raise ValueError(
            "write_cell_animation needs at least one recorded snapshot; "
            "EngineRunner.run() records them, but a caller that drives "
            "SimulationEngine.run() or engine.step() itself must call "
            "EngineRunner.record_cell_snapshot()."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    cells, immune, cell_rows, immune_rows, cell_pos, immune_pos = _plan(
        snapshots, hypoxia_threshold, mark_newborns, include_immune
    )

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1e-06)      # positions are micrometres

    times = [float(_get(s, "time", float(i))) for i, s in enumerate(snapshots)]
    n_frames = len(snapshots)
    stage.SetTimeCodesPerSecond(float(fps))
    stage.SetFramesPerSecond(float(fps))
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(n_frames - 1))
    stage.SetMetadata(
        "documentation",
        f"Cognisom {scene_name}: {n_frames} recorded frames, "
        f"t={times[0]:.3f}h..{times[-1]:.3f}h, played at {fps:g} fps. "
        "One time code per recorded snapshot.",
    )

    world = UsdGeom.Xform.Define(stage, "/World")
    # The frame -> simulated-hours map, on the stage, so a scrubbing
    # viewer can say what hour it is looking at.
    sim_time = world.GetPrim().CreateAttribute(
        "cognisom:simTimeHours", Sdf.ValueTypeNames.Float, custom=True
    )
    for f, t in enumerate(times):
        sim_time.Set(float(t), Usd.TimeCode(float(f)))

    UsdGeom.Xform.Define(stage, "/World/Lights")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.GetIntensityAttr().Set(900.0)
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    key.GetIntensityAttr().Set(2500.0)
    key.GetAngleAttr().Set(2.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 0.0))

    looks = "/World/Looks"
    UsdGeom.Scope.Define(stage, looks)

    stats = AnimationStats(
        path=str(out), n_frames=n_frames, fps=float(fps),
        start_time_code=0.0, end_time_code=float(n_frames - 1),
        frame_times_hours=times,
        n_cell_slots=cells.n_slots, n_immune_slots=immune.n_slots,
        cell_prototypes=list(cells.proto_keys),
        immune_prototypes=list(immune.proto_keys),
    )

    samples, hidden, disp, bmin, bmax = _author_track(
        stage, "/World/Cells", looks, cells, cell_rows, cell_pos, cell_radius
    )
    stats.n_position_samples += samples
    stats.n_hidden_total += hidden
    stats.max_cell_displacement = disp

    i_samples, i_hidden, i_disp, ibmin, ibmax = _author_track(
        stage, "/World/Immune", looks, immune, immune_rows, immune_pos,
        immune_radius,
    )
    stats.n_position_samples += i_samples
    stats.n_hidden_total += i_hidden
    stats.max_immune_displacement = i_disp

    mins = [b for b in (bmin, ibmin) if b is not None and np.isfinite(b).all()]
    maxs = [b for b in (bmax, ibmax) if b is not None and np.isfinite(b).all()]
    if mins:
        world_min = np.min(np.vstack(mins), axis=0)
        world_max = np.max(np.vstack(maxs), axis=0)
        stats.bbox_min = tuple(float(v) for v in world_min)
        stats.bbox_max = tuple(float(v) for v in world_max)
        if camera:
            stats.camera_path = _author_camera(
                stage, "/World/Camera", world_min, world_max,
                max(cell_radius, immune_radius) * 2.0,
            )

    layer_data = {
        "cognisom:scene": scene_name,
        "cognisom:n_frames": n_frames,
        "cognisom:fps": float(fps),
        "cognisom:frame_times_hours": Vt.FloatArray([float(t) for t in times]),
        "cognisom:hidden_slot_frames": stats.n_hidden_total,
    }
    for k, v in (metadata or {}).items():
        layer_data[f"cognisom:{k}"] = v
    stage.GetRootLayer().customLayerData = layer_data

    stage.Save()
    log.info("%s", stats.summary())
    return stats


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run a scenario and write its animation. ``-m`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a Cognisom scenario and write it to one "
                    "time-sampled USD stage."
    )
    parser.add_argument("--scenario", default="Baseline")
    parser.add_argument("--duration", type=float, default=24.0,
                        help="simulated hours")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--interval", type=float, default=None,
                        help="simulated hours between recorded frames "
                             "(default: EngineRunner's own default)")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default="run.usda")
    parser.add_argument("--no-immune", action="store_true")
    args = parser.parse_args(argv)

    import numpy as _np
    from cognisom.dashboard.engine_runner import EngineRunner

    _np.random.seed(args.seed)
    kwargs: Dict[str, Any] = {}
    if args.interval is not None:
        kwargs["snapshot_interval_hours"] = args.interval
    runner = EngineRunner(dt=args.dt, duration=args.duration,
                          scenario=args.scenario, **kwargs)
    runner.build()
    runner.run()

    stats = write_cell_animation(
        runner.cell_snapshots, args.out, fps=args.fps,
        include_immune=not args.no_immune,
        scene_name=f"{args.scenario} {args.duration:g}h",
        metadata={
            "scenario": args.scenario,
            "dt_hours": float(args.dt),
            "duration_hours": float(args.duration),
            "seed": int(args.seed),
        },
    )
    print(stats.summary())
    print(f"  cell prototypes:   {', '.join(stats.cell_prototypes)}")
    print(f"  immune prototypes: {', '.join(stats.immune_prototypes)}")
    print(f"  bbox: {stats.bbox_min} .. {stats.bbox_max}")
    print(f"  max per-frame displacement: cells "
          f"{stats.max_cell_displacement:.4f} um, immune "
          f"{stats.max_immune_displacement:.4f} um")
    # ###.### and not ###: see the module docstring. A bare ### makes
    # Apple's usdrecord collide every frame onto a couple of filenames.
    print(f"  render: usdrecord --imageWidth 960 --frames "
          f"0:{stats.end_time_code:g} --camera /World/Camera "
          f"{stats.path} frame.###.###.png")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
