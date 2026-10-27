"""Draw simulation events into a USD stage, on the frame they happened.

What this authors
-----------------
Everything here is additive and lives under its own prim paths, so it
composes with whatever else writes the stage (a cell point instancer, a
vessel network, a camera of its own).

``/World/Interactions/<Kind>``
    One :class:`UsdGeom.PointInstancer` per event kind. Every event is
    one instance, pinned to the coordinate the event carried. Markers
    appear and disappear by time-sampling ``scales`` and
    ``invisibleIds``, which is what makes a kill a *moment* rather than
    a permanent bead. Verified against Apple's ``usdrecord`` (Storm /
    Metal): an instance whose id is in ``invisibleIds`` is not drawn,
    and prototypes nested under the instancer are not drawn on their own.

``/World/Interactions/Engagements``
    A :class:`UsdGeom.BasisCurves` segment per kill, from the attacker
    to the target. Authored only for kills whose
    ``partner_provenance == 'live_at_emission'`` -- i.e. where both ends
    were read in the same step the kill fired. A kill with no attacker
    coordinate gets its burst marker and no line.

``/World/Fields/OxygenShadows`` and ``/World/Fields/HypoxicCells``
    Oxygen, made visible. A flat disc under each living cell coloured by
    *that cell's* recorded oxygen -- the coloured ground plane, with
    every coloured pixel traceable to one measurement -- plus a
    translucent dark halo on each cell whose recorded oxygen is below
    the hypoxia threshold (0.05, the number
    ``VascularModule._check_hypoxia`` uses).

``/World/Fields/OxygenSlice``
    The *solved spatial field* as a coloured plane, for when it has
    structure worth drawing. It usually does not: measured here, the
    field is flat at the arterial level to within 2.3e-06 across a run
    in which four cells were hypoxic, because SpatialModule's own
    coupling note says this geometry is not diffusion-limited and
    VascularModule decides hypoxia from Krogh distance instead. So
    :func:`author_oxygen_slice` refuses a uniform field and returns the
    measured range rather than drawing a smooth cyan sheet that would
    say "oxygen is uniform" over a run that killed cells by hypoxia.
    It is also not a ``UsdVolVolume``: that needs voxel data in an
    OpenVDB or Field3D asset and there is no ``pyopenvdb`` here to write
    one, so the volume prim would point at a file that does not exist.

``/World/Cameras/EventCam`` and its children
    A camera framing the action, a backdrop so the render is not
    transparent, and a legend quad -- a texture authored from
    :mod:`cognisom.visualization.legend_texture` and parented to the
    camera so it holds a screen corner. USD has no text primitive;
    baking the legend into a texture is the only way to make the *stage*
    self-explanatory rather than just the PNG.

Timing
------
Event times are in simulated hours; USD wants integer time codes. The
bridge is ``frame_times``: the simulated time of each recorded frame,
which is exactly what a snapshot series already knows. An event at time
``t`` is placed on the first frame whose time is ``>= t`` -- the first
frame on which a viewer could see it -- and is held for
``lifetime_frames``. Nothing is interpolated backwards onto a frame that
preceded the event.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
    USD_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on machines without usd-core
    USD_AVAILABLE = False

from .interaction_recorder import (
    IMMUNE_KILL_DEATH_CAUSES,
    KIND_ACTIVATION,
    KIND_DEATH,
    KIND_DIVISION,
    KIND_KILL,
    KIND_TRANSFORMATION,
    VisualEvent,
)

#: Oxygen level below which VascularModule calls a cell hypoxic
#: (vascular_module._check_hypoxia: ``cell.oxygen < 0.05``). Used as the
#: break point of the field colour ramp so the picture and the model
#: agree on what "hypoxic" means.
HYPOXIA_THRESHOLD = 0.05

#: Arterial oxygen, the top of the ramp (VascularModule.arterial_O2).
ARTERIAL_O2 = 0.21


@dataclass(frozen=True)
class MarkerStyle:
    """How one kind of event is drawn.

    Shape as well as colour, deliberately: a still frame has to be
    readable by someone who cannot reliably separate red from green, and
    a sphere, a cube, a cone and a disc are separable for everyone.
    """

    shape: str          # 'sphere' | 'cube' | 'cone' | 'disc'
    color: Tuple[float, float, float]
    label: str
    #: Frames the marker stays on screen, counting the frame it appears.
    lifetime_frames: int = 6
    #: Radius in world units (micrometres) at the marker's peak.
    peak_radius: float = 9.0
    #: 'burst' grows then fades; 'flash' is on hard then gone;
    #: 'collapse' starts large and shrinks to nothing.
    envelope: str = "burst"
    #: Below 1.0 the marker is translucent. The kill burst has to be
    #: bigger than a cell to read at tissue scale, and at that size an
    #: opaque sphere hides the attacker and the bond -- the two things
    #: that say *who* did it.
    opacity: float = 1.0


#: The default vocabulary. A viewer learns four glyphs and can then read
#: any frame.
DEFAULT_STYLES: Dict[str, MarkerStyle] = {
    KIND_KILL: MarkerStyle(
        # Orange rather than pure red, so it does not read as the same
        # thing as the deep red of a hypoxic cell and the hypoxic end of
        # the oxygen ramp. Two different reds in one frame is one too
        # many when the picture has to be read by someone who was not
        # told what to look for.
        shape="sphere", color=(1.00, 0.36, 0.10),
        label="Cancer cell killed", lifetime_frames=8,
        peak_radius=8.0, envelope="burst", opacity=0.55,
    ),
    KIND_ACTIVATION: MarkerStyle(
        shape="cone", color=(1.00, 0.72, 0.09),
        label="Immune cell engaged a target", lifetime_frames=5,
        peak_radius=7.0, envelope="flash",
    ),
    KIND_DIVISION: MarkerStyle(
        # Small and brief on purpose. Divisions are not rare and they
        # are not spread out: this engine seeds every cell at age 0, so
        # the population divides in synchronised waves -- measured, 59
        # divisions on one frame and 272 on another in a 48 h run. A
        # marker sized like a kill would bury every other event under a
        # white sheet four times per run.
        shape="cube", color=(0.86, 0.82, 1.00),
        label="Cell divided", lifetime_frames=2,
        peak_radius=3.2, envelope="flash",
    ),
    KIND_DEATH: MarkerStyle(
        shape="disc", color=(0.62, 0.62, 0.66),
        label="Cell died (not an immune kill)", lifetime_frames=6,
        peak_radius=6.0, envelope="collapse",
    ),
    KIND_TRANSFORMATION: MarkerStyle(
        shape="cube", color=(0.85, 0.30, 0.95),
        label="Normal cell transformed", lifetime_frames=6,
        peak_radius=7.0, envelope="burst",
    ),
}

#: Colour of the engagement bond. Deliberately the activation amber: the
#: line and the cone mean the same thing, an immune cell on a target.
BOND_COLOR = (1.00, 0.72, 0.09)


def _require_usd() -> None:
    if not USD_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "OpenUSD (usd-core) is required to author interaction geometry. "
            "pip install usd-core"
        )


# ── Frame mapping ────────────────────────────────────────────────────

class FrameMap:
    """Simulated hours -> USD time codes.

    ``frame_times[i]`` is the simulated time of frame ``i``. An event at
    time ``t`` lands on the first frame at or after ``t``: the earliest
    frame on which its consequences are on screen. An event after the
    last recorded frame is dropped rather than clamped onto the final
    frame, because clamping would pile a run's whole tail onto one image.
    """

    def __init__(self, frame_times: Sequence[float]):
        times = np.asarray(list(frame_times), dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("frame_times must be a non-empty 1-D sequence")
        if np.any(np.diff(times) < 0):
            raise ValueError("frame_times must be non-decreasing")
        self.times = times

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def first(self) -> int:
        return 0

    @property
    def last(self) -> int:
        return int(self.times.size - 1)

    def frame_of(self, t: float) -> Optional[int]:
        """First frame index at or after ``t``; ``None`` if past the end."""
        idx = int(np.searchsorted(self.times, float(t), side="left"))
        if idx >= self.times.size:
            return None
        return idx


# ── Shared helpers ───────────────────────────────────────────────────

def _emissive_material(stage, path: str, color, opacity: float = 1.0):
    """A UsdPreviewSurface that reads the same under any lighting.

    Event markers are notation, not tissue. Driving ``emissiveColor`` as
    well as ``diffuseColor`` means a kill marker is the same red whether
    the stage has one dome light or five, which matters because this
    layer composes with stages it does not own.
    """
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    col = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(col)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(col)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if opacity < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _define_prototype(stage, path: str, shape: str, radius: float):
    """One unit-ish prototype per marker kind, nested under its instancer.

    Nesting matters: a prototype authored outside the instancer renders
    on its own as well, which puts a permanent stray marker at the
    origin. Verified by render.
    """
    if shape == "sphere":
        prim = UsdGeom.Sphere.Define(stage, path)
        prim.GetRadiusAttr().Set(float(radius))
        prim.GetExtentAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(-radius, -radius, -radius), Gf.Vec3f(radius, radius, radius)]))
    elif shape == "cube":
        prim = UsdGeom.Cube.Define(stage, path)
        prim.GetSizeAttr().Set(float(radius) * 1.6)
        h = radius * 0.8
        prim.GetExtentAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(-h, -h, -h), Gf.Vec3f(h, h, h)]))
    elif shape == "cone":
        prim = UsdGeom.Cone.Define(stage, path)
        prim.GetRadiusAttr().Set(float(radius))
        prim.GetHeightAttr().Set(float(radius) * 2.2)
        prim.GetAxisAttr().Set(UsdGeom.Tokens.z)
        h = radius * 1.2
        prim.GetExtentAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(-radius, -radius, -h), Gf.Vec3f(radius, radius, h)]))
    elif shape == "disc":
        prim = UsdGeom.Cylinder.Define(stage, path)
        prim.GetRadiusAttr().Set(float(radius))
        prim.GetHeightAttr().Set(float(radius) * 0.25)
        prim.GetAxisAttr().Set(UsdGeom.Tokens.z)
        h = radius * 0.2
        prim.GetExtentAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(-radius, -radius, -h), Gf.Vec3f(radius, radius, h)]))
    else:
        raise ValueError(f"unknown marker shape {shape!r}")
    return prim


def _sample_frames(start_frames: Sequence[int], lifetime: int,
                   frame_map: "FrameMap") -> List[int]:
    """Which time codes to author samples on, for a set of timed markers.

    **USD interpolates numeric time samples linearly.** That is the
    whole reason this function exists. Authoring a sample only on the
    frames where a marker changes -- the obvious space saving -- makes
    every renderer ramp smoothly between those sparse samples, and the
    result is not subtle: measured, with samples only on frames 0 and
    181, a division burst that belongs on frame 181 was already at 45%
    of full size on frame 149, thirty-two frames and six simulated hours
    early. Fifty-nine division markers were on screen during a kill that
    had nothing to do with them.

    So each marker's window is authored *contiguously*, from the frame
    before it appears to the frame after it dies. Between two distant
    windows both bracketing samples are all-zero, so the interpolation
    USD performs there is between zero and zero, which is zero. The
    saving survives -- a run with 456 divisions in four synchronised
    bursts authors a few dozen frames, not 242 -- and nothing bleeds.
    """
    frames = {frame_map.first, frame_map.last}
    for f in start_frames:
        for k in range(-1, lifetime + 2):
            g = f + k
            if frame_map.first <= g <= frame_map.last:
                frames.add(g)
    return sorted(frames)


def envelope_scale(envelope: str, age: int, lifetime: int) -> float:
    """Scale multiplier for a marker ``age`` frames after it appeared.

    Returns 0.0 outside ``[0, lifetime)``, so a marker is visible for
    exactly the frames it is meant to be, and **strictly positive at
    age 0**, so it is visible on the frame it appears. The first version
    of this ramped from zero over the first quarter of the life, which
    made every burst marker invisible on exactly the frame the event
    happened -- the one frame the whole layer exists to show. Caught by
    rendering frame 151 of a run with a kill at frame 151 and finding no
    kill in it.
    """
    if lifetime <= 0 or age < 0 or age >= lifetime:
        return 0.0
    if envelope == "flash":
        # Full size immediately, then a quick fade. A division is an
        # instant; it should not look like a slow bloom.
        return max(0.0, 1.0 - 0.7 * (age / float(lifetime)))
    if envelope == "collapse":
        # Starts at full size and shrinks to a point: a cell going away.
        return max(0.0, 1.0 - age / float(lifetime)) ** 0.7
    # 'burst': half size on the frame of the event, full on the next,
    # then a long decay to nothing. Reads as an impact.
    peak_age = 1 if lifetime >= 3 else 0
    if age <= peak_age:
        return (age + 1) / float(peak_age + 1)
    tail = lifetime - 1 - peak_age
    if tail <= 0:
        return 1.0
    return max(0.0, (1.0 - (age - peak_age) / float(tail)) ** 0.6)


# ── Markers ──────────────────────────────────────────────────────────

def author_interaction_markers(
    stage,
    events: Sequence[VisualEvent],
    frame_map: FrameMap,
    root: str = "/World/Interactions",
    styles: Optional[Dict[str, MarkerStyle]] = None,
    scale: float = 1.0,
    drop_immune_kill_deaths: bool = True,
) -> Dict[str, Any]:
    """Author one animated PointInstancer per event kind.

    Returns a small report: how many events of each kind were drawn, how
    many were skipped and why. The caller is expected to print it --
    silence about a dropped event is how "nothing is visible" becomes
    indistinguishable from "nothing happened".
    """
    _require_usd()
    styles = dict(DEFAULT_STYLES if styles is None else styles)

    report: Dict[str, Any] = {
        "drawn": {}, "no_position": 0, "after_last_frame": 0,
        "unstyled": {}, "suppressed_duplicate_death": 0,
    }

    by_kind: Dict[str, List[Tuple[int, VisualEvent]]] = {}
    for ev in events:
        if ev.position is None:
            # hypoxia_detected and anything else that carried no
            # coordinates. Counted, never placed -- checked BEFORE the
            # style lookup so the report names the real reason.
            report["no_position"] += 1
            continue
        if ev.kind not in styles:
            report["unstyled"][ev.kind] = report["unstyled"].get(ev.kind, 0) + 1
            continue
        if (drop_immune_kill_deaths and ev.kind == KIND_DEATH
                and ev.detail.get("cause") in IMMUNE_KILL_DEATH_CAUSES):
            # The kill marker already occupies this exact coordinate at
            # this exact instant; two glyphs on one pixel is noise.
            report["suppressed_duplicate_death"] += 1
            continue
        frame = frame_map.frame_of(ev.time)
        if frame is None:
            report["after_last_frame"] += 1
            continue
        by_kind.setdefault(ev.kind, []).append((frame, ev))

    UsdGeom.Scope.Define(stage, root)

    for kind, items in sorted(by_kind.items()):
        style = styles[kind]
        name = "".join(p.capitalize() for p in kind.split("_"))
        pi_path = f"{root}/{name}"
        pi = UsdGeom.PointInstancer.Define(stage, pi_path)

        proto = _define_prototype(
            stage, f"{pi_path}/Prototypes/Marker", style.shape,
            style.peak_radius * scale,
        )
        UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(
            _emissive_material(stage, f"{pi_path}/Prototypes/Marker/Mat",
                               style.color, opacity=style.opacity)
        )
        pi.CreatePrototypesRel().SetTargets([proto.GetPath()])

        n = len(items)
        pi.CreatePositionsAttr().Set(Vt.Vec3fArray(
            [Gf.Vec3f(*(float(c) for c in ev.position)) for _f, ev in items]))
        pi.CreateProtoIndicesAttr().Set(Vt.IntArray([0] * n))
        pi.CreateIdsAttr().Set(Vt.Int64Array(list(range(n))))

        scales_attr = pi.CreateScalesAttr()
        invis_attr = pi.CreateInvisibleIdsAttr()

        for frame in sorted(_sample_frames(
                [f for f, _ev in items], style.lifetime_frames, frame_map)):
            raw = [envelope_scale(style.envelope, frame - f0,
                                  style.lifetime_frames)
                   for f0, _ev in items]
            scales_attr.Set(
                Vt.Vec3fArray([Gf.Vec3f(v, v, v) for v in raw]), frame)
            invis_attr.Set(
                Vt.Int64Array([i for i, v in enumerate(raw) if v <= 0.0]),
                frame)

        UsdGeom.PrimvarsAPI(pi.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.constant,
        ).Set(Vt.Vec3fArray([Gf.Vec3f(*style.color)]))

        pi.GetPrim().SetMetadata(
            "comment",
            f"{style.label}: {n} event(s), each pinned to the coordinate "
            f"carried in its own event payload.",
        )
        report["drawn"][kind] = n

    return report


# ── Engagement bonds ─────────────────────────────────────────────────

def author_engagement_bonds(
    stage,
    events: Sequence[VisualEvent],
    frame_map: FrameMap,
    root: str = "/World/Interactions",
    hold_frames: int = 4,
    width: float = 1.6,
) -> Dict[str, Any]:
    """A line from attacker to target, for kills where both ends are real.

    A kill event carries the *target's* coordinates and the attacker's
    *id*. The attacker's coordinates exist in a record only if the
    recorder was given a ``position_lookup`` and resolved them inside the
    event callback -- same step, nothing moved. Those kills get a line.
    Kills recorded without a lookup get no line at all, because the only
    other source for the attacker's position is wherever it has drifted
    to by now, and drawing that would claim an engagement that is not
    where the picture says it is.
    """
    _require_usd()
    report = {"drawn": 0, "no_partner_position": 0, "after_last_frame": 0}

    segments: List[Tuple[int, VisualEvent]] = []
    for ev in events:
        if ev.kind != KIND_KILL:
            continue
        if ev.position is None or ev.partner_position is None:
            report["no_partner_position"] += 1
            continue
        if ev.partner_provenance != "live_at_emission":
            report["no_partner_position"] += 1
            continue
        frame = frame_map.frame_of(ev.time)
        if frame is None:
            report["after_last_frame"] += 1
            continue
        segments.append((frame, ev))

    if not segments:
        return report

    curves = UsdGeom.BasisCurves.Define(stage, f"{root}/Engagements")
    curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([2] * len(segments)))
    pts: List[Gf.Vec3f] = []
    for _f, ev in segments:
        pts.append(Gf.Vec3f(*(float(c) for c in ev.partner_position)))
        pts.append(Gf.Vec3f(*(float(c) for c in ev.position)))
    curves.CreatePointsAttr().Set(Vt.Vec3fArray(pts))
    # `uniform` is one width per curve. `constant` is one width for the WHOLE
    # primitive, so the per-curve width array the fade writes below is ignored
    # and every bond renders at a single width for the whole run -- the fade
    # is authored correctly and simply never shown. Only rendering catches it.
    curves.SetWidthsInterpolation(UsdGeom.Tokens.uniform)

    widths_attr = curves.GetWidthsAttr() or curves.CreateWidthsAttr()
    for frame in sorted(_sample_frames(
            [f for f, _ev in segments], hold_frames, frame_map)):
        # Uniform interpolation: one width per curve.
        widths = []
        for f0, _ev in segments:
            age = frame - f0
            on = 0 <= age < hold_frames
            widths.append(float(width) * (1.0 - age / hold_frames) if on else 0.0)
        widths_attr.Set(Vt.FloatArray(widths), frame)

    UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(
        _emissive_material(stage, f"{root}/Engagements/Mat", BOND_COLOR)
    )
    curves.GetPrim().SetMetadata(
        "comment",
        "Attacker -> target at the instant of the kill. Both endpoints "
        "were read in the step the kill was emitted.",
    )
    report["drawn"] = len(segments)
    return report


# ── Oxygen field ─────────────────────────────────────────────────────

def oxygen_color(value: float,
                 hypoxia_threshold: float = HYPOXIA_THRESHOLD,
                 arterial: float = ARTERIAL_O2) -> Tuple[float, float, float]:
    """Map an oxygen concentration to a colour.

    Two bands with a hard break at the hypoxia threshold, so the eye
    reads the boundary the model actually uses rather than a smooth ramp
    that hides it. Below threshold: black -> deep red (anoxic to
    hypoxic). At or above: dark teal -> bright cyan (perfused).
    """
    v = float(value)
    if v < hypoxia_threshold:
        u = 0.0 if hypoxia_threshold <= 0 else max(0.0, v / hypoxia_threshold)
        return (0.18 + 0.62 * u, 0.02 + 0.06 * u, 0.04 + 0.06 * u)
    span = max(1e-9, arterial - hypoxia_threshold)
    u = min(1.0, (v - hypoxia_threshold) / span)
    return (0.02 + 0.02 * u, 0.14 + 0.32 * u, 0.20 + 0.40 * u)


def _slice_values(field: np.ndarray, k: int) -> np.ndarray:
    """The xy plane at voxel layer ``k``, clamped into range."""
    k = int(np.clip(k, 0, field.shape[2] - 1))
    return np.asarray(field[:, :, k], dtype=float)


def author_oxygen_slice(
    stage,
    field_frames: Sequence[Tuple[float, np.ndarray]],
    frame_map: FrameMap,
    resolution: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    z_plane: Optional[float] = None,
    slice_k: Optional[int] = None,
    path: str = "/World/Fields/OxygenSlice",
    hypoxia_threshold: float = HYPOXIA_THRESHOLD,
    arterial: float = ARTERIAL_O2,
    min_range: float = 1e-3,
) -> Dict[str, Any]:
    """The SOLVED spatial oxygen field as a coloured, animated plane.

    Not a ``UsdVolVolume``: writing one needs voxel data in an OpenVDB
    or Field3D asset and there is no ``pyopenvdb`` in this environment to
    produce one, so the volume prim would point at a file that does not
    exist and render as nothing. A plane carrying the solved values is a
    fallback that is at least true.

    **It refuses to draw a uniform field.** Measured on this engine: the
    spatial field is flat at the arterial level -- range 2.3e-06 over a
    12 h Hypoxia run in which four cells were below the hypoxia
    threshold. ``SpatialModule``'s own coupling note explains why (at
    real O2 diffusivity this tissue extent is not diffusion-limited), and
    ``VascularModule`` is what actually decides whether a cell is
    hypoxic, from its Krogh distance to the nearest vessel. Rendering
    the flat field as a smooth cyan sheet would therefore be a picture
    that says "oxygen is uniform" over a run that killed cells by
    hypoxia. Use :func:`author_cell_oxygen` for the quantity that varies.
    """
    _require_usd()
    if not field_frames:
        return {"authored": False, "reason": "no field frames captured"}

    first = np.asarray(field_frames[0][1])
    if first.ndim != 3:
        raise ValueError(f"oxygen field must be 3-D, got shape {first.shape}")
    nx, ny, nz = first.shape

    observed = 0.0
    for _t, arr in field_frames:
        a = np.asarray(arr)
        observed = max(observed, float(a.max() - a.min()))
    if observed < min_range:
        return {
            "authored": False,
            "measured_range": observed,
            "reason": (
                f"the solved spatial oxygen field is uniform (max-min = "
                f"{observed:.3g} over the run, below min_range="
                f"{min_range:g}); drawing it would show a flat sheet and "
                f"imply oxygen is uniform. Hypoxia in this model comes "
                f"from VascularModule's per-cell Krogh distance, which "
                f"author_cell_oxygen draws."
            ),
        }

    if slice_k is None:
        slice_k = nz // 2
    ox, oy, oz = (float(v) for v in origin)
    if z_plane is None:
        z_plane = oz - 0.5 * resolution

    # Vertex grid: one vertex per voxel centre, so every vertex colour
    # is a measured value and not an interpolation of one.
    points: List[Gf.Vec3f] = []
    for j in range(ny):
        for i in range(nx):
            points.append(Gf.Vec3f(
                float(ox + (i + 0.5) * resolution),
                float(oy + (j + 0.5) * resolution),
                float(z_plane),
            ))
    counts: List[int] = []
    indices: List[int] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            counts.append(4)
            indices.extend([a, a + 1, a + nx + 1, a + nx])

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)

    color_pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)

    # One colour sample per field snapshot, held until the next.
    # Nearest-previous, never interpolated: a field snapshot is a
    # measurement and the average of two measurements is not one.
    f_times = np.array([t for t, _a in field_frames], dtype=float)
    authored = 0
    for frame in range(frame_map.first, frame_map.last + 1):
        t = float(frame_map.times[frame])
        idx = int(np.searchsorted(f_times, t, side="right")) - 1
        idx = max(0, min(idx, len(field_frames) - 1))
        # A sample on every frame, even when the chosen snapshot has not
        # changed: USD interpolates colour samples linearly, so skipping
        # a frame would silently cross-fade between two measurements.
        plane = _slice_values(np.asarray(field_frames[idx][1]), slice_k)
        color_pv.Set(Vt.Vec3fArray([
            Gf.Vec3f(*oxygen_color(float(plane[i, j]), hypoxia_threshold,
                                   arterial))
            for j in range(ny) for i in range(nx)]), frame)
        authored += 1

    mesh.GetPrim().SetMetadata(
        "comment",
        f"Solved oxygen field, voxel layer k={slice_k} of {nz}, drawn as a "
        f"plane at z={z_plane:.1f}. Colour break at the hypoxia threshold "
        f"{hypoxia_threshold}. Not a UsdVolVolume: no OpenVDB writer is "
        f"available in this environment.",
    )
    return {
        "authored": True, "path": path, "slice_k": int(slice_k),
        "z_plane": float(z_plane), "samples": authored,
        "measured_range": observed, "grid": (int(nx), int(ny), int(nz)),
    }


def oxygen_frames_from_snapshots(snapshots) -> List[Tuple[Any, Any]]:
    """``[(positions (N,3), oxygen (N,)), ...]`` from CellSnapshot objects.

    Duck-typed on ``cell_positions`` and ``cell_oxygen`` so this module
    never imports the dashboard. Frames with no cells become empty
    arrays rather than being dropped, because dropping them would shift
    every later frame's index and put markers on the wrong images.
    """
    out: List[Tuple[Any, Any]] = []
    for snap in snapshots:
        pos = getattr(snap, "cell_positions", None)
        oxy = getattr(snap, "cell_oxygen", None)
        if pos is None or oxy is None or len(oxy) == 0:
            out.append((np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)))
            continue
        p = np.asarray(pos, dtype=float).reshape(-1, 3)
        o = np.asarray(oxy, dtype=float).reshape(-1)
        n = min(len(p), len(o))
        out.append((p[:n], o[:n]))
    return out


def author_cell_oxygen(
    stage,
    oxygen_frames: Sequence[Tuple[Any, Any]],
    frame_map: FrameMap,
    root: str = "/World/Fields",
    z_plane: Optional[float] = None,
    disc_radius: float = 7.0,
    halo_radius: float = 9.0,
    hypoxia_threshold: float = HYPOXIA_THRESHOLD,
    arterial: float = ARTERIAL_O2,
    draw_shadows: bool = True,
    draw_halos: bool = True,
) -> Dict[str, Any]:
    """Make oxygen -- and so hypoxia -- visible, from recorded cell state.

    Two layers, both built from ``CellSnapshot.cell_oxygen``, which is
    the per-cell level ``VascularModule`` delivered and the quantity its
    own hypoxia test reads:

    ``OxygenShadows``
        One flat disc per living cell, dropped onto a floor plane under
        the tissue and coloured by *that cell's* recorded oxygen. This is
        the coloured ground plane -- and every coloured pixel of it
        traces to one recorded measurement at one recorded position.
        Nothing is interpolated between cells, because the simulation
        has no value between cells to interpolate.

    ``HypoxicCells``
        A translucent dark halo around every cell whose recorded oxygen
        is below ``hypoxia_threshold``, at the cell's own 3-D position on
        that frame. This is the hypoxic region, where it actually is.

    Both animate: a cell that becomes hypoxic gains its halo on the
    frame its recorded oxygen crossed the threshold, not before.
    """
    _require_usd()
    if not oxygen_frames:
        return {"authored": False, "reason": "no cell oxygen frames"}

    all_pts = [p for p, _o in oxygen_frames if len(p)]
    if not all_pts:
        return {"authored": False, "reason": "no cells in any frame"}
    stacked = np.concatenate(all_pts, axis=0)
    if z_plane is None:
        z_plane = float(stacked[:, 2].min()) - 2.5 * disc_radius

    n_max = max(len(p) for p, _o in oxygen_frames)
    report: Dict[str, Any] = {"authored": True, "max_cells": int(n_max),
                              "z_plane": float(z_plane)}

    UsdGeom.Scope.Define(stage, root)

    if draw_shadows:
        path = f"{root}/OxygenShadows"
        pi = UsdGeom.PointInstancer.Define(stage, path)
        proto = UsdGeom.Cylinder.Define(stage, f"{path}/Prototypes/Disc")
        proto.GetRadiusAttr().Set(float(disc_radius))
        proto.GetHeightAttr().Set(float(disc_radius) * 0.12)
        proto.GetAxisAttr().Set(UsdGeom.Tokens.z)
        h = disc_radius * 0.1
        proto.GetExtentAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(-disc_radius, -disc_radius, -h),
            Gf.Vec3f(disc_radius, disc_radius, h)]))
        # NO material binding: per-instance colour comes from a varying
        # displayColor primvar, and a bound material would override it.
        pi.CreatePrototypesRel().SetTargets([proto.GetPath()])
        pi.CreateProtoIndicesAttr().Set(Vt.IntArray([0] * n_max))
        pi.CreateIdsAttr().Set(Vt.Int64Array(list(range(n_max))))
        pos_attr = pi.CreatePositionsAttr()
        inv_attr = pi.CreateInvisibleIdsAttr()
        col_pv = UsdGeom.PrimvarsAPI(pi.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.varying)
        for frame in range(frame_map.first, frame_map.last + 1):
            pts, oxy = (oxygen_frames[frame] if frame < len(oxygen_frames)
                        else (np.zeros((0, 3)), np.zeros((0,))))
            n = len(pts)
            # A per-instance z offset of a few thousandths of a micron:
            # the discs are coplanar by construction and overlap heavily
            # in a dense tissue, and exactly coplanar faces z-fight into
            # a speckled mess. Far below the resolution of anything the
            # picture claims, and it buys a clean floor.
            pos_attr.Set(Vt.Vec3fArray(
                [Gf.Vec3f(float(p[0]), float(p[1]),
                          float(z_plane) + 0.004 * i)
                 for i, p in enumerate(pts)]
                + [Gf.Vec3f(0.0, 0.0, float(z_plane))] * (n_max - n)), frame)
            inv_attr.Set(Vt.Int64Array(list(range(n, n_max))), frame)
            col_pv.Set(Vt.Vec3fArray(
                [Gf.Vec3f(*oxygen_color(float(v), hypoxia_threshold, arterial))
                 for v in oxy]
                + [Gf.Vec3f(0.0, 0.0, 0.0)] * (n_max - n)), frame)
        pi.GetPrim().SetMetadata(
            "comment",
            "One disc per living cell, coloured by that cell's own "
            "recorded oxygen. No value is interpolated between cells.")
        report["shadows"] = path

    if draw_halos:
        path = f"{root}/HypoxicCells"
        hyp_counts = [int((np.asarray(o) < hypoxia_threshold).sum())
                      for _p, o in oxygen_frames]
        peak = max(hyp_counts) if hyp_counts else 0
        report["peak_hypoxic_cells"] = peak
        report["frames_with_hypoxia"] = int(sum(1 for c in hyp_counts if c))
        if peak == 0:
            report["halos"] = None
            report["halo_reason"] = (
                f"no cell was ever below oxygen {hypoxia_threshold} in this "
                f"run, so no hypoxia marker is drawn")
        else:
            pi = UsdGeom.PointInstancer.Define(stage, path)
            proto = UsdGeom.Sphere.Define(stage, f"{path}/Prototypes/Halo")
            proto.GetRadiusAttr().Set(float(halo_radius))
            proto.GetExtentAttr().Set(Vt.Vec3fArray([
                Gf.Vec3f(-halo_radius, -halo_radius, -halo_radius),
                Gf.Vec3f(halo_radius, halo_radius, halo_radius)]))
            UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(_emissive_material(
                stage, f"{path}/Prototypes/Halo/Mat", (0.75, 0.06, 0.10),
                opacity=0.42))
            pi.CreatePrototypesRel().SetTargets([proto.GetPath()])
            pi.CreateProtoIndicesAttr().Set(Vt.IntArray([0] * peak))
            pi.CreateIdsAttr().Set(Vt.Int64Array(list(range(peak))))
            pos_attr = pi.CreatePositionsAttr()
            inv_attr = pi.CreateInvisibleIdsAttr()
            for frame in range(frame_map.first, frame_map.last + 1):
                pts, oxy = (oxygen_frames[frame]
                            if frame < len(oxygen_frames)
                            else (np.zeros((0, 3)), np.zeros((0,))))
                sel = [p for p, v in zip(pts, np.asarray(oxy))
                       if float(v) < hypoxia_threshold][:peak]
                pos_attr.Set(Vt.Vec3fArray(
                    [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
                     for p in sel]
                    + [Gf.Vec3f(0.0, 0.0, 0.0)] * (peak - len(sel))), frame)
                inv_attr.Set(Vt.Int64Array(list(range(len(sel), peak))), frame)
            pi.GetPrim().SetMetadata(
                "comment",
                f"Cells with recorded oxygen < {hypoxia_threshold}, the "
                f"threshold VascularModule._check_hypoxia uses. Peak {peak} "
                f"cell(s) on one frame.")
            report["halos"] = path

    return report


# ── Camera, backdrop, legend ─────────────────────────────────────────

def author_event_camera(
    stage,
    center: Sequence[float],
    radius: float,
    path: str = "/World/Cameras/EventCam",
    azimuth_deg: float = -58.0,
    elevation_deg: float = 26.0,
    focal_length: float = 35.0,
    aspect: float = 16.0 / 9.0,
    distance_scale: float = 2.5,
    backdrop_color: Tuple[float, float, float] = (0.020, 0.024, 0.038),
) -> Dict[str, Any]:
    """A camera that frames a sphere of interest, plus a backdrop.

    usdrecord auto-frames a stage with no camera, which re-frames on
    every change: markers that appear and disappear would make the
    camera breathe, and a scrubbable sequence needs a fixed viewpoint.

    The backdrop is a quad parented to the camera, far behind everything,
    so the render is a dark field rather than the transparent background
    usdrecord produces when a dome light has no texture.
    """
    _require_usd()
    cx, cy, cz = (float(v) for v in center)
    radius = max(float(radius), 1e-3)
    aperture = 20.955

    # Resolve the distance FIRST, then author everything from it. The
    # first version set the clipping range from the requested distance
    # and only afterwards pulled the camera back to fit the subject, so
    # the far plane ended up in front of the backdrop and the render
    # came out on a transparent background.
    dist = radius * float(distance_scale)
    half_fov = math.atan((aperture / 2.0) / float(focal_length))
    needed = radius / max(1e-6, math.sin(half_fov))
    if needed > dist:
        dist = needed * 1.05

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = Gf.Vec3d(
        cx + dist * math.cos(el) * math.cos(az),
        cy + dist * math.cos(el) * math.sin(az),
        cz + dist * math.sin(el),
    )
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr().Set(float(focal_length))
    cam.CreateHorizontalApertureAttr().Set(aperture)
    cam.CreateVerticalApertureAttr().Set(aperture / float(aspect))
    # Clipping tight around what is actually in frame. A near plane at
    # 0.1 against a far plane in the tens of thousands spends the whole
    # depth buffer on empty space, and the first symptom is z-fighting
    # between the coplanar oxygen discs on the floor -- measured, they
    # speckled. Everything drawn sits between 1% and 13x the subject
    # distance, the backdrop at 12x included.
    cam.CreateClippingRangeAttr().Set(
        Gf.Vec2f(float(dist * 0.01), float(dist * 13.0)))
    view = Gf.Matrix4d(1.0)
    view.SetLookAt(eye, Gf.Vec3d(cx, cy, cz), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())

    back_d = dist * 12.0
    bw = back_d * (aperture / 2.0) / float(focal_length) * 1.2
    bh = bw / float(aspect)
    backdrop = UsdGeom.Mesh.Define(stage, path + "/Backdrop")
    backdrop.CreatePointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(-bw, -bh, -back_d), Gf.Vec3f(bw, -bh, -back_d),
        Gf.Vec3f(bw, bh, -back_d), Gf.Vec3f(-bw, bh, -back_d)]))
    backdrop.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    backdrop.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    backdrop.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI.Apply(backdrop.GetPrim()).Bind(
        _emissive_material(stage, path + "/Backdrop/Mat", backdrop_color))

    return {
        "path": path, "eye": (float(eye[0]), float(eye[1]), float(eye[2])),
        "center": (cx, cy, cz), "distance": float(dist), "aspect": float(aspect),
        "focal_length": float(focal_length), "aperture": aperture,
        # Where to hang the legend quad: inside the near plane, which is
        # no longer at 0.1.
        "legend_distance": float(dist * 0.02),
    }


def author_legend(
    stage,
    camera_path: str,
    texture_file: str,
    aspect: float = 16.0 / 9.0,
    focal_length: float = 35.0,
    aperture: float = 20.955,
    width_fraction: float = 0.30,
    texture_aspect: float = 1.0,
    corner: str = "bottom-left",
    distance: float = 1.0,
    path: Optional[str] = None,
) -> str:
    """Parent a textured quad to the camera so it holds a screen corner.

    USD has no text primitive, so the only way to put words in the stage
    itself -- rather than only in a post-processed PNG -- is a texture.
    This renders in Apple's usdrecord (Storm/Metal); verified.
    """
    _require_usd()
    path = path or (camera_path + "/Legend")
    d = float(distance)
    half_w = d * (aperture / 2.0) / float(focal_length)
    half_h = half_w / float(aspect)
    w = 2.0 * half_w * float(width_fraction)
    h = w / max(1e-6, float(texture_aspect))
    margin = 0.04 * half_w
    if "left" in corner:
        x0 = -half_w + margin
    else:
        x0 = half_w - margin - w
    if "bottom" in corner:
        y0 = -half_h + margin
    else:
        y0 = half_h - margin - h

    quad = UsdGeom.Mesh.Define(stage, path)
    quad.CreatePointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(x0, y0, -d), Gf.Vec3f(x0 + w, y0, -d),
        Gf.Vec3f(x0 + w, y0 + h, -d), Gf.Vec3f(x0, y0 + h, -d)]))
    quad.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    quad.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    quad.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdGeom.PrimvarsAPI(quad.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    ).Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0),
                         Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))

    mat_path = path + "/Mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    surface = UsdShade.Shader.Define(stage, mat_path + "/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, mat_path + "/StReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, mat_path + "/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f
                        ).ConnectToSource(tex.ConnectableAPI(), "rgb")
    surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f
                        ).ConnectToSource(tex.ConnectableAPI(), "rgb")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(mat)
    return path


def legend_entries_for(
    kinds: Iterable[str],
    styles: Optional[Dict[str, MarkerStyle]] = None,
) -> List[Dict[str, Any]]:
    """Legend rows for the kinds actually present in a render.

    Only what is on screen: a legend that lists events the run never
    produced teaches the viewer to look for things that are not there.
    """
    styles = DEFAULT_STYLES if styles is None else styles
    rows: List[Dict[str, Any]] = []
    for kind in kinds:
        style = styles.get(kind)
        if style is None:
            continue
        rows.append({"shape": style.shape, "color": style.color,
                     "label": style.label})
    return rows


# ── Stage scaffolding ────────────────────────────────────────────────

def ensure_stage_timeline(stage, frame_map: FrameMap,
                          fps: float = 24.0) -> None:
    """Set start/end time codes to the frame range this layer uses."""
    _require_usd()
    stage.SetStartTimeCode(float(frame_map.first))
    stage.SetEndTimeCode(float(frame_map.last))
    stage.SetTimeCodesPerSecond(float(fps))
    stage.SetFramesPerSecond(float(fps))


def ensure_dome_light(stage, path: str = "/World/Lights/DomeLight",
                      intensity: float = 1.0):
    """A dome light at a sane intensity.

    ``SceneExporter.cells_to_usd`` authors ``intensity = 500``, which in
    Storm blows every surface to pure white -- measured: a 0.05-grey
    backdrop renders as 255,255,255. 1.0 is the value at which the
    colours authored are the colours rendered.
    """
    _require_usd()
    dome = UsdLux.DomeLight.Define(stage, path)
    dome.GetIntensityAttr().Set(float(intensity))
    return dome


def events_bounds(events: Sequence[VisualEvent],
                  extra_points: Optional[Sequence[Sequence[float]]] = None
                  ) -> Tuple[Tuple[float, float, float], float]:
    """Centre and radius covering every positioned event (and extras)."""
    pts = [e.position for e in events if e.position is not None]
    if extra_points:
        pts = list(pts) + [tuple(float(c) for c in p) for p in extra_points]
    if not pts:
        return ((0.0, 0.0, 0.0), 1.0)
    arr = np.asarray(pts, dtype=float)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0) or 1.0
    return ((float(center[0]), float(center[1]), float(center[2])), radius)
