"""Accept a USD stage from the network, and read what a viewer needs off it.

Deliberately free of ``carb`` and ``omni`` so it can be tested without
Kit. Everything here is stdlib or ``pxr``; the Kit-only work -- opening
the stage in the viewport, driving the timeline -- stays in
``usd_scene_manager``, which calls into this module.

The two jobs:

``safe_stage_name``
    An uploaded filename is attacker-controlled. This is the trust
    boundary between the HTTP body and the filesystem, so the name is
    rebuilt from scratch rather than checked for bad patterns.

``read_stage_facts``
    What ``write_run`` stamped on the stage: its frame range, the
    simulated hour of each frame, every instancer it will draw and
    its camera. A stage that lacks them still loads; it simply
    reports less.
"""

from __future__ import annotations

import math
import posixpath
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from pxr import Usd, UsdGeom, UsdLux
except ImportError:                                   # pragma: no cover
    Usd = UsdGeom = UsdLux = None                     # type: ignore

#: Layer-data key ``visualization.run_usd.write_run`` stamps with the
#: simulated time of every frame, so a viewer can report hours.
FRAME_TIMES_KEY = "cognisom:frameTimesHours"
#: Layer-data key carrying the Bio-USD vocabulary version.
VERSION_KEY = "bio:version"
#: The camera ``write_run`` authors, fitted to the run's own bounds.
PREFERRED_CAMERA = "/World/Cameras/EventCam"

#: What USD will actually open. ``.usdz`` is a package, the rest are
#: single files; all four are self-contained for a ``write_run`` stage,
#: which references nothing external.
STAGE_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")

DEFAULT_STAGE_NAME = "stage.usdc"
_MAX_NAME = 120
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_stage_name(raw: Optional[str],
                    default: str = DEFAULT_STAGE_NAME) -> str:
    """Build a filesystem-safe basename from an uploaded name.

    The name arrives over HTTP, so it is rebuilt rather than validated:
    every path separator is dropped, every character outside
    ``[A-Za-z0-9._-]`` is replaced, leading dots are stripped so nothing
    becomes a dotfile or a traversal, and a USD suffix is guaranteed.
    """
    if not raw or not str(raw).strip():
        return default

    # Split on BOTH separators: a posix host still receives Windows-style
    # names, and "a\\..\\b" must not survive as a single component.
    candidate = str(raw).strip().replace("\\", "/")
    candidate = posixpath.basename(candidate)
    candidate = _SAFE.sub("_", candidate)
    candidate = candidate.lstrip(".")                 # no dotfiles, no ".."
    if not candidate:
        return default

    stem, _, suffix = candidate.rpartition(".")
    if not stem or f".{suffix.lower()}" not in STAGE_SUFFIXES:
        # Unknown or absent suffix: keep the whole thing as the stem and
        # give it the suffix we know USD opens.
        stem, suffix = candidate, DEFAULT_STAGE_NAME.rpartition(".")[2]

    stem = stem[:_MAX_NAME] or DEFAULT_STAGE_NAME.rpartition(".")[0]
    return f"{stem}.{suffix.lower()}"


def read_stage_facts(stage: Any) -> Dict[str, Any]:
    """Read frame range, frame times, populations and camera off a stage.

    Works on any USD stage. A stage without Cognisom layer data reports
    ``frame_times == []`` and ``duration_hours is None`` rather than
    failing: the viewer can still play it, it just cannot name an hour.

    ``instancers`` is every PointInstancer on the stage. For a
    ``write_run`` stage that is four, not two: the cells, the immune
    cells, the event markers and the oxygen field.
    """
    fps = float(stage.GetTimeCodesPerSecond() or 24.0) or 24.0
    start = float(stage.GetStartTimeCode())
    end = float(stage.GetEndTimeCode())

    layer_data = dict(stage.GetRootLayer().customLayerData or {})
    raw_times = layer_data.get(FRAME_TIMES_KEY) or []
    try:
        frame_times = [float(t) for t in raw_times]
    except (TypeError, ValueError):
        frame_times = []

    # Every PointInstancer, not just the cell populations: a
    # write_run stage also instances the event markers and the
    # oxygen field, and what the renderer cares about is the full
    # list of what it will draw.
    instancers: List[str] = []
    if UsdGeom is not None:
        instancers = [str(prim.GetPath()) for prim in stage.Traverse()
                      if prim.IsA(UsdGeom.PointInstancer)]

    camera = (PREFERRED_CAMERA
              if stage.GetPrimAtPath(PREFERRED_CAMERA).IsValid() else None)

    return {
        "start_frame": start,
        "end_frame": end,
        "fps": fps,
        "n_frames": int(round(end - start)) + 1,
        "frame_times": frame_times,
        "duration_hours": frame_times[-1] if frame_times else None,
        "instancers": instancers,
        "camera": camera,
        "bio_version": layer_data.get(VERSION_KEY),
    }


def hours_at_frame(frame_times: List[float], frame: float) -> Optional[float]:
    """The simulated hour a frame sits at, interpolated between samples.

    A viewer scrubbing a 24 h run lands between frames constantly, and
    reporting the floor would make the clock stutter. Returns ``None``
    when the stage carried no frame times.
    """
    if not frame_times:
        return None
    if frame <= 0:
        return float(frame_times[0])
    last = len(frame_times) - 1
    if frame >= last:
        return float(frame_times[last])
    low = int(frame)
    frac = frame - low
    return float(frame_times[low] * (1.0 - frac) + frame_times[low + 1] * frac)


#: Dome-light intensity RTX needs to expose a scene the way Storm does
#: at 1.0. Storm's intensity is ad hoc; RTX's is physically scaled, so
#: one authored number cannot serve both renderers.
RTX_DOME_INTENSITY = 1000.0
#: Below this, a dome light was tuned for Storm and will render near
#: black under RTX.
STORM_DOME_CEILING = 50.0


def boost_lights_for_rtx(stage: Any,
                         intensity: float = RTX_DOME_INTENSITY) -> List[str]:
    """Raise Storm-tuned dome lights to an RTX-usable intensity.

    ``write_run`` authors its dome light at intensity 1.0, which is the
    value at which Storm renders the authored colours faithfully. RTX
    scales light physically, so the same stage comes back almost black:
    the geometry is all there, correctly silhouetted, and unlit.

    The edit goes in the stage's SESSION layer, so the file on disk is
    untouched and still renders correctly under Storm. A light that was
    already authored bright is left alone.

    Returns the paths it changed.
    """
    if UsdLux is None or Usd is None:
        return []
    changed: List[str] = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if not prim.IsA(UsdLux.DomeLight):
                continue
            light = UsdLux.DomeLight(prim)
            attr = light.GetIntensityAttr()
            current = attr.Get()
            if current is not None and float(current) >= STORM_DOME_CEILING:
                continue                    # authored for a lit renderer
            attr.Set(float(intensity))
            changed.append(str(prim.GetPath()))
    return changed


#: The up vector ``author_event_camera`` builds its look-at from. The
#: stage's own upAxis metadata says Y, but that camera is authored
#: Z-up, and orbiting around the wrong up rolls the horizon.
CAMERA_UP = (0.0, 0.0, 1.0)


def stage_pivot(stage: Any, time_code: Optional[float] = None
                ) -> Optional[Tuple[float, float, float]]:
    """The centre of the cell populations, to orbit around.

    Falls back to the whole of ``/World`` when there are no cells, and
    to None when nothing on the stage has an extent.
    """
    if UsdGeom is None or Usd is None:
        return None
    when = (Usd.TimeCode(float(time_code)) if time_code is not None
            else Usd.TimeCode.EarliestTime())
    cache = UsdGeom.BBoxCache(when, [UsdGeom.Tokens.default_,
                                     UsdGeom.Tokens.render])
    for path in ("/World/Cells", "/World"):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        try:
            box = cache.ComputeWorldBound(prim)
        except Exception:                             # noqa: BLE001
            continue
        if box.GetRange().IsEmpty():
            continue
        mid = box.ComputeCentroid()
        return (float(mid[0]), float(mid[1]), float(mid[2]))
    return None


def orbit_camera(stage: Any, camera_path: str,
                 pivot: Sequence[float],
                 yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                 zoom: float = 1.0) -> Optional[Dict[str, Any]]:
    """Orbit a camera around ``pivot`` and re-aim it there.

    ``author_event_camera`` writes the camera as ONE transform op (a
    matrix from ``SetLookAt``), not a translate plus a rotateXYZ. The
    extension's existing orbit reads ``ops[0]``/``ops[1]`` and gives up
    when there are fewer than two, which is why it silently does
    nothing on a Bio-USD stage. This rebuilds the matrix instead.

    ``zoom`` below 1 moves closer. Returns the new eye and distance.
    """
    if UsdGeom is None or Usd is None:
        return None
    from pxr import Gf

    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        return None
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    if not ops:
        return None

    matrix = ops[0].Get()
    if matrix is None:
        return None
    eye = matrix.ExtractTranslation()
    target = Gf.Vec3d(float(pivot[0]), float(pivot[1]), float(pivot[2]))

    offset = Gf.Vec3d(eye[0] - target[0], eye[1] - target[1], eye[2] - target[2])
    dist = offset.GetLength()
    if dist < 1e-9:
        return None

    # Spherical about the camera's own up axis (Z).
    azimuth = math.atan2(offset[1], offset[0])
    elevation = math.asin(max(-1.0, min(1.0, offset[2] / dist)))
    azimuth += math.radians(float(yaw_deg))
    elevation = max(math.radians(-89.0),
                    min(math.radians(89.0),
                        elevation + math.radians(float(pitch_deg))))
    dist = max(1e-3, dist * max(0.01, float(zoom)))

    new_eye = Gf.Vec3d(
        target[0] + dist * math.cos(elevation) * math.cos(azimuth),
        target[1] + dist * math.cos(elevation) * math.sin(azimuth),
        target[2] + dist * math.sin(elevation))

    view = Gf.Matrix4d(1.0)
    view.SetLookAt(new_eye, target, Gf.Vec3d(*CAMERA_UP))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(view.GetInverse())
    return {"eye": (float(new_eye[0]), float(new_eye[1]), float(new_eye[2])),
            "pivot": (float(target[0]), float(target[1]), float(target[2])),
            "distance": float(dist)}
