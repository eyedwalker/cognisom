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

import posixpath
import re
from typing import Any, Dict, List, Optional

try:
    from pxr import UsdGeom
except ImportError:                                   # pragma: no cover
    UsdGeom = None                                    # type: ignore

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
