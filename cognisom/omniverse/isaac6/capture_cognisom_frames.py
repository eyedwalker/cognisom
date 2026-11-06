"""Render a Cognisom run to PNG frames with RTX, offline.

Run with:
    runheadless.sh --exec /isaac-sim/.local/share/ov/data/cognisom/capture_cognisom_frames.py

The sibling ``open_cognisom_stage.py`` streams the run to a browser over
WebRTC, which is the right thing for showing someone live and the wrong
thing for producing a file: there is no frame on disk anywhere in that
path. This script opens the same stage, lights it the same way, puts the
same camera on it, and then steps the timeline one authored frame at a
time, writing each rendered frame to disk.

**It hashes what it wrote and reports how many frames are distinct.**
That is not a nicety. A previous capture in this project produced a GIF
that was one frame repeated, and it was published before anyone noticed,
because a plausible RTX image is indistinguishable from a stuck one when
you look at a single still. A capture that reports ``distinct 1/96`` has
told you it failed; a capture that reports nothing has not.

Environment:
    COGNISOM_STAGE            .usdc to render
    COGNISOM_CAPTURE_DIR      where the PNGs go
    COGNISOM_CAPTURE_SETTLE   updates to run before the FIRST capture
    COGNISOM_CAPTURE_SUBFRAMES updates per frame, for RTX accumulation
    COGNISOM_CELL_OPACITY     as in open_cognisom_stage.py
"""
import asyncio
import hashlib
import os
import time

import carb
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdLux, Vt

STAGE = os.environ.get(
    "COGNISOM_STAGE",
    "/isaac-sim/.local/share/ov/data/cognisom/run.usdc")
OUT_DIR = os.environ.get(
    "COGNISOM_CAPTURE_DIR",
    "/isaac-sim/.local/share/ov/data/cognisom/frames")
CAMERA = "/World/Cameras/EventCam"
PERSPECTIVE = "/OmniverseKit_Persp"
RTX_DOME = 1000.0
CELL_OPACITY = os.environ.get("COGNISOM_CELL_OPACITY") or None
# Updates before the first capture: shader compilation and the first RTX
# accumulation both land here, and a frame captured during either is
# incomplete rather than merely noisy.
SETTLE = int(os.environ.get("COGNISOM_CAPTURE_SETTLE", "240"))
# Updates per frame. RTX accumulates samples across updates and denoises;
# one update per frame gives a visibly noisier image than the live viewer,
# because the live viewer is accumulating continuously on a still camera.
SUBFRAMES = int(os.environ.get("COGNISOM_CAPTURE_SUBFRAMES", "12"))


def _log(msg):
    carb.log_warn(f"[cognisom-capture] {msg}")


def _light_and_dress(stage):
    """Identical treatment to open_cognisom_stage, so the file and the
    live stream are the same picture."""
    raised = 0
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DomeLight):
                attr = UsdLux.DomeLight(prim).GetIntensityAttr()
                value = attr.Get()
                if value is None or float(value) < 50.0:
                    attr.Set(RTX_DOME)
                    raised += 1
    _log(f"raised {raised} dome light(s) to {RTX_DOME}")

    if CELL_OPACITY is None:
        return
    target = float(CELL_OPACITY)
    targets = [p for p in stage.Traverse()
               if p.GetAttribute("primvars:displayOpacity")
               and p.GetAttribute("primvars:displayOpacity").HasAuthoredValue()]
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for p in targets:
            p.GetAttribute("primvars:displayOpacity").Set(
                Vt.FloatArray([target]))
    _log(f"cell opacity -> {target} on {len(targets)} prototype bodies")


def _hide_viewport_overlays():
    """Turn off everything the viewport draws that is not the scene.

    The first capture came back with Kit's construction grid and the
    world-origin axis gizmo drawn straight through the tissue -- faint
    grey lines across the slab and a red/white cross at the lower left.
    Live, in a viewport someone is navigating, that chrome is useful.
    In a file that will sit on a public page it is a tell that you are
    looking at a screenshot of a tool rather than at a render.

    ``displayOptions = 0`` clears the whole bitmask (grid, gizmos,
    selection outline, HUD) in one go, which is more reliable than
    naming each individual toggle and hoping the name is still current;
    the named settings below are belt and braces.
    """
    import carb.settings
    s = carb.settings.get_settings()
    s.set("/persistent/app/viewport/displayOptions", 0)
    for key in ("/app/viewport/grid/enabled",
                "/app/viewport/grid/showOrigin",
                "/app/viewport/outline/enabled",
                "/app/viewport/show/camera",
                "/app/viewport/show/lights",
                "/app/viewport/show/audio"):
        try:
            s.set(key, False)
        except Exception:                    # noqa: BLE001
            pass
    _log(f"viewport chrome off (displayOptions="
         f"{s.get('/persistent/app/viewport/displayOptions')})")


def _place_camera(stage):
    home = stage.GetPrimAtPath(CAMERA)
    if not home.IsValid():
        _log(f"no {CAMERA}; leaving the default view")
        return
    home_xf = UsdGeom.Xformable(home).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    persp = stage.GetPrimAtPath(PERSPECTIVE)
    if persp.IsValid():
        try:
            omni.kit.commands.execute(
                "TransformPrimCommand", path=Sdf.Path(PERSPECTIVE),
                new_transform_matrix=home_xf)
        except Exception as exc:             # noqa: BLE001
            _log(f"could not place perspective: {exc}")
    try:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = CAMERA
            _log(f"viewport camera -> {viewport.camera_path}")
    except Exception as exc:                 # noqa: BLE001
        _log(f"could not set camera: {exc}")


async def _capture(app, viewport, path):
    from omni.kit.viewport.utility import capture_viewport_to_file
    capture = capture_viewport_to_file(viewport, file_path=path)
    # wait_for_result is blocking; the async variant keeps the app ticking
    # so the write actually completes.
    if hasattr(capture, "wait_for_result"):
        for _ in range(240):
            await app.next_update_async()
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
        return os.path.exists(path)
    for _ in range(60):
        await app.next_update_async()
    return os.path.exists(path)


async def _run():
    app = omni.kit.app.get_app()
    for _ in range(120):
        await app.next_update_async()

    if not os.path.exists(STAGE):
        _log(f"stage not found: {STAGE}")
        app.post_quit()
        return

    ctx = omni.usd.get_context()
    result = await ctx.open_stage_async(STAGE)
    ok = result[0] if isinstance(result, tuple) else bool(result)
    if not ok:
        _log(f"open_stage_async failed: {result}")
        app.post_quit()
        return
    stage = ctx.get_stage()
    _log(f"opened {STAGE}")

    _light_and_dress(stage)
    _hide_viewport_overlays()
    _place_camera(stage)

    os.makedirs(OUT_DIR, exist_ok=True)

    first = int(stage.GetStartTimeCode())
    last = int(stage.GetEndTimeCode())
    _log(f"frames {first}-{last}; settling {SETTLE} updates")
    for _ in range(SETTLE):
        await app.next_update_async()

    try:
        import carb.settings
        s = carb.settings.get_settings()
        _log(f"image: dlss/execMode={s.get('/rtx/post/dlss/execMode')} "
             f"aa/op={s.get('/rtx/post/aa/op')}")
    except Exception as exc:                 # noqa: BLE001
        _log(f"could not read render settings: {exc}")

    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()

    digests = []
    t0 = time.monotonic()
    written = 0
    for i, frame in enumerate(range(first, last + 1)):
        timeline.set_current_time(frame / stage.GetTimeCodesPerSecond())
        for _ in range(SUBFRAMES):
            await app.next_update_async()

        path = os.path.join(OUT_DIR, f"frame_{i:05d}.png")
        if await _capture(app, viewport, path):
            written += 1
            with open(path, "rb") as fh:
                digests.append(hashlib.sha256(fh.read()).hexdigest())
        else:
            _log(f"frame {frame}: capture produced no file")

        if i % 10 == 0:
            rate = (i + 1) / max(1e-6, time.monotonic() - t0)
            _log(f"frame {i + 1}/{last - first + 1} ({rate:.2f} frames/s)")

    distinct = len(set(digests))
    _log(f"wrote {written} frames to {OUT_DIR}")
    _log(f"DISTINCT {distinct}/{len(digests)}")
    if distinct <= 1:
        _log("CAPTURE IS STATIC -- every frame is identical. Do not publish "
             "this. The timeline did not advance, or the capture ran before "
             "the renderer caught up.")
    elif distinct < len(digests) * 0.9:
        _log(f"WARNING: only {distinct} distinct images in {len(digests)} "
             "frames; some captures repeated a stale buffer.")
    app.post_quit()


asyncio.ensure_future(_run())
