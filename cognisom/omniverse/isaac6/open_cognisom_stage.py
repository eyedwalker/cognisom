"""Open a Cognisom Bio-USD run in stock Isaac Sim 6.0, ready to watch.

Run with:  runheadless.sh --exec /isaac-sim/.local/share/ov/data/cognisom/open_cognisom_stage.py

Four things the stock app does not do on its own:
  - open our stage (it starts on an empty scene)
  - light it for RTX: write_run authors its dome light at 1.0, the value
    Storm needs; RTX scales light physically and renders that near black
  - frame the run's own camera and play it at a pace a person can follow
  - say how fast it is actually rendering

That last one exists because "it looks fuzzy and is slow" has two
separate causes and only a measurement separates them. Softness came
from the render resolution: isaacsim.exp.base.kit ships
``rtx.post.dlss.execMode = 0`` -- DLSS PERFORMANCE -- which at a
1920x1080 output renders internally at 960x540 and AI-upscales. That
default is tuned for robotics throughput, not for a person looking at
the picture, and it is overridden on the command line (see
docker-compose.override.yml), not here. Slowness is a frame rate, so
this logs the rate rather than inferring it from GPU utilisation --
100% busy tells you nothing about whether that is 15 fps or 200.
"""
import asyncio
import os
import time

import carb
import omni.kit.app
import omni.timeline
import omni.usd
import omni.kit.commands
from pxr import Sdf, Usd, UsdGeom, UsdLux, Vt

# Every import is up here on purpose. An `import omni.timeline` inside
# _open() makes `omni` a LOCAL name for the whole function, so its first
# line -- omni.kit.app.get_app() -- raised UnboundLocalError and the
# stage never opened.

STAGE = os.environ.get(
    "COGNISOM_STAGE",
    "/isaac-sim/.local/share/ov/data/cognisom/aggressive_tumor_12h.usdc")
CAMERA = "/World/Cameras/EventCam"
# Kit's own free camera. The viewer STARTS here, placed at the home view,
# so navigation never disturbs the home camera.
PERSPECTIVE = "/OmniverseKit_Persp"
RTX_DOME = 1000.0
# 25 frames is 12 simulated hours. At the stage's 24 fps that is one
# second; 3 fps makes it about eight, which can actually be watched.
PLAYBACK_FPS = float(os.environ.get("COGNISOM_PLAYBACK_FPS", "3.0"))
# Cell bodies are authored at primvars:displayOpacity = 0.35 so the
# nucleus reads through them. That is right for a still render and
# expensive for a live one: 2,400 MUTUALLY OVERLAPPING translucent
# spheres mean every camera ray crosses many translucent layers, and
# RTX samples translucency stochastically and then denoises it -- so
# the same property that costs the frame rate is also what makes the
# picture grainy. Set COGNISOM_CELL_OPACITY=1.0 to render them opaque.
# Unset leaves the stage exactly as authored.
# `or None` so that an EMPTY value means "leave the stage as authored".
# Compose cannot really unset a variable, so the A/B that justified
# the default passes "" for the translucent arm.
CELL_OPACITY = os.environ.get("COGNISOM_CELL_OPACITY") or None
# Report the achieved rate every so often. 15 s is long enough that one
# slow stage-load frame does not dominate the average.
FPS_WINDOW_S = 15.0


def _log(msg):
    carb.log_warn(f"[cognisom] {msg}")


async def _report_render_rate(app):
    """Log the rate the app is ACHIEVING, not the rate it is asking for.

    omni.kit.livestream.core caps the run loops at 60; whether we get
    60 is a different question, and GPU utilisation cannot answer it --
    a pegged GPU is equally consistent with 200 fps of cheap frames and
    with 12 fps of expensive ones.
    """
    await asyncio.sleep(5.0)
    while True:
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < FPS_WINDOW_S:
            await app.next_update_async()
            n += 1
        dt = time.monotonic() - t0
        _log(f"render rate {n / dt:.1f} fps ({n} updates in {dt:.1f}s)")


async def _open():
    app = omni.kit.app.get_app()
    for _ in range(120):                     # let the streaming app settle
        await app.next_update_async()

    if not os.path.exists(STAGE):
        _log(f"stage not found: {STAGE}")
        return

    ctx = omni.usd.get_context()
    result = await ctx.open_stage_async(STAGE)
    ok = result[0] if isinstance(result, tuple) else bool(result)
    if not ok:
        _log(f"open_stage_async failed: {result}")
        return
    stage = ctx.get_stage()
    _log(f"opened {STAGE}")

    raised = 0
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DomeLight):
                attr = UsdLux.DomeLight(prim).GetIntensityAttr()
                value = attr.Get()
                if value is None or float(value) < 50.0:
                    attr.Set(RTX_DOME)
                    raised += 1
    _log(f"raised {raised} Storm-tuned dome light(s) to {RTX_DOME}")

    if CELL_OPACITY is not None:
        target = float(CELL_OPACITY)
        # Only the prims that AUTHOR an opacity are touched, which is
        # exactly the six cell bodies -- nuclei and markers inherit the
        # 1.0 fallback and must keep it.
        targets = [p for p in stage.Traverse()
                   if p.GetAttribute("primvars:displayOpacity")
                   and p.GetAttribute("primvars:displayOpacity").HasAuthoredValue()]
        was = [float(p.GetAttribute("primvars:displayOpacity").Get()[0])
               for p in targets]
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for p in targets:
                p.GetAttribute("primvars:displayOpacity").Set(
                    Vt.FloatArray([target]))
        now = [float(p.GetAttribute("primvars:displayOpacity").Get()[0])
               for p in targets]
        _log(f"cell opacity {sorted(set(was))} -> {sorted(set(now))} "
             f"on {len(targets)} prototype bodies")

    for _ in range(30):
        await app.next_update_async()

    # Record what the image is actually being rendered at, so the log
    # says whether the quality flags arrived rather than us assuming.
    try:
        import carb.settings
        s = carb.settings.get_settings()
        _log(f"image: dlss/execMode={s.get('/rtx/post/dlss/execMode')} "
             f"aa/op={s.get('/rtx/post/aa/op')} "
             f"aa/sharpness={s.get('/rtx/post/aa/sharpness')} "
             f"targetFps={s.get('/exts/omni.kit.livestream.app/primaryStream/targetFps')} "
             f"dynamicResize={s.get('/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize')}")
    except Exception as exc:                 # noqa: BLE001
        _log(f"could not read render settings: {exc}")

    home = stage.GetPrimAtPath(CAMERA)
    if home.IsValid():
        # Lock the home camera. Dragging while looking through EventCam
        # moved EventCam itself, so "go back to EventCam" could not undo
        # anything -- the home shot was the thing that had moved.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            lock = home.CreateAttribute("omni:kit:cameraLock",
                                        Sdf.ValueTypeNames.Bool, custom=True)
            lock.Set(True)
        _log(f"locked {CAMERA}: omni:kit:cameraLock="
             f"{home.GetAttribute('omni:kit:cameraLock').Get()}")

        # Put the free camera exactly where the home camera is, then hand
        # the viewport to it.
        home_xf = UsdGeom.Xformable(home).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        persp = stage.GetPrimAtPath(PERSPECTIVE)
        if persp.IsValid():
            try:
                omni.kit.commands.execute(
                    "TransformPrimCommand", path=Sdf.Path(PERSPECTIVE),
                    new_transform_matrix=home_xf)
                got = UsdGeom.Xformable(persp).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default())
                same = all(abs(got[i][j] - home_xf[i][j]) < 1e-3
                           for i in range(4) for j in range(4))
                _log(f"perspective placed at the home view: {same}")
            except Exception as exc:         # noqa: BLE001
                _log(f"could not place perspective: {exc}")
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = (PERSPECTIVE if persp.IsValid()
                                        else CAMERA)
                _log(f"viewport camera -> {viewport.camera_path}")
        except Exception as exc:             # noqa: BLE001
            _log(f"could not set camera: {exc}")

    try:
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_time_codes_per_second(PLAYBACK_FPS)
        timeline.set_start_time(stage.GetStartTimeCode() / PLAYBACK_FPS)
        timeline.set_end_time(stage.GetEndTimeCode() / PLAYBACK_FPS)
        timeline.set_looping(True)
        timeline.play()
        _log(f"playing frames {stage.GetStartTimeCode():.0f}-"
             f"{stage.GetEndTimeCode():.0f} at {PLAYBACK_FPS} fps, looping")
    except Exception as exc:                 # noqa: BLE001
        _log(f"timeline: {exc}")

    asyncio.ensure_future(_report_render_rate(app))


asyncio.ensure_future(_open())
