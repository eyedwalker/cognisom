"""Open a Cognisom Bio-USD run in stock Isaac Sim 6.0, ready to watch.

Run with:  runheadless.sh --exec /isaac-sim/.local/share/ov/data/cognisom/open_cognisom_stage.py

Three things the stock app does not do on its own:
  - open our stage (it starts on an empty scene)
  - light it for RTX: write_run authors its dome light at 1.0, the value
    Storm needs; RTX scales light physically and renders that near black
  - frame the run's own camera and play it at a pace a person can follow
"""
import asyncio
import os

import carb
import omni.kit.app
import omni.timeline
import omni.usd
import omni.kit.commands
from pxr import Sdf, Usd, UsdGeom, UsdLux

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
PLAYBACK_FPS = 3.0


def _log(msg):
    carb.log_warn(f"[cognisom] {msg}")


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

    for _ in range(30):
        await app.next_update_async()

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


asyncio.ensure_future(_open())
