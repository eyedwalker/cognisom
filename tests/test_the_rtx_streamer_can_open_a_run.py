"""The stage the engine writes reaches the renderer that never had one.

The gap this closes
-------------------
Two halves of this project had never met. ``visualization.run_usd``
writes a run as one USD stage -- time-sampled cell and immune
populations, event markers, the oxygen field, a fitted camera -- and the
Kit extension on the L40S renders with RTX. But every scene the
extension could show was one it BUILT itself, prim by prim, out of JSON
(``diapedesis_scene.build_scene``, ``molecular_manager``). There was no
``open_stage`` anywhere in it, so every stage this project ever wrote
was rendered by Storm, on a laptop, through ``usdrecord``.

This is the fifth instance of the same defect class as VEP, the
measured HLA typing and the NK licensing model: both ends complete,
nothing passing between them.

What this test actually exercises
--------------------------------
The REAL HTTP handler, the REAL ``UsdSceneManager`` and the REAL
``RtxStageClient``, against a REAL ``write_run`` stage. ``carb`` and
``omni`` are stubbed because they exist only inside Kit, but the stub
for ``omni.usd`` opens the stage with pxr for real, and the stub
timeline keeps real time. So everything is under test except the RTX
draw itself, which needs the GPU box.

The threading contract is exercised rather than assumed: nothing opens
a stage until a tick thread, standing in for Kit's update loop, calls
``process_pending()``.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

pytest.importorskip("pxr")
pytest.importorskip("requests")

from cognisom.omniverse.rtx_client import (  # noqa: E402
    RtxStageClient, RtxStageError)
from cognisom.visualization.interaction_recorder import VisualEvent  # noqa: E402
from cognisom.visualization.run_usd import write_run  # noqa: E402

SIM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cognisom", "omniverse", "kit_extension", "cognisom.sim", "cognisom", "sim")

def wait_for(predicate, timeout: float = 5.0, what: str = "condition"):
    """Playback is queued for the main thread, so nothing is instant."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    raise AssertionError(f"{what} did not happen within {timeout:.0f} s")


N_FRAMES = 6
FPS = 24.0
#: Hours per frame in the synthetic run, so a frame maps to a known hour.
HOURS_PER_FRAME = 0.5


# ── a real stage, without paying for an engine run ────────────────


class _Snap:
    """The fields ``write_run`` reads off an EngineRunner snapshot."""

    def __init__(self, time_h: float, n_cells: int, n_immune: int):
        self.time = time_h
        rng = np.random.default_rng(int(time_h * 10) + 7)
        self.cell_ids = list(range(n_cells))
        self.cell_positions = rng.uniform(-40, 40, (n_cells, 3))
        self.cell_types = (["cancer"] * 3 + ["normal"] * (n_cells - 3))
        self.cell_phases = ["G1"] * n_cells
        self.cell_oxygen = [40.0 - i for i in range(n_cells)]
        self.cell_age = [time_h] * n_cells
        self.cell_mhc1 = [1.0] * n_cells
        self.cell_glucose = [5.0] * n_cells
        self.cell_atp = [2.0] * n_cells
        self.immune_ids = list(range(100, 100 + n_immune))
        self.immune_positions = rng.uniform(-40, 40, (n_immune, 3))
        self.immune_types = ["T_cell"] * n_immune
        self.immune_activated = [True] * n_immune
        self.capillary_starts = [(-50.0, 0.0, 0.0)]
        self.capillary_ends = [(50.0, 0.0, 0.0)]
        self.lymph_starts = []
        self.lymph_ends = []


@pytest.fixture(scope="module")
def stage_path(tmp_path_factory):
    snaps = [_Snap(i * HOURS_PER_FRAME, 12, 4) for i in range(N_FRAMES)]
    events = [VisualEvent(kind="kill", event_type="immune_kill",
                          time=2 * HOURS_PER_FRAME, step=2,
                          position=(0.0, 0.0, 0.0), cell_id=1, partner_id=100)]
    path = str(tmp_path_factory.mktemp("stage") / "synthetic_run.usdc")
    report = write_run(path, None, snapshots=snaps, events=events, fps=FPS)
    assert report.frames == N_FRAMES
    return path


# ── the Kit side, under stubs ─────────────────────────────────────


class _FakeTimeline:
    """A timeline that keeps real time, so seeks can be read back."""

    def __init__(self):
        self.fps = 24.0
        self.start = 0.0
        self.end = 0.0
        self.current = 0.0
        self.looping = False
        self.playing = False

    def set_time_codes_per_second(self, fps): self.fps = float(fps)
    def set_start_time(self, t): self.start = float(t)
    def set_end_time(self, t): self.end = float(t)
    def set_looping(self, on): self.looping = bool(on)
    def set_current_time(self, t): self.current = float(t)
    def get_current_time(self): return self.current
    def play(self): self.playing = True
    def pause(self): self.playing = False
    def stop(self): self.playing = False
    def is_playing(self): return self.playing


class _FakeViewport:
    camera_path = "/OmniverseKit_Persp"


class _FakeUsdContext:
    """Opens the stage for real; only the viewport is imaginary."""

    def __init__(self):
        self.stage = None
        self.opened = []

    def open_stage(self, path):
        from pxr import Usd
        self.opened.append(path)
        self.stage = Usd.Stage.Open(path)
        if self.stage is None:
            raise RuntimeError(f"pxr refused to open {path}")
        return True

    def get_stage(self):
        return self.stage


def _install_kit_stubs():
    """Put fake carb/omni in sys.modules and return the handles."""
    timeline = _FakeTimeline()
    viewport = _FakeViewport()
    context = _FakeUsdContext()
    logs = []

    carb = types.ModuleType("carb")
    for level in ("info", "warn", "error"):
        carb.__dict__[f"log_{level}"] = (
            lambda msg, _l=level: logs.append((_l, str(msg))))
    sys.modules["carb"] = carb

    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.get_context = lambda: context
    omni_timeline = types.ModuleType("omni.timeline")
    omni_timeline.get_timeline_interface = lambda: timeline
    omni_kit = types.ModuleType("omni.kit"); omni_kit.__path__ = []
    omni_vp = types.ModuleType("omni.kit.viewport"); omni_vp.__path__ = []
    omni_vp_util = types.ModuleType("omni.kit.viewport.utility")
    omni_vp_util.get_active_viewport = lambda: viewport

    for name, mod in (("omni", omni), ("omni.usd", omni_usd),
                      ("omni.timeline", omni_timeline), ("omni.kit", omni_kit),
                      ("omni.kit.viewport", omni_vp),
                      ("omni.kit.viewport.utility", omni_vp_util)):
        sys.modules[name] = mod
    omni.usd = omni_usd
    omni.timeline = omni_timeline
    omni.kit = omni_kit
    omni_kit.viewport = omni_vp
    omni_vp.utility = omni_vp_util

    return timeline, viewport, context, logs


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    """The real handler + real manager, served on localhost.

    ``cognisom.sim`` is loaded under the name ``kitsim`` because inside
    Kit the extension directory is on sys.path and ``cognisom`` means
    the extension; here it already means this repo.
    """
    saved = {k: sys.modules.get(k) for k in
             ("carb", "omni", "omni.usd", "omni.timeline", "omni.kit",
              "omni.kit.viewport", "omni.kit.viewport.utility")}
    timeline, viewport, context, logs = _install_kit_stubs()

    pkg = types.ModuleType("kitsim")
    pkg.__path__ = [SIM_DIR]
    sys.modules["kitsim"] = pkg
    usm = importlib.import_module("kitsim.usd_scene_manager")
    srv = importlib.import_module("kitsim.streaming_server")

    stage_dir = str(tmp_path_factory.mktemp("stages"))
    manager = usm.UsdSceneManager(stage_dir=stage_dir)
    # Constructing the server sets the handler's class-level managers.
    # We bind the socket ourselves, to loopback only.
    srv.StreamingServer(None, molecular_manager=None,
                        usd_scene_manager=manager, port=0)
    httpd = srv._ThreadingHTTPServer(("127.0.0.1", 0), srv._StreamHandler)
    port = httpd.socket.getsockname()[1]

    stop = threading.Event()
    ticking = threading.Event()
    ticking.set()

    def serve():
        while not stop.is_set():
            httpd.handle_request()

    def tick():
        """Stand in for Kit's update loop, at roughly its rate."""
        while not stop.is_set():
            if ticking.is_set():
                manager.process_pending()
            time.sleep(0.02)

    httpd.timeout = 0.2
    threads = [threading.Thread(target=serve, daemon=True),
               threading.Thread(target=tick, daemon=True)]
    for thread in threads:
        thread.start()

    client = RtxStageClient(f"http://127.0.0.1:{port}", timeout=10.0)
    yield types.SimpleNamespace(
        client=client, manager=manager, timeline=timeline, viewport=viewport,
        context=context, logs=logs, stage_dir=stage_dir, ticking=ticking,
        intake=importlib.import_module("kitsim.stage_intake"))

    stop.set()
    for thread in threads:
        thread.join(timeout=2.0)
    httpd.server_close()
    for key, value in saved.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value
    for key in ("kitsim", "kitsim.usd_scene_manager", "kitsim.streaming_server",
                "kitsim.stage_intake"):
        sys.modules.pop(key, None)


# ── the endpoint exists and reports honestly before any load ──────


def test_the_streamer_now_has_a_stage_endpoint_and_admits_it_is_empty(box):
    status = box.client.stage_status()
    assert status["loaded"] is False
    assert status["path"] is None
    assert status["error"] is None
    assert status["load_count"] == 0
    assert status["instancers"] == []


def test_the_stage_endpoints_do_not_need_the_diapedesis_manager(box):
    """The box here was built with ``diapedesis_manager=None``. Before
    the guard was widened, every POST under /cognisom/ returned 503
    regardless of which manager it needed."""
    assert box.client._request("POST", "/cognisom/stage/pause") == {
        "status": "paused"}
    assert box.client.stage_status()["loaded"] in (True, False)
    with pytest.raises(RtxStageError, match="503|diapedesis manager"):
        box.client._request("POST", "/cognisom/diapedesis/play")


# ── a run opens, and the box reports what it is ───────────────────


def test_a_run_written_by_write_run_opens_in_the_renderer(box, stage_path):
    load = box.client.render_stage(stage_path, timeout=20.0)

    assert load.name == "synthetic_run.usdc"
    assert load.bytes_sent == os.path.getsize(stage_path)
    assert box.context.opened[-1].endswith("synthetic_run.usdc")

    assert load.n_frames == N_FRAMES
    assert load.duration_hours == pytest.approx((N_FRAMES - 1) * HOURS_PER_FRAME)
    # Four instancers, not two: write_run instances the event markers
    # and the oxygen field as well as the two cell populations, and all
    # four are what the renderer will actually draw.
    assert set(load.instancers) == {
        "/World/Cells", "/World/ImmuneCells",
        "/World/Interactions/Kill", "/World/Fields/OxygenShadows"}, load.instancers
    assert load.status["bio_version"], "the Bio-USD version travels with the stage"
    assert load.status["fps"] == FPS


def test_the_timeline_is_handed_the_stages_own_range(box, stage_path):
    box.client.render_stage(stage_path, timeout=20.0)
    assert box.timeline.fps == FPS
    assert box.timeline.start == pytest.approx(0.0)
    assert box.timeline.end == pytest.approx((N_FRAMES - 1) / FPS)
    assert box.timeline.looping is True


def test_the_runs_own_camera_is_selected(box, stage_path):
    """``write_run`` fits a camera to the run's bounds, markers included.
    Framing one here from scratch would throw that away."""
    box.viewport.camera_path = "/OmniverseKit_Persp"
    load = box.client.render_stage(stage_path, timeout=20.0)
    assert load.status["camera"] == "/World/Cameras/EventCam"
    assert box.viewport.camera_path == "/World/Cameras/EventCam"


def test_a_stage_is_reported_loaded_only_once_it_really_is(box, stage_path):
    """``status`` is what a caller polls before screenshotting. Reporting
    ``loaded`` while the timeline and camera were still being set let a
    caller capture the PREVIOUS scene believing it was this one -- the
    race that the camera test above caught."""
    box.client.stop()
    box.viewport.camera_path = "/OmniverseKit_Persp"
    box.timeline.fps = 1.0
    load = box.client.render_stage(stage_path, timeout=20.0)
    # The first status that said "loaded" must already have had both.
    assert load.status["fps"] == FPS
    assert box.timeline.fps == FPS
    assert box.viewport.camera_path == "/World/Cameras/EventCam"


# ── the threading contract ────────────────────────────────────────


def test_nothing_opens_until_the_main_thread_ticks(box, stage_path):
    """The HTTP thread may only queue. If it opened the stage itself,
    Kit would crash on a USD call from the wrong thread -- and the test
    for that is that the load does NOT happen while ticking is stopped.
    """
    box.client.stop()
    before = len(box.context.opened)
    box.ticking.clear()
    try:
        ack = box.client.upload_stage(stage_path, name="queued.usdc")
        assert ack["path"].endswith("queued.usdc")
        time.sleep(0.3)                     # several tick intervals
        assert len(box.context.opened) == before, (
            "the HTTP thread must not have opened the stage")
        assert box.client.stage_status()["name"] != "queued.usdc"
    finally:
        box.ticking.set()
    assert box.client.wait_until_loaded("queued.usdc", timeout=10.0)
    assert len(box.context.opened) == before + 1


def test_re_uploading_a_run_waits_for_the_new_load_not_the_old(box, stage_path):
    """Matching on the filename alone returned the PREVIOUS load the
    instant polling began, so a caller re-sending a run would screenshot
    the scene it had replaced. ``load_count`` is what separates them."""
    first = box.client.render_stage(stage_path, timeout=20.0)
    was = int(first.status["load_count"])

    box.ticking.clear()
    try:
        ack = box.client.upload_stage(stage_path)
        name = os.path.basename(ack["path"])
        stale = box.client.stage_status()
        assert stale["loaded"] and stale["name"] == name, (
            "the box still reports the first load, by the same name")
        assert int(stale["load_count"]) == was
        with pytest.raises(RtxStageError, match="was not open after"):
            box.client.wait_until_loaded(name, timeout=1.0, poll=0.2,
                                         after_count=was)
    finally:
        box.ticking.set()

    fresh = box.client.wait_until_loaded(name, timeout=10.0, after_count=was)
    assert int(fresh["load_count"]) == was + 1


# ── playback ──────────────────────────────────────────────────────


def test_play_on_upload_starts_the_timeline_on_the_main_thread(box, stage_path):
    box.client.stop()
    wait_for(lambda: not box.timeline.playing, what="the stop")
    box.client.render_stage(stage_path, play=True, timeout=20.0)
    wait_for(lambda: box.timeline.playing, what="playback")
    wait_for(lambda: box.client.stage_status()["playing"] is True,
             what="status catching up")
    box.client.pause()
    wait_for(lambda: not box.timeline.playing, what="the pause")


def test_commands_are_applied_in_the_order_they_were_issued(box, stage_path):
    """Opens and playback commands share one queue for this reason.

    With the open held in its own slot, every queued command was applied
    AFTER it regardless of when it was issued, so a ``stop()`` sent
    before an upload cancelled the play that upload asked for -- the
    stage loaded and sat on frame 0, looking like play-on-open was
    simply broken.
    """
    box.ticking.clear()
    try:
        box.client.stop()                   # issued FIRST
        box.client.upload_stage(stage_path, name="ordered.usdc", play=True)
    finally:
        box.ticking.set()                   # both drain on one tick

    wait_for(lambda: box.client.stage_status().get("name") == "ordered.usdc",
             what="the load")
    wait_for(lambda: box.timeline.playing,
             what="playback, which the earlier stop must not have cancelled")
    box.client.pause()
    wait_for(lambda: not box.timeline.playing, what="the pause")


def test_a_seek_clamps_to_the_stage_and_reports_the_simulated_hour(box, stage_path):
    box.client.render_stage(stage_path, timeout=20.0)

    # The clamp is arithmetic, so the frame comes back at once; only the
    # timeline call waits for the main thread.
    result = box.client.seek(2)
    assert result["frame"] == 2.0
    wait_for(lambda: box.timeline.current == pytest.approx(2.0 / FPS),
             what="the seek")
    wait_for(lambda: box.client.stage_status()["simulated_hours"]
             == pytest.approx(2 * HOURS_PER_FRAME),
             what="the reported hour")  # frame 2 of this run is hour 1.0

    # Past the end, it lands on the last frame rather than off the stage.
    assert box.client.seek(9999)["frame"] == float(N_FRAMES - 1)
    assert box.client.seek(-5)["frame"] == 0.0


def test_a_scrub_between_frames_reports_an_hour_between_them(box, stage_path):
    """A viewer scrubbing a 24 h run lands between frames constantly;
    reporting the floor would make the clock stutter."""
    box.client.render_stage(stage_path, timeout=20.0)
    box.client.seek(1.5)
    wait_for(lambda: box.client.stage_status()["simulated_hours"]
             == pytest.approx(1.5 * HOURS_PER_FRAME),
             what="a fractional hour")


def test_stop_returns_to_the_first_frame(box, stage_path):
    box.client.render_stage(stage_path, timeout=20.0)
    box.client.seek(4)
    wait_for(lambda: box.timeline.current == pytest.approx(4.0 / FPS),
             what="the seek")
    box.client.stop()
    wait_for(lambda: box.timeline.current == pytest.approx(0.0),
             what="the rewind")
    assert box.timeline.playing is False


# ── a frame is not evidence of a render ───────────────────────────


def test_a_placeholder_frame_is_not_handed_back_as_a_render(box, stage_path):
    """``/frame.jpg`` answers whether or not RTX is up.

    When it is not, the box serves a valid ~35 KB JPEG reading "Cognisom
    Kit Server / Waiting for data...". A saved file and a byte count
    therefore prove nothing, which is how a broken renderer could read
    as a working one. The client checks what the box says it is drawing
    with before handing the bytes over.
    """
    box.client.render_stage(stage_path, timeout=20.0)
    assert box.client.renderer() == "pil", (
        "this harness has no RTX, and the box says so")

    with pytest.raises(RtxStageError, match="placeholder"):
        box.client.frame_jpeg(require_rtx=True)

    # It is still available on request -- it is a real JPEG, just not a
    # render of anything.
    jpeg = box.client.frame_jpeg()
    assert jpeg[:2] == b"\xff\xd8", "a JPEG, which is exactly the problem"


def test_the_box_reports_which_renderer_made_the_frame(box):
    assert box.client.kit_status()["renderer"] in ("rtx", "pil")


# ── refusals ──────────────────────────────────────────────────────


def test_a_body_that_is_not_a_stage_is_reported_not_silently_kept(box):
    """pxr refuses it, and the box says so through status rather than
    sitting on the previous scene as though the new one had loaded."""
    name = "not_really.usdc"
    box.client._request("POST", "/cognisom/stage",
                        params={"name": name},
                        data=b"this is not a USD file at all",
                        headers={"Content-Type": "application/octet-stream"})
    deadline = time.time() + 10.0
    while time.time() < deadline:
        status = box.client.stage_status()
        if status.get("error"):
            break
        time.sleep(0.1)
    assert status["error"], "a stage that will not open must surface its error"
    assert status["name"] != name

    # and the client raises that reason instead of waiting out its timeout
    started = time.time()
    with pytest.raises(RtxStageError, match="could not open"):
        box.client.wait_until_loaded(name, timeout=30.0)
    assert time.time() - started < 5.0, "it must not wait out the timeout"


def test_an_empty_body_is_refused_with_a_reason(box):
    with pytest.raises(RtxStageError, match="bytes of a USD stage"):
        box.client._request("POST", "/cognisom/stage", data=b"")


def test_the_client_refuses_a_file_that_is_not_usd_before_sending_it(box, tmp_path):
    junk = tmp_path / "frames.json"
    junk.write_text("{}")
    with pytest.raises(ValueError, match="not a USD stage"):
        box.client.upload_stage(str(junk))


def test_an_unknown_stage_subpath_is_a_404_not_a_500(box):
    with pytest.raises(RtxStageError, match="404"):
        box.client._request("POST", "/cognisom/stage/rewind")


def test_no_method_the_http_thread_can_reach_calls_into_kit():
    """The contract, checked structurally rather than trusted.

    Kit's timeline is a main-thread API. ``DiapedesisManager`` gets away
    with ``play()`` from the HTTP thread because it only sets a flag;
    this manager would be calling into Kit, so playback queues a command
    and ``status`` reads values cached on the last tick. That is easy to
    undo by accident, so this walks the AST instead of believing the
    docstring.
    """
    import ast

    main_thread_only = {
        "process_pending", "_run_command", "_refresh_cache", "_try_open",
        "_open", "_apply_timeline", "_select_camera"}

    source = os.path.join(SIM_DIR, "usd_scene_manager.py")
    with open(source) as handle:
        tree = ast.parse(handle.read())
    klass = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef)
                 and node.name == "UsdSceneManager")

    offenders = []
    for func in klass.body:
        if not isinstance(func, ast.FunctionDef):
            continue
        if func.name in main_thread_only:
            continue
        body = ast.unparse(func)
        if "omni" in body:
            offenders.append(func.name)
    assert not offenders, (
        "these are reachable from the HTTP thread and call into Kit: "
        f"{offenders}. Queue the work for process_pending() instead.")


def test_stage_intake_stays_importable_without_kit():
    """It holds the logic worth testing, which is why it must not grow a
    ``carb`` or ``omni`` import."""
    import ast

    with open(os.path.join(SIM_DIR, "stage_intake.py")) as handle:
        tree = ast.parse(handle.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not {"carb", "omni"} & imported, imported


# ── the trust boundary ────────────────────────────────────────────


def test_an_uploaded_name_cannot_escape_the_stage_directory(box, stage_path):
    """The name arrives over HTTP. It is rebuilt, not validated."""
    with open(stage_path, "rb") as handle:
        body = handle.read()
    ack = box.client._request(
        "POST", "/cognisom/stage",
        params={"name": "../../../../tmp/escaped.usdc"},
        data=body, headers={"Content-Type": "application/octet-stream"})
    assert os.path.dirname(ack["path"]) == box.stage_dir
    assert os.path.basename(ack["path"]) == "escaped.usdc"
    assert not os.path.exists("/tmp/escaped.usdc")


@pytest.mark.parametrize("raw,expected", [
    (None, "stage.usdc"),
    ("", "stage.usdc"),
    ("run.usdc", "run.usdc"),
    ("RUN.USDA", "RUN.usda"),
    ("../../etc/passwd", "passwd.usdc"),
    ("..\\..\\windows\\sys.usdc", "sys.usdc"),
    ("/abs/path/scene.usda", "scene.usda"),
    ("..", "stage.usdc"),
    (".hidden.usdc", "hidden.usdc"),
    ("no_suffix", "no_suffix.usdc"),
    ("frames.json", "frames.json.usdc"),
])
def test_every_uploaded_name_becomes_one_safe_basename(box, raw, expected):
    assert box.intake.safe_stage_name(raw) == expected


def test_a_shell_metacharacter_cannot_survive_a_name(box):
    name = box.intake.safe_stage_name("a b; rm -rf ~/*.usdz")
    assert ";" not in name and " " not in name and "~" not in name
    assert name.endswith(".usdz")


def test_an_oversized_upload_is_refused_before_it_reaches_the_disk(box):
    """A gigabyte arriving over HTTP is a mistake, not a run, and the
    refusal has to come before the bytes are written."""
    module = sys.modules["kitsim.usd_scene_manager"]
    original = module.MAX_UPLOAD_BYTES
    module.MAX_UPLOAD_BYTES = 64
    try:
        with pytest.raises(ValueError, match="over the"):
            box.manager.save_upload(b"x" * 128, name="big.usdc")
        assert not os.path.exists(os.path.join(box.stage_dir, "big.usdc"))
        # and just under the limit still lands
        assert box.manager.save_upload(b"x" * 32, name="small.usdc")
    finally:
        module.MAX_UPLOAD_BYTES = original
