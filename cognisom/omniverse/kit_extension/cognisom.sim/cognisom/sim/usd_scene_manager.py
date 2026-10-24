"""Open a Bio-USD stage in the RTX viewport, and drive its timeline.

Why this exists
---------------
Every other scene in this extension is BUILT, prim by prim, from JSON:
``diapedesis_scene.build_scene`` defines meshes, materials and
instancers in process, and ``molecular_manager`` does the same for PDB
structures. That is why the RTX streamer has only ever shown content
this extension made itself.

Meanwhile the simulation side writes real USD.
``visualization.run_usd.write_run`` authors a whole run as one stage:
cells and immune cells as time-sampled PointInstancers, interaction
markers on the frames their events happened, the oxygen field, and a
camera framed on the run's own bounds. Nothing here could open one, so
every such stage has only ever been rendered by Storm, on a laptop,
through ``usdrecord``.

This manager is the missing link: upload a stage, and RTX renders what
the engine actually simulated.

Threading
---------
Same contract as ``DiapedesisManager``: HTTP handlers run on their own
thread and may only QUEUE work. ``process_pending()`` runs on the Kit
main thread from the extension's update loop, and is the only place
that touches USD or the timeline.

That covers playback too, which is the part easy to get wrong.
``DiapedesisManager.play()`` is safe from any thread because it only
sets a flag; this one would be calling into Kit's timeline, so
``play``/``pause``/``stop``/``seek`` queue a command instead. ``seek``
still returns the clamped frame straight away, since clamping is
arithmetic. ``status()`` likewise reads values cached on the last
tick rather than asking the timeline from the wrong thread.

Testing
-------
The logic worth testing -- filename safety and what gets read off a
stage -- lives in ``stage_intake``, which imports neither ``carb`` nor
``omni`` and so runs in the ordinary test suite. What is left here is
the Kit API calls themselves, which only a GPU box can exercise.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

import carb
import omni.usd

from .stage_intake import (PREFERRED_CAMERA, boost_lights_for_rtx,
                           hours_at_frame, read_stage_facts, safe_stage_name)

#: Where uploaded stages land. Kept off ``/tmp`` by default on the box so
#: a stage survives a tmpfs clean mid-session.
DEFAULT_STAGE_DIR = os.environ.get(
    "COGNISOM_STAGE_DIR", "/tmp/cognisom_stages")

#: Refuse an upload larger than this. A 500k-cell run writes ~30 MB; a
#: gigabyte arriving over HTTP is a mistake, not a run.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


class UsdSceneManager:
    """Opens Bio-USD stages and drives playback. One per extension."""

    def __init__(self, stage_dir: str = DEFAULT_STAGE_DIR) -> None:
        self._stage_dir = stage_dir
        self._lock = threading.Lock()
        # ONE queue for everything, opens included, so commands are
        # applied in the order they were issued. With the open held
        # separately, a stop() issued before an upload was applied
        # AFTER it and cancelled the play the upload asked for.
        #
        # Kit's timeline is a main-thread API: DiapedesisManager gets
        # away with play() from any thread because it only sets a
        # flag, and this one would be calling into Kit.
        self._commands: Deque[tuple] = deque()
        # Refreshed on every tick so status() answers without
        # calling Kit from the HTTP thread, and cheaply: status is
        # polled far more often than a frame changes.
        self._last_frame: float = 0.0
        self._last_playing: bool = False
        self._loaded_path: Optional[str] = None
        self._error: Optional[str] = None
        self._facts: Dict[str, Any] = {}
        self._loaded_at: float = 0.0
        self._load_count: int = 0

    # ── called from the HTTP thread ───────────────────────────────

    def save_upload(self, body: bytes, name: Optional[str] = None) -> str:
        """Write an uploaded stage to disk and return its path.

        Runs on the HTTP thread: it only touches the filesystem, never
        USD. The name is rebuilt by ``safe_stage_name`` because it
        arrived over the network.
        """
        if not body:
            raise ValueError("empty body: expected the bytes of a USD stage")
        if len(body) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"stage is {len(body) / 1e6:.0f} MB, over the "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit")
        os.makedirs(self._stage_dir, exist_ok=True)
        path = os.path.join(self._stage_dir, safe_stage_name(name))
        with open(path, "wb") as handle:
            handle.write(body)
        carb.log_info(f"[cognisom.usd] saved {len(body)} bytes to {path}")
        return path

    def request_open(self, path: str, play: bool = False) -> None:
        """Queue a stage for the main thread. Thread-safe.

        ``play`` starts the timeline as soon as the stage opens. It
        rides the queue rather than being called by the HTTP thread,
        which must not touch the timeline at all.
        """
        with self._lock:
            self._commands.append(("open", (path, bool(play))))
            # Clear any previous failure now, so a caller polling status
            # after this upload is not shown the last stage's error.
            self._error = None
        carb.log_info(
            f"[cognisom.usd] queued stage {path} (play={bool(play)})")

    # ── Kit main thread only ──────────────────────────────────────

    def process_pending(self) -> bool:
        """Open any queued stage, run queued commands, refresh the cache.

        MUST run on the Kit main thread. This is the only place that
        touches USD or the timeline.
        """
        with self._lock:
            commands, self._commands = self._commands, deque()

        opened = False
        for name, argument in commands:
            if name == "open":
                opened = self._try_open(*argument) or opened
            else:
                self._run_command(name, argument)
        self._refresh_cache()
        return opened

    def _run_command(self, name: str, argument: Any) -> None:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        try:
            if name == "play":
                timeline.play()
            elif name == "pause":
                timeline.pause()
            elif name == "stop":
                timeline.stop()
                if self._facts:
                    timeline.set_current_time(
                        self._facts["start_frame"] / self._facts["fps"])
            elif name == "seek" and self._facts:
                timeline.set_current_time(
                    float(argument) / self._facts["fps"])
        except Exception as exc:                      # noqa: BLE001
            carb.log_warn(f"[cognisom.usd] {name} failed: {exc}")

    def _refresh_cache(self) -> None:
        """Read the timeline once per tick, on the thread allowed to."""
        if not self._facts:
            return
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            self._last_frame = (timeline.get_current_time()
                                * self._facts["fps"])
            self._last_playing = bool(timeline.is_playing())
        except Exception:                             # noqa: BLE001
            pass

    def _try_open(self, path: str, play: bool) -> bool:
        try:
            self._open(path)
            if play:
                self._run_command("play", None)
            return True
        except Exception as exc:                      # noqa: BLE001
            # A stage that will not open has to say so through status:
            # otherwise the viewport sits on the previous scene and the
            # caller believes the new one is showing.
            self._error = f"{type(exc).__name__}: {exc}"
            carb.log_error(f"[cognisom.usd] failed to open {path}: {self._error}")
            return False

    def _open(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        context = omni.usd.get_context()
        context.open_stage(path)
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"open_stage({path}) produced no stage")

        facts = read_stage_facts(stage)
        # write_run's dome light is authored at intensity 1.0, the value
        # at which STORM renders the authored colours faithfully. RTX
        # scales light physically, so the same stage arrives almost
        # black -- every cell correctly silhouetted and unlit. Raised in
        # the session layer, so the file on disk stays Storm-correct.
        lit = boost_lights_for_rtx(stage)
        if lit:
            carb.log_warn(f"[cognisom.usd] raised {len(lit)} dome light(s) "
                          f"for RTX: {', '.join(lit)}")
        self._apply_timeline(facts)
        self._select_camera(facts)

        # Commit LAST. A caller polling status while the open is
        # half done would otherwise be told the stage is loaded
        # before its timeline and camera were set, and would
        # screenshot the previous scene believing it was this one.
        self._facts = facts
        self._loaded_path = path
        self._loaded_at = time.time()
        self._load_count += 1
        carb.log_warn(
            f"[cognisom.usd] opened {os.path.basename(path)}: "
            f"{facts['n_frames']} frames @ {facts['fps']:g}fps, "
            f"{len(facts['instancers'])} instancer(s), "
            f"{facts['duration_hours']} h simulated, "
            f"bio:version {facts.get('bio_version')}")

    def _apply_timeline(self, facts: Dict[str, Any]) -> None:
        import omni.timeline

        fps = facts["fps"]
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_time_codes_per_second(fps)
        timeline.set_start_time(facts["start_frame"] / fps)
        timeline.set_end_time(facts["end_frame"] / fps)
        timeline.set_looping(True)
        timeline.set_current_time(facts["start_frame"] / fps)

    def _select_camera(self, facts: Dict[str, Any]) -> None:
        """Use the run's own camera when the stage authored one."""
        if not facts.get("camera"):
            carb.log_info("[cognisom.usd] no EventCam; keeping the viewport camera")
            return
        try:
            import omni.kit.viewport.utility as vp_util

            viewport = vp_util.get_active_viewport()
            if viewport is not None:
                viewport.camera_path = PREFERRED_CAMERA
                carb.log_info(f"[cognisom.usd] camera -> {PREFERRED_CAMERA}")
        except Exception as exc:                      # noqa: BLE001
            # Headless Kit has bitten this extension before. A camera we
            # cannot set is not a reason to fail the load.
            carb.log_warn(f"[cognisom.usd] could not set camera: {exc}")

    # ── playback ──────────────────────────────────────────────────

    def _queue(self, name: str, argument: Any = None) -> None:
        with self._lock:
            self._commands.append((name, argument))

    def play(self) -> None:
        """Queue playback. Safe from any thread."""
        self._queue("play")

    def pause(self) -> None:
        """Queue a pause. Safe from any thread."""
        self._queue("pause")

    def stop(self) -> None:
        """Queue a stop and a rewind. Safe from any thread."""
        self._queue("stop")

    def seek(self, frame: float) -> float:
        """Queue a seek and return the frame it will land on.

        The clamp is arithmetic, so it is computed here and returned to
        the caller immediately; only the timeline call waits for the
        main thread.
        """
        if not self._facts:
            raise RuntimeError("no stage is loaded")
        frame = max(self._facts["start_frame"],
                    min(float(frame), self._facts["end_frame"]))
        self._queue("seek", frame)
        return frame

    # ── reporting ─────────────────────────────────────────────────

    def current_frame(self) -> float:
        """The frame as of the last tick. No Kit call, so any thread."""
        return self._last_frame if self._facts else 0.0

    def is_playing(self) -> bool:
        """As of the last tick. No Kit call, so any thread."""
        return self._last_playing

    def status(self) -> Dict[str, Any]:
        facts = self._facts
        frame = self.current_frame()
        return {
            "loaded": self._loaded_path is not None,
            "path": self._loaded_path,
            "name": (os.path.basename(self._loaded_path)
                     if self._loaded_path else None),
            "error": self._error,
            "load_count": self._load_count,
            "loaded_at": self._loaded_at,
            "playing": self.is_playing(),
            "current_frame": frame,
            # The number a biologist wants: where in the SIMULATION this
            # frame sits, not where in the file.
            "simulated_hours": hours_at_frame(facts.get("frame_times", []), frame),
            "start_frame": facts.get("start_frame"),
            "end_frame": facts.get("end_frame"),
            "n_frames": facts.get("n_frames"),
            "fps": facts.get("fps"),
            "duration_hours": facts.get("duration_hours"),
            "instancers": facts.get("instancers", []),
            "camera": facts.get("camera"),
            "bio_version": facts.get("bio_version"),
            "stage_dir": self._stage_dir,
        }

    def clear(self) -> None:
        self._loaded_path = None
        self._facts = {}
        self._error = None
        self._last_frame = 0.0
        self._last_playing = False
        with self._lock:
            self._commands.clear()
