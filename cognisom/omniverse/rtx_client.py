"""Push a Bio-USD stage to the RTX box and drive it.

What this closes
----------------
``visualization.run_usd.write_run`` writes a run as one USD stage, and
the Kit extension on the L40S renders with RTX. Until now nothing
joined them: every stage this project has written was rendered by
Storm, locally, through ``usdrecord``, while the RTX streamer showed
only scenes the extension built for itself out of JSON.

This is the repo half of the join. ``UsdSceneManager`` on the box is the
other half.

Usage::

    from cognisom.visualization.run_usd import write_run
    from cognisom.omniverse.rtx_client import RtxStageClient

    report = write_run("/tmp/run.usdc", runner)
    client = RtxStageClient()                 # reads GPU_HOST / KIT_SERVER_URL
    status = client.render_stage("/tmp/run.usdc", play=True)
    open("frame.jpg", "wb").write(client.frame_jpeg())

Or from the shell, which also prints what the box made of the stage::

    python -m cognisom.omniverse.rtx_client /tmp/run.usdc --play --save frame.jpg

The stage travels as the request body. A ``write_run`` stage references
nothing external, so one upload is the whole scene -- no asset
resolution on the far side, and no ``scp``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Suffixes the box will open. Matches ``stage_intake.STAGE_SUFFIXES``.
STAGE_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")

DEFAULT_TIMEOUT = 30.0
#: An upload of tens of megabytes over the open internet needs longer
#: than a status poll.
UPLOAD_TIMEOUT = 600.0


class RtxStageError(RuntimeError):
    """The box refused a request, or never finished loading a stage."""


@dataclass
class StageLoad:
    """What came back from pushing one stage."""
    path: str
    name: str
    bytes_sent: int
    seconds: float
    status: Dict[str, Any] = field(default_factory=dict)

    @property
    def instancers(self) -> list:
        """Every PointInstancer the box will draw. For a write_run
        stage that is four: cells, immune cells, markers, oxygen."""
        return list(self.status.get("instancers") or [])

    @property
    def n_frames(self) -> Optional[int]:
        return self.status.get("n_frames")

    @property
    def duration_hours(self) -> Optional[float]:
        return self.status.get("duration_hours")

    def summary(self) -> str:
        drawn = ", ".join(os.path.basename(p) for p in self.instancers) or "none"
        return (f"{self.name}: {self.bytes_sent / 1e6:.2f} MB in "
                f"{self.seconds:.1f} s, {self.n_frames} frames, "
                f"{self.duration_hours} h simulated, drawing [{drawn}]")


class RtxStageClient:
    """Talks to ``/cognisom/stage*`` on the Kit streaming server."""

    def __init__(self, base_url: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        if base_url is None:
            from cognisom.infrastructure.gpu_connector import get_kit_server_url
            base_url = get_kit_server_url()
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout

    # ── plumbing ──────────────────────────────────────────────────

    def _request(self, method: str, path: str, *, params=None, data=None,
                 headers=None, timeout: Optional[float] = None, raw=False):
        import requests

        url = f"{self.base_url}{path}"
        response = requests.request(
            method, url, params=params, data=data, headers=headers,
            timeout=timeout or self.timeout)
        if response.status_code >= 400:
            # Surface the server's own message: it distinguishes "no
            # manager" from "that is not a stage" from "no stage loaded".
            detail = response.text[:400]
            try:
                detail = response.json().get("error", detail)
            except Exception:                         # noqa: BLE001
                pass
            raise RtxStageError(
                f"{method} {path} -> HTTP {response.status_code}: {detail}")
        if raw:
            return response.content
        return response.json()

    # ── the box ───────────────────────────────────────────────────

    def kit_status(self) -> Dict[str, Any]:
        """The extension's own health, which predates this endpoint."""
        return self._request("GET", "/status")

    def is_available(self) -> bool:
        try:
            self.kit_status()
            return True
        except Exception:                             # noqa: BLE001
            return False

    # ── stages ────────────────────────────────────────────────────

    def upload_stage(self, path: str, *, name: Optional[str] = None,
                     play: bool = False) -> Dict[str, Any]:
        """POST a stage file's bytes. Returns the server's acknowledgement.

        The open happens later, on the Kit main thread, so a successful
        return means "accepted", not "showing". Use ``render_stage`` to
        wait for it.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        suffix = os.path.splitext(path)[1].lower()
        if suffix not in STAGE_SUFFIXES:
            raise ValueError(
                f"{path} is not a USD stage (expected one of "
                f"{', '.join(STAGE_SUFFIXES)})")
        with open(path, "rb") as handle:
            body = handle.read()
        if not body:
            raise ValueError(f"{path} is empty")
        return self._request(
            "POST", "/cognisom/stage",
            params={"name": name or os.path.basename(path),
                    "play": "1" if play else "0"},
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=UPLOAD_TIMEOUT)

    def stage_status(self) -> Dict[str, Any]:
        return self._request("GET", "/cognisom/stage/status")

    def play(self) -> Dict[str, Any]:
        return self._request("POST", "/cognisom/stage/play")

    def pause(self) -> Dict[str, Any]:
        return self._request("POST", "/cognisom/stage/pause")

    def stop(self) -> Dict[str, Any]:
        return self._request("POST", "/cognisom/stage/stop")

    def seek(self, frame: float) -> Dict[str, Any]:
        return self._request("POST", "/cognisom/stage/seek",
                             params={"frame": frame})

    def renderer(self) -> str:
        """``"rtx"`` once the box has produced a real RTX frame, else
        ``"pil"``.

        This matters more than it looks. When RTX is not up, /frame.jpg
        still returns a perfectly valid 35 KB JPEG -- a dark placeholder
        reading "Cognisom Kit Server / Waiting for data..." -- so a
        saved file and a byte count are NOT evidence that anything was
        rendered.
        """
        return str(self.kit_status().get("renderer") or "unknown")

    def frame_jpeg(self, *, require_rtx: bool = False) -> bytes:
        """The current viewport as JPEG.

        With ``require_rtx``, raises rather than handing back the
        placeholder frame the box serves when RTX is not up.
        """
        if require_rtx:
            renderer = self.renderer()
            if renderer != "rtx":
                raise RtxStageError(
                    f"the box is serving {renderer} frames, not RTX: this "
                    f"would be the placeholder image, not a render")
        return self._request("GET", "/frame.jpg", raw=True)

    # ── the whole trip ────────────────────────────────────────────

    def wait_until_loaded(self, name: str, *, timeout: float = 120.0,
                          poll: float = 0.5,
                          after_count: Optional[int] = None) -> Dict[str, Any]:
        """Poll until the box reports this stage open, or raise.

        ``after_count`` is the box's ``load_count`` from before the
        upload. Without it, re-sending a stage the box ALREADY has open
        returns the moment polling starts -- the previous load satisfies
        a match on the name -- and the caller screenshots the old scene.
        The counter is what distinguishes this load from that one.

        Raises on the server's own error rather than timing out on it:
        a stage that will not open says so immediately, and waiting two
        minutes to report it would hide the reason.
        """
        deadline = time.time() + timeout
        last: Dict[str, Any] = {}
        while time.time() < deadline:
            last = self.stage_status()
            if last.get("error"):
                raise RtxStageError(f"the box could not open {name}: "
                                    f"{last['error']}")
            fresh = (after_count is None
                     or int(last.get("load_count") or 0) > after_count)
            if last.get("loaded") and last.get("name") == name and fresh:
                return last
            time.sleep(poll)
        raise RtxStageError(
            f"{name} was not open after {timeout:.0f} s; last status {last}")

    def render_stage(self, path: str, *, play: bool = False,
                     timeout: float = 120.0) -> StageLoad:
        """Upload a stage, wait for RTX to have it, and report."""
        started = time.time()
        try:
            before = int(self.stage_status().get("load_count") or 0)
        except RtxStageError:
            # No stage endpoint on this box: let the upload report that.
            before = None
        ack = self.upload_stage(path, play=play)
        name = os.path.basename(ack.get("path") or path)
        status = self.wait_until_loaded(name, timeout=timeout,
                                        after_count=before)
        load = StageLoad(path=path, name=name,
                         bytes_sent=int(ack.get("bytes", 0)),
                         seconds=time.time() - started, status=status)
        logger.info("rendered %s", load.summary())
        return load


def main(argv=None) -> int:                           # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="Push a Bio-USD stage to the RTX box and report what it made of it.")
    parser.add_argument("stage", help="a .usdc/.usda/.usdz written by write_run")
    parser.add_argument("--url", default=None,
                        help="Kit server base URL (default: GPU_HOST/KIT_SERVER_URL)")
    parser.add_argument("--play", action="store_true", help="start the timeline on open")
    parser.add_argument("--seek", type=float, default=None, help="jump to this frame")
    parser.add_argument("--save", default=None, help="write the rendered frame here")
    parser.add_argument("--allow-placeholder", action="store_true",
                        help="save the frame even when the box is not "
                             "rendering with RTX (it serves a placeholder)")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = RtxStageClient(args.url)
    renderer = client.renderer()
    print(f"box: {client.base_url}  renderer: {renderer}")
    if renderer != "rtx":
        print("  NOTE: this box is not producing RTX frames. It will still "
              "serve a valid JPEG -- a placeholder -- so a saved image "
              "would not be a render.")
    load = client.render_stage(args.stage, play=args.play, timeout=args.timeout)
    print(load.summary())
    if args.seek is not None:
        print(f"seek -> {client.seek(args.seek)}")
    if args.save:
        try:
            jpeg = client.frame_jpeg(require_rtx=not args.allow_placeholder)
        except RtxStageError as exc:
            print(f"did not save: {exc}")
            print("  pass --allow-placeholder to save it anyway")
            return 1
        with open(args.save, "wb") as handle:
            handle.write(jpeg)
        print(f"wrote {args.save} ({len(jpeg)} bytes, {client.renderer()})")
    return 0


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())
