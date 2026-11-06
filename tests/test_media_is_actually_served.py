"""The app's /media endpoint must return bytes, not a 500.

cognisom.com served **HTTP 500 for every `st.video` and `st.image`** and
nothing in this suite noticed, because nothing here starts the app's HTTP
server. The page rendered, the element was emitted, the file was read and
hashed correctly -- and the bytes never came back. Counting video elements
over the websocket said "1 video element" and was consistent with a player
that shows 0:00 and never plays.

The cause was a *transitive* dependency. streamlit 1.60.0's
MediaAwareGZipMiddleware subclasses Starlette's responders and, for the
audio/ and video/ content types it excludes from compression, swallows the
ASGI `http.response.start` message instead of forwarding it. From starlette
1.8.0 that is fatal. Measured, by fetching the real media URL:

    starlette  1.4.1  1.5.0  1.6.0  1.7.0 | 1.8.0  1.9.0  1.10.0
    st.video     200    200    200    200 |   500    500     500
    st.image     200    200    200    200 |   500    500     500

See requirements.txt for why streamlit cannot simply be upgraded past it
(mhcflurry -> tensorflow-cpu<2.16 -> protobuf<5 -> streamlit==1.60.0).

``test_a_real_server_serves_the_recording`` is the only test here that would
have caught it. The two version checks are cheap guards that fail
everywhere, including on a dev machine whose streamlit is new enough that
the bug does not reproduce.
"""
from __future__ import annotations

import asyncio
import http.client
import socket
import subprocess
import sys
import textwrap
import time
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
RECORDING = ROOT / "cognisom" / "dashboard" / "assets" / "demo_run.mp4"

#: The first starlette that breaks streamlit 1.60.0's media endpoint.
FIRST_BROKEN_STARLETTE = "1.8.0"

#: The first streamlit that rewrote the middleware and no longer cares.
FIRST_FIXED_STREAMLIT = (1, 61)


def _installed(name):
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None


def _parts(v):
    out = []
    for chunk in v.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def test_requirements_forbids_the_starlette_that_breaks_media():
    """The pin is a property: starlette 1.8.0 must not be installable."""
    from packaging.requirements import Requirement

    specs = []
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        if req.name.lower() == "starlette":
            specs.append(req.specifier)

    assert specs, (
        "requirements.txt no longer constrains starlette. It is transitive, "
        "so without a pin pip takes the newest, and streamlit 1.60.0 then "
        "returns HTTP 500 for every st.video and st.image."
    )
    for spec in specs:
        assert not spec.contains(FIRST_BROKEN_STARLETTE), (
            f"starlette {FIRST_BROKEN_STARLETTE} satisfies {spec}, and that "
            "version breaks the media endpoint on streamlit 1.60.0"
        )


def test_the_installed_combination_is_not_the_known_broken_one():
    """Fails on a dev machine too, where the bug would not reproduce."""
    st_v, sl_v = _installed("streamlit"), _installed("starlette")
    if st_v is None or sl_v is None:
        pytest.skip("streamlit/starlette not installed")

    broken_streamlit = _parts(st_v)[:2] < FIRST_FIXED_STREAMLIT
    broken_starlette = _parts(sl_v) >= _parts(FIRST_BROKEN_STARLETTE)

    assert not (broken_streamlit and broken_starlette), (
        f"streamlit {st_v} with starlette {sl_v} serves HTTP 500 for every "
        "st.video and st.image. Either keep starlette below "
        f"{FIRST_BROKEN_STARLETTE} or move streamlit to >= 1.61 (which needs "
        "protobuf >= 5.26.1, so tensorflow-cpu must move off < 2.16 first)."
    )


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _health(port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", "/_stcore/health")
            if c.getresponse().status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


async def _video_url(port):
    """Drive one session so the script runs and registers the media file."""
    import websockets
    from streamlit.proto.BackMsg_pb2 import BackMsg
    from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

    async with websockets.connect(f"ws://127.0.0.1:{port}/_stcore/stream",
                                  open_timeout=30, max_size=None) as ws:
        msg = BackMsg()
        msg.rerun_script.query_string = ""
        await ws.send(msg.SerializeToString())
        url = None
        for _ in range(3000):
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            fm = ForwardMsg()
            fm.ParseFromString(raw if isinstance(raw, bytes) else raw.encode())
            kind = fm.WhichOneof("type")
            if kind == "delta" and fm.delta.HasField("new_element"):
                el = fm.delta.new_element
                if el.WhichOneof("type") == "video":
                    url = el.video.url
            if kind == "script_finished":
                break
        return url


def test_a_real_server_serves_the_recording(tmp_path):
    """Start Streamlit for real and fetch the bytes it advertises.

    A minimal app rather than the dashboard: this is a test of the media
    pipeline (streamlit + starlette + the ASGI stack), and the dashboard's
    imports would make it slow without testing anything more.
    """
    pytest.importorskip("websockets")
    pytest.importorskip("streamlit")
    assert RECORDING.exists(), f"{RECORDING} is missing"
    expected = RECORDING.read_bytes()

    app = tmp_path / "media_app.py"
    app.write_text(textwrap.dedent(f"""
        import streamlit as st
        st.video({str(RECORDING)!r})
    """))

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(app),
         "--server.port", str(port), "--server.headless", "true",
         "--server.address", "127.0.0.1",
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert _health(port, timeout=60), "streamlit never became healthy"

        url = asyncio.run(_video_url(port))
        assert url, "the page emitted no video element"

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("GET", url)
        resp = conn.getresponse()
        body = resp.read()

        assert resp.status == 200, (
            f"GET {url} returned {resp.status} with body {body[:120]!r}. "
            "A 500 here is the starlette/streamlit media bug: the element "
            "renders and the player stays empty. See this module's docstring."
        )
        assert len(body) == len(expected), (
            f"served {len(body)} bytes, file is {len(expected)}")
        assert body[:4] == expected[:4]
        assert expected[4:8] == b"ftyp", "the asset is not an MP4 any more"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
