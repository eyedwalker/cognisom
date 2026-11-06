"""Two videos on one page claim to be the same simulation.

The public demo page shows a matplotlib recording and an RTX render side
by side. That juxtaposition is a claim -- that they are the same run --
and a reader can check part of it by counting the red crosses against the
kill count the page states.

Both are produced from ONE engine state by ``record_demo_run.py --usd``:
the stage is exported from the run already in memory, not re-simulated
from the same seed. These tests hold the artefacts to that claim, because
the cheapest way to break it is to regenerate one video and not the other.

The USD stage itself is not committed -- it is 27 MB and the page does not
serve it -- so what is checked here is the sidecar the exporter wrote
beside the videos, plus the videos' own headers.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parent.parent / "cognisom" / "dashboard" / "assets"
PLOT_VIDEO = ASSETS / "demo_run.mp4"
RTX_VIDEO = ASSETS / "rtx_demo.mp4"
FACTS = ASSETS / "demo_run.json"


@pytest.fixture(scope="module")
def facts():
    assert FACTS.exists(), f"{FACTS} is missing"
    return json.loads(FACTS.read_text())


def test_both_videos_ship():
    """`*.mp4` is gitignored; these two survive on an explicit negation.

    If that negation is ever dropped the files vanish from the image and
    the page degrades to a warning, with nothing failing anywhere else.
    """
    for path in (PLOT_VIDEO, RTX_VIDEO):
        assert path.exists(), f"{path.name} is missing from the assets"
        assert path.stat().st_size > 50_000, (
            f"{path.name} is {path.stat().st_size} bytes, which is too small "
            "to be a 10-second recording -- a failed transfer writes a "
            "plausible-looking short file")


@pytest.mark.parametrize("path", [PLOT_VIDEO, RTX_VIDEO],
                         ids=lambda p: p.name)
def test_each_video_is_really_an_mp4(path):
    """A truncated download is a file too, and `curl -f` returns 0 on one."""
    head = path.read_bytes()[:12]
    assert head[4:8] == b"ftyp", (
        f"{path.name} does not start with an MP4 box; it is "
        f"{head!r}")


def test_the_stage_the_rtx_video_came_from_counted_the_engines_kills(facts):
    """The claim a viewer can actually check, by counting crosses.

    The engine's own total, the markers drawn into the matplotlib
    recording, and the markers authored into the USD stage that was
    ray-traced must all be one number. They were three separate code
    paths and two of them used to double every T-cell kill.
    """
    engine = facts["immune_kills"]
    plotted = facts["kill_markers"]
    staged = facts.get("usd_kill_markers")

    assert staged is not None, (
        "demo_run.json has no usd_kill_markers, so the RTX video's stage "
        "was not exported from this run -- regenerate with "
        "`record_demo_run.py --usd`")
    assert plotted == engine, (
        f"the recording draws {plotted} kill markers against {engine} "
        "kills the engine counted")
    assert staged == engine, (
        f"the ray-traced stage draws {staged} kill markers against "
        f"{engine} kills the engine counted -- the two videos on the page "
        "would show different numbers of crosses")


def test_the_duplicates_were_suppressed_in_both_paths(facts):
    """Equal counts could also mean neither path deduplicated.

    Both writers must report having dropped the same duplicates, so this
    cannot pass by the doubling being absent from the run.
    """
    assert facts["duplicate_kill_markers_dropped"] > 0, (
        "no duplicate kill markers were dropped in this run, so the "
        "agreement above does not demonstrate deduplication -- pick a run "
        "with T-cell kills in it")
    assert (facts["usd_duplicate_kills_suppressed"]
            == facts["duplicate_kill_markers_dropped"]), (
        f"the recording dropped {facts['duplicate_kill_markers_dropped']} "
        f"duplicates and the stage dropped "
        f"{facts['usd_duplicate_kills_suppressed']} -- they are supposed to "
        "be the same events from the same run")


def test_the_run_was_scored_by_the_real_predictor(facts):
    """The PWM fallback is a different biology, not a rounding difference.

    Measured 2026-11-06 on this exact configuration: the fallback gave
    115 kills and a cancer population ending at 65, against 98 and 79
    under MHCflurry. A page that says "real patient callset" must not be
    illustrated by the approximation.
    """
    assert facts["predictor"] == "mhcflurry", (
        f"the shipped recording was scored by {facts['predictor']!r}. "
        "Re-record with $HOME/.venvs/mhcflurry/bin/python (3.11); ./venv "
        "is 3.13 and falls back to the PWM scorer silently")


def test_the_page_shows_both_and_says_which_is_which():
    """A render beside a result invites the reader to believe the beauty
    IS the result. The page has to name each one."""
    page = (Path(__file__).resolve().parent.parent / "cognisom" / "dashboard"
            / "_pages" / "43_public_demo.py").read_text()

    assert "_RTX_VIDEO" in page, "the page does not reference the RTX video"
    assert page.count("st.video(") >= 2, "the page does not show both videos"
    assert "What the model computes" in page, (
        "the recording is not labelled as the computed result")
    assert "ray-traced" in page, "the render is not labelled as a render"
