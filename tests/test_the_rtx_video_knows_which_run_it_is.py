"""The page says the two videos are one run. Something has to carry that.

Why this file exists
--------------------
``test_the_two_demo_videos_are_one_run.py`` checks that the engine's kill
count, the markers drawn into the recording and the markers authored into
the exported USD stage are one number. All three come from the SAME
sidecar, written by one script in one pass. So it catches a bug in the
exporter, which is what it was written for -- and cannot catch the
likelier accident:

    re-record the 2D video, do not re-render the RTX one.

Then the sidecar describes the new run, its three counts still agree with
each other, that test passes, and the page shows the new recording beside
a render of a DIFFERENT simulation while saying they are the same. A test
that goes green on a false page is worse than no test.

The RTX video is made on another machine: the stage is copied to a GPU
box, Isaac Sim renders PNGs, ffmpeg encodes them. Nothing on that path
carried the run's identity. ``demo_provenance`` adds a fingerprint over
the facts that define a run, the RTX side records it in a manifest at the
moment it renders, and these tests hold the two to each other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognisom.dashboard import demo_provenance as dp

ASSETS = Path(__file__).resolve().parent.parent / "cognisom" / "dashboard" / "assets"


@pytest.fixture(scope="module")
def facts():
    path = ASSETS / "demo_run.json"
    assert path.exists(), f"{path} is missing"
    return json.loads(path.read_text())


# ── the fingerprint has to be both stable and sensitive ──────────────

def test_the_same_sidecar_always_fingerprints_the_same(facts):
    """Re-reading the file must not move the hash, or every comparison
    below is a coin flip."""
    again = json.loads((ASSETS / "demo_run.json").read_text())
    assert dp.run_fingerprint(facts) == dp.run_fingerprint(again)
    assert len(dp.run_fingerprint(facts)) == 16


@pytest.mark.parametrize("key,value", [
    ("seed", 999),
    ("scenario", "Hypoxia"),
    ("duration_hours", 24.0),
    # "compressed", not "measured": the shipped recording now USES the
    # measured clock, so substituting "measured" changed nothing and
    # this case passed while asserting nothing. A case whose new value
    # equals the current one is vacuous, and the guard below makes that
    # a failure rather than a silent pass.
    ("clock", "compressed"),
    ("immune_kills", 1),
    ("usd_kill_markers", 1),
    ("patient_id", "someone_else"),
    ("predictor", "pwm"),
])
def test_changing_what_the_run_was_changes_the_fingerprint(facts, key, value):
    """Each of these means the RTX render is of another simulation."""
    assert facts.get(key) != value, (
        f"this case substitutes {key}={value!r}, which is what the "
        "shipped recording already says -- the case tests nothing. Pick "
        "a value the recording does not have.")
    changed = dict(facts)
    changed[key] = value
    assert dp.run_fingerprint(changed) != dp.run_fingerprint(facts), (
        f"{key} changed and the fingerprint did not: a re-render would "
        "not be demanded when it should be")


@pytest.mark.parametrize("key,value", [
    ("fps", 60),
    ("video_mb", 99.9),
    ("tissue_caption", "reworded"),
    ("patient_headline", "reworded"),
])
def test_recolouring_the_video_does_not_demand_a_re_render(facts, key, value):
    """The opposite failure: a fingerprint that moves on anything at all
    would cry wolf, and the next person would delete the check."""
    changed = dict(facts)
    changed[key] = value
    assert dp.run_fingerprint(changed) == dp.run_fingerprint(facts), (
        f"{key} is cosmetic but moved the fingerprint")


def test_a_new_fact_cannot_quietly_match_an_old_run(facts):
    """A key the recorder does not write yet counts as None, not as
    absent, so adding one re-fingerprints every run from then on."""
    material = dp.fingerprint_material(facts)
    for key in dp.FINGERPRINT_KEYS:
        assert key in material, key


# ── the shipped pair ─────────────────────────────────────────────────

def test_the_rtx_video_ships_with_a_manifest():
    path = ASSETS / dp.RTX_MANIFEST_NAME
    assert path.exists(), (
        f"{dp.RTX_MANIFEST_NAME} is missing. The RTX video on the public "
        "page cannot be shown to be the recorded run without it -- "
        "re-render it and write the manifest (demo_provenance.manifest_for)"
    )


def test_the_two_shipped_videos_are_provably_one_run(facts):
    """THE CLAIM THE PAGE MAKES, checked rather than assumed.

    None here means 'no manifest', which is not agreement. It fails for
    the same reason: the page may not say 'the same run' about a pairing
    nobody recorded.
    """
    verdict = dp.videos_are_one_run(facts, ASSETS)
    manifest = dp.load_rtx_manifest(ASSETS) or {}
    assert verdict is True, (
        "the RTX video was not rendered from the recorded run.\n"
        f"  demo_run.json : {dp.run_fingerprint(facts)}\n"
        f"  rtx_demo.json : {manifest.get('run_fingerprint')}\n"
        "Re-render the RTX video from this run's stage, or stop the page "
        "claiming the two videos are one simulation."
    )


def test_the_manifest_says_what_it_rendered_and_proves_it_moved():
    """A capture that renders one frame 1,681 times looks identical in a
    still. The capture logs DISTINCT n/m for that reason; the manifest
    has to carry it, or the check only exists in a log nobody keeps."""
    manifest = dp.load_rtx_manifest(ASSETS)
    assert manifest, "no manifest"
    frames = manifest.get("frames")
    distinct = manifest.get("distinct_frames")
    assert isinstance(frames, int) and frames > 100, frames
    assert distinct == frames, (
        f"the capture wrote {frames} frames of which {distinct} were "
        "distinct: a stuck render looks exactly like a working one in a "
        "single still, which is how a wrong GIF was published here before"
    )


def test_both_videos_run_for_the_same_length_of_time(facts):
    """The page puts them side by side, so the same instant must be the
    same instant. The RTX capture renders every authored frame, so its
    fps has to be the recording's fps times the snapshot stride."""
    manifest = dp.load_rtx_manifest(ASSETS)
    assert manifest, "no manifest"
    plot_seconds = facts["frames"] / facts["fps"]
    rtx_seconds = manifest["frames"] / manifest["fps"]
    assert rtx_seconds == pytest.approx(plot_seconds, rel=0.02), (
        f"the recording runs {plot_seconds:.1f} s and the RTX video "
        f"{rtx_seconds:.1f} s; a viewer comparing them at a given moment "
        "would be comparing different hours of the run"
    )
