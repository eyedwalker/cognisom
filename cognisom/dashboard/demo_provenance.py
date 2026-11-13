"""The two demo videos claim to be one run. This is what checks it.

The hole this closes
--------------------
The public demo page shows a matplotlib recording beside an RTX render
and says they are the same simulation. The recording and its sidecar are
written together by ``record_demo_run.py``, so they cannot disagree. The
RTX video is made somewhere else entirely: the run exports a USD stage,
the stage is copied to a GPU box, Isaac Sim renders it to PNGs and
ffmpeg encodes them. Nothing carried the run's identity along that path.

So re-recording the 2D video without re-rendering the RTX one left a
page pairing two different runs, and the existing test could not see it:
it compares the sidecar's ``usd_kill_markers`` against the engine's kill
count, and BOTH of those come from the new run. The test passes while
the page's central claim quietly becomes false. A test that goes green
on a false page is worse than no test, and this repository has been
bitten by that shape more than once.

What this adds is a fingerprint. ``run_fingerprint`` hashes the facts
that define a run -- its seed, its clock, its patient, its treatment,
what the engine counted -- from the sidecar the recorder already writes.
The RTX side records the same fingerprint in a manifest beside its video
at the moment it renders. If the two ever differ, the page says so and
the test fails, naming both.

Why the fingerprint is derived from the sidecar rather than written by
the recorder: the recorder is often mid-run when this is needed, and a
format change would mean re-recording. Deriving it means any existing
sidecar can be fingerprinted after the fact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

#: The facts that make a run THIS run. Ordered, so the hash is stable,
#: and deliberately narrow: anything here that changes means the RTX
#: render is of a different simulation and must be redone. Cosmetic
#: fields -- fps, video_mb, colours, captions -- are NOT here, because
#: re-encoding a video does not make it another run.
FINGERPRINT_KEYS = (
    "seed",
    "scenario",
    "tissue",
    "duration_hours",
    "dt_hours",
    "patient_id",
    "clock",
    "tumour_fraction",
    "reactive_t_fraction",
    "cells_start",
    "cancer_start",
    # What the engine counted. Two runs with identical settings and
    # different outcomes are different runs, and this is what makes the
    # fingerprint depend on the simulation and not only on its config.
    "immune_kills",
    "divisions",
    "deaths",
    "usd_kill_markers",
    "predictor",
)

#: Filename of the manifest written beside the RTX video.
RTX_MANIFEST_NAME = "rtx_demo.json"


def _normalise(value: Any) -> Any:
    """Make a value hash-stably, without being clever about it."""
    if isinstance(value, float):
        # The sidecar is JSON, so a float has already been through a
        # round trip; round to a fixed precision so re-reading the same
        # file cannot change the hash.
        return round(value, 9)
    if isinstance(value, dict):
        return {k: _normalise(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value


def fingerprint_material(facts: Dict[str, Any]) -> Dict[str, Any]:
    """Exactly what the fingerprint is taken over, for error messages.

    A missing key is recorded as None rather than skipped, so that
    adding a field to the recorder changes the fingerprint for every
    run from then on instead of silently matching older ones.
    """
    material = {key: _normalise(facts.get(key)) for key in FINGERPRINT_KEYS}
    # The treatment, flattened to the arms and drugs that were run. The
    # whole treatment dict carries notes and prose that may be reworded
    # without the run changing.
    arms = facts.get("treatment_arms") or []
    material["arms"] = [
        {"drugs": sorted(a.get("drugs") or []),
         "recorded": bool(a.get("recorded"))}
        for a in sorted(arms, key=lambda a: sorted(a.get("drugs") or []))
    ]
    sub = facts.get("subclone") or None
    material["subclone"] = (
        None if not sub
        else {"label": sub.get("label"),
              "founding_fraction": _normalise(sub.get("founding_fraction"))}
    )
    return material


def run_fingerprint(facts: Dict[str, Any]) -> str:
    """A short, stable id for the run a sidecar describes."""
    material = fingerprint_material(facts)
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_rtx_manifest(assets_dir: Path) -> Optional[Dict[str, Any]]:
    """The RTX video's manifest, or None when it was never written."""
    path = Path(assets_dir) / RTX_MANIFEST_NAME
    try:
        return json.loads(path.read_text())
    except Exception:                                        # noqa: BLE001
        return None


def videos_are_one_run(facts: Dict[str, Any],
                       assets_dir: Path) -> Optional[bool]:
    """Whether the RTX video was rendered from the recorded run.

    None means UNKNOWN -- there is no manifest, so the question cannot
    be answered. The caller must not read that as agreement: an unknown
    pairing is exactly the state this module exists to stop shipping
    silently.
    """
    manifest = load_rtx_manifest(assets_dir)
    if not manifest:
        return None
    recorded = manifest.get("run_fingerprint")
    if not recorded:
        return None
    return bool(recorded == run_fingerprint(facts))


def load_facts(assets_dir: Path) -> Optional[Dict[str, Any]]:
    """The recorded run's sidecar, or None when it is not there."""
    path = Path(assets_dir) / "demo_run.json"
    try:
        return json.loads(path.read_text())
    except Exception:                                        # noqa: BLE001
        return None


def duration_caption(facts: Optional[Dict[str, Any]]) -> str:
    """How long the recorded run covers, in words, from the sidecar.

    Why this is a function and not a sentence in a page
    ---------------------------------------------------
    Three public places describe this video: the landing page, the
    roadmap page and the demo page. Two of them said "Twenty-four
    simulated hours" for a day after the demo was re-recorded as a
    simulated WEEK, because the sentence was typed into each page by
    hand and only the demo page was taught to read the sidecar.

    Nobody noticed, because a wrong duration on a video still looks
    like a video. So the words come from the recording's own facts, and
    a run of any length describes itself correctly.

    Returns "" when there is no sidecar, so a caller can leave the
    claim out entirely rather than guess.
    """
    hours = (facts or {}).get("duration_hours")
    try:
        hours = float(hours)
    except (TypeError, ValueError):
        return ""
    if hours <= 0:
        return ""
    if hours < 48:
        return f"{hours:g} simulated hours"
    days = hours / 24.0
    if abs(days - round(days)) < 0.05:
        days = int(round(days))
        if days == 7:
            return "A simulated week"
        return f"{days} simulated days"
    return f"{days:.1f} simulated days"


def manifest_for(facts: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """The manifest to write beside a freshly rendered RTX video."""
    payload: Dict[str, Any] = {
        "run_fingerprint": run_fingerprint(facts),
        "fingerprint_keys": list(FINGERPRINT_KEYS),
        "of_stage": facts.get("usd_stage"),
        "note": (
            "Written when the RTX video was rendered. If "
            "run_fingerprint stops matching demo_run.json, the two "
            "videos on the public page are of different runs and the "
            "page must stop saying otherwise."
        ),
    }
    payload.update(extra)
    return payload
