"""What Cognisom simulates today, and what it is built toward. No login.

Why this page exists
--------------------
The recorded demo shows one run. It cannot show what the model leaves
out, and a viewer who sees a convincing simulation will assume it
includes things it does not -- a lymph node, a fibroblast, a drug
reaching the tissue. This page says, row by row, what is built, what is
partial and what is absent, and links each built row to the test in the
public repository that holds it.

Why it reads a module instead of saying things
----------------------------------------------
Every row comes from ``cognisom/dashboard/capabilities.py``, and
``tests/test_the_roadmap_cites_its_evidence.py`` fails the build if a row
marked built cites no evidence or cites a file that does not exist. The
page cannot claim more than the repository shows.

Like the demo page it calls no gate, so it must stay read-only: it
starts no simulation and reads no patient data.
"""

import os
from pathlib import Path

import streamlit as st

import cognisom.dashboard as _dashboard_pkg
from cognisom.dashboard.capabilities import (
    ABSENT, BUILT, LAYERS, OUTCOME_MODEL_NOTE, PARTIAL, SCALE_NOTE,
    STATUS_MEANING, TARGETS, counts,
)
from cognisom.dashboard.demo_provenance import duration_caption, load_facts
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="Cognisom | What it simulates",
                     page_icon="\U0001f9ec", layout="wide")

# Resolved from the package, not __file__: app.py runs this page with
# exec() on the public route, where __file__ is app.py's path.
_ASSETS = Path(_dashboard_pkg.__file__).resolve().parent / "assets"
_VIDEO = _ASSETS / "demo_run.mp4"
_RTX_VIDEO = _ASSETS / "rtx_demo.mp4"

REPO = "https://github.com/eyedwalker/cognisom/blob/main/"

_BADGE = {
    BUILT: ":green-badge[Built]",
    PARTIAL: ":orange-badge[Partial]",
    ABSENT: ":gray-badge[Not yet]",
}

# ── Header ───────────────────────────────────────────────────────────

st.title("What Cognisom simulates, and what it can't yet")
st.markdown(
    "The goal is a simulation of human tissue that responds to cancer and "
    "to immunotherapy the way a real patient's tumour does: built from "
    "the patient's own genome, cell by cell, with every number traceable "
    "to a measurement. This page is the honest distance between that goal "
    "and what runs today."
)
st.caption(
    "Cognisom is a research platform. Nothing here is a clinical finding, "
    "a diagnosis, or a treatment recommendation."
)

# ── The videos ───────────────────────────────────────────────────────

v1, v2 = st.columns(2)
with v1:
    if _VIDEO.exists():
        st.video(str(_VIDEO))
        # From the recording's sidecar, not typed here: this said
        # "Twenty-four simulated hours" for a day after the demo was
        # re-recorded as a simulated week.
        _dur = duration_caption(load_facts(_ASSETS))
        _lead = f"{_dur} of a" if _dur else "A recorded run of a"
        st.caption(f"**One run, plotted.** {_lead} "
                   "tumour and its immune infiltrate, with the counts "
                   "beside it.")
with v2:
    if _RTX_VIDEO.exists():
        st.video(str(_RTX_VIDEO))
        st.caption("**The same run, ray-traced** in NVIDIA Isaac Sim from "
                   "the Bio-USD file the run exported.")
st.markdown("[What am I looking at? The recorded demo, explained "
            "→](/?page=demo)")

st.divider()

# ── Summary ──────────────────────────────────────────────────────────

n = counts()
# Deliberately NOT four bare numbers side by side. "Built 35" next to
# "External tests passed 1" invites a comparison between two quantities
# that are not on the same scale -- one counts mechanisms, the other
# counts falsifiable predictions tested against outside data -- so each
# count is stated together with what it means.
st.markdown(
    "- **" + str(n[BUILT]) + " built** \u2014 wired into runs, with a "
    "measured or published basis, and held by a test named in the "
    "Evidence column below.\n"
    "- **" + str(n[PARTIAL]) + " partial** \u2014 present but limited: an "
    "estimate, an optional feature, or a mechanism that exists and does "
    "not yet change outcomes.\n"
    "- **" + str(n[ABSENT]) + " not yet** \u2014 not modelled.\n\n"
    "**Built is not validated, and these are not comparable counts.** "
    "Whether the simulation *predicts* anything is a separate question, "
    "answered in the Validation section below: **one** falsifiable "
    "prediction has held against outside human data, and one "
    "pre-specified outcome test was null."
)

# ── Where it is going ────────────────────────────────────────────────

st.info(OUTCOME_MODEL_NOTE)

st.subheader("What we are building toward")
st.caption(
    "In order, with what each one needs first. The sequencing is the "
    "plan; the tables below are the inventory it is drawn from."
)
for i, (title, body) in enumerate(TARGETS, 1):
    st.markdown(f"**{i}. {title}.** {body}")

st.markdown(SCALE_NOTE)

st.divider()

# ── The map ──────────────────────────────────────────────────────────

st.subheader("Everything, layer by layer")
st.markdown(
    " · ".join(f"{_BADGE[s]} {STATUS_MEANING[s]}"
               for s in (BUILT, PARTIAL, ABSENT))
)
st.caption(
    "Each built row links to the test, or the analysis script, in the "
    "public repository that holds it. A test fails the build if a built "
    "row cites nothing."
)


def _evidence(paths):
    return ", ".join(f"[{Path(p).name}]({REPO}{p})" for p in paths) or "—"


def _cell(text: str) -> str:
    # A pipe would end the table cell early.
    return (text or "").replace("|", "/")


for layer in LAYERS:
    st.markdown(f"#### {layer.title}")
    st.caption(layer.question)
    # Three columns rather than four, with the badge above the name in
    # the same cell: a four-column table of prose is unreadable at phone
    # width, and the status is what a reader scans for first.
    lines = ["| Capability | What it does, and its limits | Evidence |",
             "|---|---|---|"]
    for row in layer.rows:
        lines.append(
            f"| {_BADGE[row.status]}<br>**{_cell(row.name)}** | "
            f"{_cell(row.note)} | {_evidence(row.evidence)} |")
    st.markdown("\n".join(lines))

st.divider()
st.markdown(
    "The cell data in both videos is written in **Bio-USD**, an open "
    "convention for putting biological cells in OpenUSD files. "
    "[Read the standard and its code examples →](/?page=biousd)"
)
# Dated by COMMIT, not by a build date: the image's build date falls back
# to today's date when nothing passes one, so it cannot prove freshness.
# The commit is baked in and can be checked against the public repository,
# which is what lets a reader watch this audit move over time.
_sha = os.environ.get("COGNISOM_GIT_SHA", "")
if _sha and _sha != "dev":
    _stamp = ("as of commit [`" + _sha[:7] + "`]"
              "(https://github.com/eyedwalker/cognisom/commit/" + _sha + ")")
else:
    _stamp = "running from a working tree rather than a tagged commit"
st.caption(
    "This audit is generated from `cognisom/dashboard/capabilities.py`, "
    + _stamp + ", in [github.com/eyedwalker/cognisom]"
    "(https://github.com/eyedwalker/cognisom) (Apache 2.0). Every Built "
    "row's Evidence column links to the test that holds it at that "
    "commit, so the audit can be re-checked, and its movement over time "
    "can be read from the repository's history."
)
