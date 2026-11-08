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

from pathlib import Path

import streamlit as st

import cognisom.dashboard as _dashboard_pkg
from cognisom.dashboard.capabilities import (
    ABSENT, BUILT, LAYERS, PARTIAL, STATUS_MEANING, TARGETS, counts,
)
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

st.title("What Cognisom simulates — and what it doesn't yet")
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
        st.caption("**One run, plotted.** Twenty-four simulated hours of a "
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
m1, m2, m3, m4 = st.columns(4)
m1.metric("Built", n[BUILT])
m2.metric("Partial", n[PARTIAL])
m3.metric("Not yet", n[ABSENT])
m4.metric("External tests passed", 1)
st.markdown(
    "**Built is not validated.** A built row is wired into runs, has a "
    "measured or published basis, and is held by a test. Whether the "
    "simulation *predicts* anything is a separate question, answered in "
    "the Validation section: one falsifiable prediction has held against "
    "human tumours so far, and one pre-specified outcome test did not."
)

# ── Where it is going ────────────────────────────────────────────────

st.subheader("What we are building toward")
for i, (title, body) in enumerate(TARGETS, 1):
    st.markdown(f"**{i}. {title}.** {body}")

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
    lines = ["| | Capability | What it does, and its limits | Evidence |",
             "|---|---|---|---|"]
    for row in layer.rows:
        lines.append(
            f"| {_BADGE[row.status]} | {_cell(row.name)} | "
            f"{_cell(row.note)} | {_evidence(row.evidence)} |")
    st.markdown("\n".join(lines))

st.divider()
st.markdown(
    "The cell data in both videos is written in **Bio-USD**, an open "
    "convention for putting biological cells in OpenUSD files. "
    "[Read the standard and its code examples →](/?page=biousd)"
)
st.caption(
    "Source of this page: `cognisom/dashboard/capabilities.py` in "
    "[github.com/eyedwalker/cognisom](https://github.com/eyedwalker/cognisom). "
    "Apache 2.0."
)
