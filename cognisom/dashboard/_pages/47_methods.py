"""How the simulation works, and how far to trust each part of it. No login.

Why this page exists
--------------------
The demo shows a tumour shrinking and cells dying, and a viewer has no
way to tell which of that is biology, which is an estimate, and which is
a plotted number that drives nothing. A scientist asked, reasonably, to
see how each quantity is computed before believing any of it.

So this page answers four questions in order: what the engine computes
each step, which of it actually changes a run, what every number on the
kill path rests on, and what is known to be wrong. ``docs/simulation_
methods.md`` carries the same material as prose for reading offline;
this page carries the parts a test can check.

Why it reads modules instead of saying things
---------------------------------------------
Rows come from ``cognisom/dashboard/methods.py``, and the signal, kill
and validation tables are read from the engine's own
``signalling.inventory()``, ``kill_calibration.calibration_report()``
and ``capabilities.LAYERS`` rather than retyped, so a number cannot be
right in the engine and wrong on this page.
``tests/test_the_methods_page_matches_the_engine.py`` fails the build if
a module an ordinary run enables has no row, or if a row cites a file
that is not in the repository.

Like the demo and roadmap pages it calls no gate, so it must stay
read-only: it starts no simulation and reads no patient data.
"""

import os

import streamlit as st

from cognisom.dashboard.methods import (
    CHANGES, LIMITATIONS, MODULES, NARRATIVE, OFF, REPORTS_ONLY,
    WIRING_MEANING, kill_rows, modules_by_wiring, signal_rows,
    status_counts, validation_rows,
)
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="Cognisom | How it works",
                     page_icon="\U0001f9ec", layout="wide")

REPO = "https://github.com/eyedwalker/cognisom/blob/main/"

_WIRING_BADGE = {
    CHANGES: ":green-badge[Changes a run]",
    REPORTS_ONLY: ":orange-badge[Reported only]",
    OFF: ":gray-badge[Off by default]",
}

#: Kill-path provenance vocabulary, from KillParameter. An unknown
#: status falls back to a grey badge carrying the raw word rather than
#: being dropped, so a new label shows up on the page instead of
#: silently reading as "fine".
_STATUS_BADGE = {
    "sourced": ":green-badge[Sourced]",
    "derived": ":green-badge[Derived]",
    "calibration-choice": ":orange-badge[Calibration choice]",
    "ESTIMATE": ":orange-badge[Estimate]",
    "placeholder": ":gray-badge[Placeholder]",
}

#: capabilities.py's vocabulary, matching the roadmap page's badges so a
#: reader who has seen one table reads the other the same way.
_CAPABILITY_BADGE = {
    "built": ":green-badge[Built]",
    "partial": ":orange-badge[Partial]",
    "absent": ":gray-badge[Not yet]",
}

#: What each one means, since "derived" and "calibration choice" are not
#: self-explanatory and the difference is the whole point of the table.
_STATUS_MEANING = {
    "sourced": "taken directly from a published measurement, with that "
               "paper's range",
    "derived": "computed from sourced numbers by a stated model, not "
               "fitted",
    "calibration-choice": "chosen so a mechanism reproduces a measured "
                          "behaviour; the choice is the model's, the "
                          "behaviour is not",
    "ESTIMATE": "no measurement found; a plausible value inside a stated "
                "range, labelled so it is never mistaken for one",
    "placeholder": "a stand-in, pending evidence",
}


def _link(path: str) -> str:
    return f"[`{path}`]({REPO}{path})"


def _cell(text: str) -> str:
    # A pipe would end a markdown table cell early.
    return (text or "").replace("|", "/")


# ── Header ───────────────────────────────────────────────────────────

st.title("How the simulation works")
st.markdown(
    "Cognisom simulates prostate tissue one cell at a time, from a "
    "patient's own sequencing. This page is for someone deciding "
    "whether any of it is worth believing: what is computed each step, "
    "**which of it actually changes a result**, what every number rests "
    "on, and what is known to be wrong."
)
st.caption(
    "Cognisom is a research platform. Nothing here is a clinical "
    "finding, a diagnosis, or a treatment recommendation."
)

st.info(
    "**The short version.** Each cell carries its own genome, clock, "
    "receptor state and position. Each step, the vasculature sets what "
    "every cell is fed, the diffusing signals are solved, each cell's "
    "androgen-receptor ODE is integrated, immune cells patrol and "
    "engage what they can recognise, and cells divide or die by gates "
    "whose sources are listed below. Nothing is drawn from a curve fit "
    "to an outcome: the engine has no outcome model, which is also why "
    "it cannot yet predict one."
)

# ── 1. What runs ─────────────────────────────────────────────────────

st.divider()
st.header("1. What the engine computes each step")

_changes = modules_by_wiring(CHANGES)
_reports = modules_by_wiring(REPORTS_ONLY)
_off = modules_by_wiring(OFF)
_inert = (f"{len(_reports)} runs, costs time and reports numbers that "
          "nothing reads back into a cell's fate"
          if len(_reports) == 1 else
          f"{len(_reports)} run, cost time and report numbers that "
          "nothing reads back into a cell's fate")
st.markdown(
    f"An ordinary run registers **{len(_changes) + len(_reports)} "
    f"modules**, which update in order inside one time step. "
    f"**{len(_changes)} of them change what a cell does.** "
    f"{_inert}, and {len(_off)} more can be switched on. That "
    "distinction is on every row, because a plot that moves is not "
    "evidence that anything depends on it."
)
st.markdown(" · ".join(f"{_WIRING_BADGE[w]} {WIRING_MEANING[w]}"
                       for w in (CHANGES, REPORTS_ONLY, OFF)))

lines = ["| Module | What it computes | Code |", "|---|---|---|"]
for row in MODULES:
    lines.append(
        f"| {_WIRING_BADGE[row.wiring]}<br>**{_cell(row.title)}** | "
        f"{_cell(row.computes)} | {_link(row.source)} |")
st.markdown("\n".join(lines))

st.subheader("How each one gets its answer")
st.caption(
    "The part a reader is most likely to want to check, and what to "
    "distrust about it. These were behind expanders first. A methods "
    "page whose method is eleven clicks deep is a methods page nobody "
    "reads, so they are open."
)
for row in MODULES:
    st.markdown(f"#### {row.title} &nbsp;·&nbsp; `{row.key}` "
                f"&nbsp;{_WIRING_BADGE[row.wiring]}")
    if row.worked_example:
        st.markdown(f"**How.** {row.worked_example}")
    if row.caveat:
        st.warning(f"**What to distrust.** {row.caveat}")
    st.caption(f"Code: {_link(row.source)}")

# ── 2. How cells talk to each other ──────────────────────────────────

st.divider()
st.header("2. What diffuses, and who reads it")

signals = signal_rows()
_wired = sum(1 for r in signals if r["Changes a run"] == "yes")
st.markdown(
    f"Cells influence each other through molecules that spread through "
    f"the tissue. The engine solves **{len(signals)} fields** every "
    f"step, and **{_wired} of them change what a cell does**. The rest "
    "are solved and consumed by nothing, which is said here rather "
    "than left out. Oxygen is the one to read carefully: the field "
    "exists, but a cell's oxygen comes from a separate Krogh-cylinder "
    "calculation, not from the field."
)
# A markdown table, not a dataframe: "Made by" and "Read by" are
# sentences, and a dataframe clips them to one line and scrolls the
# rows -- which hides exactly the column this section exists for.
lines = ["| Signal | Made by | Read by | Transport |", "|---|---|---|---|"]
for row in signals:
    badge = (":green-badge[Changes a run]" if row["Changes a run"] == "yes"
             else ":orange-badge[Read by nothing]")
    lines.append(
        f"| {badge}<br>**{_cell(str(row['Signal']))}**<br>"
        f"`{_cell(str(row['Field']))}` | {_cell(str(row['Made by']))} | "
        f"{_cell(str(row['Read by']))} | {_cell(str(row['Transport']))} "
        f"({_cell(str(row['Status']))}) |")
st.markdown("\n".join(lines))
st.caption(
    "Read from " + _link("cognisom/modules/signalling.py") +
    " when the page loads. A test fails the build if the engine solves "
    "a field with no entry in that inventory."
)

# ── 3. Provenance ────────────────────────────────────────────────────

st.divider()
st.header("3. Where the numbers come from")

counts = status_counts()
kills = kill_rows()
st.markdown(
    f"Killing a cancer cell is the simulation's load-bearing claim, so "
    f"every number on that path carries its own source and a plausible "
    f"range. There are **{len(kills)}** of them, and the split is the "
    "honest state of the model:"
)
cols = st.columns(max(len(counts), 1))
for col, (status, n) in zip(cols, sorted(counts.items())):
    col.metric(status.capitalize(), n)
st.markdown("\n".join(
    f"- {_STATUS_BADGE.get(s, ':gray-badge[' + s + ']')} "
    f"— {_STATUS_MEANING.get(s, 'no description for this label')}"
    for s in sorted(counts)))

lines = ["| Parameter | Value | Plausible range | Where it comes from |",
         "|---|---|---|---|"]
for row in kills:
    badge = _STATUS_BADGE.get(str(row["Status"]),
                              f":gray-badge[{row['Status']}]")
    lines.append(
        f"| {badge}<br>**{_cell(str(row['Parameter']))}** | "
        f"{_cell(str(row['Value']))} | "
        f"{_cell(str(row['Plausible range']))} | "
        f"{_cell(str(row['Source']))} |")
st.markdown("\n".join(lines))
st.caption(
    "Read from " +
    _link("cognisom/engine/py/immune/kill_calibration.py") +
    " when the page loads. A range is the published confidence "
    "interval where there is one, and the span the evidence allows "
    "where there is not. An estimate is labelled an estimate; it is "
    "never quietly promoted by being used."
)

st.markdown(
    "**The same discipline runs through the rest of the engine.** Every "
    "cell parameter carries one of five labels — measured, derived, "
    "estimate, model choice, placeholder — and the "
    "[cell page](/?page=cells) shows all of them with their sources."
)

# ── 4. Validation ────────────────────────────────────────────────────

st.divider()
st.header("4. What has been tested against data outside the model")

st.markdown(
    "This is the section that matters most, and it is the shortest. "
    "A mechanism being built, sourced and tested is not evidence that "
    "the model predicts anything; only a falsifiable test against data "
    "the model never saw is."
)
lines = ["| Claim | What it rests on | Evidence |", "|---|---|---|"]
for row in validation_rows():
    ev = ", ".join(_link(p) for p in row["Evidence"]) or "—"
    badge = _CAPABILITY_BADGE.get(str(row["Status"]),
                                  f":gray-badge[{row['Status']}]")
    lines.append(
        f"| {badge}<br>**{_cell(row['Claim'])}** | "
        f"{_cell(row['What it rests on'])} | {ev} |")
st.markdown("\n".join(lines))
st.caption(
    "Read from " + _link("cognisom/dashboard/capabilities.py") +
    ", the same source as the [roadmap page](/?page=roadmap), so the "
    "two cannot disagree about what has been validated."
)

# ── 5. Limitations ───────────────────────────────────────────────────

st.divider()
st.header("5. What is known to be wrong")

st.markdown(
    f"**{len(LIMITATIONS)} things** a reader should know before taking "
    "any output seriously. Each says what would have to change to lift "
    "it, because a limitation with no remedy is an excuse rather than a "
    "plan. They are listed in full rather than folded away: a reader "
    "who has to click ten times to find out what is wrong will not "
    "click."
)
lines = ["| What is wrong | Why it matters | What would lift it |",
         "|---|---|---|"]
for lim in LIMITATIONS:
    lines.append(f"| **{_cell(lim.what)}** | {_cell(lim.why_it_matters)} "
                 f"| {_cell(lim.to_lift_it)} |")
st.markdown("\n".join(lines))

# ── 6. How to check it ───────────────────────────────────────────────

st.divider()
st.header("6. How to check any of this yourself")

st.markdown(
    "Everything above is generated from the code in "
    "[github.com/eyedwalker/cognisom]"
    "(https://github.com/eyedwalker/cognisom) (Apache 2.0). Three "
    "things are worth doing in order:\n\n"
    "1. **Read the narrative.** " + _link(NARRATIVE) + " walks one "
    "cell through one step in prose, with the file and function for "
    "each stage.\n"
    "2. **Read a module.** Every row in section 1 links to the file "
    "that computes it. The docstrings carry the measurements and the "
    "reasoning, including the mistakes — several say what was "
    "previously wrong and how it was found.\n"
    "3. **Run the tests.** `pytest tests -q` runs the suite that holds "
    "every claim on this page. Tests are named for the claim they "
    "hold, so `tests/test_a_macrophage_needs_the_target_marked.py` is "
    "the one that stops a macrophage eating anything unmarked."
)
st.markdown(
    "The cell data in the demo videos is written in **Bio-USD**, an "
    "open convention for putting biological cells in OpenUSD files, so "
    "a run's output can be opened in other tools. "
    "[Read the standard →](/?page=biousd)"
)

# Dated by COMMIT, not a build date: the image's build date falls back to
# today when nothing passes one, so it cannot prove freshness.
_sha = os.environ.get("COGNISOM_GIT_SHA", "")
if _sha and _sha != "dev":
    _stamp = ("as of commit [`" + _sha[:7] + "`]"
              "(https://github.com/eyedwalker/cognisom/commit/" + _sha + ")")
else:
    _stamp = "running from a working tree rather than a tagged commit"
st.caption(
    "This page is generated from `cognisom/dashboard/methods.py` and "
    "read live from the engine's own modules, " + _stamp + ". "
    "[What it simulates, and what it can't yet](/?page=roadmap) · "
    "[The cells](/?page=cells) · [Recorded demo](/?page=demo)"
)
