"""
The cells — what is simulated, how big each is, and how it is modelled
======================================================================

A public page (``?page=cells``). One section per cell type the engine
runs: what the cell does in the body, how the engine represents that,
every number it uses with a plain status (measured, derived, estimate,
model choice, placeholder), and what is missing.

The numbers are not typed here. ``cognisom.dashboard.cell_catalog`` reads
them from the modules the engine runs, so this page cannot drift from the
simulation. It starts no simulation and reads no patient data.
"""

import streamlit as st

from cognisom.dashboard.cell_catalog import (
    NOT_CELLS, STATUS_MEANING, STATUS_ORDER, build_catalog, status_counts,
)
from cognisom.dashboard.page_config import safe_set_page_config
from cognisom.modules import signalling

safe_set_page_config(page_title="Cognisom | The cells",
                     page_icon="\U0001f9ec", layout="wide")

_BADGE = {
    "measured": ":green-badge[Measured]",
    "derived": ":green-badge[Derived]",
    "estimate": ":orange-badge[Estimate]",
    "model choice": ":blue-badge[Model choice]",
    "placeholder": ":gray-badge[Placeholder]",
}


@st.cache_resource(show_spinner="Reading the cell models…")
def _catalog():
    return build_catalog()


def _swatch(colour) -> str:
    r, g, b = (int(round(255 * c)) for c in colour)
    return (f"<span style='display:inline-block;width:0.85em;height:0.85em;"
            f"border-radius:50%;background:rgb({r},{g},{b});"
            f"margin-right:0.45em;vertical-align:baseline'></span>")


entries = _catalog()
total = sum(e.count for e in entries)

# ── Header ───────────────────────────────────────────────────────────

st.title("The cells in the simulation")
st.markdown(
    f"Cognisom simulates tissue one cell at a time. There are "
    f"**{len(entries)} kinds of cell**, and this page says for each one "
    "how big it is, what it does in the body, how the simulation "
    "represents that, and where every number comes from."
)
st.caption(
    "Cognisom is a research platform. Nothing here is a clinical finding, "
    "a diagnosis, or a treatment recommendation."
)

# ── At a glance ──────────────────────────────────────────────────────

st.subheader("At a glance")
st.markdown(
    f"Counts are for the default run: {total:,} cells on a tissue layout "
    "measured from a human prostate section."
)
rows = []
for e in entries:
    rows.append({
        "Cell": e.name,
        "Diameter (µm)": e.diameter_um,
        "Size is": e.diameter_status.capitalize(),
        "Cells in a default run": e.count,
        "Share of all cells": e.count / total,
        "Role": {"normal": "The tissue", "cancer": "The tumour",
                 "T_cell": "Specific killer", "NK_cell": "Innate killer",
                 "macrophage": "Engulfs; helps or hinders"}[e.key],
    })
st.dataframe(
    rows, hide_index=True, use_container_width=True,
    column_config={
        "Diameter (µm)": st.column_config.NumberColumn(format="%.1f"),
        "Share of all cells": st.column_config.NumberColumn(format="percent"),
    },
)

# ── How to read the status labels ────────────────────────────────────

counts = status_counts(entries)
n_values = sum(counts.values())
st.subheader("How much of this is measured")
st.markdown(
    f"The five cell types use **{n_values} numbers** between them. Each "
    "carries one of five labels, and the split is the honest state of "
    "the model:"
)
cols = st.columns(len(STATUS_ORDER))
for col, status in zip(cols, STATUS_ORDER):
    col.metric(status.capitalize(), counts[status])
for status in STATUS_ORDER:
    st.markdown(f"{_BADGE[status]} {STATUS_MEANING[status]}")

# ── One section per cell ─────────────────────────────────────────────

for e in entries:
    st.divider()
    st.markdown(f"## {_swatch(e.colour)}{e.name}", unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("Diameter", f"{e.diameter_um:g} µm")
    m[1].metric("In a default run", f"{e.count:,}")
    m[2].metric("Share of all cells", f"{e.count / total:.1%}")
    m[3].metric("Drawn as", e.shape.split(" (")[0].replace("_", " "))
    st.caption(
        f"Size: {e.diameter_status} ({e.diameter_source.rstrip('.')}). "
        f"Drawn shape: {e.shape.replace('_', ' ')}."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**What it does in the body**")
        st.markdown(e.does)
    with right:
        st.markdown("**How we simulate it**")
        st.markdown(e.simulated)

    st.markdown("**The numbers it runs on**")
    # A static table, not a dataframe: the sources are sentences and a
    # dataframe clips them to one line and scrolls the rows.
    st.table(
        [{"Quantity": p.name, "Value": p.value,
          "Status": p.status.capitalize(), "Where it comes from": p.source}
         for p in e.parameters])

    st.markdown("**What is missing or weak**")
    st.markdown("\n".join(f"- {item}" for item in e.missing))

# ── How cells talk to each other ─────────────────────────────────────

st.divider()
st.subheader("How the cells talk to each other")
signals = signalling.inventory()
in_use = sum(1 for entry in signals if entry.wired)
st.markdown(
    f"Cells influence each other through molecules that spread through "
    f"the tissue. The simulation carries **{len(signals)} of these**, and "
    f"{in_use} of them change what a cell does in a default run. The "
    "others are calculated and currently read by nothing, which is said "
    "here rather than left out."
)
st.table(
    [{"Signal": entry.molecule,
      "Made by": entry.made_by,
      "Read by": entry.read_by,
      "What it changes": entry.effect,
      "In use": "Yes" if entry.wired else "No"}
     for entry in signals])
st.markdown("**Signals a real tumour uses that are not simulated yet**")
st.table(
    [{"Signal": name, "What it does, and what stands in for it today": what}
     for name, what in signalling.NOT_YET])

# ── What is not a cell ───────────────────────────────────────────────

st.divider()
st.subheader("What is not simulated as a cell")
st.markdown(
    "Several things a tumour contains are either represented some other "
    "way or not at all."
)
st.table(
    [{"In a real tumour": name, "In the simulation": how}
     for name, how in NOT_CELLS])

st.caption(
    "Every value on this page is read from the simulation's own code when "
    "the page loads; none is typed into the page. "
    "[What it simulates, and what it can't yet](/?page=roadmap) · "
    "[Recorded demo](/?page=demo)"
)
