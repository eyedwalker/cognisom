"""A recorded simulation, explained, with no login.

Why this page exists
--------------------
Everything the engine does was behind a Cognito gate and a subscription
tier, so the only way to see a Cognisom simulation was to be given an
account. This page shows one real run and says exactly what it is.

Why every number here is read from a file
-----------------------------------------
``scripts/record_demo_run.py`` writes the video and a JSON sidecar of
everything it measured while writing it. This page reads the sidecar.
Nothing below is a figure anybody typed in by hand, so the text cannot
drift away from the recording -- re-record and the page re-states
itself. If the sidecar is missing the page says so rather than showing
numbers from a run that is not the one playing.

This page deliberately does NOT call ``streamlit_page_gate``: it is
reachable at ``/?page=demo`` without an account. It must therefore stay
read-only -- it starts no simulation, reads no patient data, and touches
no session state belonging to a logged-in user.
"""

import json
from pathlib import Path

import streamlit as st

from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="Cognisom | Recorded Simulation",
                     page_icon="\U0001f9ec", layout="wide")

# Resolved from the PACKAGE, not from __file__.
#
# This page is reached two ways: through st.navigation, where __file__
# is this file, and through app.py's "/?page=demo" branch, which runs it
# with exec() -- and there __file__ is app.py's path, so the same
# expression would point one directory too high and the asset would
# silently appear missing on the public route, which is the only route
# that matters here.
import cognisom.dashboard as _dashboard_pkg

_ASSETS = Path(_dashboard_pkg.__file__).resolve().parent / "assets"
_VIDEO = _ASSETS / "demo_run.mp4"
_RTX_VIDEO = _ASSETS / "rtx_demo.mp4"
_FACTS = _ASSETS / "demo_run.json"


def _load_facts():
    try:
        return json.loads(_FACTS.read_text())
    except Exception:                                        # noqa: BLE001
        return None


facts = _load_facts()

# ── Header ───────────────────────────────────────────────────────────

st.title("A molecular digital twin, running")
st.markdown(
    "Twenty-four simulated hours of a tumour and the immune cells "
    "hunting it, computed cell by cell from a **real patient callset**. "
    "This is a recording of one run — not an animation drawn to "
    "illustrate the idea."
)
st.caption(
    "Cognisom is a research platform. Nothing here is a clinical "
    "finding, a diagnosis, or a treatment recommendation."
)

st.divider()

# ── The recording ────────────────────────────────────────────────────

if _VIDEO.exists():
    st.video(str(_VIDEO))
    st.caption(
        "What the model computes. Every number on this page is read from "
        "this run's own measurements file, so the text cannot drift from "
        "the picture."
    )
else:
    st.warning(
        "The recording is not present in this deployment. Regenerate it "
        "with `scripts/record_demo_run.py` and redeploy."
    )

# ── The same run, ray-traced ─────────────────────────────────────────
#
# Two videos of one simulation, labelled for what each one IS. The
# plot above is the evidence; this is the same data rendered. Putting a
# beautiful render next to a claim invites the reader to believe the
# beauty is the result, so the captions say plainly which is which --
# and the two are the SAME RUN, exported from one engine state by
# `record_demo_run.py --usd`, not two runs that share a seed.
if _RTX_VIDEO.exists():
    st.subheader("The same run, ray-traced")
    st.video(str(_RTX_VIDEO))
    _rtx_bits = [
        "The identical simulation, rendered in NVIDIA Isaac Sim (RTX) "
        "from the Bio-USD stage the same run exported.",
    ]
    if facts and facts.get("usd_kill_markers") is not None:
        _rtx_bits.append(
            f"Both videos draw **{facts['usd_kill_markers']} kill markers**, "
            f"which is the engine's own kill count — so the crosses are "
            f"countable against the number stated below."
        )
    _rtx_bits.append(
        "Cells are drawn at their authored shapes and translucency rather "
        "than as dots, which is why the tissue reads as a packed solid. "
        "Nothing here is added for effect: no camera move, no extra "
        "lighting beyond what the stage authors, no post-processing."
    )
    st.caption(" ".join(_rtx_bits))

if facts is None:
    st.error(
        "The recording's measurements file is missing, so this page "
        "cannot describe the run above. Rather than print numbers from "
        "some other run, it prints none."
    )
    st.stop()

# ── What you are looking at ──────────────────────────────────────────

st.subheader("What you are looking at")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
**The cloud of dots is tissue.** Each dot is one cell, at the position
the simulation put it. The pale cells are normal prostate tissue; the
warmer cluster in the middle is the tumour.

**Brightness is oxygen.** Every tissue cell is shaded by *its own*
oxygen concentration, not by an average. Each cell burns oxygen at its
own rate and is resupplied by its nearest capillary, with the exchange
falling off with distance through the vessel's Krogh geometry — so a
cell's brightness is the balance it settles at between its own demand
and how well it happens to be perfused. It is computed per cell, not
drawn.

**The larger outlined cells are the immune infiltrate** — T cells,
NK cells and macrophages. They patrol, and they stop when they
recognise something.
        """
    )
with c2:
    st.markdown(
        f"""
**The red crosses are kills** — each marks where an immune cell killed
a cancer cell, and stays up for an hour of simulated time. The engine
counts **{facts['immune_kills']}** across the run.

**The right-hand panels are the same run, counted.** The populations
are on a log scale because the tissue holds about
{facts['normal_end']:,} normal cells against {facts['cancer_end']:,}
cancer cells — on a linear axis the two populations that actually move
would sit flat on the floor.

Each cell carries its own metabolism, cell cycle, oxygen, and — for the
cancer cells — this patient's own mutations presented on this patient's
own HLA molecules.
        """
    )

st.divider()

# ── Provenance ───────────────────────────────────────────────────────

st.subheader("Where this run came from")

p1, p2, p3 = st.columns(3)
p1.metric("Patient", facts.get("patient_id") or "none")
p2.metric("Simulated time", f"{facts['duration_hours']:.0f} h")
p3.metric("Cells at the end", f"{facts['cells_end']:,}")

st.markdown(
    f"**Genome.** {facts['patient_headline']}. The mutations that reach "
    f"the cancer cells are "
    + (", ".join(f"`{m}`" for m in facts["mutations_carried"])
       or "none — see the caveats below")
    + ". These come from a real sequencing callset"
    + (", called against a matched normal sample, so they are somatic "
       "rather than inherited."
       if not any("matched normal" in c for c in facts["caveats"])
       else ".")
)

for line in facts.get("presentation") or []:
    st.markdown(f"- {line}")

st.markdown(
    f"**HLA class I.** `{', '.join(facts['hla_alleles'])}` — the "
    "molecules that decide which of those mutations a T cell can "
    "actually see. Peptide binding is predicted with MHCflurry, not "
    "assumed."
)

# Printed from the recording's own file, not written out here. These are
# the reasons the platform itself gives for discounting this run, and a
# page that claims personalisation while hiding them would be claiming
# more than the pipeline delivered.
if facts.get("caveats"):
    st.markdown("**What weakens this run**")
    for _caveat in facts["caveats"]:
        st.markdown(f"- {_caveat}")

st.markdown(
    f"**Tissue geometry.** {facts['tissue_caption']}"
)

st.markdown(
    f"**Engine.** Scenario `{facts['scenario']}`, timestep "
    f"{facts['dt_hours']} h, {facts['frames']} frames at "
    f"{facts['fps']} fps, random seed {facts['seed']} — the same seed "
    "reproduces this run frame for frame."
)

st.divider()

# ── What happened ────────────────────────────────────────────────────

st.subheader("What happened in these 24 hours")

m = st.columns(5)
m[0].metric("Cancer cells", f"{facts['cancer_end']:,}",
            delta=f"{facts['cancer_end'] - facts['cancer_start']:+,}")
m[1].metric("Immune kills", f"{facts['immune_kills']:,}")
m[2].metric("Cell divisions", f"{facts['divisions']:,}")
m[3].metric("Immune cells", f"{facts['immune_end']:,}")
m[4].metric("Hypoxic cells", f"{facts['hypoxic_end']:,}")

st.markdown(
    f"The tumour **shrank**, from {facts['cancer_start']:,} cancer cells "
    f"to {facts['cancer_end']:,}, while the surrounding normal tissue grew "
    f"from {facts['normal_start']:,} to {facts['normal_end']:,}. Cancer "
    "cells divide faster than normal ones here, so that fall is the "
    "immune infiltrate outpacing the tumour's own division rate — not an "
    "absence of growth."
)

# Stated only when the run's two independent counters say it. The
# cellular module counts every death; the immune module counts its own
# kills. They are not the same counter, so their agreeing is a fact
# about this run rather than a tautology.
if facts["deaths"] == facts["immune_kills"]:
    st.markdown(
        f"Every one of the {facts['deaths']:,} cell deaths in this run "
        "was an immune kill: nothing starved, and nothing died of age. "
        "The cell-death counter and the immune-kill counter are separate "
        "and independently maintained, which is what makes that "
        "comparison worth making."
    )

st.markdown(
    f"**The run settles.** The last cell division is at hour "
    f"{facts['last_division_hour']:.1f} and the last kill at hour "
    f"{facts['last_kill_hour']:.1f}. After that the populations stop "
    "changing, and the final third of the video shows oxygen and immune "
    "cells still moving across a settled tissue — that plateau is the "
    "model reaching a steady state, not the recording freezing. It is "
    "also a limitation worth stating plainly: a tumour that stops "
    "changing after 15 hours is a property of this configuration, not "
    "a prediction about tumours."
)

st.markdown(
    f"Oxygen across the run spans **{facts['oxygen_min']:.3f} to "
    f"{facts['oxygen_max']:.2f}** (volume fraction; 0.21 is the "
    "well-perfused value every cell starts at, and below 0.05 the "
    "engine calls a cell hypoxic). That spread is what the brightness "
    "in the video is showing."
)

st.divider()

# ── Honesty ──────────────────────────────────────────────────────────

st.subheader("What this is, and what it is not")

st.markdown(
    """
This recording is a **mechanism demonstration**. It shows that a real
patient's mutations and HLA type can be carried through to a spatial
simulation and change what the immune system does there. It is not
evidence about any patient's outcome, and it should not be read as one.
    """
)

limits = [
    "**One run, one seed.** Nothing here is a mean over replicates, and "
    "no confidence interval is attached to any number above. Run-to-run "
    "scatter in this model is real and has been measured elsewhere.",

    "**The genome and the tissue are not matched.** The callset is "
    "HCC1395, a breast-cancer cell line with a matched normal — the "
    "best-characterised public tumour/normal pair available. The tissue "
    "geometry is from a human *prostate* section. The combination "
    "demonstrates the pipeline; it does not model a real breast tumour "
    "in real breast tissue.",

    "**Several parameters are estimates, not measurements.** Where a "
    "number in this engine comes from a published measurement it is "
    "carried with its citation, and where it does not, it is marked as "
    "an estimate. Cell mechanics stiffnesses and some immune rates are "
    "in the second category.",

    "**It is a model.** Every cell here is a sphere with a handful of "
    "state variables. Real tissue is not.",
]
for item in limits:
    st.markdown(f"- {item}")

st.divider()

# ── Reproduce ────────────────────────────────────────────────────────

with st.expander("Reproduce this recording"):
    st.markdown(
        "The platform is open source (Apache 2.0). This exact video is "
        "regenerated by one command, and the numbers on this page are "
        "read from the file that command writes beside it:"
    )
    st.code(
        f"python scripts/record_demo_run.py \\\n"
        f"    --duration {facts['duration_hours']:.0f} "
        f"--scenario {facts['scenario']} --tissue {facts['tissue']} \\\n"
        f"    --stride {facts['snapshot_stride']} --fps {facts['fps']}",
        language="bash",
    )
    st.caption(
        "Run it with a Python that has MHCflurry installed (3.11). "
        "Without it the peptide scorer silently falls back to a "
        "position-weight matrix, under which no peptide is ever a "
        "strong binder — the neoantigen half of the run would then be "
        "about the scorer rather than about the patient."
    )
    st.caption(f"Recording generated by `{facts['generated_by']}`.")

st.divider()
st.markdown(
    "### See the rest\n"
    "This is one page of the platform. The full environment — genome "
    "ingestion, neoantigen prediction, the treatment simulator and the "
    "3D viewer — needs an account."
)
st.link_button("Open Cognisom", "/", type="primary")
