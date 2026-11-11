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
    if facts and facts.get("subclone"):
        # The stage draws cells by type, and a carrier is the same type
        # as its neighbour. Said, so nobody looks for rings here.
        _rtx_bits.append(
            "This view draws cells by type, so it does not mark which "
            "cancer cells carry the resistance variant; the yellow rings "
            "are in the first video only."
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

# Whether the typing is THIS patient's is the difference between a
# personalised run and a demonstration, so the page states which it is
# from the recording's own flag rather than leaving a reader to assume.
# It read as a population default until 2026-11-09, not because the
# typing was missing -- it was in the repository all along -- but because
# the lookup was keyed "SEQC2" and the run asked for "HCC1395". The
# caveat printed below was describing a dictionary-key mismatch as a
# limitation of the data.
_hla_line = (
    "measured from this patient's own sequencing reads (OptiType on the "
    "matched normal), homozygous at all three loci"
    if facts.get("hla_measured") else
    "a population-frequency default, NOT this patient's genotype"
)
st.markdown(
    f"**HLA class I.** `{', '.join(facts['hla_alleles'])}` — "
    f"{_hla_line}. These are the molecules that decide which of those "
    "mutations a T cell can actually see, and peptide binding to them is "
    "predicted with MHCflurry, not assumed."
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

st.subheader(f"What happened in these {facts['duration_hours']:.0f} hours")

m = st.columns(5)
m[0].metric("Cancer cells", f"{facts['cancer_end']:,}",
            delta=f"{facts['cancer_end'] - facts['cancer_start']:+,}")
m[1].metric("Immune kills", f"{facts['immune_kills']:,}")
m[2].metric("Cell divisions", f"{facts['divisions']:,}")
m[3].metric("Immune cells", f"{facts['immune_end']:,}")
m[4].metric("Hypoxic cells", f"{facts['hypoxic_end']:,}")

# Said only if the recording says it: a re-record with another patient or
# drug must not inherit a sentence about a tumour that shrank.
if facts["cancer_end"] < facts["cancer_start"]:
    st.markdown(
        f"The tumour **shrank**, from {facts['cancer_start']:,} cancer "
        f"cells to {facts['cancer_end']:,}, while the surrounding normal "
        f"tissue went from {facts['normal_start']:,} to "
        f"{facts['normal_end']:,}. "
        # With a subclone on a drug, most cancer cells are blocked and
        # barely dividing, so "outpacing the division rate" would credit
        # the immune system with what the drug is doing.
        + ("The deaths are immune kills; the drug is what keeps most of "
           "the tumour from replacing them, by blocking the cells with "
           "the ordinary receptor. The carriers are not blocked, which is "
           "the section below."
           if facts.get("subclone") else
           "Cancer cells divide faster than normal ones here, so that "
           "fall is the immune infiltrate outpacing the tumour's own "
           "division rate — not an absence of growth.")
    )
else:
    st.markdown(
        f"The tumour went from {facts['cancer_start']:,} cancer cells to "
        f"{facts['cancer_end']:,}; the surrounding normal tissue from "
        f"{facts['normal_start']:,} to {facts['normal_end']:,}."
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
    f"changing after {facts['last_kill_hour']:.0f} hours is a property of "
    "this configuration, not a prediction about tumours."
)

# ── The drug ─────────────────────────────────────────────────────────
#
# Shown only when the recording was made with a drug, and written from
# the recording's own arms. The video is ONE run; the other arms are the
# same patient, seed and tissue with a different drug choice, and appear
# in the video only as lines in the PSA panel.
_arms = facts.get("treatment_arms") or []
_recorded = next((a for a in _arms if a.get("recorded")), None)
if _recorded:
    st.divider()
    st.subheader("The drug, and this patient's own receptor")
    _sub = facts.get("subclone")
    if _sub:
        # ── a resistant subclone, and a drug that selects for it ─────
        from cognisom.genomics.demo_patients import (
            SU2C_AR_F877L_CLONALITY as _clonality)
        _variant = _sub["label"].replace("_", " ")
        _none = next((a for a in _arms if not a["drugs"]), None)
        st.markdown(
            f"In this run the patient is given "
            f"**{' + '.join(_recorded['drugs'])}**, and the resistance "
            f"variant **{_variant}** starts in "
            f"**{_sub['founding_fraction']:.0%}** of the cancer cells "
            "rather than all of them. The drug then does two opposite "
            "things in one tumour: it blocks the cells with the ordinary "
            "receptor and drives the ones carrying the variant. The video "
            "rings the carriers in yellow; the two lower panels plot what "
            "the whole tumour puts out (PSA, the protein a clinic follows) "
            "and what share of it the carriers have become."
        )
        for _note in _recorded.get("notes") or []:
            st.markdown(f"- {_note}")

        st.table([
            {"Arm": a["label"] + (" (the video)" if a["recorded"] else ""),
             "Tumour PSA, start": f"{a['psa_tumour_start']:,.0f}",
             "Lowest (at hour)": (f"{a['psa_tumour_low']:,.0f} "
                                  f"({a['psa_tumour_low_hour']:.0f} h)"),
             "End": f"{a['psa_tumour_end']:,.0f}",
             "Carriers, start": f"{a['carrier_share_start']:.0%}",
             "Carriers, end": f"{a['carrier_share_end']:.0%}",
             "Cancer cells at the end": a["cancer_end"]}
            for a in _arms
        ])

        _start = _recorded["psa_tumour_start"]
        _low = _recorded["psa_tumour_low"]
        _end = _recorded["psa_tumour_end"]
        if _start > 0 and _low < 0.9 * _start and _end > 1.5 * _low:
            st.markdown(
                f"On {' + '.join(_recorded['drugs'])} the tumour's PSA "
                f"**falls {1 - _low / _start:.0%}** by hour "
                f"{_recorded['psa_tumour_low_hour']:.0f} and then **climbs "
                f"to {_end / _low:.1f}x its lowest point**"
                + (", past where it started" if _end > _start else
                   f", {_end / _start:.0%} of where it started and still "
                   "rising when the run ends")
                + f", while the carriers go from "
                f"{_recorded['carrier_share_start']:.0%} to "
                f"{_recorded['carrier_share_end']:.0%} of the tumour. A "
                "response, then progression on the same drug: the drug has "
                "not stopped working on the cells it worked on, the tumour "
                "has become a different population."
            )
        for a in _arms:
            if a is _recorded or not a["drugs"]:
                continue
            if (a["psa_tumour_end"] < 0.5 * a["psa_tumour_start"]
                    and abs(a["carrier_share_end"]
                            - a["carrier_share_start"]) < 0.15):
                st.markdown(
                    f"**{a['label']}** blocks both kinds of cell in the "
                    f"model: tumour PSA ends at "
                    f"{a['psa_tumour_end'] / a['psa_tumour_start']:.0%} of "
                    f"its start and the carriers stay near "
                    f"{a['carrier_share_end']:.0%}, so nothing is selected. "
                    "That drug-specific difference is sourced (Moilanen "
                    "2015, PMID 26137992)."
                )
        if _none is not None:
            st.markdown(
                f"With no drug the carriers end at "
                f"{_none['carrier_share_end']:.0%}. Both kinds of cell grow "
                "alike untreated, so any movement there is the chance of "
                "which few cells the immune system happened to kill, not "
                "selection."
            )

        st.markdown("**Where the starting share comes from: it is a "
                    "scenario, not a reading.**")
        st.markdown(
            f"This patient's biopsy read the variant at an allele fraction "
            f"of {_clonality['vaf']:.3f}. That is not a "
            f"{_clonality['vaf']:.0%} subclone: his AR gene is amplified, "
            f"about {_clonality['ar_copies_per_tumour_cell']:.1f} copies "
            f"per tumour cell at a sample purity of "
            f"{_clonality['purity']:.2f}, so the fraction counts gene "
            f"copies, not cells. What follows from his data is only that "
            f"between **{_clonality['carrier_fraction_min']:.0%} and "
            f"{_clonality['carrier_fraction_max']:.0%}** of his tumour "
            "cells carried it — and the biopsy was taken *after* he had "
            "received this drug class, so that is the share after "
            "selection. How rare the variant was before treatment is not "
            "in the data. "
            f"The {_sub['founding_fraction']:.0%} this run starts from is "
            "chosen to show the mechanism; the run ends at "
            f"{_recorded['carrier_share_end']:.0%}, "
            + ("inside" if _recorded["carrier_share_end"]
               >= _clonality["carrier_fraction_min"] else "below")
            + " the range his biopsy allows."
        )
        st.caption(
            "A mechanism demonstration, not a treatment recommendation. "
            "The carriers are spread evenly through the tumour at the "
            "start, where a real subclone is one contiguous patch. The "
            "activation factor is one published reporter measurement and "
            "the blockade strength is an estimate. The engine's clock is "
            "compressed by design, so these hours stand for a much longer "
            "clinical course. One run per arm, one seed."
        )
    else:
        _none = next((a for a in _arms if not a["drugs"]), None)
        st.markdown(
            f"In this run the patient is given "
            f"**{' + '.join(_recorded['drugs'])}**. What a drug does here is "
            "resolved against this patient's own androgen receptor (AR), and "
            "the bottom-right panel of the video plots the AR pathway's "
            "output — PSA, the protein AR switches on and the one a clinic "
            "follows — for this run beside the same patient, seed and tissue "
            "under each other choice."
        )
        for _note in _recorded.get("notes") or []:
            st.markdown(f"- {_note}")

        st.table([
            {"Arm": a["label"] + (" (the video)" if a["recorded"] else ""),
             "PSA per cell, start": f"{a['psa_start']:.1f}",
             "PSA per cell, end": f"{a['psa_end']:.1f}",
             "Cancer cells at the end": a["cancer_end"],
             "Immune kills": a["immune_kills"]}
            for a in _arms
        ])

        if _none and _none["psa_end"] > 0:
            _ratio = _recorded["psa_end"] / _none["psa_end"]
            if _ratio > 1.5:
                st.markdown(
                    f"On {' + '.join(_recorded['drugs'])} this patient's PSA "
                    f"ends at **{_ratio:.1f}x** the no-drug arm: the drug is "
                    "driving the receptor it was designed to block. "
                    + ("The tumour itself ends at the same size as with no "
                       "drug — which is what the source measured (Balbas "
                       "2013: resistant tumours on enzalutamide grew like "
                       "untreated ones, not faster). "
                       if _recorded["cancer_end"] == _none["cancer_end"] else "")
                    + "Rising PSA on treatment, with no benefit, is the "
                    "clinical picture of this resistance."
                )
            elif _ratio < 0.67:
                st.markdown(
                    f"On {' + '.join(_recorded['drugs'])} this patient's PSA "
                    f"ends at **{_ratio:.2f}x** the no-drug arm: the drug "
                    "blocks the receptor."
                )
            for a in _arms:
                if a is _recorded or not a["drugs"]:
                    continue
                _r = a["psa_end"] / _none["psa_end"]
                if _r < 0.67:
                    st.markdown(
                        f"**{a['label']}** still blocks this receptor in the "
                        f"model: PSA ends at {_r:.2f}x the no-drug arm and "
                        f"the tumour at {a['cancer_end']:,} cells against "
                        f"{_none['cancer_end']:,}. That drug-specific "
                        "difference is sourced (Moilanen 2015, PMID 26137992)."
                    )
        st.caption(
            "A mechanism demonstration, not a treatment recommendation. The "
            "activation factor is one published reporter measurement; the "
            "blockade strength is an estimate; one run per arm, one seed."
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

    # Not written out here: which genome is playing is a fact about the
    # recording, and this paragraph named a breast cell line for as long
    # as it was typed by hand.
    "**The genome and the tissue are not one biopsy.** The callset and "
    "the tissue geometry come from different sources, and the lines "
    "under *What weakens this run* above are this recording's own "
    "account of how they differ. The combination demonstrates the "
    "pipeline; it does not reconstruct this patient's tumour.",

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
        f"    --stride {facts['snapshot_stride']} --fps {facts['fps']}"
        + "".join(f" \\\n    --drug {d}"
                  for d in ((facts.get("treatment") or {}).get("drugs") or []))
        + "".join(f" \\\n    --compare-drug {a['label']}"
                  for a in (facts.get("treatment_arms") or [])
                  if a["drugs"] and not a["recorded"])
        + (f" \\\n    --subclone {facts['subclone']['label']}="
           f"{facts['subclone']['founding_fraction']:g}"
           if facts.get("subclone") else ""),
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
    "This run shows what the model does. What it leaves out matters as "
    "much: the roadmap lists, layer by layer, what is built, what is "
    "partial and what is not modelled yet, each built row linked to the "
    "test that holds it. The cells in both videos are stored in Bio-USD, "
    "an open format for biological cells in OpenUSD."
)
_r1, _r2, _r3 = st.columns(3)
with _r1:
    st.link_button("What it simulates, and what it doesn't yet",
                   "/?page=roadmap", use_container_width=True)
with _r2:
    st.link_button("Bio-USD: the standard and its code", "/?page=biousd",
                   use_container_width=True)
with _r3:
    st.link_button("Open Cognisom", "/", type="primary",
                   use_container_width=True)
st.caption(
    "The full environment — genome ingestion, neoantigen prediction, the "
    "treatment simulator and the 3D viewer — needs an account."
)
