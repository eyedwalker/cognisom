#!/usr/bin/env python
"""Record one real simulation as a video, plus the facts about it.

Why a script and not a notebook
-------------------------------
The public explainer page shows a recording and states numbers about it.
If those numbers were typed by hand they would drift from the video the
moment anything changed. So this script writes BOTH: the frames, and a
JSON sidecar of everything it measured while writing them. The page
reads the sidecar. Nothing on that page is a number somebody remembered.

What it draws, and where the colours come from
---------------------------------------------
Colours are taken from ``lineage_appearance`` and shaded by
``run_usd.shade_by_oxygen`` -- the same two functions the USD/RTX path
uses -- so the video and the 3D stage cannot disagree about what a cell
looks like. Tissue-cell brightness is that cell's own oxygen.

Usage
-----
    $HOME/.venvs/mhcflurry/bin/python scripts/record_demo_run.py \
        --duration 24 --tissue section --out cognisom/dashboard/assets

Run it with a Python that has MHCflurry (3.11). Under the PWM fallback
no peptide is ever a strong binder, so the neoantigen half of the
recording would be about the scorer rather than the patient.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognisom.genomics.demo_patients import (        # noqa: E402
    DEFAULT_DEMO_PATIENT, DEMO_PATIENTS)

#: Fixed so the recording can be regenerated frame for frame. The engine
#: draws from one shared RNG stream, so this seeds the whole run.
SEED = 7

#: Streamlit's dark surface, so the video sits on the page rather than
#: floating in a white box.
BG = "#0e1117"
PANEL = "#1a1d26"
INK = "#e6e6e6"
MUTED = "#8b8f9e"

KILL_MARKER_HOURS = 1.0   # how long a kill cross stays on screen


def _load_patient(which: str):
    """The demo callset, or None for the generic genome.

    Goes through ``genomics.demo_patients`` rather than naming a VCF and
    a patient id here. That module owns the cohort, so the id this script
    builds under is the id the HLA registry is consulted with -- the two
    drifting apart is exactly what put the demo on a population-default
    panel while the measured typing sat in the repository -- and the
    patient's provenance disclosures (biopsy site, prior treatment,
    genome build) ride along on the profile instead of having to be
    remembered by each caller.
    """
    if which == "none":
        return None
    from cognisom.genomics.demo_patients import load_demo_patient
    return load_demo_patient(which)


def _appearance_cache():
    from cognisom.visualization.lineage_appearance import appearance_for
    cache: dict = {}

    def colour_of(token: str):
        if token not in cache:
            cache[token] = appearance_for(token).colour
        return np.asarray(tuple(cache[token]), dtype="float32")

    return colour_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=24.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--scenario", default="Immunotherapy")
    ap.add_argument("--tissue", default="section")
    # Prostate by default: the tissue geometry, the AR pathway and most
    # of the drug library are prostate, so a breast cell line was the
    # odd one out. "none" still runs the generic genome.
    ap.add_argument("--patient", default=DEFAULT_DEMO_PATIENT,
                    choices=sorted(DEMO_PATIENTS) + ["none"])
    # Off by default: the published recording is an untreated-by-drug
    # run and adding a drug changes what it shows.
    ap.add_argument("--drug", action="append", default=[], metavar="NAME",
                    help="give the patient this library drug (repeatable). "
                         "Resolved against the patient's own receptor, so "
                         "enzalutamide on AR F877L is applied as the "
                         "agonist it is.")
    ap.add_argument("--compare-drug", action="append", default=[],
                    metavar="NAME",
                    help="with --drug: also run this drug on the same "
                         "patient, seed and tissue, and plot its AR-pathway "
                         "output (PSA) beside the recorded run and a no-drug "
                         "arm. The video still shows ONE run; the others "
                         "appear only as lines in the pathway panel.")
    ap.add_argument("--subclone", action="append", default=[],
                    metavar="LABEL=FRACTION",
                    help="put this patient mutation in only FRACTION of the "
                         "founding cancer cells (e.g. AR_F877L=0.1), in "
                         "every arm. With a drug that is an agonist on that "
                         "variant, the carriers are driven and the rest are "
                         "blocked, and the run shows selection. The fraction "
                         "is a SCENARIO: see "
                         "demo_patients.SU2C_AR_F877L_CLONALITY for what the "
                         "patient's data does and does not say.")
    ap.add_argument("--clock", default="compressed",
                    choices=("compressed", "measured"),
                    help="which clock cells divide on. 'measured' is the "
                         "labelling-index turnover (cancer ~46 days), for "
                         "a run that stands for days or weeks of a patient")
    ap.add_argument("--tumour-fraction", type=float, default=None,
                    metavar="F",
                    help="share of tissue cells that are cancer (the "
                         "presets seed ~4%%). The total is kept")
    ap.add_argument("--stride", type=int, default=2,
                    help="use every Nth snapshot as a video frame")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="cognisom/dashboard/assets")
    ap.add_argument("--name", default="demo_run")
    ap.add_argument("--usd", default=None, metavar="PATH",
                    help="also export this SAME run to a USD stage. The two "
                         "outputs then show one simulation rather than two "
                         "that happen to use the same settings -- which is "
                         "the only honest way to put an RTX render beside "
                         "the matplotlib recording on one page.")
    args = ap.parse_args()

    from cognisom.dashboard.engine_runner import EngineRunner
    from cognisom.dashboard.sim_request import (
        describe_subject, presentation_lines, tissue_caption)
    from cognisom.visualization.run_usd import shade_by_oxygen, OXYGEN_FULL

    patient = _load_patient(args.patient)
    subject = describe_subject(patient)
    print(f"subject: {subject.headline}")

    clonal = {}
    for item in args.subclone:
        label, sep, fraction = item.partition("=")
        if not sep:
            ap.error(f"--subclone wants LABEL=FRACTION, got {item!r}")
        clonal[label.strip()] = float(fraction)
    subclone_label = next(iter(clonal), None)

    def seeded_runner(drugs):
        # Seeded HERE, immediately before construction, for every arm:
        # the arms are only comparable if each starts from the same
        # stream, and loading the patient must not be able to shift it.
        random.seed(SEED)
        np.random.seed(SEED)
        return EngineRunner(dt=args.dt, duration=args.duration,
                            scenario=args.scenario, tissue=args.tissue,
                            patient=patient, drugs=drugs or None,
                            clonal_fractions=clonal or None,
                            clock=args.clock,
                            tumour_fraction=args.tumour_fraction)

    if args.compare_drug and not args.drug:
        ap.error("--compare-drug needs --drug: there is no recorded "
                 "treatment to compare against")

    runner = seeded_runner(args.drug)
    if runner.treatment is not None:
        print(f"treatment: {', '.join(runner.treatment.drugs)}")
        for note in runner.treatment.notes:
            print(f"  note: {note}")
    runner.build()

    def progress(cur, total):
        if cur % 40 == 0 or cur == total:
            print(f"  step {cur}/{total}  t={cur*args.dt:.1f}h", flush=True)

    print(f"running {args.duration} h on {args.tissue} ...")
    runner.run(progress_callback=progress)

    snaps = runner.cell_snapshots[::max(1, args.stride)]
    ts = runner.get_time_series()

    # ── The drug pathway: the same patient, other choices ────────────
    #
    # One run cannot show what a drug did, only what happened. So when a
    # drug is given the same patient is also run with no drug and with
    # each --compare-drug, from the same seed, and the AR pathway's
    # output (PSA) is plotted for all of them. PSA is the right readout
    # because it is what the pathway transcribes and what a clinic
    # follows -- and for enzalutamide on AR F877L the cell counts do NOT
    # separate from no drug (Balbas 2013: growth equivalent to vehicle)
    # while PSA triples, so a cell-count panel alone would show nothing.
    def arm_record(label, r, recorded):
        series = r.get_time_series()
        return {
            "label": label,
            "drugs": list(r.treatment.drugs) if r.treatment else [],
            "recorded": recorded,
            "notes": list(r.treatment.notes) if r.treatment else [],
            "time": [float(v) for v in series["time"]],
            "psa": [float(v) for v in series["avg_psa"]],
            # What the whole tumour puts out. Per-cell PSA cannot show a
            # response followed by progression, because that is a story
            # about how many cells there are of each kind.
            "psa_tumour": [float(v) for v in series["psa_cancer_total"]],
            "carrier_share": [
                (c / (c + o) if (c + o) else 0.0)
                for c, o in zip(series["n_subclone_carriers"],
                                series["n_subclone_others"])],
            "cancer_end": int(series["n_cancer"][-1]),
            "immune_kills": int(series["total_kills"][-1]),
        }

    arms = []
    if runner.treatment is not None:
        arms.append(arm_record(" + ".join(runner.treatment.drugs),
                               runner, True))
        for label, drugs in ([("No drug", None)]
                             + [(d, [d]) for d in args.compare_drug]):
            print(f"comparison arm: {label} ...", flush=True)
            other = seeded_runner(drugs)
            other.build()
            other.run()
            arms.append(arm_record(
                " + ".join(other.treatment.drugs) if other.treatment
                else label, other, False))
            a = arms[-1]
            print(f"  cancer {a['cancer_end']}, kills {a['immune_kills']}, "
                  f"PSA {a['psa'][0]:.1f} -> {a['psa'][-1]:.1f}")
    #: recorded run bright; no drug grey; alternatives in teal/amber.
    arm_colours = ["#f5c542", "#8b8f9e", "#4dd0c8", "#c792ea", "#7fb069"]
    from cognisom.visualization.interaction_recorder import dedupe_kill_events

    raw_events = list(getattr(runner.recorder, "events", []) or [])
    # A T-cell kill is emitted as both cancer_killed and
    # cell_killed_by_tcell, so without this the video draws two crosses
    # on one pixel and the page states a kill count the engine does not
    # agree with.
    events = dedupe_kill_events(raw_events)
    n_dropped = len(raw_events) - len(events)
    kinds = sorted({e.kind for e in events})
    print(f"{len(runner.cell_snapshots)} snapshots -> {len(snaps)} frames; "
          f"event kinds: {kinds}; {n_dropped} duplicate kill markers dropped")

    kills = [e for e in events
             if "kill" in e.kind.lower() and e.position is not None]

    # ── Fixed camera: computed once over every frame, so the view never
    #    jumps. A moving camera in a scientific recording makes growth
    #    and migration impossible to read.
    allpos = [s.cell_positions for s in snaps if s.cell_positions is not None
              and len(s.cell_positions)]
    allpos += [s.immune_positions for s in snaps
               if s.immune_positions is not None and len(s.immune_positions)]
    stacked = np.vstack([np.asarray(p, dtype="float64") for p in allpos])
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    pad = 0.06 * max(hi - lo).item()
    lo, hi = lo - pad, hi + pad

    colour_of = _appearance_cache()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    tmp = Path(tempfile.mkdtemp(prefix="cognisom_rec_"))
    time_arr = np.asarray(ts["time"], dtype="float64")

    try:
        for i, snap in enumerate(snaps):
            t = float(snap.time)
            show_clone = bool(arms and subclone_label)
            n_rows = 4 if show_clone else (3 if arms else 2)
            fig = plt.figure(figsize=(12.8, 7.2 if show_clone else 6.4),
                             dpi=100, facecolor=BG)
            gs = gridspec.GridSpec(n_rows, 2, width_ratios=[2.05, 1.0],
                                   height_ratios=[1] * n_rows, figure=fig,
                                   wspace=0.14,
                                   hspace=(0.95 if show_clone
                                           else 0.62 if arms else 0.34),
                                   left=0.00, right=0.965,
                                   top=0.835, bottom=0.095)

            # ── 3D tissue ────────────────────────────────────────────
            ax = fig.add_subplot(gs[:, 0], projection="3d")
            ax.set_facecolor(BG)

            pos = np.asarray(snap.cell_positions, dtype="float64") \
                if snap.cell_positions is not None else np.zeros((0, 3))
            if len(pos):
                tokens = list(snap.cell_types)
                base = np.vstack([colour_of(tok) for tok in tokens])
                oxy = [float(v) if v is not None else OXYGEN_FULL
                       for v in (snap.cell_oxygen or [OXYGEN_FULL] * len(pos))]
                rgb = shade_by_oxygen(base, oxy)
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=np.clip(rgb, 0, 1),
                           s=9, alpha=0.88, linewidths=0, depthshade=False)

            # The resistant cells, ringed: selection is a statement
            # about WHICH cells are left, and a count cannot show that.
            flags = list(getattr(snap, "cell_subclone", None) or [])
            if len(pos) and len(flags) == len(pos) and any(flags):
                cp = pos[np.asarray(flags, dtype=bool)]
                ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], s=70,
                           facecolors="none", edgecolors="#f5c542",
                           linewidths=1.4, depthshade=False)

            ipos = np.asarray(snap.immune_positions, dtype="float64") \
                if snap.immune_positions is not None else np.zeros((0, 3))
            if len(ipos):
                irgb = np.vstack([colour_of(tok) for tok in snap.immune_types])
                ax.scatter(ipos[:, 0], ipos[:, 1], ipos[:, 2],
                           c=np.clip(irgb, 0, 1), s=46, alpha=1.0,
                           edgecolors="white", linewidths=0.4,
                           depthshade=False)

            recent = [e for e in kills if 0.0 <= t - e.time <= KILL_MARKER_HOURS]
            if recent:
                kp = np.asarray([e.position for e in recent], dtype="float64")
                ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], marker="x",
                           c="#ff4d6d", s=120, linewidths=2.0, depthshade=False)

            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect((hi - lo), zoom=1.5)
            ax.set_axis_off()
            ax.view_init(elev=24, azim=-55)   # fixed, see above

            # ── populations ──────────────────────────────────────────
            def strip(cell, series, title, log=False, lines=None):
                a = fig.add_subplot(cell)
                a.set_facecolor(PANEL)
                for key, label, col in series:
                    a.plot(time_arr, ts[key], color=col, lw=1.8, label=label)
                for x, y, label, col, lw in (lines or []):
                    a.plot(x, y, color=col, lw=lw, label=label)
                if log:
                    # ~1,950 normal cells against ~120 cancer: on a
                    # linear axis the two populations that actually move
                    # sit flat on the floor and the panel says nothing.
                    a.set_yscale("log")
                a.axvline(t, color=INK, lw=1.0, alpha=0.55)
                a.set_title(title, color=INK, fontsize=11, loc="left", pad=22)
                # which="both": a log axis spanning under a decade labels
                # its MINOR ticks, and those were drawn in the default
                # black, half outside the panel.
                a.tick_params(colors=MUTED, labelsize=8, which="both")
                a.tick_params(labelsize=7, which="minor")
                for s in a.spines.values():
                    s.set_color("#2b2f3a")
                a.grid(color="#262a34", lw=0.6)
                a.set_xlim(time_arr[0], time_arr[-1])
                # Above the axes, not inside: an inside legend sat on
                # top of the very lines it labels.
                a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005),
                         fontsize=8, frameon=False, labelcolor=INK,
                         ncols=3, borderaxespad=0.0, handlelength=1.2,
                         columnspacing=1.0)
                return a

            strip(gs[0, 1], [("n_normal", "Normal", "#3498db"),
                             ("n_cancer", "Cancer", "#e74c3c"),
                             ("n_immune", "Immune", "#2ecc71")],
                  "Cell populations (log scale)", log=True)
            a2 = strip(gs[1, 1], [("total_kills", "Cumulative immune kills",
                                   "#ff4d6d")], "Immune killing")
            if arms:
                psa_key = "psa_tumour" if show_clone else "psa"
                a3 = strip(
                    gs[2, 1], [],
                    ("Tumour PSA output (log scale)" if show_clone else
                     "AR pathway output: PSA per cell (log scale)"), log=True,
                    lines=[(arm["time"], arm[psa_key],
                            arm["label"] + (" (this video)"
                                            if arm["recorded"] else ""),
                            arm_colours[j % len(arm_colours)],
                            2.4 if arm["recorded"] else 1.6)
                           for j, arm in enumerate(arms)])
                if show_clone:
                    a4 = strip(
                        gs[3, 1], [],
                        f"Share of cancer cells carrying "
                        f"{subclone_label.replace('_', ' ')}",
                        lines=[(arm["time"],
                                [100.0 * v for v in arm["carrier_share"]],
                                arm["label"], arm_colours[j % len(arm_colours)],
                                2.4 if arm["recorded"] else 1.6)
                               for j, arm in enumerate(arms)])
                    a4.set_ylim(0, 100)
                    a4.set_ylabel("%", color=MUTED, fontsize=8)
                    a4.set_xlabel("simulated hours", color=MUTED, fontsize=9)
                else:
                    a3.set_xlabel("simulated hours", color=MUTED, fontsize=9)
            else:
                a2.set_xlabel("simulated hours", color=MUTED, fontsize=9)

            fig.suptitle(
                f"{subject.headline}   ·   {args.scenario}"
                + (f" + {' + '.join(runner.treatment.drugs)}"
                   if runner.treatment is not None else "")
                + f" on the {args.tissue} geometry",
                color=INK, fontsize=11, x=0.02, ha="left", y=0.955)
            # The clock is its own right-aligned text: appended to the
            # title it was the first thing clipped when the title grew.
            fig.text(0.965, 0.955, f"t = {t:5.2f} h", color=INK,
                     fontsize=12, ha="right", va="center")
            fig.text(0.02, 0.018,
                     "Tissue-cell brightness is that cell's own oxygen.  "
                     "Red crosses: a cancer cell killed by an immune cell "
                     "within the last hour.  "
                     + ("Yellow rings: cancer cells carrying the resistance "
                        "variant.  " if show_clone else "")
                     + "Research use only.",
                     color=MUTED, fontsize=8 if show_clone else 9, ha="left")

            fig.savefig(tmp / f"f{i:05d}.png", facecolor=BG)
            plt.close(fig)
            if i % 20 == 0:
                print(f"  frame {i}/{len(snaps)}", flush=True)

        outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
        mp4 = outdir / f"{args.name}.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("ffmpeg not found; frames left in", tmp)
            return 2
        cmd = [ffmpeg, "-y", "-framerate", str(args.fps),
               "-i", str(tmp / "f%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "scale=1280:-2", "-crf", "26",
               "-movflags", "+faststart", str(mp4)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        size_mb = mp4.stat().st_size / 1e6
        print(f"wrote {mp4}  ({size_mb:.2f} MB)")

        # ── the facts, measured here and nowhere else ───────────────
        n_cancer = ts["n_cancer"]; n_normal = ts["n_normal"]
        facts = {
            "generated_by": "scripts/record_demo_run.py",
            "seed": SEED,
            "video": mp4.name,
            "video_mb": round(size_mb, 2),
            "fps": args.fps,
            "frames": len(snaps),
            "snapshot_stride": args.stride,
            "scenario": args.scenario,
            "tissue": args.tissue,
            "tissue_caption": tissue_caption(args.tissue),
            "duration_hours": args.duration,
            "dt_hours": args.dt,
            "clock": args.clock,
            "tumour_fraction": args.tumour_fraction,
            "patient_headline": subject.headline,
            "personalised": subject.personalised,
            "patient_id": subject.patient_id,
            "mutations_carried": subject.mutation_labels,
            "hla_alleles": subject.hla_alleles,
            "hla_measured": subject.hla_measured,
            "caveats": subject.caveats,
            "treatment": (runner.treatment.summary()
                          if runner.treatment is not None else None),
            # Every arm of the pathway panel, so the page states the
            # comparison from this file and not from memory. The first
            # is the run the video shows.
            "treatment_arms": [
                {"label": a["label"], "drugs": a["drugs"],
                 "recorded": a["recorded"], "notes": a["notes"],
                 "cancer_end": a["cancer_end"],
                 "immune_kills": a["immune_kills"],
                 "psa_start": round(a["psa"][0], 3),
                 "psa_end": round(a["psa"][-1], 3),
                 "psa_tumour_start": round(a["psa_tumour"][0], 1),
                 "psa_tumour_low": round(min(a["psa_tumour"]), 1),
                 "psa_tumour_low_hour": round(
                     a["time"][a["psa_tumour"].index(min(a["psa_tumour"]))], 1),
                 "psa_tumour_end": round(a["psa_tumour"][-1], 1),
                 "carrier_share_start": round(a["carrier_share"][0], 4),
                 "carrier_share_end": round(a["carrier_share"][-1], 4)}
                for a in arms],
            # Which mutation was put in only part of the tumour, and how
            # much of it. A scenario, not a reading -- see the page.
            "subclone": ({"label": subclone_label,
                          "founding_fraction": clonal[subclone_label]}
                         if subclone_label else None),
            "presentation": presentation_lines(runner.presentation_report),
            "cells_start": int(n_cancer[0] + n_normal[0]),
            "cells_end": int(n_cancer[-1] + n_normal[-1]),
            "cancer_start": int(n_cancer[0]),
            "cancer_end": int(n_cancer[-1]),
            "normal_start": int(n_normal[0]),
            "normal_end": int(n_normal[-1]),
            "immune_end": int(ts["n_immune"][-1]),
            "immune_kills": int(ts["total_kills"][-1]),
            "divisions": int(ts["total_divisions"][-1]),
            "deaths": int(ts["total_deaths"][-1]),
            "metastases": int(ts["total_metastases"][-1]),
            "hypoxic_end": int(ts["hypoxic_regions"][-1]),
            "event_kinds": kinds,
            "kill_markers": len(kills),
            "duplicate_kill_markers_dropped": n_dropped,
            # When the run stops changing. The populations settle well
            # before the clock runs out, and a viewer watching a still
            # picture for the last quarter of the video deserves to be
            # told that is the model and not the player.
            "last_kill_hour": round(max((e.time for e in kills),
                                        default=0.0), 2),
            "last_division_hour": round(
                max((e.time for e in events if e.kind == "division"),
                    default=0.0), 2),
            "last_death_hour": round(
                max((e.time for e in events if e.kind == "death"),
                    default=0.0), 2),
            "oxygen_min": round(min(
                float(v) for s in snaps for v in (s.cell_oxygen or [])
                if v is not None), 4),
            "oxygen_max": round(max(
                float(v) for s in snaps for v in (s.cell_oxygen or [])
                if v is not None), 4),
            "predictor": "mhcflurry" if not os.environ.get(
                "COGNISOM_ALLOW_PWM_FALLBACK") else "pwm-fallback-allowed",
        }
        if args.usd:
            # Exported from the run already in memory, not re-simulated.
            # A second run with the same seed would almost certainly match,
            # but "almost certainly" is not something to put under two
            # videos that claim to be the same run.
            from cognisom.visualization.run_usd import write_run

            usd_path = Path(args.usd)
            usd_path.parent.mkdir(parents=True, exist_ok=True)
            usd_report = write_run(str(usd_path), runner)
            usd_kills = usd_report.markers["drawn"].get("kill", 0)
            facts["usd_stage"] = usd_path.name
            facts["usd_kill_markers"] = int(usd_kills)
            facts["usd_duplicate_kills_suppressed"] = int(
                usd_report.markers.get("suppressed_duplicate_kill", 0))
            print(f"wrote {usd_path} ({usd_path.stat().st_size / 1e6:.1f} MB)")
            print(f"  kill markers in the stage: {usd_kills}  "
                  f"(video markers: {len(kills)}; engine: {facts['immune_kills']})")
            if int(usd_kills) != int(facts["immune_kills"]):
                print("  WARNING: the stage and the engine disagree on kills. "
                      "Do not render this until they agree -- a viewer can "
                      "count the crosses.")

        facts_path = outdir / f"{args.name}.json"
        facts_path.write_text(json.dumps(facts, indent=2) + "\n")
        print(f"wrote {facts_path}")
        print(json.dumps({k: facts[k] for k in
                          ("cancer_start", "cancer_end", "immune_kills",
                           "divisions", "deaths", "oxygen_min", "oxygen_max")},
                         indent=2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
