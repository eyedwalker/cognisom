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
    """The demo callset, or None for the generic genome."""
    if which == "none":
        return None
    from cognisom.genomics.patient_profile import PatientProfileBuilder
    root = Path(__file__).resolve().parent.parent
    vcf = root / "cognisom" / "genomics" / "seqc2_demo.vep.vcf"
    return PatientProfileBuilder(cancer_type="breast").from_vcf_file(
        str(vcf), patient_id="HCC1395")


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
    ap.add_argument("--patient", default="hcc1395", choices=["hcc1395", "none"])
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

    random.seed(SEED)
    np.random.seed(SEED)

    from cognisom.dashboard.engine_runner import EngineRunner
    from cognisom.dashboard.sim_request import (
        describe_subject, presentation_lines, tissue_caption)
    from cognisom.visualization.run_usd import shade_by_oxygen, OXYGEN_FULL

    patient = _load_patient(args.patient)
    subject = describe_subject(patient)
    print(f"subject: {subject.headline}")

    runner = EngineRunner(dt=args.dt, duration=args.duration,
                          scenario=args.scenario, tissue=args.tissue,
                          patient=patient)
    runner.build()

    def progress(cur, total):
        if cur % 40 == 0 or cur == total:
            print(f"  step {cur}/{total}  t={cur*args.dt:.1f}h", flush=True)

    print(f"running {args.duration} h on {args.tissue} ...")
    runner.run(progress_callback=progress)

    snaps = runner.cell_snapshots[::max(1, args.stride)]
    ts = runner.get_time_series()
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
            fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)
            gs = gridspec.GridSpec(2, 2, width_ratios=[2.05, 1.0],
                                   height_ratios=[1, 1], figure=fig,
                                   wspace=0.14, hspace=0.34,
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
            def strip(cell, series, title, log=False):
                a = fig.add_subplot(cell)
                a.set_facecolor(PANEL)
                for key, label, col in series:
                    a.plot(time_arr, ts[key], color=col, lw=1.8, label=label)
                if log:
                    # ~1,950 normal cells against ~120 cancer: on a
                    # linear axis the two populations that actually move
                    # sit flat on the floor and the panel says nothing.
                    a.set_yscale("log")
                a.axvline(t, color=INK, lw=1.0, alpha=0.55)
                a.set_title(title, color=INK, fontsize=11, loc="left", pad=22)
                a.tick_params(colors=MUTED, labelsize=8)
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
            a2.set_xlabel("simulated hours", color=MUTED, fontsize=9)

            fig.suptitle(
                f"{subject.headline}   ·   {args.scenario} on the "
                f"{args.tissue} geometry   ·   t = {t:5.2f} h",
                color=INK, fontsize=13, x=0.02, ha="left", y=0.955)
            fig.text(0.02, 0.018,
                     "Tissue-cell brightness is that cell's own oxygen.  "
                     "Red crosses: a cancer cell killed by an immune cell "
                     "within the last hour.  Research use only.",
                     color=MUTED, fontsize=9, ha="left")

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
            "patient_headline": subject.headline,
            "personalised": subject.personalised,
            "patient_id": subject.patient_id,
            "mutations_carried": subject.mutation_labels,
            "hla_alleles": subject.hla_alleles,
            "hla_measured": subject.hla_measured,
            "caveats": subject.caveats,
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
