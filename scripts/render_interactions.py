#!/usr/bin/env python3
"""Run the engine, record what happens, and render it as images.

    PYTHONPATH=. python3 scripts/render_interactions.py \
        --scenario Immunotherapy --duration 48 --seed 7 --out out/interactions

Produces, in ``--out``:

    interactions.usda   the stage -- opens in Blender, Houdini, Maya,
                        Omniverse, or usdview if you have one
    legend.png          the legend texture the stage references
    frame.####.png      the rendered frames
    events.json         every recorded event, with time and position
    manifest.json       what was drawn, what was not, and why

Rendering uses ``usdrecord``, which ships with macOS at
``/usr/bin/usdrecord`` and needs no GPU box. ``--no-render`` writes the
stage and stops.

A note on how rare these events are. At the shipped presets, immune
kills are the thing you came to see and they are scarce: a 6 h
Immunotherapy run (the dashboard default) produces **zero** kills and
zero divisions, and a 24 h run produces one activation and no kills. The
first kill in the measured 48 h run lands at t=30.0 h and the one dense
cluster -- six kills -- at t=36.0-36.5 h. So ``--duration 48`` is the
shortest setting at which an immune kill is visible at all, and
``--window 35.5:37`` is where to look.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from cognisom.visualization.interaction_recorder import (
    InteractionRecorder, KIND_KILL,
)
from cognisom.visualization import interaction_usd as iusd
from cognisom.visualization.interaction_usd import FrameMap
from cognisom.visualization.legend_texture import render_legend_png

USDRECORD_CANDIDATES = ("/usr/bin/usdrecord", "usdrecord")


# ── Context geometry ────────────────────────────────────────────────
#
# The markers need something to happen *to*. This draws the cell
# population as two point instancers so a kill burst reads as a kill
# burst and not as a dot in a void. It is deliberately minimal and lives
# in this script rather than in the library: a full cell renderer with
# per-state colouring is a separate concern, and when one lands this
# function should be deleted in favour of it.

# Deliberately desaturated, and deliberately NOT red: the event markers
# own the saturated end of the palette, and a dark-red cancer cell next
# to a bright-red kill marker is exactly the confusion this render has
# to avoid. Cancer is violet-grey here, which no marker uses.
_CONTEXT_COLORS = {
    "normal": (0.17, 0.25, 0.38),
    "cancer": (0.36, 0.22, 0.42),
    "T_cell": (0.16, 0.40, 0.26),
    "NK_cell": (0.34, 0.38, 0.22),
    "macrophage": (0.38, 0.28, 0.18),
}


def author_context_cells(stage, snapshots, frame_map: FrameMap,
                         root: str = "/World/ContextCells",
                         cell_radius: float = 3.0) -> Dict[str, Any]:
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    groups: Dict[str, List[List[Tuple[float, float, float]]]] = {}
    for snap in snapshots:
        per_frame: Dict[str, List[Tuple[float, float, float]]] = {}
        if snap.cell_positions is not None:
            for pos, ctype in zip(snap.cell_positions, snap.cell_types):
                per_frame.setdefault(ctype, []).append(
                    (float(pos[0]), float(pos[1]), float(pos[2])))
        if snap.immune_positions is not None:
            for pos, itype in zip(snap.immune_positions, snap.immune_types):
                per_frame.setdefault(itype, []).append(
                    (float(pos[0]), float(pos[1]), float(pos[2])))
        for key in set(list(per_frame) + list(groups)):
            groups.setdefault(key, []).append(per_frame.get(key, []))

    UsdGeom.Scope.Define(stage, root)
    report = {}
    for ctype, frames in groups.items():
        n_max = max((len(f) for f in frames), default=0)
        if n_max == 0:
            continue
        safe = ctype.replace(" ", "_")
        pi_path = f"{root}/{safe}"
        pi = UsdGeom.PointInstancer.Define(stage, pi_path)
        proto = UsdGeom.Sphere.Define(stage, f"{pi_path}/Prototypes/Cell")
        proto.GetRadiusAttr().Set(float(cell_radius))
        proto.GetExtentAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(-cell_radius, -cell_radius, -cell_radius),
            Gf.Vec3f(cell_radius, cell_radius, cell_radius)]))
        colour = _CONTEXT_COLORS.get(ctype, (0.35, 0.35, 0.40))
        mat = UsdShade.Material.Define(stage, f"{pi_path}/Prototypes/Cell/Mat")
        sh = UsdShade.Shader.Define(stage, f"{pi_path}/Prototypes/Cell/Mat/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*colour))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(proto).Bind(mat)
        pi.CreatePrototypesRel().SetTargets([proto.GetPath()])
        pi.CreateProtoIndicesAttr().Set(Vt.IntArray([0] * n_max))
        pi.CreateIdsAttr().Set(Vt.Int64Array(list(range(n_max))))
        pos_attr = pi.CreatePositionsAttr()
        inv_attr = pi.CreateInvisibleIdsAttr()
        for frame in range(frame_map.first, frame_map.last + 1):
            pts = frames[frame] if frame < len(frames) else []
            padded = list(pts) + [(0.0, 0.0, 0.0)] * (n_max - len(pts))
            pos_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in padded]), frame)
            inv_attr.Set(Vt.Int64Array(list(range(len(pts), n_max))), frame)
        report[ctype] = n_max
    return report


# ── Simulation ───────────────────────────────────────────────────────

def run_simulation(scenario: str, duration: float, dt: float,
                   record_interval: int, seed: Optional[int]):
    from cognisom.dashboard.engine_runner import EngineRunner

    if seed is not None:
        np.random.seed(int(seed))

    runner = EngineRunner(dt=dt, duration=duration, scenario=scenario,
                          record_interval=record_interval)
    runner.build()

    recorder = InteractionRecorder.for_engine(runner.engine)

    # Field snapshots are taken on the same cadence as the cell
    # snapshots, by wrapping the runner's own recorder. Doing it this
    # way rather than editing EngineRunner keeps the two concerns apart:
    # nothing in the dashboard path changes.
    original = runner._record_cell_snapshot

    def record_both():
        original()
        recorder.capture_engine_fields(runner.engine)

    runner._record_cell_snapshot = record_both
    try:
        runner.run()
    finally:
        runner._record_cell_snapshot = original

    return runner, recorder


# ── Stage ────────────────────────────────────────────────────────────

def build_stage(out_dir: Path, runner, recorder, *,
                window: Optional[Tuple[float, float]] = None,
                with_context: bool = True,
                with_field: bool = True,
                focus: str = "tissue",
                aspect: float = 16.0 / 9.0) -> Dict[str, Any]:
    from pxr import Usd, UsdGeom

    out_dir.mkdir(parents=True, exist_ok=True)
    usda_path = out_dir / "interactions.usda"
    if usda_path.exists():
        usda_path.unlink()

    snapshots = list(runner.cell_snapshots)
    frame_times = [float(s.time) for s in snapshots]
    if not frame_times:
        raise RuntimeError(
            "the run recorded no cell snapshots, so there is no timeline "
            "to hang events on")
    frame_map = FrameMap(frame_times)

    stage = Usd.Stage.CreateNew(str(usda_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1e-6)   # positions are micrometres
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    iusd.ensure_stage_timeline(stage, frame_map)
    iusd.ensure_dome_light(stage)

    events = recorder.events
    marker_report = iusd.author_interaction_markers(stage, events, frame_map)
    bond_report = iusd.author_engagement_bonds(stage, events, frame_map)

    slice_report: Dict[str, Any] = {"authored": False,
                                    "reason": "field rendering disabled"}
    cell_o2_report: Dict[str, Any] = {"authored": False}
    if with_field:
        oxygen = recorder.field_frames.get("oxygen")
        if oxygen:
            meta = recorder.field_meta["oxygen"]
            slice_report = iusd.author_oxygen_slice(
                stage, oxygen, frame_map, resolution=meta["resolution"],
                origin=meta["origin"])
        else:
            slice_report = {"authored": False,
                            "reason": "no spatial module / no field captured"}
        cell_o2_report = iusd.author_cell_oxygen(
            stage, iusd.oxygen_frames_from_snapshots(snapshots), frame_map)

    context_report: Dict[str, Any] = {}
    if with_context:
        context_report = author_context_cells(stage, snapshots, frame_map)

    # Frame on the cells, not only on the markers: a camera that frames
    # the events alone zooms into whichever corner the kills happened in
    # and loses the tissue around them.
    extra: List[Sequence[float]] = []
    for snap in snapshots:
        if snap.cell_positions is not None and len(snap.cell_positions):
            arr = np.asarray(snap.cell_positions, dtype=float)
            extra.append(arr.min(axis=0))
            extra.append(arr.max(axis=0))
    if focus == "kills":
        kills = [e for e in events if e.kind == KIND_KILL]
        if kills:
            center, radius = iusd.events_bounds(kills)
            # x3 so the kill sites sit inside visible tissue rather than
            # filling the frame: a close-up with no surrounding cells
            # shows a red ball and nothing to read it against.
            radius = max(radius, 30.0) * 3.0
        else:
            center, radius = iusd.events_bounds(events, extra_points=extra)
            radius *= 1.15
    elif focus == "events":
        center, radius = iusd.events_bounds(events)
        radius = max(radius, 18.0) * 1.3
    else:
        center, radius = iusd.events_bounds(events, extra_points=extra)
        radius *= 1.15
    cam = iusd.author_event_camera(stage, center, radius, aspect=aspect)

    kinds = [k for k in marker_report["drawn"]]
    entries = iusd.legend_entries_for(kinds)
    if bond_report.get("drawn"):
        entries.append({
            "shape": "disc", "color": iusd.BOND_COLOR,
            "label": "Attacker → target, at the moment of the kill",
        })
    if cell_o2_report.get("halos"):
        entries.append({
            "shape": "sphere", "color": (0.75, 0.06, 0.10),
            "label": (f"Hypoxic cell (O₂ < {iusd.HYPOXIA_THRESHOLD:g}), "
                      f"peak {cell_o2_report.get('peak_hypoxic_cells', 0)} "
                      f"on one frame"),
        })
    span = recorder.time_span()
    counts = recorder.counts()
    subtitle = (
        f"{runner.scenario_name} · {runner.duration:g} h · dt={runner.dt:g} h · "
        f"{len(snapshots)} frames"
    )
    footer = (
        "Markers sit where the event was emitted, not where the cell is now. "
        f"Recorded {len(events)} events over {span[0]:.2f}-{span[1]:.2f} h: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    )
    if not slice_report.get("authored") and "measured_range" in slice_report:
        footer += (
            f" | solved spatial O2 field not drawn: uniform to "
            f"{slice_report['measured_range']:.1e}"
        )
    legend_png = out_dir / "legend.png"
    _path, legend_aspect = render_legend_png(
        entries, str(legend_png),
        title="Cognisom — what the cells are doing",
        subtitle=subtitle, footer=footer,
        oxygen_ramp=bool(cell_o2_report.get("shadows")
                         or slice_report.get("authored")),
        oxygen_caption=("Oxygen, per cell (disc under each cell)"
                        if cell_o2_report.get("shadows")
                        else "Oxygen field (plane under the tissue)"),
    )
    iusd.author_legend(
        stage, cam["path"], "legend.png", aspect=aspect,
        focal_length=cam["focal_length"], aperture=cam["aperture"],
        texture_aspect=legend_aspect, width_fraction=0.34,
        distance=cam["legend_distance"],
    )

    stage.Save()

    frame_range = (frame_map.first, frame_map.last)
    if window:
        lo = frame_map.frame_of(window[0])
        hi = frame_map.frame_of(window[1])
        frame_range = (frame_map.first if lo is None else lo,
                       frame_map.last if hi is None else hi)

    return {
        "usda": str(usda_path),
        "legend": str(legend_png),
        "camera": cam,
        "frames": len(frame_map),
        "frame_times": frame_times,
        "frame_range": [int(frame_range[0]), int(frame_range[1])],
        "markers": marker_report,
        "bonds": bond_report,
        "oxygen_slice": slice_report,
        "cell_oxygen": cell_o2_report,
        "context_cells": context_report,
        "event_counts": counts,
        "global_events": len(recorder.global_events),
    }


# ── Rendering ────────────────────────────────────────────────────────

def find_usdrecord() -> Optional[str]:
    for cand in USDRECORD_CANDIDATES:
        if os.path.isabs(cand):
            if os.path.exists(cand):
                return cand
        else:
            found = shutil.which(cand)
            if found:
                return found
    return None


def render_frames(usda: str, out_dir: Path, camera_path: str,
                  frame_range: Tuple[int, int], width: int = 1280,
                  complexity: str = "high",
                  prefix: str = "frame") -> Dict[str, Any]:
    """Render a frame range, one usdrecord invocation per frame.

    One invocation per frame is not a style choice. Apple's usdrecord
    (0.25.2, /usr/bin/usdrecord) formats the frame number into the
    "####" placeholder with what is evidently a %g conversion at one
    significant figure: time code 10 is written "1e+01", 12, 13 and 14
    all become "1e+01" and overwrite each other, and 176 becomes
    "2e+02". Measured, not assumed. Any range crossing 10 therefore
    silently collapses onto a handful of files. Rendering one time code
    at a time into a scratch directory and moving the single result into
    place is the only way to get a correctly numbered sequence out of
    this binary.
    """
    exe = find_usdrecord()
    if exe is None:
        return {"rendered": False,
                "reason": "usdrecord not found; stage written, render skipped"}

    tmp = out_dir / "_usdrecord_tmp"
    produced: List[str] = []
    failures: List[Dict[str, Any]] = []
    for frame in range(int(frame_range[0]), int(frame_range[1]) + 1):
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        cmd = [
            exe, "--imageWidth", str(width), "--camera", camera_path,
            "--complexity", complexity, "-f", str(frame),
            usda, str(tmp / "r.####.png"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        made = sorted(tmp.glob("r.*.png"))
        if not made:
            failures.append({"frame": frame, "returncode": proc.returncode,
                             "stderr_tail": (proc.stderr or "")[-400:]})
            continue
        target = out_dir / f"{prefix}.{frame:05d}.png"
        shutil.move(str(made[0]), str(target))
        produced.append(str(target))
    if tmp.exists():
        shutil.rmtree(tmp)

    return {
        "rendered": bool(produced),
        "exe": exe,
        "per_frame_invocation": True,
        "n_images": len(produced),
        "first": produced[0] if produced else None,
        "last": produced[-1] if produced else None,
        "failures": failures[:5],
        "n_failures": len(failures),
    }


# ── CLI ──────────────────────────────────────────────────────────────

def _parse_window(text: Optional[str]) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    lo, _sep, hi = text.partition(":")
    return (float(lo), float(hi))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="Immunotherapy")
    ap.add_argument("--duration", type=float, default=48.0,
                    help="hours; immune kills first appear around 30 h")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--record-interval", type=int, default=4,
                    help="steps between recorded frames")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="out/interactions")
    ap.add_argument("--window", default=None,
                    help="render only this simulated-time span, 'LO:HI' in hours")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-context", action="store_true",
                    help="omit the cell population (markers only)")
    ap.add_argument("--focus", choices=("tissue", "events", "kills"),
                    default="tissue",
                    help="what the camera frames: the whole tissue, every "
                         "positioned event, or just the kill sites")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    runner, recorder = run_simulation(
        args.scenario, args.duration, args.dt, args.record_interval, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    recorder.write_json(str(out_dir / "events.json"))

    info = build_stage(out_dir, runner, recorder,
                       window=_parse_window(args.window),
                       with_context=not args.no_context,
                       focus=args.focus)

    if not args.no_render:
        info["render"] = render_frames(
            info["usda"], out_dir, info["camera"]["path"],
            tuple(info["frame_range"]), width=args.width)

    kills = [e for e in recorder.events if e.kind == KIND_KILL]
    info["kill_times"] = [round(e.time, 3) for e in kills]
    info["engagements"] = [
        {"killer": g.killer_id, "target": g.target_id,
         "activated_at": round(g.activated_at, 3),
         "killed_at": round(g.killed_at, 3),
         "duration_h": round(g.duration_hours, 3)}
        for g in recorder.engagements()
    ]
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(info, fh, indent=1, default=str)

    print(json.dumps({k: v for k, v in info.items()
                      if k != "frame_times"}, indent=1, default=str))
    if not kills:
        print(
            "\nNO IMMUNE KILL WAS RECORDED. At the shipped presets this is "
            "expected below ~30 h of simulated time; try "
            "--scenario Immunotherapy --duration 48.",
            file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
