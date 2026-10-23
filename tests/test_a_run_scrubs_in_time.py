"""A run exported as a timeline: cells on their frames, events on theirs.

The recorder, the marker writer, the bonds, the oxygen layers and the
camera each existed and were tested, and nothing attached the recorder
to a run or wrote a run out; every Bio-USD export was a still. This is
the end-to-end: a real 2 h Baseline run, written once, read back frame
by frame against the snapshots it came from.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

pytest.importorskip("pxr")

from cognisom.dashboard.engine_runner import EngineRunner  # noqa: E402
from cognisom.visualization.interaction_usd import FrameMap  # noqa: E402
from cognisom.visualization.run_usd import (  # noqa: E402
    CELLS_PATH, IMMUNE_PATH, write_run)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    np.random.seed(3)
    runner = EngineRunner(dt=0.05, duration=2.0, scenario="Baseline",
                          snapshot_interval_hours=0.1)
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    path = str(tmp_path_factory.mktemp("run") / "run.usdc")
    report = write_run(path, runner)
    from pxr import Usd
    stage = Usd.Stage.Open(path)
    return runner, path, report, stage


def _instancer(stage, path):
    from pxr import UsdGeom
    return UsdGeom.PointInstancer(stage.GetPrimAtPath(path))


# ── the runner records ─────────────────────────────────────────────


def test_the_runner_attaches_a_recorder_and_it_saw_the_run(run):
    runner, _, report, _ = run
    assert runner.recorder is not None
    assert len(runner.recorder.events) == report.events_recorded > 0
    assert report.events_by_kind.get("kill", 0) > 0, "Baseline kills its cancer in 2 h"


def test_the_recorder_is_cleared_when_a_run_restarts():
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    np.random.seed(4)
    runner = EngineRunner(dt=0.05, duration=0.3, scenario="Baseline")
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
        first = list(runner.recorder.events)
        runner.run()
    # The second run draws different random numbers, so its count is its
    # own; what must not happen is the first run's records surviving.
    assert first and runner.recorder.events
    assert not any(ev is old for ev in runner.recorder.events for old in first), (
        "a second run must not carry the first run's events")


# ── the cells are where the snapshots say, when they say ───────────


def test_one_frame_per_snapshot(run):
    runner, _, report, stage = run
    assert report.frames == len(runner.cell_snapshots) == 21
    assert stage.GetStartTimeCode() == 0.0
    assert stage.GetEndTimeCode() == float(report.frames - 1)
    times = list(stage.GetRootLayer().customLayerData["cognisom:frameTimesHours"])
    assert times == pytest.approx([s.time for s in runner.cell_snapshots])


@pytest.mark.parametrize("k", [0, 10, 20])
def test_visible_cells_on_a_frame_are_the_snapshot_s_cells_at_their_positions(run, k):
    from pxr import UsdGeom
    runner, _, _, stage = run
    snap = runner.cell_snapshots[k]
    inst = _instancer(stage, CELLS_PATH)
    ids = list(UsdGeom.PrimvarsAPI(inst.GetPrim()).GetPrimvar("bio:cell:id").Get())
    pos = np.asarray(inst.GetPositionsAttr().Get(float(k)))
    hidden = set(inst.GetInvisibleIdsAttr().Get(float(k)) or [])
    visible_ids = {ids[i] for i in range(len(ids)) if i not in hidden}
    assert visible_ids == set(snap.cell_ids)
    where = {cid: snap.cell_positions[j] for j, cid in enumerate(snap.cell_ids)}
    for i, cid in enumerate(ids):
        if i not in hidden:
            assert np.allclose(pos[i], where[cid], atol=1e-5)


def test_a_cell_that_died_is_hidden_after_and_visible_before(run):
    from pxr import UsdGeom
    runner, _, _, stage = run
    first = set(runner.cell_snapshots[0].cell_ids)
    last = set(runner.cell_snapshots[-1].cell_ids)
    gone = sorted(first - last)
    assert gone, "Baseline loses cells in 2 h (ten cancer cells are killed)"
    inst = _instancer(stage, CELLS_PATH)
    ids = list(UsdGeom.PrimvarsAPI(inst.GetPrim()).GetPrimvar("bio:cell:id").Get())
    i = ids.index(gone[0])
    assert i not in set(inst.GetInvisibleIdsAttr().Get(0.0) or [])
    assert i in set(inst.GetInvisibleIdsAttr().Get(float(len(runner.cell_snapshots) - 1)) or [])
    # and the default (untimed) view is the last frame
    assert i in set(inst.GetInvisibleIdsAttr().Get() or [])


def test_immune_cells_are_a_second_population(run):
    from pxr import UsdGeom
    runner, _, report, stage = run
    inst = _instancer(stage, IMMUNE_PATH)
    api = UsdGeom.PrimvarsAPI(inst.GetPrim())
    assert report.immune_cells == len(list(api.GetPrimvar("bio:cell:id").Get())) > 0
    assert set(api.GetPrimvar("bio:cell:type").Get()) == {"immune"}
    assert api.HasPrimvar("cognisom:immune:type")
    snap = runner.cell_snapshots[0]
    hidden = set(inst.GetInvisibleIdsAttr().Get(0.0) or [])
    assert len(list(api.GetPrimvar("bio:cell:id").Get())) - len(hidden) == len(snap.immune_ids)


# ── events land on their frames ────────────────────────────────────


def test_every_positioned_event_inside_the_run_is_drawn(run):
    runner, _, report, _ = run
    fm = FrameMap([s.time for s in runner.cell_snapshots])
    expected = {}
    for ev in runner.recorder.events:
        if ev.position is None or fm.frame_of(ev.time) is None:
            continue
        if ev.kind == "death" and (ev.detail or {}).get("cause") == "immune_killed":
            continue   # drawn once, as the kill
        expected[ev.kind] = expected.get(ev.kind, 0) + 1
    assert report.markers["drawn"] == expected


def test_a_kill_is_on_its_frame_and_not_the_frame_before(run):
    runner, _, _, stage = run
    fm = FrameMap([s.time for s in runner.cell_snapshots])
    kills = sorted((e for e in runner.recorder.events if e.kind == "kill"), key=lambda e: e.time)
    first = kills[0]
    f = fm.frame_of(first.time)
    inst = _instancer(stage, "/World/Interactions/Kill")
    n = len(inst.GetPositionsAttr().Get())
    hidden_before = set(inst.GetInvisibleIdsAttr().Get(float(f - 1)) or [])
    hidden_on = set(inst.GetInvisibleIdsAttr().Get(float(f)) or [])
    assert len(hidden_before) == n, "no kill is visible before the first kill"
    assert len(hidden_on) == n - 1, "exactly the first kill is visible on its frame"
    # the marker sits where the event said the target was
    pos = np.asarray(inst.GetPositionsAttr().Get())
    shown = next(i for i in range(n) if i not in hidden_on)
    assert np.allclose(pos[shown], first.position, atol=1e-4)


def test_engagement_lines_need_both_ends(run):
    _, _, report, _ = run
    assert report.bonds["drawn"] == report.events_by_kind["kill"]
    assert report.bonds["no_partner_position"] == 0


# ── and it is still Bio-USD ───────────────────────────────────────


def test_the_timeline_validates_as_bio_usd(run):
    from cognisom.biousd.standard.validate import validate
    _, path, _, _ = run
    report = validate(path)
    assert report.ok, [str(i) for i in report.issues]


def test_nothing_to_write_is_an_error(tmp_path):
    class _Empty:
        cell_snapshots = []
    with pytest.raises(ValueError):
        write_run(str(tmp_path / "x.usdc"), _Empty())
