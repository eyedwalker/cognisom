"""A USD export of a run must be the run, not one frame of it.

``SceneExporter.cells_to_usd`` sets startTimeCode=0 and endTimeCode=1 and
authors no time samples at all, so the only thing time-shaped about it is
the metadata.  These tests pin the sequence: one stage, one time sample
per recorded frame, positions that genuinely differ between samples, and
cells that disappear when they die two ways over -- ``invisibleIds`` for
a viewer that honours it and a zero scale for one that does not, because
a corpse left on screen is a fabricated cell.

They also pin the two findings behind the design:

* ``cell.phase`` never leaves 'G1' anywhere in the engine, so the
  "dividing" colour is derived from the recorded id sequence instead.
* ``SimulationEngine.run()`` records nothing by itself.  Every caller
  that drives the engine that way -- the five files in
  cognisom/scenarios/, the patent pipeline page, the module demos --
  produced a run with no frames; it now takes the callback that
  TissueEngine.run has always had.
"""
from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cognisom.dashboard.engine_runner import CellSnapshot, EngineRunner
from cognisom.visualization.usd_animation import (
    _split_key,
    write_cell_animation,
)


# ── Fixtures: a five-frame run with a death, a birth and a hypoxic cell ──

def _snapshots():
    """Cells 1 and 2 live throughout; 2 drifts.  Cell 3 dies after frame
    1.  Cell 4 is born at frame 3.  Cell 1 goes hypoxic at frame 2.
    """
    frames = []
    for f in range(5):
        ids = [1, 2]
        pos = [[100.0, 100.0, 50.0], [110.0 + f, 100.0, 50.0]]
        oxy = [0.21 if f < 2 else 0.01, 0.19]
        types = ["normal", "cancer"]
        if f < 2:
            ids.append(3)
            pos.append([90.0, 100.0, 50.0])
            oxy.append(0.20)
            types.append("normal")
        if f >= 3:
            ids.append(4)
            pos.append([112.0 + f, 102.0, 50.0])
            oxy.append(0.18)
            types.append("cancer")
        frames.append(
            CellSnapshot(
                time=0.5 * f,
                cell_positions=np.array(pos, dtype=np.float64),
                cell_types=types,
                cell_phases=["G1"] * len(ids),
                cell_oxygen=oxy,
                cell_mhc1=[1.0] * len(ids),
                cell_ids=ids,
                cell_glucose=[5.0] * len(ids),
                cell_atp=[900.0] * len(ids),
                cell_age=[float(f)] * len(ids),
                immune_positions=np.array(
                    [[120.0, 100.0 + f, 50.0], [80.0, 90.0 - f, 48.0]]
                ),
                immune_types=["T_cell", "macrophage"],
                immune_activated=[f >= 3, False],
                immune_ids=[7, 8],
            )
        )
    return frames


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out = tmp_path_factory.mktemp("usdanim") / "run.usda"
    stats = write_cell_animation(_snapshots(), str(out), fps=12.0)
    return stats, Usd.Stage.Open(str(out))


def _cells(stage):
    return UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/Cells"))


# ── ONE: there is a time axis, and samples hang off it ──────────────

def test_one_stage_carries_one_time_sample_per_recorded_frame(written):
    stats, stage = written
    cells = _cells(stage)
    assert stats.n_frames == 5
    pos = cells.GetPositionsAttr()
    assert pos.GetNumTimeSamples() == 5
    assert pos.GetTimeSamples() == [0.0, 1.0, 2.0, 3.0, 4.0]
    for attr in (cells.GetProtoIndicesAttr(), cells.GetScalesAttr(),
                 cells.GetInvisibleIdsAttr()):
        assert attr.GetNumTimeSamples() == 5, attr.GetName()


def test_the_stage_plays_at_a_stated_rate_over_the_recorded_range(written):
    stats, stage = written
    assert stage.GetStartTimeCode() == 0.0
    assert stage.GetEndTimeCode() == 4.0
    assert stage.GetTimeCodesPerSecond() == 12.0
    assert stage.GetFramesPerSecond() == 12.0
    # The frame -> simulated-hours map is on the stage, not in a filename.
    sim_time = stage.GetPrimAtPath("/World").GetAttribute("cognisom:simTimeHours")
    assert sim_time.GetNumTimeSamples() == 5
    assert sim_time.Get(Usd.TimeCode(0.0)) == pytest.approx(0.0)
    assert sim_time.Get(Usd.TimeCode(4.0)) == pytest.approx(2.0)


def test_positions_genuinely_differ_between_samples(written):
    """A static scene with a time axis bolted on would pass every test
    above.  This is the one that fails for it."""
    stats, stage = written
    pos = _cells(stage).GetPositionsAttr()
    first = np.array(pos.Get(Usd.TimeCode(0.0)))
    middle = np.array(pos.Get(Usd.TimeCode(2.0)))
    last = np.array(pos.Get(Usd.TimeCode(4.0)))
    assert first.shape == middle.shape == last.shape
    assert not np.allclose(first, middle)
    assert not np.allclose(middle, last)
    # Cell 2 drifts by 1 um per frame, and that is what is in the stage.
    ids = list(_cells(stage).GetIdsAttr().Get())
    slot = ids.index(2)
    assert last[slot][0] - first[slot][0] == pytest.approx(4.0)
    assert stats.max_cell_displacement == pytest.approx(1.0)


# ── TWO: deaths and births, with a fixed-length array ───────────────

def test_a_dead_cell_is_hidden_by_invisibleids_and_by_a_zero_scale(written):
    """Both, deliberately: invisibleIds is the USD idiom and the zero
    scale is what a viewer that ignores it falls back to.  Either alone
    leaves a corpse in some viewer."""
    stats, stage = written
    cells = _cells(stage)
    ids = list(cells.GetIdsAttr().Get())
    slot = ids.index(3)

    for frame in (0.0, 1.0):
        hidden = list(cells.GetInvisibleIdsAttr().Get(Usd.TimeCode(frame)))
        assert 3 not in hidden
        scale = cells.GetScalesAttr().Get(Usd.TimeCode(frame))[slot]
        assert tuple(scale) == (1.0, 1.0, 1.0)
        assert cells.ComputeMaskAtTime(Usd.TimeCode(frame))[slot] is True

    for frame in (2.0, 3.0, 4.0):
        hidden = list(cells.GetInvisibleIdsAttr().Get(Usd.TimeCode(frame)))
        assert 3 in hidden, frame
        scale = cells.GetScalesAttr().Get(Usd.TimeCode(frame))[slot]
        assert tuple(scale) == (0.0, 0.0, 0.0), frame
        assert cells.ComputeMaskAtTime(Usd.TimeCode(frame))[slot] is False


def test_a_cell_born_mid_run_is_absent_until_it_exists(written):
    stats, stage = written
    cells = _cells(stage)
    ids = list(cells.GetIdsAttr().Get())
    assert ids == [1, 2, 3, 4]          # first-appearance order
    slot = ids.index(4)

    for frame in (0.0, 1.0, 2.0):
        assert 4 in list(cells.GetInvisibleIdsAttr().Get(Usd.TimeCode(frame)))
        assert tuple(cells.GetScalesAttr().Get(Usd.TimeCode(frame))[slot]) == \
            (0.0, 0.0, 0.0)
    for frame in (3.0, 4.0):
        assert 4 not in list(cells.GetInvisibleIdsAttr().Get(Usd.TimeCode(frame)))
        assert cells.ComputeMaskAtTime(Usd.TimeCode(frame))[slot] is True


def test_the_id_array_is_the_engine_ids_and_does_not_move(written):
    """Slot order is allocated once, so an id in the stage is the id in
    the run and invisibleIds means what it says."""
    stats, stage = written
    ids_attr = _cells(stage).GetIdsAttr()
    assert ids_attr.GetNumTimeSamples() == 0        # uniform, not animated
    assert list(ids_attr.Get()) == [1, 2, 3, 4]
    assert stats.n_cell_slots == 4
    # 4 slots x 5 frames = 20 slot-frames. Cell 3 is dead for 3 of them
    # and cell 4 unborn for 3; the two immune cells never leave.
    assert stats.n_hidden_total == 6


# ── THREE: state is visible, and it comes from the data ─────────────

def test_hypoxia_and_birth_colour_from_recorded_state(written):
    stats, stage = written
    cells = _cells(stage)
    protos = stats.cell_prototypes
    assert "normal" in protos and "cancer" in protos
    assert "normal_hypoxic" in protos
    assert "cancer_newborn" in protos

    ids = list(cells.GetIdsAttr().Get())
    hypoxic = protos.index("normal_hypoxic")
    normal = protos.index("normal")
    slot = ids.index(1)
    assert cells.GetProtoIndicesAttr().Get(Usd.TimeCode(0.0))[slot] == normal
    assert cells.GetProtoIndicesAttr().Get(Usd.TimeCode(2.0))[slot] == hypoxic

    # One prototype prim per key, as children of the instancer, so Hydra
    # draws them only through the instancing.
    targets = cells.GetPrototypesRel().GetTargets()
    assert len(targets) == len(protos)
    for target in targets:
        assert str(target).startswith("/World/Cells/Protos/")
        assert stage.GetPrimAtPath(target).IsA(UsdGeom.Sphere)


def test_activated_immune_cells_are_their_own_prototype(written):
    stats, stage = written
    immune = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/Immune"))
    assert "T_cell" in stats.immune_prototypes
    assert "T_cell_activated" in stats.immune_prototypes
    ids = list(immune.GetIdsAttr().Get())
    slot = ids.index(7)
    before = immune.GetProtoIndicesAttr().Get(Usd.TimeCode(2.0))[slot]
    after = immune.GetProtoIndicesAttr().Get(Usd.TimeCode(3.0))[slot]
    assert before != after
    assert stats.immune_prototypes[after] == "T_cell_activated"
    # Immune cells are the only thing in the scene that actually moves.
    assert stats.max_immune_displacement > 0.5


def test_prototype_keys_split_back_into_type_and_state():
    assert _split_key("normal") == ("normal", "")
    assert _split_key("normal_hypoxic") == ("normal", "hypoxic")
    assert _split_key("T_cell") == ("T_cell", "")
    assert _split_key("T_cell_activated") == ("T_cell", "activated")
    assert _split_key("macrophage_activated") == ("macrophage", "activated")


def test_newborn_is_derived_from_ids_because_phase_never_changes():
    """If this fails, CellularModule has grown real cell-cycle phases and
    the dividing colour should be keyed on cell.phase instead of on the
    id sequence.  Until then, a phase-based colour would be dead code:
    every cell in every snapshot of a 24 h run reads 'G1'.
    """
    source = (REPO_ROOT / "cognisom" / "modules" / "cellular_module.py").read_text()
    assert "phase='G1'" in source, "cells are no longer born in G1"
    assert re.search(r"\.phase\s*=[^=]", source) is None, (
        "CellularModule now reassigns cell.phase, so the cell cycle is real: "
        "colour the dividing state from cell.phase instead of from the id "
        "sequence, and delete this test"
    )


# ── FOUR: a camera, framed on every frame ───────────────────────────

def test_every_cell_in_every_frame_is_inside_the_camera_frustum(written):
    """usdrecord auto-frames when there is no camera, per frame.  An
    explicit camera that frames the union of all frames is what keeps a
    sequence from drifting out of shot."""
    stats, stage = written
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera"))
    assert camera, "no /World/Camera"
    frustum = camera.GetCamera(Usd.TimeCode(0.0)).frustum
    for snap in _snapshots():
        for point in np.vstack([snap.cell_positions, snap.immune_positions]):
            assert frustum.Intersects(Gf.Vec3d(*[float(v) for v in point])), point


def test_the_stage_records_what_it_is(written):
    stats, stage = written
    data = stage.GetRootLayer().customLayerData
    assert data["cognisom:n_frames"] == 5
    assert data["cognisom:fps"] == 12.0
    assert list(data["cognisom:frame_times_hours"]) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z


def test_an_empty_run_is_refused_rather_than_written(tmp_path):
    with pytest.raises(ValueError, match="at least one recorded snapshot"):
        write_cell_animation([], str(tmp_path / "nothing.usda"))


# ── FIVE: the recording interval, and the caller that never fired ───

def test_the_default_interval_gives_a_watchable_number_of_frames():
    """It was every 10 steps -- 49 frames for a 24 h run at dt=0.05, and
    whatever the time-series cadence happened to be elsewhere.  It is now
    a simulated-hours interval of its own.
    """
    np.random.seed(12345)
    runner = EngineRunner(dt=0.05, duration=2.0, scenario="Baseline")
    assert runner.snapshot_interval_hours == 0.1
    runner.build()
    runner.run()

    times = [s.time for s in runner.cell_snapshots]
    assert len(times) == 21, times            # 0.0, 0.1, ... 2.0
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(2.0)
    gaps = np.diff(times)
    assert gaps.min() == pytest.approx(0.1, abs=1e-6)
    assert gaps.max() == pytest.approx(0.1, abs=1e-6)

    # The same interval over a 24 h run is 241 frames: ten seconds at
    # 24 fps, not 17,000 frames and not 49.
    frames_24h = int(24.0 / runner.snapshot_interval_hours) + 1
    assert 100 <= frames_24h <= 300


def test_the_time_series_cadence_is_not_the_frame_rate():
    """The two were the same line of code, so asking for coarse plots
    also threw away the animation."""
    np.random.seed(12345)
    runner = EngineRunner(dt=0.05, duration=1.0, scenario="Baseline",
                          record_interval=20, snapshot_interval_hours=0.1)
    runner.build()
    runner.run()
    # The history keeps its own, untouched cadence: t=0, step 0 and the
    # final step.
    assert len(runner.history) == 3
    assert len(runner.cell_snapshots) == 11


def test_a_caller_that_drives_the_engine_itself_can_record():
    """SimulationEngine.run() records nothing on its own -- that is why a
    6 h run through cognisom/scenarios/ produces no frames.  It now
    takes the callback TissueEngine.run always had."""
    np.random.seed(12345)
    runner = EngineRunner(dt=0.05, duration=1.0, scenario="Baseline",
                          snapshot_interval_hours=0.2)
    runner.build()

    buf = io.StringIO()
    with redirect_stdout(buf):
        runner.engine.run(duration=0.5)
    assert runner.cell_snapshots == []        # the engine records nothing

    with redirect_stdout(buf):
        runner.engine.run(duration=0.5, callback=runner.record_cell_snapshot)
    times = [s.time for s in runner.cell_snapshots]
    assert len(times) >= 2, times
    assert times[-1] > times[0]
