"""Cells start staggered through their cycle, so divisions spread through a run.

Both engines started every cell at age 0 with its lineage's division time,
so a whole lineage divided on one step. In the 24 h nine-module Baseline all
80 normal cells divided on the final step -- whether that step fell inside
the run depended on whether the float sum of the steps reached 24.0, so at
dt = 0.05 the population doubled and at dt = 0.02 it did not -- and every
daughter, starting at arterial O2, made half the tissue look perfectly
oxygenated in the end-of-run snapshot.

Initial ages now come from the steady-state age distribution of a cycling
population (core.cell_cycle.steady_state_ages), and daughters inherit their
parent's oxygen and glucose. 'synchronous' / asynchronous_cell_cycle=False
remain for tests whose subject is the division clock itself.
"""
from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pytest

from cognisom.core import SimulationConfig, SimulationEngine
from cognisom.core.cell_cycle import steady_state_ages
from cognisom.core.event_bus import EventTypes
from cognisom.core.tissue_config import TissueScaleConfig
from cognisom.core.tissue_engine import TissueSimulationEngine
from cognisom.modules.cellular_module import CellularModule
from cognisom.modules.vascular_module import VascularModule


# ── the distribution ────────────────────────────────────────────────────

def test_ages_follow_the_steady_state_distribution():
    period = 24.0
    ages = steady_state_ages(np.full(200_000, period),
                             np.random.default_rng(1).random(200_000))
    assert ages.min() >= 0.0 and ages.max() < period
    # Mean T(1 - ln2)/ln2; CDF 2(1 - 2^(-a/T)), so 2 - sqrt(2) below T/2.
    assert ages.mean() / period == pytest.approx((1 - math.log(2)) / math.log(2), abs=3e-3)
    assert (ages < period / 2).mean() == pytest.approx(2 - math.sqrt(2), abs=3e-3)


# ── the module engine ───────────────────────────────────────────────────

def _baseline(dt, **cellular):
    np.random.seed(12345)
    divisions = []
    with contextlib.redirect_stdout(io.StringIO()):
        engine = SimulationEngine(SimulationConfig(dt=dt, duration=24.0))
        engine.register_module("cellular", CellularModule,
                               {"n_normal_cells": 80, "n_cancer_cells": 10, **cellular})
        engine.register_module("vascular", VascularModule, {"n_capillaries": 8})
        engine.initialize()
        engine.modules["vascular"].set_cellular_module(engine.modules["cellular"])
        engine.event_bus.subscribe(EventTypes.CELL_DIVIDED,
                                   lambda d: divisions.append(round(engine.time, 4)))
        for _ in range(int(round(24.0 / dt))):
            engine.step()
    return engine.modules["cellular"], divisions


def _normal(cel):
    return sum(1 for c in cel.cells.values() if c.alive and c.cell_type == "normal")


def test_a_day_of_growth_no_longer_depends_on_the_time_step():
    coarse, _ = _baseline(0.05)
    fine, _ = _baseline(0.02)
    assert _normal(coarse) == _normal(fine) == 160
    # Synchronised, the same day at dt = 0.02 fell one rounding short of 24 h.
    synchronised, _ = _baseline(0.02, initial_cell_cycle="synchronous")
    assert _normal(synchronised) == 80


def test_divisions_are_spread_through_the_run():
    _, staggered = _baseline(0.05)
    _, synchronised = _baseline(0.05, initial_cell_cycle="synchronous")
    assert max(np.unique(synchronised, return_counts=True)[1]) >= 80   # one step
    assert max(np.unique(staggered, return_counts=True)[1]) <= 8
    assert min(staggered) < 6.0 < 18.0 < max(staggered)


def test_staggering_does_not_touch_the_global_stream():
    """The ring layout's draw order is what the Krogh calibration was
    measured on, so the ages come from a generator of their own."""
    def layout(mode):
        np.random.seed(3)
        with contextlib.redirect_stdout(io.StringIO()):
            cel = CellularModule({"n_normal_cells": 20, "n_cancer_cells": 5,
                                  "initial_cell_cycle": mode})
            cel.initialize()
        return [c.position.tolist() for c in cel.cells.values()], np.random.random()

    assert layout("asynchronous") == layout("synchronous")


def test_synchronous_remains_available_and_anything_else_is_refused():
    with contextlib.redirect_stdout(io.StringIO()):
        cel = CellularModule({"n_normal_cells": 5, "n_cancer_cells": 2,
                              "initial_cell_cycle": "synchronous"})
        cel.initialize()
    assert all(c.age == 0.0 for c in cel.cells.values())
    with pytest.raises(ValueError, match="initial_cell_cycle"):
        CellularModule({"initial_cell_cycle": "sometimes"})


def test_a_daughter_inherits_its_parents_oxygen_and_glucose():
    with contextlib.redirect_stdout(io.StringIO()):
        cel = CellularModule({"n_normal_cells": 1, "n_cancer_cells": 0})
        cel.initialize()
    parent_id = next(iter(cel.cells))
    parent = cel.cells[parent_id]
    parent.oxygen, parent.glucose = 0.031, 1.7
    cel._divide_cell(parent_id)
    daughter = next(c for i, c in cel.cells.items() if i != parent_id)
    assert (daughter.oxygen, daughter.glucose) == (0.031, 1.7)


# ── the tissue engine ───────────────────────────────────────────────────

def _tissue(asynchronous, seed=4):
    cfg = TissueScaleConfig(n_cells=400, grid_shape=(20, 20, 20),
                            tissue_size_um=(200.0, 200.0, 200.0), n_gpus=2, dt=0.5,
                            enable_diffusion=False, enable_mechanics=False,
                            enable_ode=False, seed=seed,
                            asynchronous_cell_cycle=asynchronous)
    eng = TissueSimulationEngine(cfg)
    eng.initialize()
    return eng


def _column(eng, col):
    return np.concatenate([eng._backend.to_numpy(ca.state)[:ca.n_real, col]
                           for ca in eng.cell_arrays])


def test_tissue_timers_start_staggered_within_each_lineage_period():
    eng = _tissue(True)
    timers = _column(eng, 7)
    types = np.concatenate([eng._backend.to_numpy(ca.cell_types)[:ca.n_real]
                            for ca in eng.cell_arrays])
    cancer = types == TissueSimulationEngine._CANCER_TYPE_INDEX
    assert timers[cancer].max() < eng._config.division_time_cancer_hr
    assert timers[~cancer].max() < eng._config.division_time_normal_hr
    assert np.unique(timers).size == timers.size          # no two alike
    assert np.array_equal(timers, _column(eng, 4))       # age since birth
    assert not _column(_tissue(False), 7).any()           # the old start


def test_staggering_leaves_the_tissue_layout_alone():
    def layout(eng):
        return np.concatenate([eng._backend.to_numpy(ca.positions)[:ca.n_real]
                               for ca in eng.cell_arrays])
    assert np.array_equal(layout(_tissue(True)), layout(_tissue(False)))


def test_tissue_cancer_starts_dividing_within_the_first_hours():
    """Synchronised, no cell is due before 12 h; staggered, some are due at once."""
    def grown(asynchronous):
        eng = _tissue(asynchronous)
        start = sum(ca.n_real for ca in eng.cell_arrays)
        for _ in range(6):                                  # 3 h
            eng.step()
        return sum(ca.n_real for ca in eng.cell_arrays) - start
    assert grown(False) == 0
    assert grown(True) > 0


def test_the_setting_travels_with_the_config_json():
    cfg = TissueScaleConfig(asynchronous_cell_cycle=False)
    assert TissueScaleConfig.from_json(cfg.to_json()).asynchronous_cell_cycle is False
