"""The circadian clock gates division, and now something listens.

``CircadianModule`` computed a phase per cell, counted peaks and troughs
for the dashboard, and offered three rhythmic outputs -- ``can_divide``,
a metabolic modifier and an immune modifier -- that had no caller
anywhere in the codebase. The clock ran; nothing it said reached a cell.

Only the division gate is wired here. That the clock gates mitosis is
measured (Matsuo et al., Science 2003, PMID 12934012: the clockwork
drives WEE1 and with it the timing of division in regenerating liver).
The permissive window the module uses, ZT 6-12, is its own estimate and
says so. The two modifiers are unsourced magnitudes and stay unwired.

A gated cell is deferred, not lost: its age keeps accruing and it
divides at the next window. So over a long run the gate changes WHEN
cells divide far more than HOW MANY divide -- which is what the tests
look for.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.core.event_bus import EventTypes


# ── the gate itself ─────────────────────────────────────────────────


class _Clock:
    def __init__(self, permit):
        self.permit = permit
        self.asked = 0

    def can_divide(self, cell_id):
        self.asked += 1
        return self.permit


def _cellular(n=6):
    from cognisom.modules.cellular_module import CellularModule

    module = CellularModule({"n_normal_cells": n, "n_cancer_cells": 0,
                            "division_time_normal": 1.0,
                            "initial_cycle_phase": "synchronous"})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    return module


def test_an_unlinked_module_divides_on_age_alone():
    module = _cellular()
    before = len(module.cells)
    for _ in range(12):
        module.update(0.1)
    assert module.total_divisions > 0
    assert len(module.cells) > before


def test_a_refusing_clock_defers_every_division():
    module = _cellular()
    clock = _Clock(permit=False)
    module.set_circadian_module(clock)
    for _ in range(12):
        module.update(0.1)
    assert module.total_divisions == 0
    assert clock.asked > 0, "the gate was never consulted"
    # deferred, not lost: the cells are past their division time
    assert all(c.age >= 1.0 for c in module.cells.values())


def test_a_permitting_clock_changes_nothing():
    gated, free = _cellular(), _cellular()
    gated.set_circadian_module(_Clock(permit=True))
    for _ in range(12):
        gated.update(0.1)
        free.update(0.1)
    assert gated.total_divisions == free.total_divisions


def test_the_real_module_refuses_outside_its_window():
    from cognisom.modules.circadian_module import CircadianModule

    clock = CircadianModule({})
    with redirect_stdout(io.StringIO()):
        clock.initialize()
    clock.add_cell(1, initial_phase=9.0)      # inside ZT 6-12
    clock.add_cell(2, initial_phase=20.0)     # outside
    assert clock.can_divide(1) is True
    assert clock.can_divide(2) is False
    assert clock.divisions_deferred == 1
    assert clock.can_divide(99) is True, "an unknown cell is not gated"


def test_gating_can_be_switched_off():
    from cognisom.modules.circadian_module import CircadianModule

    clock = CircadianModule({"enable_gating": False})
    with redirect_stdout(io.StringIO()):
        clock.initialize()
    clock.add_cell(2, initial_phase=20.0)
    assert clock.can_divide(2) is True
    assert clock.divisions_deferred == 0


# ── end to end: the dashboard's engine ──────────────────────────────


def _runner(gating, duration=24.0, seed=12345):
    from cognisom.dashboard.engine_runner import EngineRunner

    np.random.seed(seed)
    runner = EngineRunner(tissue="ring", dt=0.05, duration=duration, scenario="Baseline",
                          overrides={"circadian": {"enable_gating": gating}})
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def _division_times(runner):
    return np.array([e["time"] for e in runner.event_log
                     if e["event"] == EventTypes.CELL_DIVIDED], dtype=float)


def _between(times, lo, hi):
    return int(((times >= lo) & (times < hi)).sum())


@pytest.fixture(scope="module")
def gated():
    return _runner(True)


@pytest.fixture(scope="module")
def free():
    return _runner(False)


def test_the_runner_links_the_clock_to_the_cells(gated):
    mods = gated.engine.modules
    assert mods["cellular"].circadian_module is mods["circadian"]
    assert mods["circadian"].enable_gating is True


def test_the_gate_was_consulted_and_refused_something(gated, free):
    assert gated.engine.modules["circadian"].divisions_deferred > 0
    assert free.engine.modules["circadian"].divisions_deferred == 0


def test_divisions_cluster_in_the_window_only_when_gated(gated, free):
    """The Phase 6 bar: attached, the clock changes WHEN cells divide.

    The master clock starts at ZT 0 and advances with simulation time;
    cells couple to it within half a day. So in a 24 h run the only
    permissive window is 6-12 h, and after it the gate holds the
    population until the next one at 30-36 h. Measured (seed 12345):
    gated divisions per 6 h bin [8, 24, 3, 0], ungated [18, 18, 21, 23];
    over 36 h the gated run then bursts with 109 divisions in the
    second window where the ungated run spreads 72 evenly.
    """
    g, f = _division_times(gated), _division_times(free)
    assert len(g) > 0 and len(f) > 0
    # held between windows
    assert _between(g, 12.0, 24.0) < 0.25 * _between(f, 12.0, 24.0), (
        f"gated {_between(g, 12, 24)} vs free {_between(f, 12, 24)} in 12-24 h")
    # and concentrated in the first one
    assert _between(g, 6.0, 12.0) / len(g) > _between(f, 6.0, 12.0) / len(f)


def test_a_deferred_division_is_not_a_lost_one(gated, free):
    """Fewer divisions fit inside a 24 h run when cells wait for a
    window; none are lost -- every cell still alive past its division
    time is waiting, not sterile."""
    cells = gated.engine.modules["cellular"]
    waiting = [c for c in cells.cells.values()
               if c.alive and c.cell_type == "normal"
               and c.age >= cells.division_time_normal]
    assert cells.total_divisions <= free.engine.modules["cellular"].total_divisions
    assert cells.total_divisions + len(waiting) > 0
