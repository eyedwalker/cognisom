"""An NK cell kills a few targets and is then spent. It is not tireless.

Why this file exists
--------------------
Nothing limited how many targets one NK cell could kill. Each kill cost
2-4 of 63 granules, a reserve that never binds, so an NK cell beside a
nodule killed for as long as the run lasted. Measured on the demo patient
(section, Baseline, real division clock): 6 NK cells made 51 of 80 kills
in 30 h -- 6.8 kills per cell per day -- and the tumour was cleared in
every arm, with or without a drug. No longer run could show a course of
disease while that held.

Imaging of single cells says otherwise, and both papers were read in full
text before a number was taken from them: Prager 2019 (PMID 31270246), an
average of 2.53 kills per IL-2-activated NK cell over 15-17 h, most cells
three or fewer, the most active six; Bhat & Watzl 2007 (PMID 17389917),
spent cells survive depleted and recover by day 2 with IL-2.
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from cognisom.engine.py.immune import kill_calibration as kc
from cognisom.modules.immune_module import ImmuneModule


def _module(**config) -> ImmuneModule:
    with contextlib.redirect_stdout(io.StringIO()):
        module = ImmuneModule(config)
    module.emit_event = lambda *a, **k: None      # no bus in a unit test
    return module


def _target(i: int = 0):
    return SimpleNamespace(cell_id=10_000 + i, position=np.zeros(3))


def _nk(module: ImmuneModule):
    return module.immune_cells[module.add_immune_cell([50, 50, 50], "NK_cell")]


# ── the drawn capacity is the measured distribution ──────────────────

def test_the_capacity_draw_has_the_measured_mean_and_ceiling():
    rng = np.random.default_rng(1)
    draws = np.array([kc.draw_serial_kill_capacity(rng.poisson)
                      for _ in range(100_000)])
    assert draws.min() == 1, "a cell that can kill nothing is a different claim"
    assert draws.max() == kc.NK_SERIAL_KILLS_MAX == 6
    # Prager: "an average of 2.53 kills/NK cell". The cap costs under 1%.
    assert draws.mean() == pytest.approx(kc.NK_SERIAL_KILLS_MEAN.value, rel=0.02)
    # Prager: "the majority of NK cells killing three or fewer targets".
    assert (draws <= 3).mean() > 0.5


def test_the_numbers_say_how_good_they_are():
    report = kc.calibration_report()
    assert report["nk_serial_kills_mean"]["status"] == "sourced"
    assert "31270246" in report["nk_serial_kills_mean"]["source"]
    # Recovery was measured in a dish with added IL-2. It must not read
    # as a tissue measurement.
    assert report["nk_recovery_hours"]["status"] == "ESTIMATE"


# ── the cell ─────────────────────────────────────────────────────────

def test_an_nk_cell_goes_quiet_at_its_capacity_and_not_before():
    module = _module()
    cell = _nk(module)
    capacity = cell.serial_kill_capacity
    assert 1 <= capacity <= kc.NK_SERIAL_KILLS_MAX
    for i in range(capacity):
        assert not module._is_spent(cell), f"spent after {i} of {capacity}"
        module._kill_target(cell, _target(i))
    assert module._is_spent(cell)
    assert module.total_nk_spent == 1
    assert module.get_state()["n_nk_spent"] == 1


def test_a_spent_cell_recovers_as_a_fresh_one():
    module = _module()
    cell = _nk(module)
    for i in range(cell.serial_kill_capacity):
        module._kill_target(cell, _target(i))
    module.clock_h += kc.NK_RECOVERY_HOURS.value - 0.01
    assert module._is_spent(cell)
    module.clock_h += 0.02
    assert not module._is_spent(cell)
    assert cell.serial_kills == 0 and cell.spent_until is None
    assert 1 <= cell.serial_kill_capacity <= kc.NK_SERIAL_KILLS_MAX


def test_the_population_of_cells_matches_the_paper_not_just_the_helper():
    module = _module()
    capacities = [_nk(module).serial_kill_capacity for _ in range(4_000)]
    assert np.mean(capacities) == pytest.approx(
        kc.NK_SERIAL_KILLS_MEAN.value, rel=0.04)


def test_only_nk_cells_are_limited():
    """The measurement is of NK cells. A T cell's limit is its clone's
    exhaustion clock, and a macrophage's is not measured at all."""
    module = _module()
    for lineage in ("T_cell", "macrophage"):
        cell = module.immune_cells[module.add_immune_cell([50, 50, 50], lineage)]
        assert cell.serial_kill_capacity is None
        for i in range(20):
            module._kill_target(cell, _target(i))
        assert not module._is_spent(cell)


def test_the_limit_can_be_turned_off_and_then_nothing_is_ever_spent():
    module = _module(nk_serial_kill_limit=False)
    cell = _nk(module)
    assert cell.serial_kill_capacity is None
    for i in range(20):
        module._kill_target(cell, _target(i))
    assert not module._is_spent(cell)
    assert module.get_state()["nk_serial_kill_limit"] is False


def test_drawing_a_capacity_does_not_touch_the_shared_random_stream():
    """Recognition and contact times draw from numpy's global stream; a
    capacity drawn from it would move every pinned run for a reason
    unrelated to the limit itself."""
    np.random.seed(5)
    before = np.random.random()
    np.random.seed(5)
    module = _module()
    _nk(module)
    module2 = _module(nk_serial_kill_limit=False)
    np.random.seed(5)
    _nk(module)
    assert np.random.random() == before
    del module2
