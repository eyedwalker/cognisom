"""Androgen withdrawal reaches cancer-cell fate: slower cycling for all,
death for the dependent third, nothing for normal epithelium.

The anchor is human and verified: Westin et al. 1995 (Am J Pathol
146:1368, PMID 7778676) biopsied 18 prostatic tumours the day before
and seven days after castration -- proliferation (Ki-67) fell in 15,
the apoptotic index rose in 6 (the grade 1-2, p53/Bcl-2/c-myc-negative
ones), and 3 were unaffected. Page 2006 (PMID 16882745) found benign
epithelium's proliferation and apoptosis unchanged 28 days in. The
magnitudes here are estimates and the module says so; the directions,
the split and the sparing of normal cells are the paper's.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

pytest.importorskip("scipy")

from cognisom.core.event_bus import EventTypes  # noqa: E402
from cognisom.modules import cellular_module as cm  # noqa: E402

MODS = {"cellular": True, "spatial": True, "vascular": True, "ar_signaling": True,
        "immune": False, "lymphatic": False, "molecular": False, "epigenetic": False,
        "circadian": False, "morphogen": False, "cell_mechanics": False}


def _run(scenario, hours, seed=11, **enabled):
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(seed)
    runner = EngineRunner(dt=0.05, duration=hours, scenario=scenario,
                          modules_enabled={**MODS, **enabled})
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


@pytest.fixture(scope="module")
def baseline():
    # 72 h, not 96: at 96 the untreated arm reaches 2,560 cells and the
    # batched AR solve made this fixture 106 s against the suite's 120 s
    # per-test timeout -- it passed alone and timed out under full-suite
    # load. 72 h keeps every claim below (640 against 41 cancer cells,
    # two withdrawal deaths) at half the cost.
    return _run("Baseline", 72.0)


@pytest.fixture(scope="module")
def castrate():
    return _run("Androgen Deprivation", 72.0)


def _state(runner):
    return runner.engine.modules["cellular"].get_state()


# ── an intact run is untouched ─────────────────────────────────────


@pytest.mark.slow
def test_an_intact_run_divides_exactly_as_it_did_without_the_coupling():
    """Same seed, AR module on and off: identical counts. The dead band
    (AR_INTACT_ABOVE) is what makes this exact rather than close."""
    with_ar = _run("Baseline", 24.0)
    without = _run("Baseline", 24.0, ar_signaling=False)
    for key in ("n_cancer", "n_normal", "total_divisions", "total_deaths"):
        assert _state(with_ar)[key] == _state(without)[key], key
    assert _state(with_ar)["androgen_withdrawal_deaths"] == 0


@pytest.mark.slow
def test_baseline_cancer_doubles_every_twelve_hours_for_three_days(baseline):
    s = _state(baseline)
    assert s["n_cancer"] == 10 * 2 ** 6 and s["n_normal"] == 80 * 2 ** 3
    assert s["androgen_withdrawal_deaths"] == 0


# ── castration: slower, deadlier for a third, normal cells spared ──


@pytest.mark.slow
def test_castration_slows_the_tumour_and_spares_normal_epithelium(baseline, castrate):
    b, c = _state(baseline), _state(castrate)
    assert c["n_cancer"] < 0.1 * b["n_cancer"], (c["n_cancer"], b["n_cancer"])
    assert c["n_cancer"] > 10, "the tumour still grows, just slowly -- Westin's Ki-67 fell, it did not vanish"
    assert c["n_normal"] == b["n_normal"], "Page 2006: benign epithelium is unchanged"
    assert c["total_divisions"] < b["total_divisions"]


@pytest.mark.slow
def test_only_dependent_cells_die_of_withdrawal_and_only_under_withdrawal(baseline, castrate):
    assert _state(baseline)["androgen_withdrawal_deaths"] == 0
    withdrawn = [e for e in castrate.event_log
                 if e["event"] == EventTypes.CELL_DIED and e["data"].get("cause") == "androgen_withdrawal"]
    assert len(withdrawn) == _state(castrate)["androgen_withdrawal_deaths"] > 0
    assert all(e["data"]["cell_type"] == "cancer" for e in withdrawn)


@pytest.mark.slow
def test_the_response_unfolds_over_days_not_steps(castrate):
    """AR-DHT decays with a 20 h time constant: signalling leaves the
    dead band within two hours and reaches its castrate floor over days,
    so the tumour still doubles in its first day (4x would be untouched,
    1x frozen) and grows less in its third than in its first."""
    ts = castrate.get_time_series()
    t = np.asarray(ts["time"]); n = np.asarray(ts["n_cancer"], float)
    day1 = n[np.searchsorted(t, 24.0)] / n[0]
    day3 = n[-1] / n[np.searchsorted(t, 48.0)]
    assert 1.5 < day1 < 4.0, f"day-1 growth {day1:.2f}x"
    assert day3 < day1, "growth has slowed by the third day"


# ── the dependent third ────────────────────────────────────────────


def test_a_third_of_cancer_cells_are_dependent_deterministically_and_clonally():
    module = cm.CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0})
    ids = [module.add_cell(position=np.array([10.0 * i, 10.0, 10.0]), cell_type="cancer") for i in range(30)]
    dependent = [i for i in ids if module.cells[i].ar_dependent]
    assert len(dependent) == 10
    # a second module with the same ids makes the same choice: no RNG
    again = cm.CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0})
    ids2 = [again.add_cell(position=np.array([10.0 * i, 10.0, 10.0]), cell_type="cancer") for i in range(30)]
    assert [again.cells[i].ar_dependent for i in ids2] == [module.cells[i].ar_dependent for i in ids]
    normal = module.add_cell(position=np.array([1.0, 1.0, 1.0]), cell_type="normal")
    assert module.cells[normal].ar_dependent is False
    parent = dependent[0]
    module._divide_cell(parent)
    daughter = max(module.cells)
    assert module.cells[daughter].ar_dependent is True


def test_unlinked_the_cellular_module_cannot_see_androgen():
    module = cm.CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0})
    assert module._ar_signalling(0) is None
