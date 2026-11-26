"""Serum PSA is a different quantity from the PSA the cells hold, and
the engine now reports the one a clinic measures.

Every clinical PSA criterion -- a fall of 50% or more, time to nadir,
progression as a rise from nadir, doubling time -- is defined on serum
PSA: secreted protein, in blood, cleared with its own half-life. The
engine's number was the intracellular pool summed over the tumour, so
none of those criteria could be applied to it. This is the step
between the two, and these tests hold it to three claims:

  1. the clearance half-life is the measured one (Stamey 1987);
  2. it is an output only -- nothing reads it back, so a run cannot be
     moved by it;
  3. it lags the tumour by days, not weeks, so it changes the KIND of
     quantity reported and not the conclusions.
"""

from __future__ import annotations

import math

import pytest

from cognisom.engine.py.pharmacology import serum_marker as sm


# ── the arithmetic ─────────────────────────────────────────────────


def test_the_half_life_is_the_measured_one_and_says_where_from():
    assert sm.SERUM_PSA_HALF_LIFE_DAYS == 2.2
    assert sm.SERUM_PSA_STATUS == "sourced"
    assert "2442609" in sm.SERUM_PSA_SOURCE
    report = sm.serum_report()
    assert "not ng/mL" in str(report["units"]), (
        "the pool is a fold change; a reader must not take it for a "
        "concentration")


def test_with_the_source_removed_serum_halves_every_half_life():
    """The measurement itself: Stamey's half-life is PSA falling after
    the prostate is taken out, i.e. the source set to zero at once."""
    s = 100.0
    s = sm.advance_serum(s, 0.0, 2.2 * 24.0, 2.2)
    assert s == pytest.approx(50.0, rel=1e-12)
    s = sm.advance_serum(s, 0.0, 2.2 * 24.0, 2.2)
    assert s == pytest.approx(25.0, rel=1e-12)


def test_a_steady_tumour_gives_a_steady_reading():
    """In balance, nothing moves: the pool settles AT the source."""
    assert sm.advance_serum(40.0, 40.0, 500.0, 2.2) == pytest.approx(40.0)


def test_the_result_does_not_depend_on_the_step_size():
    """An exact update, not Euler: 1,000 small steps and one large one
    agree, so the reading is not a function of how the run is stepped."""
    one = sm.advance_serum(100.0, 30.0, 50.0, 2.2)
    many = 100.0
    for _ in range(1000):
        many = sm.advance_serum(many, 30.0, 0.05, 2.2)
    assert many == pytest.approx(one, rel=1e-9)


def test_it_cannot_overshoot_however_long_the_step():
    for dt in (1.0, 1e3, 1e6):
        s = sm.advance_serum(100.0, 30.0, dt, 2.2)
        assert 30.0 <= s <= 100.0


def test_no_half_life_means_no_lag_and_a_bad_one_is_refused():
    assert sm.advance_serum(100.0, 30.0, 0.05, None) == 30.0
    with pytest.raises(ValueError):
        sm.advance_serum(100.0, 30.0, 0.05, 0.0)
    with pytest.raises(ValueError):
        sm.advance_serum(100.0, 30.0, 0.05, -1.0)


# ── in a run ───────────────────────────────────────────────────────


MODS = {"cellular": True, "spatial": True, "vascular": True,
        "ar_signaling": True, "immune": False, "lymphatic": False,
        "molecular": False, "epigenetic": False, "circadian": False,
        "morphogen": False, "cell_mechanics": False}


def _run(drugs, serum_half_life="default", hours=48.0):
    import random

    import numpy as np

    from cognisom.dashboard.engine_runner import EngineRunner

    random.seed(11)
    np.random.seed(11)
    overrides = {}
    if serum_half_life != "default":
        overrides = {"ar_signaling": {
            "serum_psa_half_life_days": serum_half_life}}
    runner = EngineRunner(tissue="ring", dt=0.05, duration=hours,
                          scenario="Baseline", modules_enabled=MODS,
                          drugs=drugs, overrides=overrides,
                          record_interval=20)
    runner.build()
    runner.run()
    return runner.get_time_series()


@pytest.fixture(scope="module")
def treated():
    pytest.importorskip("scipy")
    return _run(("Enzalutamide",))


@pytest.mark.slow
def test_serum_psa_starts_in_balance_with_the_tumour(treated):
    """A tumour that existed before the run was already secreting. A
    pool started from zero would rise for a week and look like biology."""
    assert treated["serum_psa_relative"][0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
def test_serum_psa_falls_under_the_drug_and_lags_the_tumour(treated):
    tumour = [v / treated["psa_cancer_total"][0]
              for v in treated["psa_cancer_total"]]
    serum = treated["serum_psa_relative"]
    # NON-VACUITY: the drug has to have moved the tumour's PSA, or a
    # lag has nothing to lag behind.
    assert tumour[-1] < 0.9, tumour[-1]
    assert serum[-1] < 1.0, "serum PSA did not respond at all"
    assert serum[-1] > tumour[-1], (
        "while the tumour's PSA is falling, blood still carries what "
        "was secreted earlier, so serum must read HIGHER than the "
        f"tumour: serum {serum[-1]:.4f}, tumour {tumour[-1]:.4f}")
    # And it lags by days, not weeks: 48 h is 0.91 half-lives, so serum
    # has closed at least a third of the gap by now.
    closed = (1.0 - serum[-1]) / (1.0 - tumour[-1])
    assert 0.2 < closed < 0.95, closed


@pytest.mark.slow
def test_serum_psa_is_an_output_and_moves_nothing(treated):
    """Nothing reads it back. Turning the clearance lag off must leave
    every cell-level outcome identical -- not close, identical -- while
    the serum reading itself does change."""
    no_lag = _run(("Enzalutamide",), serum_half_life=None)
    for key in ("n_cancer", "total_divisions", "total_deaths",
                "psa_cancer_total", "total_kills"):
        assert no_lag[key] == treated[key], key
    assert no_lag["serum_psa_relative"] != treated["serum_psa_relative"]
    # with no lag, serum IS the tumour total
    tumour = [v / no_lag["psa_cancer_total"][0]
              for v in no_lag["psa_cancer_total"]]
    assert no_lag["serum_psa_relative"][-1] == pytest.approx(
        tumour[-1], rel=1e-3)


def test_the_closed_form_lag_matches_the_half_life():
    """Against a tumour whose PSA steps down and stays down, serum
    closes half the gap per half-life -- the property the run-level
    test above leans on, checked without a run."""
    s = 1.0
    for _ in range(int(2.2 * 24 / 0.05)):
        s = sm.advance_serum(s, 0.4, 0.05, 2.2)
    assert s == pytest.approx(0.7, abs=1e-3)
    assert math.isclose((1.0 - s) / (1.0 - 0.4), 0.5, abs_tol=2e-3)
