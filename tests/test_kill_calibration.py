"""Who kills, and how often: per-contact killing from data.

Before this, NK cells and macrophages killed on 80% of contacts -- a
number with no source -- and on the 12 h slab runs they did 92-98% of all
kills, clearing Baseline and Immunotherapy completely. The T-cell rule at
full signal gave 0.99 per contact, where imaging of CTLs against solid
tumour cells finds <10% in vitro and ~5% in vivo (Weigelin 2021).

Now (engine/py/immune/kill_calibration.py):

* T cells integrate sublethal perforin hits that decay with a 56.7 min
  half-life and kill at the damage three hits 50 min apart leave
  (Weigelin 2021, PMID 34471116);
* NK cells and macrophages kill at their own per-contact numbers, both
  ESTIMATE with a range;
* the slab's infiltrate is split T/NK/macrophage as TCGA prostate tumours
  are, with macrophages drawn M1/M0/M2 from the same cohort -- a
  population prior, labelled as one;
* a cell dies of oxygen only when anoxic (<0.01% O2, Papandreou 2005),
  at a rate, instead of instantly at ~1.3% O2.
"""

from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pytest

from cognisom.engine.py.immune import kill_calibration as kc
from cognisom.modules.cellular_module import (
    ANOXIC_O2_LEVEL, STARVATION_GLUCOSE_MM, CellState)
from cognisom.modules.immune_module import ImmuneModule


def _immune(**config):
    with contextlib.redirect_stdout(io.StringIO()):
        module = ImmuneModule({"n_t_cells": 0, "n_nk_cells": 0,
                               "n_macrophages": 0, **config})
        module.initialize()
    return module


def _target():
    return CellState(cell_id=1, position=np.zeros(3), cell_type="cancer",
                     phase="G1")


# ── every number has a source or a range ─────────────────────────────────

def test_every_kill_number_is_sourced_or_an_estimate_with_a_range():
    report = kc.calibration_report()
    assert set(report) == {
        "tcell_hits_per_contact", "tcell_damage_half_life_min",
        "tcell_lethal_damage", "nk_kill_per_contact",
        "macrophage_kill_per_contact"}
    for name, entry in report.items():
        low, high = entry["range"]
        assert low <= entry["value"] <= high, name
        assert entry["status"] in {"sourced", "derived", "ESTIMATE"}, name
        assert entry["source"], name
    assert report["nk_kill_per_contact"]["status"] == "ESTIMATE"
    assert report["macrophage_kill_per_contact"]["status"] == "ESTIMATE"


def test_the_lethal_damage_is_three_hits_fifty_minutes_apart():
    expected = 1 + 2 ** (-50 / 56.7) + 2 ** (-100 / 56.7)
    assert kc.LETHAL_DAMAGE.value == pytest.approx(expected)
    assert kc.LETHAL_DAMAGE.value == pytest.approx(1.84, abs=0.005)


def test_the_hit_rate_reproduces_a_number_it_was_not_fitted_to():
    # Fitted: 40% of cognate contacts carry a perforin event. Not fitted:
    # how many single contacts kill on their own. Two hits in one contact
    # exceed 1.84, so that is P(N >= 2), and Weigelin measured <10% for
    # B16F10 and MV3 at first encounter.
    m = kc.HITS_PER_CONTACT.value
    assert 1 - math.exp(-m) == pytest.approx(0.40)
    p_two = 1 - math.exp(-m) * (1 + m)
    assert 0.05 < p_two < 0.10


# ── T cells: additive cytotoxicity ───────────────────────────────────────

def test_one_cognate_contact_rarely_kills():
    np.random.seed(0)
    module = _immune()
    lethal = sum(module._deliver_perforin_hits(_target(), 1.0, 0.0)
                 for _ in range(20_000))
    assert 0.07 < lethal / 20_000 < 0.11


def test_damage_decays_with_the_measured_half_life():
    assert kc.damage_now(1.0, 0.0, 56.7 / 60) == pytest.approx(0.5)
    assert kc.damage_now(2.0, 1.0, 1.0) == 2.0


def test_three_hits_kill_inside_fifty_minutes_and_not_outside():
    module = _immune()
    fast, slow = _target(), _target()
    # One hit per contact, exactly: a Poisson whose every draw is 1.
    module.tcell_hits_per_contact = 1.0
    import cognisom.modules.immune_module as im
    real = im.np.random.poisson
    im.np.random.poisson = lambda mean: 1
    try:
        assert not module._deliver_perforin_hits(fast, 1.0, 0.0)
        assert not module._deliver_perforin_hits(fast, 1.0, 45 / 60)
        assert module._deliver_perforin_hits(fast, 1.0, 90 / 60)
        assert not module._deliver_perforin_hits(slow, 1.0, 0.0)
        assert not module._deliver_perforin_hits(slow, 1.0, 1.0)
        assert not module._deliver_perforin_hits(slow, 1.0, 2.0)
    finally:
        im.np.random.poisson = real


def test_no_signal_no_hits():
    np.random.seed(1)
    module = _immune()
    target = _target()
    assert not any(module._deliver_perforin_hits(target, 0.0, 0.0)
                   for _ in range(100))
    assert target.cytotoxic_damage == 0.0 and module.total_perforin_hits == 0


def test_the_old_bernoulli_rule_is_one_flag_away():
    assert _immune().tcell_additive_cytotoxicity is True
    assert _immune(tcell_additive_cytotoxicity=False).tcell_additive_cytotoxicity is False


# ── innate arm ───────────────────────────────────────────────────────────

def test_the_innate_arm_no_longer_falls_back_to_point_eight():
    module = _immune()
    assert module.nk_kill_probability == kc.NK_KILL_PER_CONTACT.value == 0.15
    assert module.macrophage_kill_probability == kc.MACROPHAGE_KILL_PER_CONTACT.value
    assert module.kill_probability == 0.8          # unknown lineages only


def test_a_config_that_names_a_shared_probability_still_gets_it():
    module = _immune(kill_probability=0.95)
    assert module.nk_kill_probability == module.macrophage_kill_probability == 0.95
    module = _immune(kill_probability=0.95, nk_kill_probability=0.3)
    assert module.nk_kill_probability == 0.3


def test_the_run_reports_what_its_kill_numbers_stand_on():
    state = _immune().get_innate_state()
    assert state["kill_calibration"] == kc.calibration_report()
    assert state["tcell_additive_cytotoxicity"] is True


# ── composition ──────────────────────────────────────────────────────────

def test_the_slab_infiltrate_is_split_as_prostate_tumours_are():
    from cognisom.dashboard.engine_runner import SCENARIOS, SLAB_IMMUNE_TOTAL
    immune = SCENARIOS["TissueSlab"]["immune"]
    counts = {k: immune[k] for k in ("n_t_cells", "n_nk_cells", "n_macrophages")}
    assert counts == {"n_t_cells": 19, "n_nk_cells": 6, "n_macrophages": 35}
    assert sum(counts.values()) == SLAB_IMMUNE_TOTAL
    assert kc.split_immune(60) == counts


def test_presets_keep_their_character_lineage_by_lineage():
    from cognisom.dashboard.engine_runner import SCENARIOS, slab_variant
    base = slab_variant(SCENARIOS["Baseline"])["immune"]
    therapy = slab_variant(SCENARIOS["Immunotherapy"])["immune"]
    weak = slab_variant(SCENARIOS["Aggressive Tumor"])["immune"]
    assert therapy["n_t_cells"] == round(base["n_t_cells"] * 30 / 12)
    assert weak["n_macrophages"] == round(base["n_macrophages"] * 3 / 6)
    for immune in (base, therapy, weak):
        assert immune["macrophage_polarization_mix"] == kc.PRAD_MACROPHAGE_POLARIZATION
        assert "Thorsson" in immune["macrophage_polarization_source"]


def test_macrophages_are_drawn_per_cell_from_the_mix():
    module = _immune(macrophage_polarization_mix=kc.PRAD_MACROPHAGE_POLARIZATION,
                     macrophage_polarization_source=kc.PRAD_SOURCE)
    pols = [module._initial_macrophage_polarization() for _ in range(4000)]
    assert set(pols) == {-1.0, 0.0, 1.0}
    assert np.mean(np.array(pols) == 1.0) == pytest.approx(0.703, abs=0.03)
    assert np.mean(np.array(pols) == -1.0) == pytest.approx(0.186, abs=0.03)


def test_a_population_prior_is_never_reported_as_the_patient():
    module = _immune(n_macrophages=4,
                     macrophage_polarization_mix=kc.PRAD_MACROPHAGE_POLARIZATION,
                     macrophage_polarization_source=kc.PRAD_SOURCE)
    state = module.get_innate_state()
    assert "population_prior" in str(state)
    assert "patient_measured" not in str(state)


def test_no_mix_still_means_unpolarized():
    module = _immune()
    assert module._initial_macrophage_polarization() == 0.0


# ── oxygen ───────────────────────────────────────────────────────────────

def test_the_anoxic_line_is_one_hundredth_of_a_percent_oxygen():
    # 0.21 == arterial 100 mmHg; 0.01% of 713 mmHg dry gas is 0.071 mmHg.
    mmhg = ANOXIC_O2_LEVEL / 0.21 * 100.0
    assert mmhg == pytest.approx(0.0713, abs=1e-4)


def test_the_renderer_draws_dying_where_the_engine_kills():
    from cognisom.visualization.exporters import (
        DYING_GLUCOSE_THRESHOLD, DYING_O2_THRESHOLD)
    assert DYING_O2_THRESHOLD == pytest.approx(ANOXIC_O2_LEVEL)
    assert DYING_GLUCOSE_THRESHOLD == STARVATION_GLUCOSE_MM


# ── what it does to a run (slow) ─────────────────────────────────────────

def _twelve_hours(preset, **immune):
    from collections import Counter
    from cognisom.core.event_bus import EventTypes
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(1)
    runner = EngineRunner(scenario=preset, duration=12.0, dt=0.05,
                          record_interval=1000, snapshot_interval_hours=None,
                          overrides={"immune": immune} if immune else None)
    runner.build()
    start = runner.engine.modules["cellular"].get_state()["n_cancer"]
    with contextlib.redirect_stdout(io.StringIO()):
        runner.run()
    kills = Counter(e["data"].get("killer_type") for e in runner.event_log
                    if e["event"] == EventTypes.CANCER_KILLED)
    return start, runner.engine.modules["cellular"].get_state()["n_cancer"], kills


@pytest.mark.slow
@pytest.mark.timeout(400)   # one 12 h slab run, ~90 s alone
def test_an_untreated_tumour_is_no_longer_cleared_in_half_a_day():
    # Measured, 12 h, seeds 1-3: Baseline 80 -> 116/114/116. Under the
    # replaced model the same runs ended at 3 (and Immunotherapy at 0).
    start, end, _ = _twelve_hours("Baseline")
    assert start == 80
    assert end > start


@pytest.mark.slow
@pytest.mark.timeout(400)   # one 12 h slab run, ~90 s alone
def test_under_anti_pd1_the_t_cells_do_most_of_the_killing():
    # Measured, 12 h, seeds 1-3: T 57%, NK 28%, macrophage 16% of kills.
    # Before this file every preset's kills were 92-98% innate, and the
    # T-cell compartment had exhausted itself within minutes.
    _, end, kills = _twelve_hours("Immunotherapy")
    assert kills["T_cell"] > kills["NK_cell"] + kills["macrophage"] * 0.5
    assert kills["T_cell"] == max(kills.values())
    assert end > 0


@pytest.mark.slow
@pytest.mark.timeout(400)   # one 12 h slab run, ~90 s alone
def test_the_replaced_model_still_clears_everything():
    # The sweep's anchor: the old flat 0.8 innate rate, one Bernoulli draw
    # per T-cell contact and the five-contact exhaustion count.
    _, end, _ = _twelve_hours(
        "Immunotherapy", nk_kill_probability=0.8,
        macrophage_kill_probability=0.8, tcell_additive_cytotoxicity=False,
        tcell_exhaustion_onset_hours=None,
        macrophage_polarization_mix={"M0": 1.0})
    assert end <= 5
