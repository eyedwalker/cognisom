"""Phase 4, first signal: testosterone, as transport plus one measured fact.

The field is fed at the capillary wall with free plasma testosterone and
drawn on by every cell; tissue DHT at each cell follows that supply plus
an intracrine share that castration does not touch. Three papers set the
numbers and one of them is the test:

* Titus 2005 (PMID 16000557): benign prostate carries 13.7 pmol/g DHT.
* Page 2006 (PMID 16882745): castration drops serum T 94% but tissue DHT
  only 80% in healthy men.
* Nishiyama 2004 (PMID 15534082): in patients, tissue DHT stays at ~25%
  while serum DHT falls to 7.5%.

A vessel-fed field on its own predicts 6%. That is the negative result
this file pins alongside the model that passes.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.modules import cellular_module as cm
from cognisom.modules.vascular_module import (
    EUGONADAL_FREE_TESTOSTERONE_NM, EUGONADAL_TOTAL_TESTOSTERONE_NM,
    FREE_TESTOSTERONE_FRACTION)


def _run(scenario, hours, seed=11, **overrides):
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(seed)
    runner = EngineRunner(tissue="ring", dt=0.05, duration=hours, scenario=scenario,
                          modules_enabled={"cellular": True, "spatial": True,
                                           "vascular": True, "immune": False,
                                           "lymphatic": False, "molecular": False,
                                           "epigenetic": False, "circadian": False,
                                           "morphogen": False, "cell_mechanics": False},
                          **overrides)
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


def _alive(runner):
    return [c for c in runner.engine.modules["cellular"].cells.values() if c.alive]


# ── the supply ─────────────────────────────────────────────────────


def test_the_wall_is_held_at_free_not_total_testosterone():
    runner = _run("Baseline", 0.2)
    vascular = runner.engine.modules["vascular"]
    assert vascular.arterial_testosterone_nM == pytest.approx(
        EUGONADAL_TOTAL_TESTOSTERONE_NM * FREE_TESTOSTERONE_FRACTION)
    assert vascular.arterial_testosterone_nM == pytest.approx(0.34)
    assert all(c.testosterone == pytest.approx(0.34) for c in vascular.capillaries.values())


def test_the_field_is_flat_at_the_wall_level_and_says_so():
    """phi << 1: transport beats consumption, so every cell sees the
    plasma free level. That uniformity is the model's prediction for
    intraprostatic free testosterone, stated, not hidden."""
    runner = _run("Baseline", 1.0)
    spatial = runner.engine.modules["spatial"]
    state = spatial.get_state()
    assert "testosterone" in spatial.fields
    assert state["fields"]["testosterone"]["thiele_modulus"] < 0.1
    seen = [c.testosterone_nM for c in _alive(runner)]
    assert all(v is not None for v in seen), "the field was never pushed to a cell"
    assert min(seen) > 0.95 * 0.34 and max(seen) <= 0.34 * (1 + 1e-6)
    assert state["interstitial_testosterone_nM_mean"] == pytest.approx(0.34, rel=0.05)


# ── the tissue level ───────────────────────────────────────────────


def test_eugonadal_tissue_dht_is_the_titus_value():
    runner = _run("Baseline", 6.0)
    dht = np.array([c.dht_nM for c in _alive(runner)])
    assert dht.mean() == pytest.approx(cm.TISSUE_DHT_REFERENCE_NM, rel=0.02)
    assert runner.get_time_series()["avg_dht_nM"][-1] == pytest.approx(dht.mean(), rel=1e-6)


def test_castration_leaves_about_a_quarter_of_tissue_dht():
    """Serum T to 6%; tissue DHT to 0.20 + 0.80 x 0.06 = 0.248 of baseline.
    Page measured 0.20 (healthy men, 28 d), Nishiyama 0.25 (patients, 6 mo)."""
    runner = _run("Androgen Deprivation", 24.0)
    assert runner.engine.modules["vascular"].arterial_testosterone_nM == pytest.approx(0.34 * 0.06, rel=1e-6)
    dht = np.array([c.dht_nM for c in _alive(runner)])
    fraction = dht.mean() / cm.TISSUE_DHT_REFERENCE_NM
    assert fraction == pytest.approx(0.248, abs=0.01)
    assert 0.20 <= fraction <= 0.30, "outside the Page-Nishiyama bracket"


def test_transport_alone_predicts_six_percent_which_is_wrong(monkeypatch):
    """The negative result: with no intracrine share the field reproduces
    the serum change, not the tissue measurement."""
    monkeypatch.setattr(cm, "DHT_INTRACRINE_FRACTION", 0.0)
    runner = _run("Androgen Deprivation", 24.0)
    dht = np.array([c.dht_nM for c in _alive(runner)])
    fraction = dht.mean() / cm.TISSUE_DHT_REFERENCE_NM
    assert fraction == pytest.approx(0.06, abs=0.01)
    assert fraction < 0.20, "transport alone must NOT be able to match Page 2006"


def test_the_tissue_relaxes_over_the_stated_time_not_instantly():
    """At 2 h after a step to castrate supply, DHT is 1/e of the way; a
    short run must therefore sit between the two levels."""
    runner = _run("Androgen Deprivation", float(cm.DHT_RELAXATION_H))
    dht = np.array([c.dht_nM for c in _alive(runner)])
    fraction = dht.mean() / cm.TISSUE_DHT_REFERENCE_NM
    expected = 0.248 + (1.0 - 0.248) * np.exp(-1.0)
    assert fraction == pytest.approx(expected, abs=0.03)


def test_a_cell_with_no_field_holds_its_reference():
    cell = cm.CellState(cell_id=1, position=np.zeros(3), cell_type="normal", phase="G1")
    assert cell.testosterone_nM is None and cell.dht_nM == cm.TISSUE_DHT_REFERENCE_NM
    module = cm.CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0})
    module._update_androgen(cell, 10.0)
    assert cell.dht_nM == cm.TISSUE_DHT_REFERENCE_NM
