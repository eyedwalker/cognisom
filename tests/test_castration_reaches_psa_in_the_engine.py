"""The testosterone signal has a consumer: the AR/PSA pathway, per cell.

The library-sourced AR system existed, was pinned in isolation, and ran
in no engine. It now runs in every live cell with that cell's tissue DHT
as the supply bound of the AR-DHT complex, so a castrate vessel reaches
PSA through the field, the tissue DHT, and the receptor -- and what it
finds there is the K_psa saturation the 10-17 note recorded: steady PSA
falls about 10% where the clinic sees more than 90%.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

pytest.importorskip("scipy")

from cognisom.core.event_bus import EventBus, EventTypes  # noqa: E402
from cognisom.modules import cellular_module as cm  # noqa: E402
from cognisom.modules.ar_signaling_module import ARSignalingModule  # noqa: E402

MODULES = {"cellular": True, "spatial": True, "vascular": True, "ar_signaling": True,
           "immune": False, "lymphatic": False, "molecular": False, "epigenetic": False,
           "circadian": False, "morphogen": False, "cell_mechanics": False}


def _run(scenario, hours, seed=11):
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(seed)
    runner = EngineRunner(tissue="ring", dt=0.05, duration=hours, scenario=scenario, modules_enabled=MODULES)
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    return runner


@pytest.fixture(scope="module")
def baseline():
    return _run("Baseline", 24.0)


@pytest.fixture(scope="module")
def castrate():
    return _run("Androgen Deprivation", 24.0)


# ── it runs in the engine, in equilibrium ──────────────────────────


def test_every_live_cell_has_a_receptor_state(baseline):
    ar = baseline.engine.modules["ar_signaling"]
    alive = [c.cell_id for c in baseline.engine.modules["cellular"].cells.values() if c.alive]
    assert set(ar.ids) == set(alive) and len(alive) > 0
    assert ar.failed_solves == 0


def test_cells_start_in_the_eugonadal_steady_state_and_stay_there(baseline):
    """A run that began from zeros would show PSA climbing for days and
    call it biology. At eugonadal supply nothing should move."""
    ar = baseline.engine.modules["ar_signaling"]
    psa0 = float(ar.eugonadal_state[ar._i["PSA"]])
    series = baseline.get_time_series()["avg_psa"]
    assert series[0] == pytest.approx(psa0, rel=1e-3)
    assert series[-1] == pytest.approx(psa0, rel=1e-2)
    assert ar.get_state()["ar_dht_mean"] == pytest.approx(cm.TISSUE_DHT_REFERENCE_NM, rel=0.02)


def test_the_complex_is_bounded_by_the_largest_supply_a_cell_has_had(baseline, castrate):
    """The coupling's invariant: DHT_input = k_complex_deg x dht_nM, so the
    AR-DHT complex can never exceed the tissue DHT -- of the LARGEST
    supply the cell has seen, since the complex decays with a 20 h time
    constant and a castrate cell is still carrying what the eugonadal
    supply built. At steady state it equals the tissue DHT exactly."""
    for runner in (baseline, castrate):
        ar = runner.engine.modules["ar_signaling"]
        for cell_id in ar.ids:
            assert ar.species_of(cell_id, "AR_DHT") <= cm.TISSUE_DHT_REFERENCE_NM * (1 + 1e-3)
    ar = baseline.engine.modules["ar_signaling"]
    cells = baseline.engine.modules["cellular"].cells
    for cell_id in ar.ids:
        assert ar.species_of(cell_id, "AR_DHT") == pytest.approx(cells[cell_id].dht_nM, rel=0.02)


def test_it_is_cheap_enough_to_be_on_by_default(baseline):
    assert baseline.engine.modules["ar_signaling"].get_state()["rhs_evals_per_step"] < 100


# ── castration reaches PSA, by the saturated amount ───────────────

def test_castration_lowers_psa_through_an_unsaturated_term(baseline, castrate):
    """Tissue DHT 13.7 -> 3.4 nM, and PSA follows it down.

    Until the Km was sourced this test asserted the opposite -- that PSA
    moved only ~10%, because K_psa came from an in-vitro naked-DNA
    affinity (0.5 nM) and left the induction term 97.6% saturated. With
    the in-cell apparent Kd (130 nM, PMID 26657626) the term is
    responsive and PSA tracks the androgen.

    A Hill-1 term cannot take PSA below the androgen ratio itself --
    0.248 here -- for any Km, which is worth knowing: the clinic's >90%
    fall needs the tumour to lose cells too, and that is modelled
    separately.
    """
    ar_b = baseline.engine.modules["ar_signaling"].get_state()
    ar_c = castrate.engine.modules["ar_signaling"].get_state()
    ratio = ar_c["psa_mean"] / ar_b["psa_mean"]
    assert 0.35 < ratio < 0.75, f"PSA ratio at 24 h {ratio:.3f}"
    assert ratio > 0.248, "and not below the androgen ratio -- Hill-1 cannot"

def test_the_castrate_complex_is_falling_toward_its_new_bound(baseline, castrate):
    b = baseline.engine.modules["ar_signaling"].get_state()["ar_dht_mean"]
    c = castrate.engine.modules["ar_signaling"].get_state()["ar_dht_mean"]
    bound = 0.248 * cm.TISSUE_DHT_REFERENCE_NM
    assert bound < c < b


# ── population bookkeeping ────────────────────────────────────────


class _Cell:
    def __init__(self, cell_id, dht=cm.TISSUE_DHT_REFERENCE_NM):
        self.cell_id, self.dht_nM, self.alive = cell_id, dht, True


class _Cellular:
    def __init__(self, cells):
        self.cells = {c.cell_id: c for c in cells}


@pytest.fixture(scope="module")
def module():
    m = ARSignalingModule({})
    bus = EventBus()
    m.set_event_bus(bus)
    with redirect_stdout(io.StringIO()):
        m.initialize()
    return m, bus


def test_a_daughter_inherits_the_parents_receptor_state_and_a_dead_cell_is_dropped(module):
    m, bus = module
    cells = [_Cell(1, dht=3.0), _Cell(2)]
    m.set_cellular_module(_Cellular(cells))
    for c in cells:
        m.add_cell(c.cell_id)
    for _ in range(40):                      # 2 h: cell 1 drifts castrate-ward
        m.update(0.05)
    assert m.species_of(1, "AR_DHT") < m.species_of(2, "AR_DHT")
    bus.emit(EventTypes.CELL_DIVIDED, {"cell_id": 1, "daughter_id": 3,
                                       "cell_type": "normal", "position": [0, 0, 0]})
    bus.process_events()
    assert np.allclose(m.state_of(3), m.state_of(1))
    bus.emit(EventTypes.CELL_DIED, {"cell_id": 2, "cell_type": "normal",
                                    "cause": "test", "position": [0, 0, 0]})
    bus.process_events()
    assert 2 not in m.index and set(m.ids) == {1, 3}
    assert m.state_of(1) is not None and m.state_of(3) is not None
