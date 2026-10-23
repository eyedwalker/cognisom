"""Phase 4, third signal: IFN-gamma as a field, and the range it buys.

The immune module has used IFN-gamma for a while as a COUNT of activated
T cells within 30 um -- the measured paracrine range (Oyler-Yaniv et al.,
Immunity 2017;46:609-20, PMID 28389069) -- and said plainly that it was
not a concentration. That paper's actual finding is that cytokine
propagation is a diffusion-CONSUMPTION mechanism whose length scale
sqrt(D/k) shrinks as the consuming cell density rises, which a fixed
radius cannot express.

So: a field secreted by activated T and NK cells, consumed by every
cell, pushed to each cell, and one consumer -- tumour MHC-I rising with
the local level (Naganuma 1998, PMID 9821108: mean +50% across five
glioma lines; Martini 2010, PMID 20304037: same direction in a prostate
model, no magnitude). The gain and the saturation point are estimates
and the module says so.

What this file pins is the mechanism and the two numbers that matter:
that the range follows sqrt(D/k_eff) and therefore shortens with
density, and that a BYSTANDER tumour cell -- one no T cell is touching
-- is made more visible, which the contact-only PD-L1 bump cannot do.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

import cognisom.modules.spatial_module as sm
from cognisom.modules import cellular_module as cm
from cognisom.modules.spatial_module import SpatialModule


# ── the field: range follows diffusion over consumption ────────────


class _Cell:
    def __init__(self, position, cell_type="normal"):
        self.position = np.asarray(position, dtype=float)
        self.cell_type = cell_type
        self.alive = True
        self.oxygen, self.glucose = 0.21, 5.0


class _Cellular:
    def __init__(self, cells):
        self.cells = {i: c for i, c in enumerate(cells)}


def _solved_field(n_cells, source_rate=100.0):
    module = SpatialModule({})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    rng = np.random.default_rng(0)
    points = rng.uniform([30, 30, 15], [170, 170, 85], size=(n_cells, 3))
    module.set_cellular_module(_Cellular([_Cell(p) for p in points]))
    module.register_secretion(
        "ifn_gamma", lambda: [(np.array([100.0, 100.0, 50.0]), source_rate)])
    module._sync_sources_sinks()
    for field in module.fields.values():
        field.update(module.update_interval)
    return module, module.fields["ifn_gamma"]


def _decay_length(n_cells):
    """The theory the solver is checked against: L = sqrt(D / k_eff)."""
    module = SpatialModule({})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    v_cell = (4.0 / 3.0) * np.pi * module.cell_radius_um ** 3
    v_voxel = module.resolution ** 3
    nx, ny, nz = module.grid_size
    k_eff = sm.IFN_GAMMA_UPTAKE_PER_H * (v_cell / v_voxel) * n_cells / (nx * ny * nz)
    return float(np.sqrt(sm.IFN_GAMMA_DIFFUSION_UM2_S * 3600.0 / k_eff))


def _profile(field, distances=(0, 20, 40, 80)):
    values = [field.get_concentration_at(np.array([100.0 + d, 100.0, 50.0]))
              for d in distances]
    return [v / values[0] for v in values]


def test_the_field_falls_off_and_falls_off_faster_when_cells_are_denser():
    _, sparse = _solved_field(90)
    _, dense = _solved_field(3000)
    p_sparse, p_dense = _profile(sparse), _profile(dense)
    assert p_sparse == sorted(p_sparse, reverse=True), "monotone decreasing"
    assert p_dense == sorted(p_dense, reverse=True)
    # at every distance past the source voxel the dense tissue has less
    for a, b in zip(p_sparse[1:], p_dense[1:]):
        assert b < a
    assert p_dense[2] < 0.1, "40 um into dense tissue the signal is nearly gone"


def test_the_range_is_the_one_the_constant_claims():
    """200/h is chosen to put the range at Oyler-Yaniv's tens of um in
    real tissue, not in the presets' sparse block. Hold both ends."""
    assert 40.0 <= _decay_length(3000) <= 60.0, "dense: tens of micrometres"
    assert _decay_length(90) > 200.0, "1% packed: a broad field, stated not tuned"
    # and the solver agrees with the formula to within a voxel's blur
    _, dense = _solved_field(3000)
    at_L = dense.get_concentration_at(np.array([100.0 + _decay_length(3000), 100.0, 50.0]))
    at_0 = dense.get_concentration_at(np.array([100.0, 100.0, 50.0]))
    assert 0.0 < at_L / at_0 < 0.2


def test_no_activated_effector_means_no_signal_at_all():
    module = SpatialModule({})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    module.set_cellular_module(_Cellular([_Cell([100.0, 100.0, 50.0])]))
    module.register_secretion("ifn_gamma", lambda: [])
    module._sync_sources_sinks()
    for field in module.fields.values():
        field.update(module.update_interval)
    assert module.fields["ifn_gamma"].get_concentration_at(
        np.array([100.0, 100.0, 50.0])) == 0.0


# ── the source: activated T and NK cells only ─────────────────────


def test_only_activated_t_and_nk_cells_secrete():
    from cognisom.modules.immune_module import ImmuneModule
    module = ImmuneModule({"n_t_cells": 0, "n_nk_cells": 0, "n_macrophages": 0})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    t_on = module.add_immune_cell([10.0, 10.0, 10.0], "T_cell")
    t_off = module.add_immune_cell([20.0, 10.0, 10.0], "T_cell")
    nk_on = module.add_immune_cell([30.0, 10.0, 10.0], "NK_cell")
    mac = module.add_immune_cell([40.0, 10.0, 10.0], "macrophage")
    for cid in (t_on, nk_on, mac):
        module.immune_cells[cid].activated = True
    sources = module.ifn_gamma_sources_now()
    assert len(sources) == 2 and module.ifn_gamma_sources == 2
    placed = {tuple(np.round(p, 1)) for p, _ in sources}
    assert (10.0, 10.0, 10.0) in placed and (30.0, 10.0, 10.0) in placed
    assert all(r == module.ifn_gamma_secretion_per_h for _, r in sources)
    module.immune_cells[t_on].in_blood = True
    assert len(module.ifn_gamma_sources_now()) == 1, "a cell in a vessel is not in tissue"


def test_linking_the_spatial_module_registers_the_provider():
    from cognisom.modules.immune_module import ImmuneModule
    module = ImmuneModule({"n_t_cells": 0, "n_nk_cells": 0, "n_macrophages": 0})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    spatial = SpatialModule({})
    with redirect_stdout(io.StringIO()):
        spatial.initialize()
    module.set_spatial_module(spatial)
    assert spatial._secretion_providers.get("ifn_gamma")


# ── the consumer: MHC-I, on bystanders too ────────────────────────


def _cell(mhc1=0.3, ifn=None):
    c = cm.CellState(cell_id=1, position=np.zeros(3), cell_type="cancer", phase="G1")
    c.mhc1_expression = mhc1
    c.ifn_gamma_level = ifn
    return c


def test_mhc1_rises_toward_the_measured_gain_and_returns_to_baseline():
    module = cm.CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0})
    cell = _cell(mhc1=0.3, ifn=1.0)
    # 48 h = 8 relaxation times; at 20 h it is only 96% of the way, which
    # is the point of having a time constant at all.
    for _ in range(960):
        module._update_mhc1(cell, 0.05)
    assert cell.mhc1_baseline == pytest.approx(0.3)
    assert cell.mhc1_expression == pytest.approx(0.3 * (1 + cm.MHC1_IFN_GAIN), rel=1e-3)
    cell.ifn_gamma_level = 0.0
    for _ in range(960):
        module._update_mhc1(cell, 0.05)
    assert cell.mhc1_expression == pytest.approx(0.3, rel=1e-3), "back to ITS baseline, not 1.0"


def test_the_response_takes_hours_not_steps():
    module = cm.CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0})
    cell = _cell(mhc1=0.3, ifn=1.0)
    module._update_mhc1(cell, 0.05)
    assert cell.mhc1_expression < 0.31, "one step moves it barely"
    for _ in range(int(cm.MHC1_RESPONSE_H / 0.05) - 1):
        module._update_mhc1(cell, 0.05)
    full = 0.3 * cm.MHC1_IFN_GAIN
    assert cell.mhc1_expression == pytest.approx(0.3 + full * (1 - 1 / np.e), rel=0.02)


def test_a_cell_with_no_field_is_untouched():
    module = cm.CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0})
    cell = _cell(mhc1=0.3, ifn=None)
    for _ in range(100):
        module._update_mhc1(cell, 0.05)
    assert cell.mhc1_expression == 0.3 and cell.mhc1_baseline is None


def test_a_daughter_keeps_the_level_it_returns_to():
    module = cm.CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0})
    parent_id = module.add_cell(position=np.array([10.0, 10.0, 10.0]), cell_type="cancer")
    parent = module.cells[parent_id]
    parent.mhc1_expression, parent.ifn_gamma_level = 0.3, 1.0
    for _ in range(200):
        module._update_mhc1(parent, 0.05)
    assert parent.mhc1_baseline == pytest.approx(0.3)
    module._divide_cell(parent_id)
    daughter = module.cells[max(module.cells)]
    assert daughter.mhc1_baseline == pytest.approx(0.3)
    assert daughter.mhc1_expression == pytest.approx(parent.mhc1_expression)


def test_a_bystander_is_made_visible_which_contact_alone_cannot_do():
    """Two tumour cells, one engaged and one 20 um away that no effector
    touches. The field reaches both; a contact-only mechanism reaches
    one. That difference is the point of making this a field."""
    module = SpatialModule({})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    engaged = _Cell([100.0, 100.0, 50.0], "cancer")
    bystander = _Cell([120.0, 100.0, 50.0], "cancer")
    module.set_cellular_module(_Cellular([engaged, bystander]))
    module.register_secretion(
        "ifn_gamma", lambda: [(np.array([100.0, 100.0, 50.0]), 1.0)])
    module._sync_sources_sinks()
    for field in module.fields.values():
        field.update(module.update_interval)
    module._push_fields_to_cells()
    assert engaged.ifn_gamma_level > 0.0
    assert bystander.ifn_gamma_level > 0.0, "the bystander is in the signal"
    assert bystander.ifn_gamma_level < engaged.ifn_gamma_level, "and less of it"


# ── in the engine ─────────────────────────────────────────────────


def test_the_mhc1_rise_reaches_the_t_cell_kill_rule():
    """What the signal buys, measured where it is live: a tumour cell at
    0.45 is several times more killable by a TCR-matched T cell than the
    same cell at 0.30. This is the Hill rule in tcell_kill, and it is the
    only path in the engine that reads mhc1_expression."""
    from cognisom.engine.py.immune.tcell_kill import kill_probability
    low = kill_probability(affinity=0.5, mhc_level=0.30,
                           costimulation=1.0, checkpoint_block=0.0)
    high = kill_probability(affinity=0.5, mhc_level=0.45,
                            costimulation=1.0, checkpoint_block=0.0)
    assert high > 3.0 * low, (low, high)


def test_in_the_shipped_preset_the_rise_changes_no_outcome_and_why():
    """The honest other half. Measured over 8 h with 300 cancer cells:
    MHC-I rises 0.300 -> 0.380, and the kill count does not move, because
    every kill in this configuration is made by an NK cell or a
    macrophage -- and the NK path falls back to a flat baseline when the
    patient's class-I typing is not on record, so it never reads
    mhc1_expression. The T-cell path reads it and is the one that is
    inert here. Wiring NK missing-self to patient HLA is the next step,
    and this test is what will catch it when that changes."""
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from collections import Counter
    from cognisom.core.event_bus import EventTypes
    from cognisom.dashboard.engine_runner import EngineRunner
    over = {"cellular": {"n_cancer_cells": 300, "n_normal_cells": 100},
            "immune": {"n_t_cells": 12, "n_nk_cells": 6, "n_macrophages": 4}}
    results = {}
    for label, extra in (("on", {}), ("off", {"ifn_gamma_secretion_per_h": 0.0})):
        np.random.seed(5)
        config = {k: dict(v) for k, v in over.items()}
        config["immune"].update(extra)
        runner = EngineRunner(dt=0.05, duration=8.0, scenario="Baseline", overrides=config)
        with redirect_stdout(io.StringIO()):
            runner.build()
            runner.run()
        killers = Counter(e["data"].get("killer_type")
                          for e in runner.event_log
                          if e["event"] == EventTypes.CANCER_KILLED)
        cancer = [c for c in runner.engine.modules["cellular"].cells.values()
                  if c.alive and c.cell_type == "cancer"]
        results[label] = (float(np.mean([c.mhc1_expression for c in cancer])), killers)
    (mhc_on, killers_on), (mhc_off, killers_off) = results["on"], results["off"]
    assert mhc_off == pytest.approx(0.30, abs=1e-6), "no signal, no change"
    assert mhc_on > 0.36, f"the field raised MHC-I to {mhc_on:.3f}"
    assert sum(killers_on.values()) == sum(killers_off.values())
    assert killers_on["T_cell"] == 0, "no T-cell kill here: that is why nothing moved"
    assert killers_on["NK_cell"] > 0


def test_a_run_carries_the_field_to_every_cell():
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(7)
    runner = EngineRunner(dt=0.05, duration=2.0, scenario="Baseline")
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    cells = [c for c in runner.engine.modules["cellular"].cells.values() if c.alive]
    assert cells and all(c.ifn_gamma_level is not None for c in cells)
    state = runner.engine.modules["spatial"].get_state()
    assert "interstitial_ifn_gamma_mean" in state
    assert "ifn_gamma" in runner.engine.modules["spatial"].fields
