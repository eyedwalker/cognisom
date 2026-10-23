"""Phase 4, second signal: lactate -- and what transport says about it.

The field is fed by every cell at its production rate and held at the
vessels at the lactate the blood carries, which is a budget, not a
Dirichlet at arterial level: C_tissue = C_arterial + P / F (production
per minute of tissue over perfusion). T and NK killing is gated on the
interstitial lactate at the effector (Fischer 2007, PMID 17255361: 50%
loss of cytotoxic activity; Brand 2016, PMID 27641098: T and NK alike).

The finding this file pins: on the shipped block, at ~1% of a cell per
voxel and the pool model's production rate, the field is flat at blood
level and the gate is exactly 1.0 -- a sparse, perfused block is not
acidic, and no immune number moves. Walenta's 6-10 umol/g (PMID
10706105) needs a packed tumour AND a glycolytic production rate, and
the budget says precisely how much of each.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.engine.py.immune.lactate import lactate_attenuation
from cognisom.modules.spatial_module import SpatialModule


# ── the gate ───────────────────────────────────────────────────────


def test_the_gate_is_one_at_normal_lactate_and_half_at_twenty():
    assert lactate_attenuation(None) == 1.0
    assert lactate_attenuation(0.0) == 1.0
    assert lactate_attenuation(1.0) == 1.0
    assert lactate_attenuation(2.0) == 1.0
    assert lactate_attenuation(11.0) == pytest.approx(0.75)
    assert lactate_attenuation(20.0) == pytest.approx(0.5)
    assert lactate_attenuation(40.0) == pytest.approx(0.5), "Fischer's 50% is the ceiling"
    assert lactate_attenuation(20.0, max_suppression=0.0) == 1.0


class _Field:
    def __init__(self, value):
        self.value = value

    def get_concentration_at(self, position):
        return self.value


class _Spatial:
    def __init__(self, value):
        self.fields = {"lactate": _Field(value)}


def test_the_immune_module_reads_the_field_at_the_effector():
    from cognisom.modules.immune_module import ImmuneModule
    module = ImmuneModule({"n_t_cells": 0, "n_nk_cells": 0, "n_macrophages": 0})

    class _Effector:
        position = np.array([50.0, 50.0, 50.0])
    assert module._lactate_attenuation_at(_Effector()) == 1.0, "no field, no effect"
    module.set_spatial_module(_Spatial(1.0))
    assert module._lactate_attenuation_at(_Effector()) == 1.0
    module.set_spatial_module(_Spatial(20.0))
    assert module._lactate_attenuation_at(_Effector()) == pytest.approx(0.5)


# ── the budget ─────────────────────────────────────────────────────


class _Cell:
    def __init__(self, cell_type):
        self.cell_type, self.alive = cell_type, True


class _Cellular:
    def __init__(self, n_cancer, n_normal=0):
        self.cells = {i: _Cell("cancer" if i < n_cancer else "normal") for i in range(n_cancer + n_normal)}


def _spatial(**config):
    module = SpatialModule(config)
    with redirect_stdout(io.StringIO()):
        module.initialize()
    return module


def test_the_shipped_block_cannot_be_acidic_at_the_pool_rate():
    """1% packing, 0.3 mM/h on the cell volume: P/F is a few thousandths
    of a millimolar. Flat at blood level, gate exactly 1.0."""
    sp = _spatial()
    sp.set_cellular_module(_Cellular(n_cancer=10, n_normal=80))
    level = sp._lactate_vessel_level()
    assert sp.lactate_budget["packing_fraction"] < 0.02
    assert level == pytest.approx(1.0, abs=0.01)
    assert lactate_attenuation(level) == 1.0


def test_walenta_s_range_needs_a_packed_tumour_at_a_glycolytic_rate():
    """70% packing at prostate perfusion (0.15 mL/mL/min) reaches 6-10 mM
    only at ~100 mM/h on the cell volume -- a glycolytic rate, 300x the
    pool model's. Both halves are required: dense at the pool rate, or
    sparse at the glycolytic rate, stay under the 2 mM onset."""
    v_cell = (4.0 / 3.0) * np.pi * 5.0 ** 3
    v_domain = 200 * 200 * 100 * 1.0  # default grid at 10 um: 20x20x10 voxels of 1000 um^3
    sp = _spatial()
    nx, ny, nz = sp.grid_size
    v_domain = nx * ny * nz * sp.resolution ** 3
    n_dense = int(0.7 * v_domain / v_cell)
    dense_glycolytic = _spatial(lactate_production_cancer_per_h=100.0)
    dense_glycolytic.set_cellular_module(_Cellular(n_cancer=n_dense))
    level = dense_glycolytic._lactate_vessel_level()
    assert 6.0 <= level <= 10.0, f"{level:.2f} mM"
    dense_pool = _spatial()
    dense_pool.set_cellular_module(_Cellular(n_cancer=n_dense))
    assert dense_pool._lactate_vessel_level() < 2.0
    sparse_glycolytic = _spatial(lactate_production_cancer_per_h=100.0)
    sparse_glycolytic.set_cellular_module(_Cellular(n_cancer=10, n_normal=80))
    assert sparse_glycolytic._lactate_vessel_level() < 2.0


def test_the_budget_is_fick_s_principle():
    sp = _spatial(lactate_production_cancer_per_h=60.0)
    sp.perfusion_ml_per_ml_min = 0.2
    sp.blood_lactate_mM = 1.0
    sp.set_cellular_module(_Cellular(n_cancer=1000))
    level = sp._lactate_vessel_level()
    b = sp.lactate_budget
    assert level == pytest.approx(1.0 + b["production_mM_per_min"] / 0.2)
    v_cell = (4.0 / 3.0) * np.pi * sp.cell_radius_um ** 3
    nx, ny, nz = sp.grid_size
    assert b["production_mM_per_min"] == pytest.approx(60.0 * 1000 * v_cell / (nx * ny * nz * sp.resolution ** 3) / 60.0)


# ── in the engine ─────────────────────────────────────────────────


def test_the_field_is_flat_and_the_kills_are_unchanged_in_baseline():
    os.environ.setdefault("COGNISOM_ALLOW_PWM_FALLBACK", "1")
    from cognisom.dashboard.engine_runner import EngineRunner
    np.random.seed(7)
    runner = EngineRunner(dt=0.05, duration=2.0, scenario="Baseline")
    with redirect_stdout(io.StringIO()):
        runner.build()
        runner.run()
    state = runner.engine.modules["spatial"].get_state()
    assert state["lactate_budget"]["vessel_mM"] == pytest.approx(1.0, abs=0.01)
    assert state["interstitial_lactate_mM_mean"] == pytest.approx(1.0, abs=0.01)
    cells = [c for c in runner.engine.modules["cellular"].cells.values() if c.alive]
    assert all(c.lactate_mM is not None for c in cells)
    # the same seed without the gate gives the same kills: it is inert here
    np.random.seed(7)
    off = EngineRunner(dt=0.05, duration=2.0, scenario="Baseline",
                       overrides={"immune": {"lactate_max_suppression": 0.0}})
    with redirect_stdout(io.StringIO()):
        off.build()
        off.run()
    assert (runner.engine.modules["immune"].get_state()["total_kills"]
            == off.engine.modules["immune"].get_state()["total_kills"])
