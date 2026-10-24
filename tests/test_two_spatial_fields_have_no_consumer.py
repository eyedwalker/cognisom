"""Two of SpatialModule's six fields are solved and read by nobody.

                     solved   consumed by
    oxygen             yes    -- nothing
    glucose            yes    -- nothing
    cytokine           yes    immune chemotaxis (gradient)
    testosterone       yes    cell.testosterone_nM -> tissue DHT -> AR
    lactate            yes    cell.lactate_mM -> the T/NK kill gate
    ifn_gamma          yes    cell.ifn_gamma_level -> MHC-I, polarisation

The two with no consumer are the two oldest. Both describe quantities
the engine does model -- just by another route: cell.oxygen and
cell.glucose are written by VascularModule._exchange_with_tissue over a
Krogh distance and drained by CellularModule._update_metabolism.

Below, the oxygen case in detail, because it also could not make a cell
hypoxic if something did read it.

This is a guard, not a fix. It states two measured facts so neither can
be rediscovered as news, and so that a change which repairs either is
recognised as a change.

ONE: no consumer. ``SpatialModule`` creates and solves an ``oxygen``
field every update and reports ``interstitial_O2_mean`` from it, but
``cell.oxygen`` is written by ``VascularModule._exchange_with_tissue``
(per-cell delivery over a Krogh distance) and drained by
``CellularModule._update_metabolism``. The field is not in
``FIELDS_TO_CELLS``, and no module outside spatial_module reads it. So
hypoxia in this engine is real but comes from the vascular path; the
field is scenery.

TWO: it is far too weak to matter anyway. The field's sink is taken
from the cellular module's intracellular pool rate -- ``cell.oxygen -=
0.15 * dt`` -- and that is a different quantity from volumetric tissue
consumption. Converted: 0.15 field-units/h over a cell's own volume is
0.0198 mmHg/s of tissue at 100% packing, against the 8.0 mmHg/s that
``core.tissue_config`` derives and bounds with Movsas (PMID 11586096)
and Gullino. 403x low packed solid; 37,700x low at the presets' ~1%.

The same category error produced the lactate result this morning: the
pool rate is ~300x below a glycolytic tumour's production. A field's
per-cell rate is not the cell's own bookkeeping rate, and this repo has
now been bitten by that twice.

Fixing it is a separate change with consequences -- a field that can
make cells hypoxic would move most of the pinned 24 h numbers and give
the engine two competing oxygen paths to reconcile, which is a Phase 8
question, not a constant to bump.
"""

from __future__ import annotations

import inspect
import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.modules import spatial_module as sm
from cognisom.modules.spatial_module import SpatialModule

ARTERIAL_FIELD_UNITS = 0.21      # the module's own 0.21 == arterial
ARTERIAL_MMHG = 100.0            # the mapping vascular_module documents
SOURCED_TISSUE_MMHG_PER_S = 8.0  # core.tissue_config, Movsas/Gullino-bounded


def _volumetric_mmhg_per_s(field_units_per_h: float) -> float:
    """A per-cell field-unit drain, read as tissue consumption at 100%
    packing, in mmHg/s."""
    return field_units_per_h / ARTERIAL_FIELD_UNITS * ARTERIAL_MMHG / 3600.0


# ── one: nothing reads it ─────────────────────────────────────────


def test_neither_oxygen_nor_glucose_is_pushed_to_any_cell():
    pushed = {name for name, _ in SpatialModule.FIELDS_TO_CELLS}
    assert "oxygen" not in pushed and "glucose" not in pushed
    assert pushed == {"testosterone", "lactate", "ifn_gamma"}, (
        "if either joins this list, that field has a consumer and this "
        "guard should be rewritten, not deleted")


def test_the_fields_that_do_have_consumers_still_have_them():
    """The other side of the ledger, so this guard cannot be read as
    'the spatial module does nothing'."""
    from cognisom.modules.immune_module import ImmuneModule
    module = ImmuneModule({"n_t_cells": 0, "n_nk_cells": 0, "n_macrophages": 0})
    assert module.chemotaxis_field == "cytokine", "the chemokine field is read"
    pushed = {name for name, _ in SpatialModule.FIELDS_TO_CELLS}
    assert pushed == {"testosterone", "lactate", "ifn_gamma"}


def test_cell_oxygen_and_glucose_come_from_the_vascular_path_not_the_fields():
    from cognisom.modules.vascular_module import VascularModule
    from cognisom.modules.cellular_module import CellularModule

    writes = inspect.getsource(VascularModule._exchange_with_tissue)
    assert "cell.oxygen +=" in writes, "the vascular module is the supplier"
    assert "cell.glucose +=" in writes, "and of glucose, by the same route"
    drains = inspect.getsource(CellularModule._update_metabolism)
    assert "cell.oxygen -=" in drains, "the cellular module is the sink"
    assert "cell.glucose -=" in drains
    # and neither consults a field
    for source in (writes, drains):
        assert "get_concentration" not in source


def test_both_are_still_solved_and_reported_so_they_look_alive():
    module = SpatialModule({})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    state = module.get_state()
    for name in ("oxygen", "glucose"):
        assert name in module.fields
        assert name in state["fields"], "reported like any other field"
    assert "interstitial_O2_mean" not in state, (
        "only reported once cells are attached -- see the module's get_state")


# ── two: and it could not bite if it were read ────────────────────


def test_the_fields_oxygen_sink_is_hundreds_of_times_too_weak():
    packed = _volumetric_mmhg_per_s(sm.SpatialModule.O2_DEMAND_NORMAL)
    assert packed == pytest.approx(0.0198, abs=5e-4)
    ratio = SOURCED_TISSUE_MMHG_PER_S / packed
    assert 350 < ratio < 450, f"{ratio:,.0f}x too low at 100% packing"


def test_at_the_shipped_packing_it_is_four_orders_out():
    module = SpatialModule({})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    v_cell = (4.0 / 3.0) * np.pi * module.cell_radius_um ** 3
    nx, ny, nz = module.grid_size
    packing = 90 * v_cell / (nx * ny * nz * module.resolution ** 3)
    assert packing == pytest.approx(0.011, abs=0.003), "the presets' density"
    effective = _volumetric_mmhg_per_s(sm.SpatialModule.O2_DEMAND_NORMAL) * packing
    assert SOURCED_TISSUE_MMHG_PER_S / effective > 10_000


def test_the_per_cell_rate_is_not_absurd_which_is_why_it_survived():
    """0.15/h is a reasonable rate for a CELL draining its own pool -- a
    cell cut off from supply hits the 0.05 hypoxia flag in about an hour.
    It is only wrong as a per-VOLUME tissue sink. That is the category
    error, and it is why reading it looks fine."""
    hours_to_flag = (ARTERIAL_FIELD_UNITS - 0.05) / sm.SpatialModule.O2_DEMAND_NORMAL
    assert 0.8 < hours_to_flag < 1.5


def test_the_field_stays_arterial_even_in_a_packed_block():
    """The consequence, end to end: solve the field with cells at 80%
    packing and it does not move off the vessel level."""
    class _Cell:
        def __init__(self, position):
            self.position = np.asarray(position, dtype=float)
            self.cell_type = "cancer"
            self.alive = True
            self.oxygen, self.glucose = 0.21, 5.0

    class _Cellular:
        def __init__(self, cells):
            self.cells = {i: c for i, c in enumerate(cells)}

    class _Capillary:
        start = np.array([20.0, 100.0, 50.0])
        end = np.array([180.0, 100.0, 50.0])
        oxygen, glucose = 0.21, 5.0

        def get_position(self, t):
            return self.start + t * (self.end - self.start)

    class _Vascular:
        capillaries = {0: _Capillary()}
        arterial_O2, arterial_glucose = 0.21, 5.0

    module = SpatialModule({})
    with redirect_stdout(io.StringIO()):
        module.initialize()
    nx, ny, nz = module.grid_size
    v_cell = (4.0 / 3.0) * np.pi * module.cell_radius_um ** 3
    n = int(0.8 * nx * ny * nz * module.resolution ** 3 / v_cell)
    rng = np.random.default_rng(0)
    points = rng.uniform([20, 20, 10], [180, 180, 90], size=(n, 3))
    module.set_cellular_module(_Cellular([_Cell(p) for p in points]))
    module.set_vascular_module(_Vascular())
    module._sync_sources_sinks()
    module.fields["oxygen"].update(module.update_interval)
    far = module.fields["oxygen"].get_concentration_at(np.array([100.0, 180.0, 50.0]))
    assert far > 0.95 * 0.21, (
        f"{n} cells at 80% packing leave the far corner at {far:.4f} of 0.21 -- "
        "a tissue consuming oxygen would not")
