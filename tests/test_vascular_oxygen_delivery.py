"""The vascular module must keep normal tissue alive, and must still let a
hypoxic core happen.

Before the Krogh recalibration, the per-cell O2 conductance in the Baseline
scenario was 0.0926 /h against a normal-cell demand of 0.15 /h, so the
steady state 0.21 - 0.15/0.0926 = -1.41 did not exist and every cell was
dead by 9.5 h. These tests pin the three things that went wrong: the
magnitude of the conductance, its distance dependence, and its independence
from the time step.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.core import SimulationConfig, SimulationEngine
from cognisom.modules.cellular_module import CellularModule
from cognisom.modules.vascular_module import VascularModule

# cellular_module._update_metabolism / gpu.cell_ops.update_metabolism_vectorized
O2_DEMAND_NORMAL = 0.15      # per hour
O2_DEMAND_CANCER = 0.10
GLUCOSE_DEMAND_CANCER = 0.50  # mM per hour
# cellular_module.update death test
O2_DEATH = 0.02
GLUCOSE_DEATH = 0.5
# vascular_module._check_hypoxia
O2_HYPOXIC = 0.05
ARTERIAL_O2 = 0.21
ARTERIAL_GLUCOSE = 5.0


BASELINE_DT = 0.05


def _baseline(duration=24.0, dt=BASELINE_DT, seed=12345, vascular=None, cellular=None):
    """Run the dashboard Baseline preset and hand back both modules."""
    np.random.seed(seed)
    with redirect_stdout(io.StringIO()):
        engine = SimulationEngine(SimulationConfig(dt=dt, duration=duration))
        engine.register_module(
            "cellular", CellularModule,
            cellular or {"n_normal_cells": 80, "n_cancer_cells": 10},
        )
        engine.register_module(
            "vascular", VascularModule, vascular or {"n_capillaries": 8},
        )
        engine.initialize()
        vas = engine.modules["vascular"]
        cel = engine.modules["cellular"]
        vas.set_cellular_module(cel)
        for _ in range(int(round(duration / dt))):
            engine.step()
    return vas, cel


def _alive(cel, cell_type):
    return [c for c in cel.cells.values() if c.alive and c.cell_type == cell_type]


def test_normal_tissue_survives_a_day_of_the_baseline_scenario():
    """The headline regression: 0/80 normal cells used to be left at 24 h."""
    vas, cel = _baseline()
    normal = _alive(cel, "normal")
    cancer = _alive(cel, "cancer")

    # 80 seeded and division_time_normal = 24 h, so the cohort roughly
    # doubles. The exact count is not the invariant and pinning it makes the
    # test brittle against any unrelated change to division timing -- what
    # matters is that nothing dies and the population grows.
    assert cel.total_deaths == 0, f"{cel.total_deaths} cells died"
    assert len(normal) >= 80, (
        f"the seeded cohort did not survive: {len(normal)} normal cells left"
    )
    assert cel.total_divisions > 0, "no normal cell managed to divide"

    o2 = np.array([c.oxygen for c in normal])
    assert o2.min() > O2_DEATH, "a normal cell is at the death threshold"
    assert not (o2 < O2_HYPOXIC).any(), "a Baseline normal cell is hypoxic"
    # Each settled cell sits at its own Krogh steady state. This replaced a
    # band on the mean, 0.15-0.19, read off a run in which every normal cell
    # divided on the final step and each daughter started at arterial O2:
    # half the tissue was one step old and sitting at 0.21, and the cells
    # that had actually lived there averaged 0.126. Read after the exchange
    # -- metabolism and exchange are applied in turn -- a settled cell sits
    # above the continuous steady state by at most one step's consumption.
    # "Settled" is 8 h since the cell's last division: a daughter starts at
    # its parent's O2, a few um away, and near the Krogh radius, where the
    # conductance is 0.79 /h, it takes that long to forget the difference.
    settled = [c for c in normal if c.age > 8.0]
    assert len(settled) > len(normal) // 3
    gap = np.array([c.oxygen - vas.get_O2_at_position(c.position)
                    for c in settled])
    one_step = O2_DEMAND_NORMAL * BASELINE_DT
    assert gap.min() >= -1e-6, gap.min()
    assert gap.max() <= one_step + 1e-5, gap.max()   # residual relaxation

    gluc = np.array([c.glucose for c in normal])
    assert gluc.min() > GLUCOSE_DEATH
    # Interstitial glucose is ~80-95% of plasma in real tissue.
    assert 0.80 * ARTERIAL_GLUCOSE < gluc.mean() < 0.97 * ARTERIAL_GLUCOSE

    # Delivery must now cover demand rather than fall ~99% short of it
    # (2.159 units over 24 h before the fix).
    assert vas.total_O2_delivered > 250.0, vas.total_O2_delivered

    # Nothing may exceed what the blood carries.
    assert o2.max() <= ARTERIAL_O2 + 1e-12
    assert np.array([c.oxygen for c in cancer]).max() <= ARTERIAL_O2 + 1e-12


def test_exchange_radius_is_the_lethal_corner():
    """A cell at the exchange radius must sit at its own death threshold.

    That identity is what fixes permeability_O2 and permeability_glucose
    with no free parameter: permeability = q_max / (C_arterial - C_death).
    """
    with redirect_stdout(io.StringIO()):
        vas = VascularModule()           # must construct with no config
        vas.initialize()
    cap = next(iter(vas.capillaries.values()))

    # The weight is normalised to exactly 1.0 at the tissue radius.
    assert cap.conductance_weight(cap.exchange_radius) == pytest.approx(1.0)

    # ... so the steady state there is the death threshold, for the most
    # demanding cell type in each species.
    k_o2 = cap.permeability_O2 * 1.0
    assert ARTERIAL_O2 - O2_DEMAND_NORMAL / k_o2 == pytest.approx(O2_DEATH, abs=1e-6)

    k_gluc = cap.permeability_glucose * 1.0
    assert (ARTERIAL_GLUCOSE - GLUCOSE_DEMAND_CANCER / k_gluc
            == pytest.approx(GLUCOSE_DEATH, abs=1e-6))

    # The weight rises monotonically towards the vessel and never inverts.
    radii = np.linspace(cap.radius_um + 0.5, cap.exchange_radius, 60)
    weights = [cap.conductance_weight(r) for r in radii]
    assert all(a >= b for a, b in zip(weights, weights[1:]))


def test_a_hypoxic_core_is_still_reachable():
    """Fixing delivery must not make hypoxia impossible.

    One normal cell at each radial distance from a single straight
    capillary: normoxic near it, hypoxic but alive in the outer half of the
    Krogh cylinder, dead beyond it.
    """
    np.random.seed(7)
    with redirect_stdout(io.StringIO()):
        cel = CellularModule({
            "n_normal_cells": 0, "n_cancer_cells": 0,
            "division_time_normal": 1e9, "division_time_cancer": 1e9,
        })
        cel.initialize()
        vas = VascularModule({"n_capillaries": 1})
        vas.initialize()
    vas.set_cellular_module(cel)
    vas.capillaries.clear()
    vas.next_capillary_id = 0
    vas.add_capillary([-200.0, 0.0, 0.0], [200.0, 0.0, 0.0])

    # 40um rather than 45um for the outer probe: the calibration puts the
    # steady state at exactly the death threshold at r = 50um, so a cell at
    # 45um sits ~0.001 above it and its fate depends on the O(q*dt) offset
    # of the time-stepping scheme. 40um has a 20% margin and still lies in
    # the outer, hypoxic half of the cylinder.
    ids = {d: cel.add_cell(position=[0.0, float(d), 0.0], cell_type="normal")
           for d in (10, 20, 30, 40, 60)}

    dt = 0.05
    for _ in range(int(round(24.0 / dt))):
        cel.update(dt)
        vas.update(dt)

    def state(d):
        cid = ids[d]
        cell = cel.cells.get(cid)
        return cell if (cell is not None and cell.alive) else None

    # Inner half of the cylinder: alive and normoxic.
    for d in (10, 20):
        cell = state(d)
        assert cell is not None, f"cell at {d}um died"
        assert cell.oxygen > O2_HYPOXIC, f"cell at {d}um is hypoxic"
    # Outer half: alive but hypoxic -- this is the hypoxic core.
    for d in (30, 40):
        cell = state(d)
        assert cell is not None, f"cell at {d}um died"
        assert O2_DEATH < cell.oxygen < O2_HYPOXIC, (
            f"cell at {d}um has O2 {cell.oxygen:.4f}, wanted hypoxic-but-alive"
        )
    # Beyond the Krogh radius there is no supply at all.
    assert state(60) is None, "a cell outside the exchange radius survived"
    assert vas.hypoxic_regions == 2, vas.hypoxic_regions


def test_delivery_does_not_depend_on_the_discretisation():
    """Total delivery must be set by demand, not by a loop bound.

    The old code summed over 10 arbitrary sample points per capillary, so
    delivery was proportional to n_samples and halving it flipped cells
    from normoxic to hypoxic. Sweeping dt over 20x is the equivalent test
    for the replacement.
    """
    static = {"n_normal_cells": 80, "n_cancer_cells": 10,
              "division_time_normal": 1e9, "division_time_cancer": 1e9}
    delivered, means = [], []
    for dt in (0.2, 0.05, 0.01):
        vas, cel = _baseline(duration=12.0, dt=dt, cellular=static)
        assert cel.total_deaths == 0, f"deaths at dt={dt}"
        delivered.append(vas.total_O2_delivered)
        means.append(np.mean([c.oxygen for c in _alive(cel, "normal")]))

    spread = (max(delivered) - min(delivered)) / min(delivered)
    assert spread < 0.05, f"delivery varies {spread:.1%} across dt: {delivered}"
    # The remaining drift is the O(q*dt) offset of the explicit step, and it
    # must never be large enough to change a cell's fate.
    assert min(means) > O2_HYPOXIC, means


def test_a_cell_can_never_exceed_blood_levels_at_any_time_step():
    """The transfer is clamped to the driving gradient, so the scheme is
    monotone and bounded for any K*dt -- no CFL condition, no hyperoxia."""
    for dt in (0.01, 0.5, 5.0):
        vas, cel = _baseline(duration=10 * dt, dt=dt)
        for cell in cel.cells.values():
            assert cell.oxygen <= ARTERIAL_O2 + 1e-12, (dt, cell.oxygen)
            assert cell.glucose <= ARTERIAL_GLUCOSE + 1e-12, (dt, cell.glucose)
            assert cell.lactate >= 0.0, (dt, cell.lactate)


def test_hypoxic_region_count_does_not_latch():
    """hypoxic_regions must report the current count, not the last
    non-zero one. The dashboard metric could previously only ratchet up."""
    with redirect_stdout(io.StringIO()):
        cel = CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0,
                              "division_time_normal": 1e9,
                              "division_time_cancer": 1e9})
        cel.initialize()
        vas = VascularModule({"n_capillaries": 1})
        vas.initialize()
    vas.set_cellular_module(cel)

    cid = cel.add_cell(position=[0.0, 0.0, 0.0], cell_type="normal")
    cel.cells[cid].oxygen = 0.01            # hypoxic
    vas.update(0.0)                          # dt=0 -> no transfer
    assert vas.hypoxic_regions == 1

    cel.cells[cid].oxygen = ARTERIAL_O2      # recovered
    vas.update(0.0)
    assert vas.hypoxic_regions == 0, "hypoxic_regions latched"


def test_the_calibration_still_matches_the_cellular_module():
    """The permeabilities are q_max / (C_arterial - C_death). q_max is read
    off cellular_module._update_metabolism, so measure it there rather than
    trusting the constants at the top of this file: if anyone retunes
    consumption, this is the test that notices the vascular calibration no
    longer agrees with it."""
    with redirect_stdout(io.StringIO()):
        cel = CellularModule({"n_normal_cells": 0, "n_cancer_cells": 0})
        cel.initialize()
        vas = VascularModule()
        vas.initialize()
    cap = next(iter(vas.capillaries.values()))

    demands = {}
    for cell_type in ("normal", "cancer"):
        cid = cel.add_cell(position=[0.0, 0.0, 0.0], cell_type=cell_type)
        cell = cel.cells[cid]
        o2_before, gluc_before = cell.oxygen, cell.glucose
        cel._update_metabolism(cell, 1.0)          # one hour
        demands[cell_type] = (o2_before - cell.oxygen,
                              gluc_before - cell.glucose)

    o2_max = max(d[0] for d in demands.values())
    gluc_max = max(d[1] for d in demands.values())
    assert o2_max == pytest.approx(O2_DEMAND_NORMAL)
    assert gluc_max == pytest.approx(GLUCOSE_DEMAND_CANCER)

    assert cap.permeability_O2 == pytest.approx(
        o2_max / (ARTERIAL_O2 - O2_DEATH), rel=1e-6
    )
    assert cap.permeability_glucose == pytest.approx(
        gluc_max / (ARTERIAL_GLUCOSE - GLUCOSE_DEATH), rel=1e-6
    )
