"""The blood runs down inside the engine, not only in a unit test.

``gpu/vessel_transport.py`` was built, calibrated against Movsas (PMID
11586096) and tested -- and had no importer anywhere in the codebase.
The engine's oxygen solve still held every vessel voxel at one tension
along its whole length, an infinite reservoir, which is why the tissue
could not go hypoxic at any realistic vessel spacing. This file is the
wiring: a config knob, a solver method that gathers the global problem
and marches the blood, and a re-pin that keeps the depleting profile
between solves instead of putting every vessel back to the inlet.

Each test is the Phase 6 bar -- that the module, once attached,
*changes state* -- and the one that matters most is the last: a
constant vessel and a perfused one must not produce the same tissue.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from cognisom.core.tissue_config import TissueScaleConfig
from cognisom.core.tissue_engine import TissueSimulationEngine
from cognisom.gpu.vessel_transport import ARTERIAL_PO2_MMHG, PerfusedResult

#: The three calibrated perfusions, um/s (see TissueScaleConfig).
TEXTBOOK, MUSCLE, CARCINOMA = 500.0, 162.0, 48.0

#: A 1 mm capillary, which is what the perfusions were calibrated on.
#: Extraction scales with length over velocity, so a 480 um vessel at
#: the muscle perfusion is half as spent and sits well above 30 mmHg --
#: measured, and the reason this is not a shorter grid.
GRID = (24, 24, 100)

#: OXYGEN_Q_PER_HOUR is the consumption of ONE cell occupying one 10 um
#: voxel, so the calibration's M0 = 8.0 mmHg/s is reached at one cell
#: per voxel. A test grid seeded with a few hundred cells is a few
#: percent of that and barely consumes anything: measured, 3,000 cells
#: in a 24x24x100 grid left the carcinoma perfusion at a median of 42.6
#: mmHg against the 2.4 it is calibrated to. Packing matters, and the
#: engine makes it a free number.
PACKING = 0.7


def _engine(velocity):
    cfg = TissueScaleConfig(
        n_cells=int(PACKING * np.prod(GRID)), grid_shape=GRID,
        tissue_size_um=tuple(10.0 * n for n in GRID), n_gpus=1, dt=0.05,
        duration=0.15, seed=7, enable_ode=False, enable_diffusion=True,
        enable_cell_cycle=False, enable_mechanics=False,
        capillary_velocity_um_per_s=velocity)
    with redirect_stdout(io.StringIO()):
        eng = TissueSimulationEngine(cfg)
        eng.initialize()
        eng.run()
    return eng


@pytest.fixture(scope="module")
def constant():
    return _engine(None)


@pytest.fixture(scope="module")
def textbook():
    return _engine(TEXTBOOK)


@pytest.fixture(scope="module")
def muscle():
    return _engine(MUSCLE)


@pytest.fixture(scope="module")
def perfused():
    return _engine(CARCINOMA)


def _global(engine, name="oxygen"):
    """The field and vessel mask on the config grid, ghost layers off.

    A raw partition carries ghost layers along the split axis -- 48
    becomes 52 -- so reading one and reshaping by the config shape fails
    (and did). The decomposer's gather is the API for the global view.
    """
    solver = engine._diffusion
    field = solver._decomp.gather(
        [solver._mgpu.to_numpy(a) for a in solver._fields[name]]).astype(float)
    mask = solver._decomp.gather(
        [solver._mgpu.to_numpy(a).astype(np.float64)
         for a in solver._vessels[name]]) > 0.5
    return field, mask


def _vessel_profile(engine):
    """Tension along each vessel, mmHg, as (n_vessels, length)."""
    field, mask = _global(engine)
    length = field.shape[engine.VESSEL_AXIS]
    return field[mask].reshape(-1, length) * 760.0


def test_a_constant_vessel_is_the_same_along_its_length(constant):
    """The model as it stood, pinned so the next test means something."""
    along = _vessel_profile(constant)
    assert along.shape[0] > 0
    assert np.allclose(along, along[:, :1]), "a reservoir vessel does not deplete"
    assert constant._perfusion is None


def test_a_perfused_vessel_runs_down_from_inlet_to_outlet(perfused):
    along = _vessel_profile(perfused)
    inlet, outlet = along[:, 0].mean(), along[:, -1].mean()
    assert inlet > outlet, f"inlet {inlet:.1f} <= outlet {outlet:.1f} mmHg"
    # and monotone: blood does not pick oxygen back up on the way
    drops = np.diff(along, axis=1)
    assert (drops <= 1e-6).mean() > 0.95


def test_the_engine_keeps_the_perfused_result(muscle):
    result = muscle._perfusion
    assert isinstance(result, PerfusedResult)
    assert result.converged, result.describe()
    assert 0.0 < result.max_extraction <= 1.0
    assert muscle._steady_solves["oxygen"] is result


def test_the_profile_survives_the_implicit_steps_between_solves(perfused):
    """A scalar re-pin after each step would erase the depletion.

    The run above took several implicit steps after its last perfused
    solve; if the re-pin had used the constant, every vessel voxel
    would read the inlet again.
    """
    solver = perfused._diffusion
    assert "oxygen" in solver._vessel_profiles
    along = _vessel_profile(perfused)
    assert along[:, -1].mean() < along[:, 0].mean()


def _tissue(engine):
    field, mask = _global(engine)
    return field[~mask] * 760.0


def test_perfusion_is_the_controlling_variable(textbook, muscle, perfused):
    """The Phase 6 bar: attached, it must change state -- and the right way.

    Same model, same vessels, same cells; only the blood speed differs.
    Slower blood is more spent by the time it reaches the tissue, so the
    tissue must sit lower, and the hypoxic fraction must grow.
    """
    t, m, c = (np.median(_tissue(e)) for e in (textbook, muscle, perfused))
    assert t > m > c, f"medians textbook {t:.1f}, muscle {m:.1f}, carcinoma {c:.1f}"
    assert (_tissue(perfused) < 10.0).mean() > (_tissue(muscle) < 10.0).mean()
    ex = [e._perfusion.max_extraction for e in (textbook, muscle, perfused)]
    assert ex[0] < ex[1] <= ex[2], ex


def test_the_muscle_calibration_survives_the_wiring(muscle):
    """Movsas's periprostatic muscle is 30.0 mmHg; the unit cell gave 30.0.

    The engine has lumpy per-cell uptake and 60 um spacing where the
    calibration had uniform uptake and 80 um, so a loose band is honest;
    a number outside it would mean the wiring changed the physics.
    """
    assert 20.0 < np.median(_tissue(muscle)) < 40.0


def test_a_depleting_vessel_is_not_the_old_model(constant, perfused):
    """Not a direction claim -- the old model pinned at 40 mmHg and the new
    inlet is 95, so either could sit higher -- just that they differ."""
    assert not np.allclose(_tissue(constant), _tissue(perfused), atol=0.5)


def test_the_axis_the_lattice_runs_along_is_the_axis_the_blood_marches(constant):
    """One constant; a disagreement would march blood across vessels."""
    _, mask = _global(constant)
    axis = constant.VESSEL_AXIS
    # every vessel line along VESSEL_AXIS spans the whole domain
    per_line = mask.sum(axis=axis)
    lengths = np.unique(per_line[per_line > 0])
    assert lengths.tolist() == [mask.shape[axis]], lengths


# ── the config ──────────────────────────────────────────────────────


def test_the_knobs_round_trip_through_json():
    cfg = TissueScaleConfig(capillary_velocity_um_per_s=162.0,
                            capillary_inlet_po2_mmhg=90.0, steady_interval=3)
    back = TissueScaleConfig.from_json(cfg.to_json())
    assert back.capillary_velocity_um_per_s == 162.0
    assert back.capillary_inlet_po2_mmhg == 90.0
    assert back.steady_interval == 3, "steady_interval was never serialised"


def test_the_default_is_the_old_model():
    assert TissueScaleConfig().capillary_velocity_um_per_s is None


def test_the_inlet_default_is_not_a_second_copy():
    """95 mmHg lives in vessel_transport; the config must not drift from it."""
    assert TissueScaleConfig().capillary_inlet_po2_mmhg == ARTERIAL_PO2_MMHG


def test_the_tissue_page_lets_a_user_set_perfusion():
    """The knob existed before the page did; a page that forgot it would
    build a config with the old infinite-reservoir vessels and say nothing."""
    import pathlib
    source = (pathlib.Path(__file__).resolve().parents[1] / "cognisom"
              / "dashboard" / "_pages" / "21_tissue_scale.py").read_text()
    assert "capillary_velocity_um_per_s=capillary_velocity" in source
    # and it warns about the packing trap, which is the other half
    assert "packing" in source and "0.7" in source
