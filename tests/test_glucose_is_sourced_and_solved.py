"""Glucose keeps its first-order sink, and now for a measured reason.

The oxygen work left glucose with a hand-waved justification: "GLUT1's Km is
of the same order as blood glucose itself, so the saturating and linear forms
are not far apart". That was a guess. Checked against the primary literature
it turns out to be right, which is worth pinning precisely because the
tempting secondary number would have pushed the code the other way.

Brahm 1983 (J Physiol 339:339-54, PMID 6887027) measured glucose transport in
human erythrocytes at 38 C and found simple Michaelis-Menten kinetics with
K1/2 = 6.7 mM for self-exchange and 8.2 mM for net efflux. Both sit ABOVE the
5 mM this field supplies, so the transporter is not saturated at physiological
glucose and uptake varies with concentration across the whole range -- which
is what a first-order sink says.

The widely repeated "GLUT1 Km is about 1 mM, so transport runs at an
essentially constant rate" is a low-temperature result. GLUT1 is kinetically
asymmetric and the asymmetry collapses as temperature rises: Brahm found the
maximal self-exchange flux and the maximal net efflux IDENTICAL at 38 C and
about threefold apart at 10 C. Taking the 1 mM figure would have made this
sink look saturated when at body temperature it is not -- the same shape of
error as reading a cell diameter off the Coulter column instead of the flow
column of the paper that reports both.
"""

import logging

import numpy as np
import pytest

from cognisom.core.tissue_config import DiffusionFieldConfig

#: Brahm 1983, 38 C, human erythrocytes. The field's own supply level is 5 mM.
BRAHM_SELF_EXCHANGE_MM = 6.7
BRAHM_NET_EFFLUX_MM = 8.2


def test_glucose_stays_first_order_because_its_km_is_above_supply():
    glucose = DiffusionFieldConfig.glucose()
    assert glucose.uptake_km == 0.0, (
        "glucose must keep the first-order sink; its half-saturation sits "
        "above the supply concentration, so there is no saturated plateau "
        "for a Michaelis-Menten form to capture"
    )
    assert BRAHM_SELF_EXCHANGE_MM > glucose.initial_value, (
        f"the measured K1/2 ({BRAHM_SELF_EXCHANGE_MM} mM) must exceed the "
        f"supply ({glucose.initial_value} mM) for that argument to hold"
    )
    # And oxygen is the opposite case, which is why the two differ.
    oxygen = DiffusionFieldConfig.oxygen()
    assert oxygen.uptake_km > 0
    assert oxygen.initial_value / oxygen.uptake_km > 100


def test_the_reasoning_is_recorded_with_its_source():
    """A number without its paper is the thing this repo keeps getting wrong."""
    import inspect

    source = inspect.getsource(DiffusionFieldConfig.glucose)
    assert "6887027" in source, "the PMID has to travel with the number"
    assert "6.7" in source
    assert "38 C" in source or "38C" in source, (
        "the temperature is the whole reason this value differs from the "
        "commonly quoted one"
    )


def test_a_low_temperature_km_would_have_changed_the_model():
    """Why the distinction is not pedantry.

    At 5 mM supply, a 1 mM half-saturation leaves the transporter 83%
    saturated -- nearly constant uptake, which is the saturating regime. The
    measured 6.7 mM leaves it 43% saturated, well inside the
    concentration-dependent regime. The two disagree about which sink shape
    is right, so the temperature the measurement was made at decides the
    model.
    """
    supply = DiffusionFieldConfig.glucose().initial_value

    def saturation(km):
        return supply / (km + supply)

    assert saturation(1.0) > 0.8
    assert saturation(BRAHM_SELF_EXCHANGE_MM) < 0.5


def test_the_consumption_rate_is_recorded_as_about_24x_low():
    """Derived from the sourced OCR and a measured O2:glucose ratio.

    BNID 102644 (Rathmell et al., J Immunol) gives 2-3 nmol oxygen per nmol
    glucose in T cells, so a cell respiring at a Seahorse-typical
    3.3e-18 mol O2/s consumes about 1.3e-18 mol glucose/s. In a 10 um voxel
    that is a few mM per hour against the 0.2 mM/h configured.

    The rate is deliberately NOT changed -- calibration is a biology decision
    -- but the arithmetic is checked here so the figure in the comment cannot
    drift away from what it claims to be.
    """
    ocr_mol_per_cell_per_s = 200e-12 / 60 / 1e6      # 200 pmol/min/1e6 cells
    o2_per_glucose = 2.5                             # BNID 102644, midpoint
    voxel_litres = (10e-6 ** 3) * 1000               # 10 um cube, m^3 -> L

    glucose_mol_per_s = ocr_mol_per_cell_per_s / o2_per_glucose
    mM_per_hour = glucose_mol_per_s * 3600 / voxel_litres * 1000

    glucose = DiffusionFieldConfig.glucose()
    configured = glucose.uptake_per_cell_per_hour * glucose.initial_value
    assert configured == pytest.approx(0.2)
    assert mM_per_hour / configured == pytest.approx(24, rel=0.15), (
        f"derived {mM_per_hour:.2f} mM/h against configured {configured} -- "
        f"a factor of {mM_per_hour / configured:.0f}, and the comment in "
        "tissue_config says 24"
    )


def test_the_rejected_bionumber_is_inconsistent_with_the_oxygen_figure():
    """BNID 110890 is third-hand and 42x out, so it is recorded as rejected.

    "Typical glucose consumption 0.2 pmol/hour/cell" works out to 200 mM/h in
    a 10 um voxel, which no O2:glucose ratio can reconcile with a
    Seahorse-typical oxygen consumption. Its reference is a paper on
    thermometry in single cells quoting the number from unnamed primary
    sources. The higher number has the worse provenance, which is the usual
    direction.
    """
    voxel_litres = (10e-6 ** 3) * 1000
    bnid_mM_per_hour = 0.2e-12 / voxel_litres * 1000

    ocr = 200e-12 / 60 / 1e6
    # Even at the most glycolytic ratio anyone reports, 1 O2 per glucose.
    most_glycolytic = ocr / 1.0 * 3600 / voxel_litres * 1000

    assert bnid_mM_per_hour / most_glycolytic > 10, (
        f"{bnid_mM_per_hour:.0f} mM/h against at most "
        f"{most_glycolytic:.1f} mM/h from the oxygen side"
    )


def test_glucose_is_solved_and_the_guard_goes_quiet(caplog):
    """It ran at D*dt/dx^2 = 216, and the guard said so.

    Glucose crosses the 60 um intervessel gap in L^2/D = 6 s against a 180 s
    step, so it is as settled as oxygen by the time the cells have aged once.
    Solving it removes the transient AND the directional error the guard was
    reporting.
    """
    import contextlib
    import io as _io

    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    glucose = DiffusionFieldConfig.glucose()
    assert glucose.steady_state is True
    relaxation = 60.0 ** 2 / glucose.diffusion_coeff
    assert relaxation < 60.0, (
        f"glucose relaxes in {relaxation:.1f} s; if that is no longer short "
        "against a step, solving it instead of advancing it needs rethinking"
    )

    cfg = TissueScaleConfig(
        n_cells=500, grid_shape=(16, 16, 16),
        tissue_size_um=(160., 160., 160.), n_gpus=1, dt=0.05,
        duration=1.0, seed=11, enable_ode=True, enable_diffusion=True,
        enable_cell_cycle=False, enable_mechanics=True)
    with caplog.at_level(logging.WARNING,
                         logger="cognisom.gpu.multi_gpu_diffusion"):
        with contextlib.redirect_stdout(_io.StringIO()):
            engine = TissueSimulationEngine(cfg)
            engine.initialize()
            engine.run()

    about_glucose = [r.getMessage() for r in caplog.records
                     if "glucose" in r.getMessage()]
    assert not about_glucose, about_glucose

    solved = engine._steady_solves
    assert "glucose" in solved and "oxygen" in solved
    for name, result in solved.items():
        assert result.converged, f"{name} did not converge"
        assert result.newton_steps <= 5, (name, result.newton_steps)


def test_a_deeper_sink_deepens_the_solved_depletion():
    """The solve has to respond to the demand, not just converge."""
    import contextlib
    import io as _io

    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    depletions = {}
    for scale in (1.0, 1000.0):
        cfg = TissueScaleConfig(
            n_cells=1200, grid_shape=(16, 16, 16),
            tissue_size_um=(160., 160., 160.), n_gpus=1, dt=0.05,
            duration=1.0, seed=11, enable_ode=True, enable_diffusion=True,
            enable_cell_cycle=False, enable_mechanics=True)
        cfg.fields["glucose"].uptake_per_cell_per_hour *= scale
        with contextlib.redirect_stdout(_io.StringIO()):
            engine = TissueSimulationEngine(cfg)
            engine.initialize()
            engine.run()
        depletions[scale] = engine._steady_solves["glucose"].depletion

    assert depletions[1000.0] > 100 * depletions[1.0], depletions


def test_the_cost_of_a_correct_field_is_stated_at_startup(caplog):
    """Solving beats stepping, and on the flagship preset it does not fit.

    A field that is not 312x out on its own depletion costs about 25x more per
    step than the stepped path it replaces. On the 500^3 ``full_tissue``
    preset -- the one the headline million-cell figure comes from -- that is
    roughly 33 h of solving against a 2 h ``max_runtime_hours``, where the
    stepped path took about 1.25 h.

    This is a regression introduced by the correctness work, and the only
    acceptable version of it is one the user is told about before the run
    rather than at the auto-terminate. The smaller presets fit and must stay
    quiet, or the warning becomes noise and gets ignored.
    """
    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    def warnings_for(cfg):
        engine = TissueSimulationEngine.__new__(TissueSimulationEngine)
        with caplog.at_level(logging.WARNING,
                             logger="cognisom.core.tissue_engine"):
            caplog.clear()
            engine._warn_if_the_solves_will_not_fit(cfg)
        return [r.getMessage() for r in caplog.records]

    assert warnings_for(TissueScaleConfig.small_test()) == []
    assert warnings_for(TissueScaleConfig.medium_test()) == []

    big = warnings_for(TissueScaleConfig.full_tissue())
    assert len(big) == 1, big
    assert "max_runtime_hours" in big[0]
    assert "steady_interval" in big[0], (
        "a warning without a remedy is just bad news")

    # Raising the interval is the remedy, so it has to silence the warning.
    cfg = TissueScaleConfig.full_tissue()
    cfg.steady_interval = 64
    assert warnings_for(cfg) == []


def test_the_interval_defaults_to_solving_every_step():
    """Because staleness is an error the solve does not otherwise have."""
    from cognisom.core.tissue_config import TissueScaleConfig

    assert TissueScaleConfig.small_test().steady_interval == 1
    assert TissueScaleConfig().steady_interval == 1


def test_a_longer_interval_actually_skips_solves():
    """And the field it leaves behind is the previous solve, not a transient."""
    import contextlib
    import io as _io

    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    counts = {}
    for interval in (1, 4):
        cfg = TissueScaleConfig(
            n_cells=400, grid_shape=(16, 16, 16),
            tissue_size_um=(160., 160., 160.), n_gpus=1, dt=0.05,
            duration=0.4, seed=11, enable_ode=True, enable_diffusion=True,
            enable_cell_cycle=False, enable_mechanics=True)
        cfg.steady_interval = interval
        calls = []
        with contextlib.redirect_stdout(_io.StringIO()):
            engine = TissueSimulationEngine(cfg)
            engine.initialize()
            real = engine._diffusion.solve_steady

            def counted(name, *a, _real=real, **kw):
                calls.append(name)
                return _real(name, *a, **kw)

            engine._diffusion.solve_steady = counted
            engine.run()
        counts[interval] = len(calls)

    assert counts[1] > counts[4], counts
    assert counts[4] >= 1, "it must still solve at least once"
