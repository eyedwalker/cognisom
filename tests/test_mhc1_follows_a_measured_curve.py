"""The oxygen response is measured; the baselines are still guesses.

``mhc1`` was filled from two chosen constants under an instruction to accept
a labelled estimate if no effect size could be sourced. The source then
arrived: Estephan et al., EMBO J 2025;44:903-922,
doi 10.1038/s44318-024-00319-7, Figure 2C -- HT29 cells at 21, 2, 1, 0.5 and
<0.1% O2 for 24 h, surface MHC I by flow cytometry.

Fitting the module's curve to those five points corrected three things:

    quantity        guessed        measured
    floor           0.35           0.373
    half-point      1.5% O2        0.822% O2
    shape           hyperbolic     Hill, n = 2.56

The floor was a good guess. The half-point was nearly twice too high. The
SHAPE was simply wrong, and that was the costly one -- the hyperbola put a
cell at 2% O2 down to 0.72 of its presentation where the measurement says
0.96, because a hyperbola cannot be flat at the top and steep in the middle.
Residual sums of squares across the five points: Hill 0.0006, Michaelis-Menten
0.025, the curve that shipped 0.079.

The mechanism in the old justification was wrong too. The half-point was
argued from where HIF-1alpha stabilises; the paper shows HIF1alpha-null RKO
cells and ARNT/HIF1beta-knockdown HT29 cells both still lose MHC I. The
pathway is PERK to ATF4 to SUSD6 to STUB1 to NBR1 to the lysosome. A wrong
mechanism had arrived at roughly the right number, which is the kind of thing
that only shows up when the measurement does.

These tests pin the curve against the published points, and pin that the
parts which are still estimates keep saying so.
"""

import numpy as np
import pytest

from cognisom.core import antigen_presentation as ap


#: Estephan et al., EMBO J 2025, Fig 2C. HT29, 24 h, surface MHC I by flow
#: cytometry, normalised to 21% O2. Read from the figure; exact source data
#: are at BioStudies S-SCDT-10_1038-S44318-024-00319-7.
PUBLISHED = [(21.0, 1.00), (2.0, 0.96), (1.0, 0.75), (0.5, 0.52), (0.05, 0.37)]


def test_the_curve_reproduces_the_published_dose_response():
    """The whole point: this is fitted to data, not chosen."""
    worst = 0.0
    for percent, measured in PUBLISHED:
        modelled = float(ap.hypoxic_factor(percent / 100.0))
        worst = max(worst, abs(modelled - measured))
    assert worst < 0.03, (
        f"largest departure from the published curve is {worst:.3f}; the fit "
        "no longer reproduces the measurement it claims to come from")


def test_a_hyperbola_would_not_fit_and_that_is_why_n_is_not_one():
    """The correction that mattered, kept as a test so it cannot regress.

    A hyperbola cannot be flat near 2% O2 and steep near 0.5%. Forcing n = 1
    and refitting the other two constants still leaves it far worse, so the
    cooperativity is in the data and not an artefact of the fit.
    """
    import numpy as np

    o2 = np.array([p for p, _ in PUBLISHED])
    measured = np.array([m for _, m in PUBLISHED])

    hill = np.array([float(ap.hypoxic_factor(p / 100.0)) for p in o2])
    hill_rss = float(np.sum((hill - measured) ** 2))

    # Best hyperbola through the same points.
    best = min(
        float(np.sum((floor + (1 - floor) * o2 / (o2 + k) - measured) ** 2))
        for floor in np.linspace(0.0, 0.6, 61)
        for k in np.linspace(0.05, 5.0, 100))

    assert hill_rss < best / 10, (
        f"Hill fit {hill_rss:.5f} against the best hyperbola {best:.5f}; if "
        "these have converged the exponent is no longer earning its place")
    assert ap.HILL_COEFFICIENT > 1.5


def test_the_estimates_that_remain_keep_saying_so():
    """The baselines are not measured and the caveats carry that."""
    assert any("BASELINE" in note or "baseline" in note for note in ap.CAVEATS)
    assert any("B2M" in note for note in ap.CAVEATS), (
        "the rejected data source belongs in the caveats, or someone will "
        "reach for it again")
    assert any("8-24 h" in note or "instantaneously" in note
               for note in ap.CAVEATS), (
        "the unmodelled timescale is a real limit and has to travel too")

    text = ap.describe()
    assert "measured" in text
    doc = ap.__doc__ or ""
    assert "s44318-024-00319-7" in doc, "the DOI has to travel with the curve"
    assert "HIF" in doc, (
        "the module should record that the mechanism is NOT HIF-mediated, "
        "since its previous justification said the opposite")


def test_the_curve_is_bounded_and_monotone():
    oxygen = np.linspace(0.0, 0.25, 60)
    factor = ap.hypoxic_factor(oxygen)

    assert factor[0] == pytest.approx(ap.HYPOXIC_FLOOR)
    assert np.all(np.diff(factor) > 0), "must rise with oxygen"
    assert np.all(factor <= 1.0)
    assert factor[-1] < 1.0, "it approaches 1 and should not reach it"

    # Negative oxygen is clipped rather than producing a factor below the
    # floor: a solver can hand back a tiny negative on round-off.
    assert ap.hypoxic_factor(-1e-9) == pytest.approx(ap.HYPOXIC_FLOOR)


def test_the_half_point_is_where_it_claims():
    """At K the curve sits halfway between the floor and 1."""
    half = float(ap.hypoxic_factor(ap.HYPOXIC_HALF_O2))
    expected = ap.HYPOXIC_FLOOR + 0.5 * (1.0 - ap.HYPOXIC_FLOOR)
    assert half == pytest.approx(expected)


def test_it_is_not_a_function_of_the_respiratory_state():
    """The design claim, and the reason this is its own curve.

    The mitochondrial half-saturation is 1 mmHg, so a cell at 1% O2 is still
    respiring at nearly full rate while already losing surface MHC-I. If the
    two half-points were comparable, MHC-I could be derived from the ATP
    charge and this module would not need to exist.
    """
    from cognisom.core.tissue_config import DiffusionFieldConfig

    respiratory_km = DiffusionFieldConfig.oxygen().uptake_km
    assert ap.HYPOXIC_HALF_O2 > 5 * respiratory_km, (
        f"hypoxic half-point {ap.HYPOXIC_HALF_O2:.4f} against a respiratory "
        f"km of "
        f"{respiratory_km:.4f}; if these have converged, MHC-I and the ATP "
        "charge are the same curve and one of them is wrong")

    # Concretely: at the physioxia floor, respiration is untouched and
    # presentation has already gone. The measured curve makes this gap
    # SMALLER than the estimate it replaced -- 0.76 at 1% O2 where the guess
    # said 0.61 -- so the threshold here is what the data supports, not what
    # the old model predicted.
    from cognisom.core import metabolism
    atp, _lactate, feasible = metabolism.evaluate(
        np.array([0.01]), np.array([5.0]))
    assert feasible[0] and atp[0] > 0.99, "should still be making full ATP"
    presenting = float(ap.hypoxic_factor(0.01))
    assert presenting < 0.85, (
        f"presentation is {presenting:.2f} at 1% O2; the measurement puts it "
        "near 0.76, so a quarter is already lost while ATP is untouched")

    # And where ATP finally does fall, presentation is long gone.
    atp_anoxic, _l, _f = metabolism.evaluate(np.array([0.0]), np.array([5.0]))
    assert float(ap.hypoxic_factor(0.0)) < float(atp_anoxic[0])


def test_a_tumour_cell_is_harder_to_see_than_its_neighbour():
    """The evasion route, which is the point of the column."""
    oxygen = np.full(2, DiffusionFieldConfigOxygen())
    levels = ap.surface_level(oxygen, np.array([0, 5]))
    somatic, cancer = float(levels[0]), float(levels[1])
    assert cancer < somatic
    assert somatic / cancer == pytest.approx(
        ap.DEFAULT_BASELINE / ap.BASELINE[5], rel=1e-9)
    # Immune cells are not downregulated.
    immune = ap.surface_level(oxygen[:1], np.array([6]))
    assert float(immune[0]) > somatic


def DiffusionFieldConfigOxygen():
    from cognisom.core.tissue_config import DiffusionFieldConfig
    return DiffusionFieldConfig.oxygen().initial_value


def test_hypoxia_and_malignancy_compound():
    """A tumour cell in an anoxic core is suppressed twice over."""
    perfused = float(ap.surface_level(np.array([0.11]), np.array([5]))[0])
    anoxic = float(ap.surface_level(np.array([0.0]), np.array([5]))[0])
    assert anoxic < perfused
    assert anoxic / perfused == pytest.approx(
        ap.HYPOXIC_FLOOR / float(ap.hypoxic_factor(0.11)), rel=1e-9)
    # And it stays a level, not a probability gone negative.
    assert 0.0 < anoxic < 1.0


def test_an_unknown_archetype_gets_the_somatic_baseline():
    """Not the immune one, and not zero."""
    out = ap.baseline_for(np.array([999, 0, 5]))
    assert out[0] == ap.DEFAULT_BASELINE
    assert out[1] == ap.BASELINE[0]
    assert out[2] == ap.BASELINE[5]


def test_the_column_moves_in_a_run_and_tracks_the_oxygen():
    """End to end, where there is a gradient to respond to."""
    import contextlib
    import io as _io

    from cognisom.core.tissue_config import TissueScaleConfig
    from cognisom.core.tissue_engine import TissueSimulationEngine

    cfg = TissueScaleConfig(
        n_cells=800, grid_shape=(16, 16, 16),
        tissue_size_um=(160., 160., 160.), n_gpus=1, dt=0.05,
        duration=2.0, seed=11, enable_ode=True, enable_diffusion=True,
        enable_cell_cycle=False, enable_mechanics=True)
    cfg.fields["oxygen"].uptake_per_cell_per_hour *= 50.0
    original = TissueSimulationEngine.VESSEL_SPACING_UM
    TissueSimulationEngine.VESSEL_SPACING_UM = 160.0
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            engine = TissueSimulationEngine(cfg)
            engine.initialize()
            engine.run()
            arrays = engine._cell_arrays[0]
            state = np.asarray(arrays.state[:arrays.n_real])
            types = np.asarray(engine._backend.to_numpy(
                arrays.cell_types[:arrays.n_real]))
    finally:
        TissueSimulationEngine.VESSEL_SPACING_UM = original

    mhc1 = state[:, 6]
    oxygen = state[:, 0]
    assert mhc1.std() > 1e-6, "the column is uniform, so it is not coupled"
    assert np.all(mhc1 > 0) and np.all(mhc1 <= 1.0)

    # Within one archetype, MHC-I has to rise with oxygen -- MONOTONICALLY,
    # not linearly. The curve saturates, so a Pearson correlation caps out
    # well below 1 (measured 0.85) and testing it that way would be testing
    # the wrong relationship. Rank order is exact here.
    somatic = types == 0
    if somatic.sum() > 20 and oxygen[somatic].std() > 1e-6:
        order = np.argsort(oxygen[somatic])
        ranked = mhc1[somatic][order]
        assert np.all(np.diff(ranked) >= -1e-9), (
            "MHC-I must not fall as oxygen rises within one archetype")

    # And cancer cells sit below somatic cells at comparable oxygen.
    cancer = types == 5
    if cancer.sum() > 5 and somatic.sum() > 5:
        assert mhc1[cancer].mean() < mhc1[somatic].mean()
