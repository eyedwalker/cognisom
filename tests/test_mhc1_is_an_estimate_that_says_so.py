"""MHC-I is now computed, and every number behind it is a guess.

``mhc1`` had consumers and no producer: ``cell_renderer`` offers
``color_by="mhc1"``, ``exporters`` writes an ``mhc1_expression`` column, and
both were fed by synthetic demo values. It is now computed from the archetype
and the local oxygen.

It was written under an explicit instruction to accept a labelled estimate if
no effect size could be sourced, and none could. The direction and mechanism
are solid -- hypoxia suppresses MHC-I HIF-dependently and also downregulates
TAP1/2 and LMP7 (Sethumadhavan et al., PLoS One 2017, PMID 29155844, p = 0.002
to 0.013) -- but that paper reports magnitudes only inside figure images, and
the one study with an oxygen dose-response is behind a publisher login.

So these tests do two jobs. They pin the shape of the curve, which is the part
with a rationale. And they check that the estimate keeps announcing itself,
because the failure mode for a number like this is not being wrong, it is
being quoted later as though it were measured.
"""

import numpy as np
import pytest

from cognisom.core import antigen_presentation as ap


def test_every_number_is_declared_an_estimate():
    """The caveats have to travel with the values, in the module and in
    ``describe``, because this sets a simulated immune-escape rate."""
    assert any("ESTIMATED" in note or "estimate" in note.lower()
               for note in ap.CAVEATS)
    assert any("29155844" in note or "effect size" in note
               for note in ap.CAVEATS)
    assert any("B2M" in note for note in ap.CAVEATS), (
        "the rejected data source belongs in the caveats, or someone will "
        "reach for it again")

    text = ap.describe()
    for fragment in ("ESTIMATED", "assumption, not a result"):
        assert fragment in text, text

    # And the module docstring has to carry the PMID and the admission.
    doc = ap.__doc__ or ""
    assert "29155844" in doc
    assert "ESTIMATE" in doc


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
    """At km the curve sits halfway between the floor and 1."""
    half = float(ap.hypoxic_factor(ap.HIF_HALF_O2))
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
    assert ap.HIF_HALF_O2 > 5 * respiratory_km, (
        f"HIF half-point {ap.HIF_HALF_O2:.4f} against a respiratory km of "
        f"{respiratory_km:.4f}; if these have converged, MHC-I and the ATP "
        "charge are the same curve and one of them is wrong")

    # Concretely: at the physioxia floor, respiration is near-maximal and
    # presentation is already well down.
    from cognisom.core import metabolism
    atp, _lactate, feasible = metabolism.evaluate(
        np.array([0.01]), np.array([5.0]))
    assert feasible[0] and atp[0] > 0.9, "should still be making ATP at 1% O2"
    assert float(ap.hypoxic_factor(0.01)) < 0.7, "and already presenting less"


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
