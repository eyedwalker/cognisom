"""A vessel's radius enters through its conductance, not its voxel count.

``held`` marks whole voxels as Dirichlet boundaries, so a capillary is exactly
one voxel wide and its physical diameter IS the grid resolution. Coarsening
the grid fattens every capillary and raises the vessel volume fraction as
dx^2, which is why ``test_a_vessel_is_one_voxel_wide`` had to record that a
coarser field grid is not free.

``exchange`` is the alternative. Inside a voxel the field around a cylinder is
the Krogh solution -- radial and logarithmic -- so the flux per unit vessel
length from a surface at radius r to a far field at radius R is
2 pi D (C_blood - C) / ln(R/r). Taking R as the voxel half-width and
multiplying by vessel length per unit volume gives a conductance per volume:

    beta = 2 pi D * lambda / ln(dx / 2r)          (``krogh_exchange``)

which depends on the grid only logarithmically where the whole-voxel condition
depended on it linearly, and in which r appears explicitly.

Two things this does NOT do, both measured below. It is not grid-independent,
only much less grid-dependent -- the far-field-at-the-half-width assumption is
itself approximate. And it does not unlock coarse grids at NORMAL capillary
spacing: at 60 um spacing, dx of 10/20/30 um leaves 6/3/2 voxels between
vessels, and under-resolving the profile swamps the vessel model entirely. The
grid is constrained by resolving the intervessel profile, and that constraint
binds first unless the vasculature is sparse.
"""

import numpy as np
import pytest

from cognisom.core.tissue_config import (
    CAPILLARY_PO2, OXYGEN_Q_PER_HOUR, DiffusionFieldConfig,
)
from cognisom.gpu.steady_diffusion import (
    INSULATED, krogh_exchange, solve_steady_state,
)

#: A capillary. 2.5-4 um is the usual range.
RADIUS_UM = 3.0
#: All read from the config. Literals here are second copies of calibrated
#: numbers, and they go stale silently: BLOOD was pinned at 0.11 -- the top of
#: tissue physioxia -- which is not what a capillary carries.
D = DiffusionFieldConfig.oxygen().diffusion_coeff
BLOOD = CAPILLARY_PO2
KM = DiffusionFieldConfig.oxygen().uptake_km
#: Volumetric demand, concentration per second. NOT scaled by voxel volume --
#: doing that changes the physical demand 8x per halving of dx.
VMAX = OXYGEN_Q_PER_HOUR / 3600.0  # read from config: a literal here is a second copy


def _lattice(dx, spacing_um, box_um):
    n = int(round(box_um / dx))
    stride = int(round(spacing_um / dx))
    mask = np.zeros((n, n, n), bool)
    mask[stride // 2::stride, stride // 2::stride, :] = True
    return mask


def _depletion(dx, spacing_um, box_um, sub_voxel):
    """Blood level minus the lowest tissue value, which is what matters.

    Reported as a depletion and not as a field minimum on purpose: comparing
    minima hides the error. Two solves whose minima differ by 0.6% of the
    field can differ by 2x in the depletion, and the depletion is the quantity
    that decides whether a cell is hypoxic.
    """
    mask = _lattice(dx, spacing_um, box_um)
    n = mask.shape[0]
    uptake = np.full((n,) * 3, VMAX / BLOOD)
    if sub_voxel:
        length_per_volume = np.where(mask, 1.0 / (dx * dx), 0.0)
        extra = dict(
            exchange=krogh_exchange(length_per_volume, RADIUS_UM, dx, D),
            exchange_value=BLOOD)
        free = np.ones_like(mask)
    else:
        uptake[mask] = 0.0
        extra = dict(held=mask, held_value=BLOOD)
        free = ~mask
    result = solve_steady_state(
        np.full((n,) * 3, BLOOD), D, dx, uptake_k=uptake, uptake_km=KM,
        uptake_ref=BLOOD, faces=INSULATED, **extra)
    assert result.converged, (dx, spacing_um, sub_voxel)
    return BLOOD - float(result.field[free].min())


def test_the_conductance_carries_the_radius_and_the_grid_only_in_a_log():
    """beta ~ 1/ln(dx/2r): the formula, checked against itself."""
    dx = 20.0
    lam = 1.0 / (dx * dx)
    beta = krogh_exchange(lam, RADIUS_UM, dx, D)
    expected = 2.0 * np.pi * D * lam / np.log((dx / 2) / RADIUS_UM)
    assert beta == pytest.approx(expected)

    # Halving the radius does NOT halve the conductance -- it enters in a log,
    # which is the physics and the reason a thin vessel is not a weak one.
    thinner = krogh_exchange(lam, RADIUS_UM / 2, dx, D)
    assert thinner < beta
    assert thinner > 0.5 * beta


def test_a_vessel_that_fills_its_voxel_is_refused_not_approximated():
    """At r = dx/2 the log vanishes and the conductance diverges.

    That limit is the Dirichlet condition, which the solver already has, so the
    honest response is to say so rather than return an infinity.
    """
    with pytest.raises(ValueError, match="does not fit inside"):
        krogh_exchange(1.0, 5.0, 10.0, D)        # r = half-width
    with pytest.raises(ValueError, match="does not fit inside"):
        krogh_exchange(1.0, 8.0, 10.0, D)        # r > half-width
    for bad in ((0.0, 10.0), (-1.0, 10.0)):
        with pytest.raises(ValueError, match="radius_um must be positive"):
            krogh_exchange(1.0, bad[0], bad[1], D)


def test_the_two_models_agree_where_they_describe_the_same_vessel():
    """The consistency check that makes the closure believable.

    At dx = 2r the whole-voxel capillary is exactly the true diameter, so the
    two models describe the same vessel and must give the same answer. They
    then diverge monotonically as dx/2r grows, which is precisely where the
    whole-voxel radius departs from the truth.
    """
    # 120 um apart, not 240. At the corrected consumption rate the sparser
    # lattice is fully anoxic -- BOTH models return a depletion equal to the
    # whole supply, "agree" perfectly, and the comparison measures nothing.
    # A degenerate test passes for the wrong reason, so the geometry has to
    # keep a real gradient in it.
    spacing, box = 120.0, 240.0
    gaps = {}
    for dx in (240 / 36, 8.0, 20.0, 40.0):
        whole = _depletion(dx, spacing, box, sub_voxel=False)
        sub = _depletion(dx, spacing, box, sub_voxel=True)
        gaps[dx / (2 * RADIUS_UM)] = abs(whole - sub) / sub

    ratios = sorted(gaps)
    assert gaps[ratios[0]] < 0.05, (
        f"at dx/2r = {ratios[0]:.2f} the models should agree closely; they "
        f"differ by {gaps[ratios[0]]:.1%}")
    # Monotone in dx/2r.
    values = [gaps[r] for r in ratios]
    assert values == sorted(values), gaps
    assert gaps[ratios[-1]] > 0.2, (
        "and they should disagree badly on a coarse grid, or the whole-voxel "
        "model was never the problem")


def test_the_radius_survives_coarsening_better_than_the_voxel_does():
    """Sparse vasculature, where a coarser grid is actually wanted.

    Also 120 um rather than 240, and for the same reason as above: at the
    corrected consumption rate 240 um leaves nothing to resolve.
    """
    spacing, box = 120.0, 240.0
    fine_w = _depletion(10.0, spacing, box, sub_voxel=False)
    fine_s = _depletion(10.0, spacing, box, sub_voxel=True)

    worst_w = worst_s = 0.0
    for dx in (20.0, 30.0, 40.0):
        worst_w = max(worst_w, abs(_depletion(dx, spacing, box, False)
                                   - fine_w) / fine_w)
        worst_s = max(worst_s, abs(_depletion(dx, spacing, box, True)
                                   - fine_s) / fine_s)

    assert worst_s < worst_w / 1.5, (
        f"sub-voxel drifts {worst_s:.1%} against whole-voxel {worst_w:.1%}; "
        "the closure should be markedly more stable or it is not worth it")
    # And it is NOT grid-independent. Saying so here stops the claim growing.
    assert worst_s > 0.02, (
        f"sub-voxel drift is {worst_s:.1%}; if it is now negligible the "
        "docstring's caveat is stale and should be removed")


def test_it_does_not_rescue_a_grid_that_cannot_resolve_the_profile():
    """At normal capillary spacing the profile constraint binds first.

    60 um apart and dx = 20 um leaves three voxels between vessels. No vessel
    model fixes that, and this is the reason sub-voxel vessels do not unlock
    coarse grids for normal tissue -- only for sparse, tumour-like
    vasculature.
    """
    spacing, box = 60.0, 240.0
    fine = _depletion(10.0, spacing, box, sub_voxel=True)
    coarse = _depletion(20.0, spacing, box, sub_voxel=True)
    drift = abs(coarse - fine) / fine
    assert drift > 0.1, (
        f"only {drift:.1%} drift at three voxels per intervessel gap; if the "
        "solver now copes with that, this file's conclusion is too pessimistic"
    )


def test_exchange_defaults_off_and_validates_its_input():
    """A field with no exchange must solve exactly as before."""
    import inspect

    params = inspect.signature(solve_steady_state).parameters
    assert params["exchange"].default == 0.0
    assert params["exchange_value"].default == 0.0

    n = 12
    field = np.full((n,) * 3, BLOOD)
    with pytest.raises(ValueError, match="does not match field shape"):
        solve_steady_state(field, D, 10.0, exchange=np.zeros((4, 4, 4)))
    with pytest.raises(ValueError, match="exchange must be non-negative"):
        solve_steady_state(field, D, 10.0, exchange=-1.0)


def test_exchange_drives_the_field_towards_the_blood_level():
    """The sign and the limit, on a problem with an obvious answer.

    No sink, no Dirichlet voxels, uniform exchange: the only steady state is
    the blood level everywhere, whatever the field started at.
    """
    n = 12
    result = solve_steady_state(
        np.zeros((n,) * 3), D, 10.0, exchange=1.0, exchange_value=BLOOD,
        faces=INSULATED)
    assert result.converged
    assert np.allclose(result.field, BLOOD, atol=1e-9), (
        result.field.min(), result.field.max())
