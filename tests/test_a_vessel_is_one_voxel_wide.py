"""The field resolution and the capillary diameter are the same number.

A vessel is represented by marking whole voxels as Dirichlet boundaries, so a
capillary in this model is exactly one voxel across. At the shipped 10 um
resolution that is about right -- a real capillary is 5-10 um -- which is a
piece of luck rather than a design, and it has a consequence that is easy to
miss:

**the field grid cannot be coarsened without changing the vasculature.**

This matters because coarsening is the obvious answer to a real cost problem.
Solving oxygen and glucose on the 500^3 full_tissue preset takes about 33 h
against a 2 h budget, and oxygen varies on the 60 um intervessel scale while
being resolved at 10 um, so halving the resolution looks like an 8x saving for
no physical loss. It is not. Halving the resolution doubles the capillary
diameter and quadruples the vessel volume fraction, which changes the supply,
which changes the answer.

Measured on a fixed 480 um box with vessels every 60 um and the VOLUMETRIC
demand held constant:

    dx     grid    vessel fraction   depletion
    5 um   96^3    0.69%             4.17e-03
    10 um  48^3    2.78%             3.73e-03
    20 um  24^3    11.11%            2.07e-03
    30 um  16^3    25.00%            3.85e-03

The depletion is not converging as dx falls, and at 30 um it moves back the
other way -- the signature of a confounded comparison rather than a
discretisation error. What varies is the vasculature.

(Two traps on the way to that table, both worth naming. The reaction term is a
VOLUMETRIC rate, concentration per second, so holding the physical demand
fixed means holding Vmax constant; scaling it by the voxel volume instead
changes the demand 8x per halving and manufactured an apparent 615% resolution
error. And there is no way to hold the vessel geometry fixed while changing
dx, which is the point of this file.)

Decoupling them needs a sub-voxel vessel representation -- a Dirichlet
condition or source weighted by the vessel surface area within each voxel,
rather than whole marked voxels. Until that exists, 10 um is not a free
parameter.
"""

import numpy as np
import pytest

from cognisom.core.tissue_engine import TissueSimulationEngine
from cognisom.gpu.steady_diffusion import INSULATED, solve_steady_state

BOX_UM = 240.0
VESSEL_SPACING_UM = 60.0
D = 2000.0
ARTERIAL = 0.21
KM = 1.0 / 760.0
#: Concentration per second. Volumetric, so it does NOT scale with dx.
VMAX = (0.15 / 3600.0) * 100.0


def _lattice(dx):
    """The engine's own vessel placement, at this resolution."""
    n = int(round(BOX_UM / dx))
    stride = max(1, int(round(VESSEL_SPACING_UM / dx)))
    vessels = np.zeros((n, n, n), bool)
    offset = stride // 2
    vessels[offset::stride, offset::stride, :] = True
    return vessels


def _solve(dx):
    vessels = _lattice(dx)
    n = vessels.shape[0]
    up = np.full((n, n, n), VMAX / ARTERIAL)
    up[vessels] = 0.0
    result = solve_steady_state(
        np.full((n,) * 3, ARTERIAL), D, dx, uptake_k=up, uptake_km=KM,
        uptake_ref=ARTERIAL, held=vessels, held_value=ARTERIAL,
        faces=INSULATED)
    return vessels, result


def test_the_vessel_fraction_is_set_by_the_resolution():
    """Not by the vasculature. This is the whole point.

    A capillary one voxel wide occupies (dx/spacing)^2 of the volume for
    vessels running as lines, so the fraction grows as dx^2 while the physical
    vasculature is unchanged.
    """
    fractions = {}
    for dx in (5.0, 10.0, 20.0):
        vessels = _lattice(dx)
        fractions[dx] = float(vessels.mean())

    # Each halving of dx quarters the vessel volume, to within the rounding
    # of a lattice onto an integer grid.
    assert fractions[10.0] / fractions[5.0] == pytest.approx(4.0, rel=0.35), (
        fractions)
    assert fractions[20.0] / fractions[10.0] == pytest.approx(4.0, rel=0.35), (
        fractions)
    assert fractions[20.0] > 0.08, (
        f"at 20 um the vasculature would be {100 * fractions[20.0]:.0f}% of "
        "the tissue by volume, which is an order above any real capillary bed"
    )


def test_the_shipped_resolution_is_the_one_that_makes_a_capillary_physical():
    """10 um per voxel, and a capillary is 5-10 um across.

    So the vessel width is right at the shipped resolution and wrong at any
    coarser one. If the default ever moves, this is the reason it cannot move
    upward without a sub-voxel vessel model.
    """
    from cognisom.core.tissue_config import TissueScaleConfig

    resolution = TissueScaleConfig.small_test().resolution_um
    assert resolution == pytest.approx(10.0)
    assert 5.0 <= resolution <= 10.0, (
        f"a vessel is one voxel wide, so {resolution} um per voxel makes "
        "every capillary that wide"
    )
    # And the spacing is resolvable at that resolution.
    assert TissueSimulationEngine.VESSEL_SPACING_UM / resolution >= 3, (
        "fewer than three voxels between vessels cannot carry a profile"
    )


def test_coarsening_the_grid_changes_the_answer():
    """The claim this file exists to refute: that coarsening is free.

    Held fixed: the box, the vessel spacing, and the volumetric demand. Varied:
    dx alone. If coarsening were free the depletion would be stable; it is not,
    and it does not even move monotonically, because what actually changes is
    how much of the tissue is capillary.
    """
    depletions = {}
    for dx in (5.0, 10.0, 20.0):
        _, result = _solve(dx)
        assert result.converged, dx
        depletions[dx] = result.depletion

    fine, coarse = depletions[5.0], depletions[20.0]
    assert abs(coarse - fine) / fine > 0.2, (
        f"coarsening 5 um -> 20 um moved the depletion only "
        f"{abs(coarse - fine) / fine:.1%}; if it is now stable, a sub-voxel "
        "vessel model may have landed and this test should be revisited"
    )


def test_the_volumetric_demand_must_not_be_scaled_by_the_voxel():
    """The trap that faked a 615% resolution error.

    The reaction term is a concentration per second. Multiplying it by the
    voxel volume -- a natural-looking way to "hold the demand per unit volume
    fixed" -- changes the demand by 8x for every halving of dx, so the
    resulting depletion difference is the demand and not the grid.
    """
    results = {}
    for dx in (10.0, 20.0):
        vessels = _lattice(dx)
        n = vessels.shape[0]
        # The WRONG scaling, reproduced on purpose.
        up = np.full((n, n, n), VMAX * (dx ** 3) / 1000.0 / ARTERIAL)
        up[vessels] = 0.0
        results[dx] = solve_steady_state(
            np.full((n,) * 3, ARTERIAL), D, dx, uptake_k=up, uptake_km=KM,
            uptake_ref=ARTERIAL, held=vessels, held_value=ARTERIAL,
            faces=INSULATED).depletion

    inflated = results[20.0] / results[10.0]
    assert inflated > 2.0, (
        f"the wrong scaling should inflate the coarse depletion a lot; got "
        f"{inflated:.2f}x. If it no longer does, the warning above is stale"
    )
